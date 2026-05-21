import torch
from torch import nn
from torch.nn.init import xavier_uniform_
from torch.nn.init import constant_
import math
import torch.nn.functional as F
from enum import IntEnum
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Dim(IntEnum):
    batch = 0
    seq = 1
    feature = 2


class AKT(nn.Module):
    def __init__(
        self,
        n_question,
        n_pid,
        d_model,
        n_blocks,
        dropout,
        d_ff=256,
        kq_same=1,
        final_fc_dim=512,
        num_attn_heads=8,
        separate_qa=False,
        l2=1e-5,
        emb_type="qid",
        emb_path="",
        pretrain_dim=768,
    ):
        super().__init__()

        """
        Input:
            d_model: dimension of attention block
            final_fc_dim: dimension of final fully connected net before prediction
            num_attn_heads: number of heads in multi-headed attention
            d_ff: dimension for fully connected net inside the basic block
            kq_same: if key query same, kq_same=1, else = 0

        emb_type:
            qid:
                Standard AKT qid branch.
                If n_pid > 0, it uses Rasch-style problem difficulty offset.

            qid_pdiff:
                Same as qid, but pid difficulty is also injected into attention decay.

            pure_kc:
                Pure KC ablation.
                Completely removes question/problem difficulty offset.
                One KC embedding represents all questions under that KC.
                pid_data is ignored even when n_pid > 0.

            pure_question:
                Pure question ablation.
                Completely removes KC input and problem difficulty offset.
                It uses pid_data / question id sequence as the base embedding id.
                This is useful for comparing question-only memorization against
                KC-level sharing under sparse questions.
        """

        self.model_name = "akt"
        self.n_question = n_question
        self.dropout = dropout
        self.kq_same = kq_same
        self.n_pid = n_pid
        self.l2 = l2
        self.model_type = self.model_name
        self.separate_qa = separate_qa
        self.emb_type = emb_type

        embed_l = d_model

        # ------------------------------------------------------------
        # Ablation switches
        # ------------------------------------------------------------
        self.is_pure_kc = emb_type == "pure_kc"
        self.is_pure_question = emb_type == "pure_question"

        # In pure_kc / pure_question, we deliberately disable all
        # problem/question difficulty offsets.
        #
        # pure_kc:
        #     q_data is treated as KC/concept id.
        #
        # pure_question:
        #     pid_data is treated as question/problem id and becomes the
        #     base embedding id. q_data/KC id is ignored in forward().
        self.use_pid_difficulty = (
            (self.n_pid > 0)
            and (not self.is_pure_kc)
            and (not self.is_pure_question)
        )

        # Base embedding vocabulary size.
        # For qid / pure_kc, q_data is KC/concept id, so use n_question.
        # For pure_question, pid_data is question/problem id, so use n_pid + 1.
        # The +1 is consistent with AKT's difficult_param indexing and is
        # safer when question ids use 0 as padding or are 1-based.
        if self.is_pure_question:
            if self.n_pid <= 0:
                raise ValueError(
                    "pure_question requires n_pid > 0 because it uses "
                    "pid_data / question id as the base embedding id."
                )
            self.base_n = self.n_pid + 1
        else:
            self.base_n = self.n_question

        # ------------------------------------------------------------
        # Difficulty / Rasch-style offset branch
        # Only used by qid / qid_pdiff when n_pid > 0.
        # Not used by pure_kc.
        # ------------------------------------------------------------
        if self.use_pid_difficulty:
            self.difficult_param = nn.Embedding(self.n_pid + 1, 1)

            # KC/question-level direction for problem difficulty offset.
            self.q_embed_diff = nn.Embedding(self.n_question + 1, embed_l)

            # Interaction-level direction for problem difficulty offset.
            self.qa_embed_diff = nn.Embedding(2 * self.n_question + 1, embed_l)

        # ------------------------------------------------------------
        # Base KC/question embedding branch.
        #
        # qid / pure_kc:
        #     q_data is used as the base id, usually KC/concept id.
        #
        # pure_question:
        #     pid_data is used as the base id in forward(), so the embedding
        #     table size is n_pid + 1.
        # ------------------------------------------------------------
        if emb_type.startswith("qid") or self.is_pure_kc or self.is_pure_question:
            self.q_embed = nn.Embedding(self.base_n, embed_l)

            if self.separate_qa:
                self.qa_embed = nn.Embedding(2 * self.base_n + 1, embed_l)
            else:
                self.qa_embed = nn.Embedding(2, embed_l)
        else:
            raise ValueError(
                f"Unsupported emb_type={emb_type!r}. "
                "Supported examples: 'qid', 'qid_pdiff', 'qid_avgpool', "
                "'qid_linear', 'pure_kc', 'pure_question'."
            )

        # Architecture Object. It contains stack of attention blocks.
        self.model = Architecture(
            n_question=n_question,
            n_blocks=n_blocks,
            n_heads=num_attn_heads,
            dropout=dropout,
            d_model=d_model,
            d_feature=d_model / num_attn_heads,
            d_ff=d_ff,
            kq_same=self.kq_same,
            model_type=self.model_type,
            emb_type=self.emb_type,
        )

        self.out = nn.Sequential(
            nn.Linear(d_model + embed_l, final_fc_dim),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(final_fc_dim, 256),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(256, 1),
        )

        self.reset()

    def reset(self):
        """
        Original code zeroed any parameter whose first dimension equals n_pid+1.
        That can accidentally zero unrelated embeddings in some edge cases.

        Here we only zero difficult_param when it actually exists.
        pure_kc / pure_question do not create difficult_param, so nothing is
        zeroed there.
        """
        if self.use_pid_difficulty:
            torch.nn.init.constant_(self.difficult_param.weight, 0.0)

    def base_emb(self, q_data, target):
        """
        q_data:
            Base id sequence.

            For qid / pure_kc:
                this should be the KC/concept id sequence.

            For pure_question:
                this function receives pid_data/question id sequence from
                forward(), not the original KC sequence.

        target:
            response sequence, usually 0/1.

        For pure_kc:
            q_embed_data = KC embedding
            qa_embed_data = KC embedding + response embedding
            No problem/question offset will be added in forward().

        For pure_question:
            q_embed_data = question embedding
            qa_embed_data = question embedding + response embedding
            No KC embedding and no problem/question offset will be used.
        """
        q_embed_data = self.q_embed(q_data)  # [batch, seq_len, d_model]

        if self.separate_qa:
            qa_data = q_data + self.base_n * target
            qa_embed_data = self.qa_embed(qa_data)
        else:
            qa_embed_data = self.qa_embed(target) + q_embed_data

        return q_embed_data, qa_embed_data

    def forward(self, q_data, target, pid_data=None, qtest=False):
        emb_type = self.emb_type

        # ------------------------------------------------------------
        # Base embedding.
        #
        # qid / pure_kc:
        #     use q_data as the base id, usually KC/concept id.
        #
        # pure_question:
        #     ignore q_data/KC id and use pid_data/question id as the base id.
        # ------------------------------------------------------------
        if self.is_pure_question:
            if pid_data is None:
                raise ValueError(
                    "pure_question requires pid_data because it uses "
                    "question/problem ids as the base embedding sequence."
                )
            q_embed_data, qa_embed_data = self.base_emb(pid_data, target)
        elif emb_type.startswith("qid") or self.is_pure_kc:
            q_embed_data, qa_embed_data = self.base_emb(q_data, target)
        else:
            raise ValueError(f"Unsupported emb_type={emb_type!r}")

        # ------------------------------------------------------------
        # pid_embed_data:
        #     qid / qid_pdiff with n_pid > 0:
        #         use problem difficulty offset.
        #
        #     pure_kc / pure_question:
        #         always None.
        #         This completely removes question/problem-specific shift.
        # ------------------------------------------------------------
        pid_embed_data = None

        if self.use_pid_difficulty:
            if pid_data is None:
                raise ValueError(
                    "pid_data must be provided when n_pid > 0 and emb_type is not 'pure_kc'."
                )

            q_embed_diff_data = self.q_embed_diff(q_data)
            pid_embed_data = self.difficult_param(pid_data)

            # q embedding with problem difficulty offset:
            # c_q + u_p * d_q
            q_embed_data = q_embed_data + pid_embed_data * q_embed_diff_data

            qa_embed_diff_data = self.qa_embed_diff(target)

            if self.separate_qa:
                qa_embed_data = qa_embed_data + pid_embed_data * qa_embed_diff_data
            else:
                qa_embed_data = qa_embed_data + pid_embed_data * (
                    qa_embed_diff_data + q_embed_diff_data
                )

            c_reg_loss = (pid_embed_data ** 2.0).sum() * self.l2
        else:
            # pure_kc / pure_question reach here even when self.n_pid > 0.
            # This is intentional: no problem/question difficulty offset at all.
            c_reg_loss = q_embed_data.new_tensor(0.0)

        # ------------------------------------------------------------
        # Pass to AKT architecture.
        # For pure_kc / pure_question, pid_embed_data is None, so pdiff cannot
        # affect attention.
        # ------------------------------------------------------------
        d_output = self.model(q_embed_data, qa_embed_data, pid_embed_data)

        concat_q = torch.cat([d_output, q_embed_data], dim=-1)
        output = self.out(concat_q).squeeze(-1)
        preds = torch.sigmoid(output)

        if not qtest:
            return preds, c_reg_loss
        else:
            return preds, c_reg_loss, concat_q


class Architecture(nn.Module):
    def __init__(
        self,
        n_question,
        n_blocks,
        d_model,
        d_feature,
        d_ff,
        n_heads,
        dropout,
        kq_same,
        model_type,
        emb_type,
    ):
        super().__init__()

        self.d_model = d_model
        self.model_type = model_type

        if model_type in {"akt"}:
            self.blocks_1 = nn.ModuleList(
                [
                    TransformerLayer(
                        d_model=d_model,
                        d_feature=d_model // n_heads,
                        d_ff=d_ff,
                        dropout=dropout,
                        n_heads=n_heads,
                        kq_same=kq_same,
                        emb_type=emb_type,
                    )
                    for _ in range(n_blocks)
                ]
            )

            self.blocks_2 = nn.ModuleList(
                [
                    TransformerLayer(
                        d_model=d_model,
                        d_feature=d_model // n_heads,
                        d_ff=d_ff,
                        dropout=dropout,
                        n_heads=n_heads,
                        kq_same=kq_same,
                        emb_type=emb_type,
                    )
                    for _ in range(n_blocks * 2)
                ]
            )

    def forward(self, q_embed_data, qa_embed_data, pid_embed_data):
        qa_pos_embed = qa_embed_data
        q_pos_embed = q_embed_data

        y = qa_pos_embed
        x = q_pos_embed

        # Encoder: encode historical QA information.
        for block in self.blocks_1:
            y = block(
                mask=1,
                query=y,
                key=y,
                values=y,
                pdiff=pid_embed_data,
            )

        flag_first = True

        for block in self.blocks_2:
            if flag_first:
                # Peek current question/KC.
                x = block(
                    mask=1,
                    query=x,
                    key=x,
                    values=x,
                    apply_pos=False,
                    pdiff=pid_embed_data,
                )
                flag_first = False
            else:
                # Do not peek current response.
                x = block(
                    mask=0,
                    query=x,
                    key=x,
                    values=y,
                    apply_pos=True,
                    pdiff=pid_embed_data,
                )
                flag_first = True

        return x


class TransformerLayer(nn.Module):
    def __init__(
        self,
        d_model,
        d_feature,
        d_ff,
        n_heads,
        dropout,
        kq_same,
        emb_type,
    ):
        super().__init__()

        kq_same = kq_same == 1

        self.masked_attn_head = MultiHeadAttention(
            d_model,
            d_feature,
            n_heads,
            dropout,
            kq_same=kq_same,
            emb_type=emb_type,
        )

        self.layer_norm1 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)

        self.linear1 = nn.Linear(d_model, d_ff)
        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_ff, d_model)

        self.layer_norm2 = nn.LayerNorm(d_model)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, mask, query, key, values, apply_pos=True, pdiff=None):
        """
        mask:
            0 means attention can only see past values.
            1 means attention can see current and past values.
        """

        seqlen = query.size(1)

        nopeek_mask = np.triu(
            np.ones((1, 1, seqlen, seqlen)),
            k=mask,
        ).astype("uint8")

        src_mask = (torch.from_numpy(nopeek_mask) == 0).to(query.device)

        if mask == 0:
            query2 = self.masked_attn_head(
                query,
                key,
                values,
                mask=src_mask,
                zero_pad=True,
                pdiff=pdiff,
            )
        else:
            query2 = self.masked_attn_head(
                query,
                key,
                values,
                mask=src_mask,
                zero_pad=False,
                pdiff=pdiff,
            )

        query = query + self.dropout1(query2)
        query = self.layer_norm1(query)

        if apply_pos:
            query2 = self.linear2(
                self.dropout(
                    self.activation(
                        self.linear1(query)
                    )
                )
            )
            query = query + self.dropout2(query2)
            query = self.layer_norm2(query)

        return query


class MultiHeadAttention(nn.Module):
    def __init__(
        self,
        d_model,
        d_feature,
        n_heads,
        dropout,
        kq_same,
        bias=True,
        emb_type="qid",
    ):
        super().__init__()

        self.d_model = d_model
        self.emb_type = emb_type

        if emb_type.endswith("avgpool"):
            pool_size = 3
            self.pooling = nn.AvgPool1d(
                pool_size,
                stride=1,
                padding=pool_size // 2,
                count_include_pad=False,
            )
            self.out_proj = nn.Linear(d_model, d_model, bias=bias)

        elif emb_type.endswith("linear"):
            self.linear = nn.Linear(d_model, d_model, bias=bias)
            self.out_proj = nn.Linear(d_model, d_model, bias=bias)

        elif emb_type.startswith("qid") or emb_type in {"pure_kc", "pure_question"}:
            self.d_k = d_feature
            self.h = n_heads
            self.kq_same = kq_same

            self.v_linear = nn.Linear(d_model, d_model, bias=bias)
            self.k_linear = nn.Linear(d_model, d_model, bias=bias)

            if kq_same is False:
                self.q_linear = nn.Linear(d_model, d_model, bias=bias)

            self.dropout = nn.Dropout(dropout)
            self.proj_bias = bias
            self.out_proj = nn.Linear(d_model, d_model, bias=bias)

            self.gammas = nn.Parameter(torch.zeros(n_heads, 1, 1))
            torch.nn.init.xavier_uniform_(self.gammas)

            self._reset_parameters()

        else:
            raise ValueError(
                f"Unsupported emb_type={emb_type!r} in MultiHeadAttention."
            )

    def _reset_parameters(self):
        xavier_uniform_(self.k_linear.weight)
        xavier_uniform_(self.v_linear.weight)

        if self.kq_same is False:
            xavier_uniform_(self.q_linear.weight)

        if self.proj_bias:
            constant_(self.k_linear.bias, 0.0)
            constant_(self.v_linear.bias, 0.0)

            if self.kq_same is False:
                constant_(self.q_linear.bias, 0.0)

            constant_(self.out_proj.bias, 0.0)

    def forward(self, q, k, v, mask, zero_pad, pdiff=None):
        bs = q.size(0)

        if self.emb_type.endswith("avgpool"):
            scores = self.pooling(v)
            concat = self.pad_zero(scores, bs, scores.shape[2], zero_pad)

        elif self.emb_type.endswith("linear"):
            scores = self.linear(v)
            concat = self.pad_zero(scores, bs, scores.shape[2], zero_pad)

        elif self.emb_type.startswith("qid") or self.emb_type in {"pure_kc", "pure_question"}:
            k = self.k_linear(k).view(bs, -1, self.h, self.d_k)

            if self.kq_same is False:
                q = self.q_linear(q).view(bs, -1, self.h, self.d_k)
            else:
                q = self.k_linear(q).view(bs, -1, self.h, self.d_k)

            v = self.v_linear(v).view(bs, -1, self.h, self.d_k)

            k = k.transpose(1, 2)
            q = q.transpose(1, 2)
            v = v.transpose(1, 2)

            gammas = self.gammas

            # Only emb_type containing "pdiff" uses problem difficulty
            # inside attention decay.
            #
            # pure_kc / pure_question do not contain "pdiff", and
            # pid_embed_data is None, so this branch completely disables
            # question difficulty in attention.
            if self.emb_type.find("pdiff") == -1:
                pdiff = None

            scores = attention(
                q,
                k,
                v,
                self.d_k,
                mask,
                self.dropout,
                zero_pad,
                gammas,
                pdiff,
            )

            concat = (
                scores.transpose(1, 2)
                .contiguous()
                .view(bs, -1, self.d_model)
            )

        else:
            raise ValueError(
                f"Unsupported emb_type={self.emb_type!r} in MultiHeadAttention.forward()."
            )

        output = self.out_proj(concat)

        return output

    def pad_zero(self, scores, bs, dim, zero_pad):
        if zero_pad:
            pad_zero = torch.zeros(
                bs,
                1,
                dim,
                device=scores.device,
                dtype=scores.dtype,
            )
            scores = torch.cat([pad_zero, scores[:, 0:-1, :]], dim=1)

        return scores


def attention(q, k, v, d_k, mask, dropout, zero_pad, gamma=None, pdiff=None):
    """
    Multi-head attention with AKT-style distance decay.

    For pure_kc / pure_question:
        pdiff is always None.
        Therefore only standard distance decay is used.
    """

    scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(d_k)

    bs, head, seqlen = scores.size(0), scores.size(1), scores.size(2)

    x1 = torch.arange(seqlen, device=scores.device).expand(seqlen, -1)
    x2 = x1.transpose(0, 1).contiguous()

    with torch.no_grad():
        scores_ = scores.masked_fill(mask == 0, -1e32)
        scores_ = F.softmax(scores_, dim=-1)
        scores_ = scores_ * mask.float().to(scores.device)

        distcum_scores = torch.cumsum(scores_, dim=-1)

        disttotal_scores = torch.sum(
            scores_,
            dim=-1,
            keepdim=True,
        )

        position_effect = torch.abs(x1 - x2)[None, None, :, :].float()

        dist_scores = torch.clamp(
            (disttotal_scores - distcum_scores) * position_effect,
            min=0.0,
        )

        dist_scores = dist_scores.sqrt().detach()

    m = nn.Softplus()
    gamma = -1.0 * m(gamma).unsqueeze(0)

    if pdiff is None:
        total_effect = torch.clamp(
            torch.clamp(
                (dist_scores * gamma).exp(),
                min=1e-5,
            ),
            max=1e5,
        )
    else:
        diff = pdiff.unsqueeze(1).expand(
            pdiff.shape[0],
            dist_scores.shape[1],
            pdiff.shape[1],
            pdiff.shape[2],
        )

        diff = diff.sigmoid().exp()

        total_effect = torch.clamp(
            torch.clamp(
                (dist_scores * gamma * diff).exp(),
                min=1e-5,
            ),
            max=1e5,
        )

    scores = scores * total_effect

    scores.masked_fill_(mask == 0, -1e32)
    scores = F.softmax(scores, dim=-1)

    if zero_pad:
        pad_zero = torch.zeros(
            bs,
            head,
            1,
            seqlen,
            device=scores.device,
            dtype=scores.dtype,
        )
        scores = torch.cat([pad_zero, scores[:, :, 1:, :]], dim=2)

    scores = dropout(scores)

    output = torch.matmul(scores, v)

    return output


class LearnablePositionalEmbedding(nn.Module):
    def __init__(self, d_model, max_len=512):
        super().__init__()

        pe = 0.1 * torch.randn(max_len, d_model)
        pe = pe.unsqueeze(0)

        self.weight = nn.Parameter(pe, requires_grad=True)

    def forward(self, x):
        return self.weight[:, : x.size(Dim.seq), :]


class CosinePositionalEmbedding(nn.Module):
    def __init__(self, d_model, max_len=512):
        super().__init__()

        pe = 0.1 * torch.randn(max_len, d_model)

        position = torch.arange(0, max_len).unsqueeze(1).float()

        div_term = torch.exp(
            torch.arange(0, d_model, 2).float()
            * -(math.log(10000.0) / d_model)
        )

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)

        self.weight = nn.Parameter(pe, requires_grad=False)

    def forward(self, x):
        return self.weight[:, : x.size(Dim.seq), :]