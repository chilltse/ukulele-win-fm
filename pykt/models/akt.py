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
        num_c_fm4=None,
    ):
        super().__init__()

        """
        Input:
            d_model: dimension of attention block
            final_fc_dim: dimension of final fully connected net before prediction
            num_attn_heads: number of heads in multi-headed attention
            d_ff: dimension for fully connected net inside the basic block
            kq_same: if key query same, kq_same=1, else = 0
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
        self.num_c_fm4 = num_c_fm4

        embed_l = d_model

        # ------------------------------------------------------------------
        # General factorized KC embedding branch
        # ------------------------------------------------------------------
        if emb_type == "qid_fm4":
            if num_c_fm4 is None or len(num_c_fm4) < 1:
                raise ValueError(
                    "emb_type qid_fm4 requires num_c_fm4 with at least one field"
                )

            if separate_qa:
                raise ValueError("qid_fm4 does not support separate_qa")

            # General FM-KC setting:
            # num_c_fm4 can now contain any number of factor fields.
            #
            # Example FM4:
            #   [prev_pitches, prev_strings, pitches, strings]
            #
            # Example FM5:
            #   [prev_pitches, prev_strings, pitches, strings, duration]
            self.num_fm_fields = len(num_c_fm4)

            self.kc_emb = nn.ModuleList(
                [nn.Embedding(int(n), embed_l) for n in num_c_fm4]
            )

            # Field-wise and dimension-wise learnable weights for first-order terms.
            # Shape: [num_fields, d_model]
            self.alpha_fm1 = nn.Parameter(
                torch.ones(self.num_fm_fields, embed_l)
            )

            # Start second-order FM interaction softly to avoid unstable logits.
            self.fm2_scale = nn.Parameter(torch.tensor(0.1))

            # Learnable global output scale for the FM-KC embedding.
            self.fm_out_scale = nn.Parameter(torch.tensor(1.0))

        # ------------------------------------------------------------------
        # Rasch / problem difficulty branch
        # ------------------------------------------------------------------
        if self.n_pid > 0:
            # problem difficulty scalar: u_q
            self.difficult_param = nn.Embedding(self.n_pid + 1, 1)

            # For qid_fm4, q_embed_diff is indexed by pid_data.
            # For normal qid, q_embed_diff is indexed by q_data.
            qdiff_rows = (
                self.n_pid + 1
                if emb_type == "qid_fm4"
                else self.n_question + 1
            )

            self.q_embed_diff = nn.Embedding(qdiff_rows, embed_l)

            # Keep the original AKT design here.
            self.qa_embed_diff = nn.Embedding(2 * self.n_question + 1, embed_l)

        # ------------------------------------------------------------------
        # Base embeddings
        # ------------------------------------------------------------------
        if emb_type == "qid_fm4":
            # response embedding only: target is 0 or 1
            self.qa_embed = nn.Embedding(2, embed_l)

        elif emb_type.startswith("qid"):
            self.q_embed = nn.Embedding(self.n_question, embed_l)

            if self.separate_qa:
                self.qa_embed = nn.Embedding(2 * self.n_question + 1, embed_l)
            else:
                self.qa_embed = nn.Embedding(2, embed_l)

        # ------------------------------------------------------------------
        # Architecture object
        # ------------------------------------------------------------------
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
        Important:
            Only reset difficult_param to zero.

        Do NOT reset all parameters whose first dimension equals n_pid + 1.
        In qid_fm4, q_embed_diff also has n_pid + 1 rows.
        If q_embed_diff is also zeroed, the Rasch / problem difficulty branch
        can become too weak or nearly dead at initialization.
        """
        if self.n_pid > 0:
            torch.nn.init.constant_(self.difficult_param.weight, 0.0)

    def fm_kc_embed(self, c_multi):
        """
        General factorized KC embedding.

        Input:
            c_multi: LongTensor with shape [batch_size, seq_len, num_fields]

        Meaning:
            Each time step contains multiple discrete KC fields.

            Example FM4:
                [prev_pitches, prev_strings, pitches, strings]

            Example FM5:
                [prev_pitches, prev_strings, pitches, strings, duration]

        Output:
            Tensor with shape [batch_size, seq_len, d_model]

        Design:
            1. First-order term:
                sum_i alpha_i * e_i

               This is normalized by sqrt(number_of_valid_fields).

            2. Second-order FM term:
                sum_{i<j} e_i ⊙ e_j

               This is normalized by sqrt(number_of_valid_pairs).

        Why normalization:
            If the number of factor fields increases from FM4 to FM5/FM6,
            directly summing all embeddings and pairwise interactions can change
            the embedding scale. Scale instability can shift logits and hurt
            ACC@0.5 even when AUC improves.

        Negative ids:
            Negative ids are treated as padding or missing fields.
            They are clamped to 0 for safe embedding lookup and then masked out.
        """
        num_fields = len(self.kc_emb)

        if c_multi.size(-1) != num_fields:
            raise ValueError(
                f"Expected {num_fields} KC fields, but got {c_multi.size(-1)}"
            )

        c_multi = c_multi.long()

        # [B, L, F]
        # 1 means this field is valid; 0 means padding or missing.
        field_valid = (c_multi >= 0).float()

        # Embedding lookup cannot use negative ids.
        # Invalid fields are masked out immediately after lookup.
        safe_ids = c_multi.clamp(min=0)

        embs = []
        for i in range(num_fields):
            # [B, L, D]
            e_i = self.kc_emb[i](safe_ids[..., i])

            # Mask invalid fields.
            e_i = e_i * field_valid[..., i].unsqueeze(-1)

            embs.append(e_i)

        # [B, L, F, D]
        es = torch.stack(embs, dim=2)

        # [B, L, 1]
        valid_count = field_valid.sum(dim=-1, keepdim=True)
        valid_count_safe = valid_count.clamp(min=1.0)

        # [B, L, 1]
        # 1 means this whole time step has at least one valid field.
        token_valid = (valid_count > 0).float()

        # ------------------------------------------------------------------
        # First-order term
        # ------------------------------------------------------------------
        # [1, 1, F, D]
        alpha = self.alpha_fm1.view(1, 1, num_fields, -1)

        # [B, L, D]
        first = (alpha * es).sum(dim=2)

        # Scale control:
        # avoid larger embeddings when more fields are used.
        first = first / torch.sqrt(valid_count_safe)

        # ------------------------------------------------------------------
        # Second-order FM term
        # ------------------------------------------------------------------
        # Efficient FM formula:
        #
        #   0.5 * ((sum_i e_i)^2 - sum_i(e_i^2))
        #
        # This equals:
        #
        #   sum_{i<j} e_i ⊙ e_j
        #
        # where ⊙ means element-wise product.
        s = es.sum(dim=2)
        sum_sq = (es * es).sum(dim=2)
        fm2 = 0.5 * (s * s - sum_sq)

        # Number of valid pairs:
        #   C(F, 2) = F * (F - 1) / 2
        pair_count = valid_count_safe * (valid_count_safe - 1.0) / 2.0
        pair_count_safe = pair_count.clamp(min=1.0)

        # Scale control for second-order interactions.
        fm2 = fm2 / torch.sqrt(pair_count_safe)

        out = first + self.fm2_scale * fm2
        out = self.fm_out_scale * out

        # If the whole time step is padding, output zero vector.
        out = out * token_valid

        return out

    def base_emb(self, q_data, target):
        if self.emb_type == "qid_fm4":
            q_embed_data = self.fm_kc_embed(q_data)
        else:
            q_embed_data = self.q_embed(q_data)

        if self.separate_qa:
            qa_data = q_data + self.n_question * target
            qa_embed_data = self.qa_embed(qa_data)
        else:
            # e_(ct, rt) = c_ct + g_rt
            qa_embed_data = self.qa_embed(target) + q_embed_data

        return q_embed_data, qa_embed_data

    def forward(self, q_data, target, pid_data=None, qtest=False):
        emb_type = self.emb_type

        # ------------------------------------------------------------------
        # Base question/KC embedding and QA embedding
        # ------------------------------------------------------------------
        if emb_type.startswith("qid"):
            q_embed_data, qa_embed_data = self.base_emb(q_data, target)

        pid_embed_data = None

        # ------------------------------------------------------------------
        # Problem difficulty / Rasch branch
        # ------------------------------------------------------------------
        if self.n_pid > 0:
            if pid_data is None:
                raise ValueError("pid_data must be provided when n_pid > 0")

            pid_ids = pid_data.long().clamp(min=0)

            if self.emb_type == "qid_fm4":
                # qid_fm4 uses pid ids for q_embed_diff.
                q_embed_diff_data = self.q_embed_diff(pid_ids)
            else:
                # Original AKT qid mode uses q_data for q_embed_diff.
                q_embed_diff_data = self.q_embed_diff(q_data)

            # u_q: problem difficulty scalar
            pid_embed_data = self.difficult_param(pid_ids)

            # question encoder:
            # c_ct + u_q * d_ct
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
            c_reg_loss = 0.0

        # ------------------------------------------------------------------
        # AKT architecture
        # ------------------------------------------------------------------
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

        """
        n_block: number of stacked blocks in the attention
        d_model: dimension of attention input/output
        d_feature: dimension of input in each multi-head attention part
        n_head: number of heads; n_heads * d_feature = d_model
        """

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
        # target shape: [batch_size, seq_len]
        seqlen, batch_size = q_embed_data.size(1), q_embed_data.size(0)

        qa_pos_embed = qa_embed_data
        q_pos_embed = q_embed_data

        y = qa_pos_embed
        seqlen, batch_size = y.size(1), y.size(0)
        x = q_pos_embed

        # Encoder:
        # encode QA information from time 0 to t.
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
                # Peek current question.
                # No FFN in this layer.
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
                # mask=0 means the model can only attend to previous interaction information.
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

        """
        Basic Transformer block:
            Multi-head attention
            LayerNorm
            Feed-forward network
            Dropout
        """

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
            0 means the block can only attend to past values.
            1 means the block can attend to current and past values.

        query:
            Query tensor.

        key:
            Key tensor.

        values:
            Value tensor.

        apply_pos:
            Whether to apply the feed-forward network after attention.
        """

        seqlen, batch_size = query.size(1), query.size(0)

        nopeek_mask = np.triu(
            np.ones((1, 1, seqlen, seqlen)),
            k=mask,
        ).astype("uint8")

        src_mask = (torch.from_numpy(nopeek_mask) == 0).to(device)

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

        # Residual connection + LayerNorm
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

        """
        Multi-head attention module:
            linear projections for Q/K/V
            attention
            output projection
        """

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

        elif emb_type.startswith("qid"):
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

        elif self.emb_type.startswith("qid"):
            # Linear projections and split into heads.
            k = self.k_linear(k).view(bs, -1, self.h, self.d_k)

            if self.kq_same is False:
                q = self.q_linear(q).view(bs, -1, self.h, self.d_k)
            else:
                q = self.k_linear(q).view(bs, -1, self.h, self.d_k)

            v = self.v_linear(v).view(bs, -1, self.h, self.d_k)

            # [B, H, L, D]
            k = k.transpose(1, 2)
            q = q.transpose(1, 2)
            v = v.transpose(1, 2)

            gammas = self.gammas

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

            # Concatenate heads.
            concat = (
                scores.transpose(1, 2)
                .contiguous()
                .view(bs, -1, self.d_model)
            )

        output = self.out_proj(concat)

        return output

    def pad_zero(self, scores, bs, dim, zero_pad):
        if zero_pad:
            pad_zero = torch.zeros(bs, 1, dim).to(device)
            scores = torch.cat([pad_zero, scores[:, 0:-1, :]], dim=1)

        return scores


def attention(q, k, v, d_k, mask, dropout, zero_pad, gamma=None, pdiff=None):
    """
    Multi-head attention with AKT distance decay.
    """

    scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(d_k)

    bs, head, seqlen = scores.size(0), scores.size(1), scores.size(2)

    x1 = torch.arange(seqlen).expand(seqlen, -1).to(device)
    x2 = x1.transpose(0, 1).contiguous()

    with torch.no_grad():
        scores_ = scores.masked_fill(mask == 0, -1e32)
        scores_ = F.softmax(scores_, dim=-1)

        scores_ = scores_ * mask.float().to(device)

        distcum_scores = torch.cumsum(scores_, dim=-1)

        disttotal_scores = torch.sum(
            scores_,
            dim=-1,
            keepdim=True,
        )

        position_effect = torch.abs(x1 - x2)[None, None, :, :].type(
            torch.FloatTensor
        ).to(device)

        dist_scores = torch.clamp(
            (disttotal_scores - distcum_scores) * position_effect,
            min=0.0,
        )

        dist_scores = dist_scores.sqrt().detach()

    m = nn.Softplus()

    # gamma is negative after this transformation.
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
        pad_zero = torch.zeros(bs, head, 1, seqlen).to(device)
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