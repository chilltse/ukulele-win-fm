import json
import os
import torch
from torch import nn
from torch.nn.init import xavier_uniform_
from torch.nn.init import constant_
import math
import torch.nn.functional as F
from enum import IntEnum
import numpy as np
from .utils import transformer_FFN, ut_mask, pos_encode, get_clones
from torch.nn import Module, Embedding, LSTM, Linear, Dropout, LayerNorm, TransformerEncoder, TransformerEncoderLayer, \
        MultiLabelMarginLoss, MultiLabelSoftMarginLoss, CrossEntropyLoss, BCELoss, MultiheadAttention, ModuleList, Parameter
from torch.nn.functional import one_hot, cross_entropy, multilabel_margin_loss, binary_cross_entropy

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Dim(IntEnum):
    batch = 0
    seq = 1
    feature = 2

class simpleKT(nn.Module):
    def __init__(self, n_question, n_pid, 
            d_model, n_blocks, dropout, d_ff=256, 
            loss1=0.5, loss2=0.5, loss3=0.5, start=50, num_layers=2, nheads=4, seq_len=200, 
            kq_same=1, final_fc_dim=512, final_fc_dim2=256, num_attn_heads=8, separate_qa=False, l2=1e-5, emb_type="qid", emb_path="", pretrain_dim=768, num_c_fmkc=None, dpath="", kc_tree_path=""):
        super().__init__()
        """
        Input:
            d_model: dimension of attention block
            final_fc_dim: dimension of final fully connected net before prediction
            num_attn_heads: number of heads in multi-headed attention
            d_ff : dimension for fully conntected net inside the basic block
            kq_same: if key query same, kq_same=1, else = 0
        """
        self.model_name = "simplekt"
        print(f"model_name: {self.model_name}, emb_type: {emb_type}")
        self.n_question = n_question
        self.dropout = dropout
        self.kq_same = kq_same
        self.n_pid = n_pid
        self.l2 = l2
        self.model_type = self.model_name
        self.separate_qa = separate_qa
        self.emb_type = emb_type
        self.dpath = dpath
        embed_l = d_model
        if emb_type == "qid_fmkc":
            if num_c_fmkc is None or len(num_c_fmkc) < 1:
                raise ValueError("emb_type qid_fmkc requires num_c_fmkc")
            if separate_qa:
                raise ValueError("qid_fmkc does not support separate_qa")
            self.num_c_fmkc = [int(n) for n in num_c_fmkc]
            self.num_fmkc_fields = len(self.num_c_fmkc)
            self.kc_residual_emb = nn.Embedding(self.n_question, embed_l)
            self.kc_residual_scale = nn.Parameter(torch.tensor(0.1))

            fmkc_strides = []
            stride = 1
            for n in self.num_c_fmkc:
                fmkc_strides.append(stride)
                stride *= int(n)
            self.register_buffer(
                "fmkc_strides",
                torch.tensor(fmkc_strides, dtype=torch.long),
                persistent=False,
            )

            fmkc_tuple_space = 1
            for n in self.num_c_fmkc:
                fmkc_tuple_space *= int(n)
            self.fmkc_tuple_space = int(fmkc_tuple_space)
            self.fmkc_residual_collision_risk = self.fmkc_tuple_space > int(self.n_question)
        if self.n_pid > 0:
            if emb_type.find("scalar") != -1:
                # print(f"question_difficulty is scalar")
                self.difficult_param = nn.Embedding(self.n_pid+1, 1) # 题目难度
            else:
                self.difficult_param = nn.Embedding(self.n_pid+1, embed_l) # 题目难度
            if emb_type == "qid_fmkc":
                self.q_embed_diff_fmkc = ModuleList(
                    [Embedding(n, embed_l) for n in self.num_c_fmkc]
                )
                self.alpha_qdiff_fm1 = Parameter(torch.ones(self.num_fmkc_fields, embed_l))
                self.qdiff_fm2_scale = Parameter(torch.tensor(0.1))
                self.qdiff_fm_out_scale = Parameter(torch.tensor(1.0))
                self.qa_embed_diff = nn.Embedding(2, embed_l)
            else:
                self.q_embed_diff = nn.Embedding(self.n_question+1, embed_l) # question emb, 总结了包含当前question（concept）的problems（questions）的变化
                self.qa_embed_diff = nn.Embedding(2 * self.n_question + 1, embed_l) # interaction emb, 同上
        
        if emb_type == "qid_fmkc":
            self.q_embed_fmkc = ModuleList(
                [Embedding(n, embed_l) for n in self.num_c_fmkc]
            )
            self.qa_embed_fmkc = ModuleList(
                [Embedding(n * 2, embed_l) for n in self.num_c_fmkc]
            )
            self.alpha_fm1 = Parameter(torch.ones(self.num_fmkc_fields, embed_l))
            self.alpha_qa_fm1 = Parameter(torch.ones(self.num_fmkc_fields, embed_l))
            self.fm2_scale = Parameter(torch.tensor(0.1))
            self.fm_out_scale = Parameter(torch.tensor(1.0))
            self.qa_fm2_scale = Parameter(torch.tensor(0.1))
            self.qa_fm_out_scale = Parameter(torch.tensor(1.0))
        elif emb_type == "qid_tree":
            if separate_qa:
                raise ValueError("qid_tree does not support separate_qa")
            self.residual_kc_emb = nn.Embedding(self.n_question, embed_l)
            self.response_emb = nn.Embedding(2, embed_l)
            self.tree_mlp = nn.Sequential(
                nn.Linear(embed_l, embed_l),
                nn.ReLU(),
            )
            self.edge_alpha = nn.Parameter(torch.zeros(self.n_question))

            parent_index = self._build_tree_parent_index(kc_tree_path, dpath)
            self.register_buffer(
                "parent_index",
                torch.tensor(parent_index, dtype=torch.long),
            )
            topo_index = self._build_topo_order(parent_index)
            self.register_buffer(
                "topo_index",
                torch.tensor(topo_index, dtype=torch.long),
            )
        elif emb_type.startswith("qid"):
            # n_question+1 ,d_model
            self.q_embed = nn.Embedding(self.n_question, embed_l)
            if self.separate_qa: 
                    self.qa_embed = nn.Embedding(2*self.n_question+1, embed_l)
            else: # false default
                self.qa_embed = nn.Embedding(2, embed_l)
        # Architecture Object. It contains stack of attention block
        self.model = Architecture(n_question=n_question, n_blocks=n_blocks, n_heads=num_attn_heads, dropout=dropout,
                                    d_model=d_model, d_feature=d_model / num_attn_heads, d_ff=d_ff,  kq_same=self.kq_same, model_type=self.model_type, seq_len=seq_len)

        self.out = nn.Sequential(
            nn.Linear(d_model + embed_l,
                      final_fc_dim), nn.ReLU(), nn.Dropout(self.dropout),
            nn.Linear(final_fc_dim, final_fc_dim2), nn.ReLU(
            ), nn.Dropout(self.dropout),
            nn.Linear(final_fc_dim2, 1)
        )

        self.reset()

    def reset(self):
        for p in self.parameters():
            if p.dim() > 0 and p.size(0) == self.n_pid+1 and self.n_pid > 0:
                torch.nn.init.constant_(p, 0.)

    def _resolve_tree_path(self, kc_tree_path, dpath):
        if kc_tree_path and os.path.exists(kc_tree_path):
            return kc_tree_path
        if dpath:
            default_path = os.path.join(
                dpath,
                "2_DBE_KT22_datafiles_100102_csv",
                "kc_knowledge_tree.json",
            )
            if os.path.exists(default_path):
                return default_path
        return ""

    def _build_tree_parent_index(self, kc_tree_path, dpath):
        tree_path = self._resolve_tree_path(kc_tree_path, dpath)
        if not tree_path:
            raise FileNotFoundError(
                "emb_type qid_tree requires kc_tree_path or default tree json under dpath."
            )

        tree_keyid2idx_path = os.path.join(dpath, "keyid2idx_tree.json")
        default_keyid2idx_path = os.path.join(dpath, "keyid2idx.json")
        keyid2idx_path = (
            tree_keyid2idx_path
            if os.path.exists(tree_keyid2idx_path)
            else default_keyid2idx_path
        )
        if not os.path.exists(keyid2idx_path):
            raise FileNotFoundError(
                f"emb_type qid_tree requires keyid2idx_tree.json or keyid2idx.json under dpath, "
                f"missing: {tree_keyid2idx_path} and {default_keyid2idx_path}"
            )

        with open(tree_path, "r", encoding="utf-8") as f:
            tree_data = json.load(f)
        with open(keyid2idx_path, "r", encoding="utf-8") as f:
            keyid2idx = json.load(f)

        concepts_map = keyid2idx.get("concepts", {})
        if not concepts_map:
            raise ValueError(f"{os.path.basename(keyid2idx_path)} has no `concepts` mapping for qid_tree.")

        kc_name_to_id = {}
        for item in tree_data.get("kc_index", []):
            name = str(item.get("name", "")).strip()
            kc_id = item.get("kc_id", None)
            if name and kc_id is not None:
                kc_name_to_id[name] = int(kc_id)

        child_to_parent_kc = {}
        for edge in tree_data.get("non_tree_prerequisite_edges", []):
            parent_name = str(edge.get("from", "")).strip()
            child_name = str(edge.get("to", "")).strip()
            if parent_name in kc_name_to_id and child_name in kc_name_to_id:
                child_to_parent_kc[kc_name_to_id[child_name]] = kc_name_to_id[parent_name]

        parent_index = [-1] * self.n_question
        for raw_kc, mapped_idx in concepts_map.items():
            try:
                child_kc_id = int(raw_kc)
            except Exception:
                continue

            parent_kc_id = child_to_parent_kc.get(child_kc_id, None)
            if parent_kc_id is None:
                continue

            parent_raw = str(parent_kc_id)
            if parent_raw not in concepts_map:
                continue

            cidx = int(mapped_idx)
            pidx = int(concepts_map[parent_raw])
            if 0 <= cidx < self.n_question and 0 <= pidx < self.n_question:
                parent_index[cidx] = pidx

        return parent_index

    def _build_topo_order(self, parent_index):
        n = len(parent_index)
        state = [0] * n
        order = []

        def dfs(u):
            if state[u] == 2:
                return
            if state[u] == 1:
                return
            state[u] = 1
            p = parent_index[u]
            if 0 <= p < n:
                dfs(p)
            state[u] = 2
            order.append(u)

        for i in range(n):
            dfs(i)
        return order

    def _tree_kc_table(self):
        residual = self.residual_kc_emb.weight
        node_embs = [None] * self.n_question

        for idx in self.topo_index.tolist():
            base = residual[idx]
            pidx = int(self.parent_index[idx].item())
            if 0 <= pidx < self.n_question:
                parent_emb = node_embs[pidx]
                parent_proj = self.tree_mlp(parent_emb)
                a = torch.sigmoid(self.edge_alpha[idx])
                cur = a * parent_proj + (1.0 - a) * base
            else:
                cur = base
            node_embs[idx] = cur

        return torch.stack(node_embs, dim=0)

    def tree_kc_embed(self, q_data):
        q_valid = (q_data >= 0) & (q_data < self.n_question)
        safe_q = q_data.long().clamp(0, self.n_question - 1)
        kc_table = self._tree_kc_table()
        q_embed_data = kc_table[safe_q]
        q_embed_data = q_embed_data * q_valid.unsqueeze(-1).float()
        return q_embed_data

    def base_emb(self, q_data, target, q_dense=None):
        target_valid = (target >= 0) & (target <= 1)
        safe_target = target.long().clamp(0, 1)

        if self.emb_type == "qid_fmkc":
            if q_dense is None:
                raise ValueError("qid_fmkc requires q_dense to avoid residual id collision.")
            q_embed_data = self.fm_kc_embed(q_data, dense_kc_ids=q_dense)
            qa_embed_data = self.fm_kcr_embed(q_data, target)
            qa_embed_data = qa_embed_data * (self.fmkc_token_valid(q_data) & target_valid).unsqueeze(-1).float()
        elif self.emb_type == "qid_tree":
            q_valid = (q_data >= 0) & (q_data < self.n_question)
            q_embed_data = self.tree_kc_embed(q_data)
            qa_embed_data = self.response_emb(safe_target) + q_embed_data
            qa_embed_data = qa_embed_data * (q_valid & target_valid).unsqueeze(-1).float()
        else:
            q_valid = (q_data >= 0) & (q_data < self.n_question)
            safe_q = q_data.long().clamp(0, self.n_question - 1)
            q_embed_data = self.q_embed(safe_q)  # BS, seqlen,  d_model# c_ct
            q_embed_data = q_embed_data * q_valid.unsqueeze(-1).float()
            if self.separate_qa:
                qa_data = safe_q + self.n_question * safe_target
                qa_embed_data = self.qa_embed(qa_data)
                qa_embed_data = qa_embed_data * (q_valid & target_valid).unsqueeze(-1).float()
            else:
                # BS, seqlen, d_model # c_ct+ g_rt =e_(ct,rt)
                qa_embed_data = self.qa_embed(safe_target) + q_embed_data
                qa_embed_data = qa_embed_data * (q_valid & target_valid).unsqueeze(-1).float()
        return q_embed_data, qa_embed_data

    def _check_fmkc_shape(self, c_multi):
        if c_multi.dim() != 3:
            raise ValueError(
                f"qid_fmkc expects q shape [B, L, F], got {tuple(c_multi.shape)}"
            )
        if c_multi.size(-1) != self.num_fmkc_fields:
            raise ValueError(
                f"Expected {self.num_fmkc_fields} KC fields, got {c_multi.size(-1)}"
            )

    def _fmkc_field_valid_bool(self, c_multi):
        self._check_fmkc_shape(c_multi)
        c_multi = c_multi.long()
        valid_fields = []
        for i, n in enumerate(self.num_c_fmkc):
            valid_fields.append((c_multi[..., i] >= 0) & (c_multi[..., i] < n))
        return torch.stack(valid_fields, dim=-1)

    def _fmkc_safe_ids(self, c_multi):
        self._check_fmkc_shape(c_multi)
        c_multi = c_multi.long()
        safe_fields = []
        for i, n in enumerate(self.num_c_fmkc):
            safe_fields.append(c_multi[..., i].clamp(0, n - 1))
        return torch.stack(safe_fields, dim=-1)

    def fmkc_token_valid(self, c_multi):
        return self._fmkc_field_valid_bool(c_multi).any(dim=-1)

    def _fmkc_tuple_residual_ids(self, c_multi, dense_kc_ids=None):
        self._check_fmkc_shape(c_multi)
        token_valid = self.fmkc_token_valid(c_multi)

        if dense_kc_ids is not None:
            if dense_kc_ids.dim() != 2:
                raise ValueError(
                    f"dense_kc_ids must have shape [B, L], got {tuple(dense_kc_ids.shape)}"
                )
            if dense_kc_ids.shape != c_multi.shape[:2]:
                raise ValueError(
                    f"dense_kc_ids shape {tuple(dense_kc_ids.shape)} must match q_data [B, L] {tuple(c_multi.shape[:2])}"
                )

            dense_kc_ids = dense_kc_ids.long()
            dense_valid = (dense_kc_ids >= 0) & (dense_kc_ids < self.n_question)
            valid = token_valid & dense_valid
            residual_ids = dense_kc_ids.clamp(min=0, max=self.n_question - 1)
            residual_ids = residual_ids.masked_fill(~valid, 0)
            return residual_ids, valid

        if self.fmkc_tuple_space > self.n_question:
            raise ValueError(
                "qid_fmkc residual requires q_dense when tuple space exceeds n_question. "
                f"Got tuple_space={self.fmkc_tuple_space}, n_question={self.n_question}."
            )

        safe_ids = self._fmkc_safe_ids(c_multi)
        strides = self.fmkc_strides.view(1, 1, self.num_fmkc_fields)
        residual_ids = (safe_ids * strides).sum(dim=-1)
        residual_ids = residual_ids.masked_fill(~token_valid, 0)
        return residual_ids, token_valid

    def _fm_embed(self, c_multi, emb_tables, alpha, fm2_scale, fm_out_scale, r=None):
        self._check_fmkc_shape(c_multi)
        c_multi = c_multi.long()
        field_valid_bool = self._fmkc_field_valid_bool(c_multi)
        if r is not None:
            response_valid_bool = (r >= 0) & (r <= 1)
            field_valid_bool = field_valid_bool & response_valid_bool.unsqueeze(-1)
            ri = r.long().clamp(0, 1)
        else:
            ri = None
        field_valid = field_valid_bool.float()
        safe_ids = self._fmkc_safe_ids(c_multi)
        embs = []
        for i in range(self.num_fmkc_fields):
            ids_i = safe_ids[..., i]
            if r is not None:
                ids_i = ids_i + self.num_c_fmkc[i] * ri
            e_i = emb_tables[i](ids_i)
            e_i = e_i * field_valid[..., i].unsqueeze(-1)
            embs.append(e_i)
        es = torch.stack(embs, dim=2)  # [B, L, F, D]
        valid_count = field_valid.sum(dim=-1, keepdim=True)
        valid_count_safe = valid_count.clamp(min=1.0)
        token_valid = (valid_count > 0).float()
        alpha = alpha.view(1, 1, self.num_fmkc_fields, es.size(-1))
        first = (alpha * es).sum(dim=2) / torch.sqrt(valid_count_safe)
        s = es.sum(dim=2)
        sum_sq = (es * es).sum(dim=2)
        fm2 = 0.5 * (s * s - sum_sq)
        pair_count = valid_count_safe * (valid_count_safe - 1.0) / 2.0
        pair_count_safe = pair_count.clamp(min=1.0)
        fm2 = fm2 / torch.sqrt(pair_count_safe)
        out = fm_out_scale * (first + fm2_scale * fm2)
        out = out * token_valid
        return out

    def fm_kc_embed(self, c_multi, dense_kc_ids=None):
        fm_emb = self._fm_embed(
            c_multi=c_multi,
            emb_tables=self.q_embed_fmkc,
            alpha=self.alpha_fm1,
            fm2_scale=self.fm2_scale,
            fm_out_scale=self.fm_out_scale,
            r=None,
        )
        residual_ids, residual_valid = self._fmkc_tuple_residual_ids(
            c_multi,
            dense_kc_ids=dense_kc_ids,
        )
        residual_emb = self.kc_residual_emb(residual_ids)
        residual_emb = residual_emb * residual_valid.unsqueeze(-1).float()
        return fm_emb + self.kc_residual_scale * residual_emb

    def fm_kcr_embed(self, c_multi, r):
        return self._fm_embed(
            c_multi=c_multi,
            emb_tables=self.qa_embed_fmkc,
            alpha=self.alpha_qa_fm1,
            fm2_scale=self.qa_fm2_scale,
            fm_out_scale=self.qa_fm_out_scale,
            r=r,
        )

    def fm_qdiff_embed(self, c_multi):
        return self._fm_embed(
            c_multi=c_multi,
            emb_tables=self.q_embed_diff_fmkc,
            alpha=self.alpha_qdiff_fm1,
            fm2_scale=self.qdiff_fm2_scale,
            fm_out_scale=self.qdiff_fm_out_scale,
            r=None,
        )

    def get_attn_pad_mask(self, sm):
        batch_size, l = sm.size()
        pad_attn_mask = sm.data.eq(0).unsqueeze(1)
        pad_attn_mask = pad_attn_mask.expand(batch_size, l, l)
        return pad_attn_mask.repeat(self.nhead, 1, 1)

    def forward(self, dcur, qtest=False, train=False):
        q, c, r = dcur["qseqs"].long(), dcur["cseqs"].long(), dcur["rseqs"].long()
        qshft, cshft, rshft = dcur["shft_qseqs"].long(), dcur["shft_cseqs"].long(), dcur["shft_rseqs"].long()
        pid_data = torch.cat((q[:,0:1], qshft), dim=1)
        q_data = torch.cat((c[:,0:1], cshft), dim=1)
        target = torch.cat((r[:,0:1], rshft), dim=1)
        q_dense = None
        if self.emb_type == "qid_fmkc":
            c_dense = dcur["cdense_seqs"].long() if "cdense_seqs" in dcur else None
            cshft_dense = dcur["shft_cdense_seqs"].long() if "shft_cdense_seqs" in dcur else None
            if c_dense is None or cshft_dense is None:
                raise ValueError("simplekt qid_fmkc requires concepts_dense in dataset as q_dense.")
            q_dense = torch.cat((c_dense[:, 0:1], cshft_dense), dim=1)

        emb_type = self.emb_type

        # Batch First
        if emb_type.startswith("qid"):
            q_embed_data, qa_embed_data = self.base_emb(q_data, target, q_dense=q_dense)
        if self.n_pid > 0 and emb_type.find("norasch") == -1: # have problem id
            if emb_type.find("aktrasch") == -1:
                if emb_type == "qid_fmkc":
                    q_embed_diff_data = self.fm_qdiff_embed(q_data)
                else:
                    q_embed_diff_data = self.q_embed_diff(q_data)  # d_ct 总结了包含当前question（concept）的problems（questions）的变化
                pid_embed_data = self.difficult_param(pid_data)  # uq 当前problem的难度
                q_embed_data = q_embed_data + pid_embed_data * \
                    q_embed_diff_data  # uq *d_ct + c_ct # question encoder

            else:
                if emb_type == "qid_fmkc":
                    q_embed_diff_data = self.fm_qdiff_embed(q_data)
                else:
                    q_embed_diff_data = self.q_embed_diff(q_data)  # d_ct 总结了包含当前question（concept）的problems（questions）的变化
                pid_embed_data = self.difficult_param(pid_data)  # uq 当前problem的难度
                q_embed_data = q_embed_data + pid_embed_data * \
                    q_embed_diff_data  # uq *d_ct + c_ct # question encoder

                qa_embed_diff_data = self.qa_embed_diff(
                    target)  # f_(ct,rt) or #h_rt (qt, rt)差异向量
                qa_embed_data = qa_embed_data + pid_embed_data * \
                        (qa_embed_diff_data+q_embed_diff_data)  # + uq *(h_rt+d_ct) # （q-response emb diff + question emb diff）

        # BS.seqlen,d_model
        # Pass to the decoder
        # output shape BS,seqlen,d_model or d_model//2
        y2, y3 = 0, 0
        if emb_type in ["qid", "qidaktrasch", "qid_scalar", "qid_norasch", "qid_fmkc", "qid_tree"]:
            d_output = self.model(q_embed_data, qa_embed_data)

            concat_q = torch.cat([d_output, q_embed_data], dim=-1)
            output = self.out(concat_q).squeeze(-1)
            m = nn.Sigmoid()
            preds = m(output)

        if train:
            return preds, y2, y3
        else:
            if qtest:
                return preds, concat_q
            else:
                return preds

class Architecture(nn.Module):
    def __init__(self, n_question,  n_blocks, d_model, d_feature,
                 d_ff, n_heads, dropout, kq_same, model_type, seq_len):
        super().__init__()
        """
            n_block : number of stacked blocks in the attention
            d_model : dimension of attention input/output
            d_feature : dimension of input in each of the multi-head attention part.
            n_head : number of heads. n_heads*d_feature = d_model
        """
        self.d_model = d_model
        self.model_type = model_type

        if model_type in {'simplekt'}:
            self.blocks_2 = nn.ModuleList([
                TransformerLayer(d_model=d_model, d_feature=d_model // n_heads,
                                 d_ff=d_ff, dropout=dropout, n_heads=n_heads, kq_same=kq_same)
                for _ in range(n_blocks)
            ])
        self.position_emb = CosinePositionalEmbedding(d_model=self.d_model, max_len=seq_len)

    def forward(self, q_embed_data, qa_embed_data):
        # target shape  bs, seqlen
        seqlen, batch_size = q_embed_data.size(1), q_embed_data.size(0)

        q_posemb = self.position_emb(q_embed_data)
        q_embed_data = q_embed_data + q_posemb
        qa_posemb = self.position_emb(qa_embed_data)
        qa_embed_data = qa_embed_data + qa_posemb

        qa_pos_embed = qa_embed_data
        q_pos_embed = q_embed_data

        y = qa_pos_embed
        seqlen, batch_size = y.size(1), y.size(0)
        x = q_pos_embed

        # encoder
        
        for block in self.blocks_2:
            x = block(mask=0, query=x, key=x, values=y, apply_pos=True) # True: +FFN+残差+laynorm 非第一层与0~t-1的的q的attention, 对应图中Knowledge Retriever
            # mask=0，不能看到当前的response, 在Knowledge Retrever的value全为0，因此，实现了第一题只有question信息，无qa信息的目的
            # print(x[0,0,:])
        return x

class TransformerLayer(nn.Module):
    def __init__(self, d_model, d_feature,
                 d_ff, n_heads, dropout,  kq_same):
        super().__init__()
        """
            This is a Basic Block of Transformer paper. It containts one Multi-head attention object. Followed by layer norm and postion wise feedforward net and dropout layer.
        """
        kq_same = kq_same == 1
        # Multi-Head Attention Block
        self.masked_attn_head = MultiHeadAttention(
            d_model, d_feature, n_heads, dropout, kq_same=kq_same)

        # Two layer norm layer and two droput layer
        self.layer_norm1 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)

        self.linear1 = nn.Linear(d_model, d_ff)
        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_ff, d_model)

        self.layer_norm2 = nn.LayerNorm(d_model)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, mask, query, key, values, apply_pos=True):
        """
        Input:
            block : object of type BasicBlock(nn.Module). It contains masked_attn_head objects which is of type MultiHeadAttention(nn.Module).
            mask : 0 means, it can peek only past values. 1 means, block can peek only current and pas values
            query : Query. In transformer paper it is the input for both encoder and decoder
            key : Keys. In transformer paper it is the input for both encoder and decoder
            Values. In transformer paper it is the input for encoder and  encoded output for decoder (in masked attention part)

        Output:
            query: Input gets changed over the layer and returned.

        """

        seqlen, batch_size = query.size(1), query.size(0)
        nopeek_mask = np.triu(
            np.ones((1, 1, seqlen, seqlen)), k=mask).astype('uint8')
        src_mask = (torch.from_numpy(nopeek_mask) == 0).to(device)
        if mask == 0:  # If 0, zero-padding is needed.
            # Calls block.masked_attn_head.forward() method
            query2 = self.masked_attn_head(
                query, key, values, mask=src_mask, zero_pad=True) # 只能看到之前的信息，当前的信息也看不到，此时会把第一行score全置0，表示第一道题看不到历史的interaction信息，第一题attn之后，对应value全0
        else:
            # Calls block.masked_attn_head.forward() method
            query2 = self.masked_attn_head(
                query, key, values, mask=src_mask, zero_pad=False)

        query = query + self.dropout1((query2)) # 残差1
        query = self.layer_norm1(query) # layer norm
        if apply_pos:
            query2 = self.linear2(self.dropout( # FFN
                self.activation(self.linear1(query))))
            query = query + self.dropout2((query2)) # 残差
            query = self.layer_norm2(query) # lay norm
        return query


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, d_feature, n_heads, dropout, kq_same, bias=True):
        super().__init__()
        """
        It has projection layer for getting keys, queries and values. Followed by attention and a connected layer.
        """
        self.d_model = d_model
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

        self._reset_parameters()

    def _reset_parameters(self):
        xavier_uniform_(self.k_linear.weight)
        xavier_uniform_(self.v_linear.weight)
        if self.kq_same is False:
            xavier_uniform_(self.q_linear.weight)

        if self.proj_bias:
            constant_(self.k_linear.bias, 0.)
            constant_(self.v_linear.bias, 0.)
            if self.kq_same is False:
                constant_(self.q_linear.bias, 0.)
            constant_(self.out_proj.bias, 0.)

    def forward(self, q, k, v, mask, zero_pad):

        bs = q.size(0)

        # perform linear operation and split into h heads

        k = self.k_linear(k).view(bs, -1, self.h, self.d_k)
        if self.kq_same is False:
            q = self.q_linear(q).view(bs, -1, self.h, self.d_k)
        else:
            q = self.k_linear(q).view(bs, -1, self.h, self.d_k)
        v = self.v_linear(v).view(bs, -1, self.h, self.d_k)

        # transpose to get dimensions bs * h * sl * d_model

        k = k.transpose(1, 2)
        q = q.transpose(1, 2)
        v = v.transpose(1, 2)
        # calculate attention using function we will define next
        scores = attention(q, k, v, self.d_k,
                           mask, self.dropout, zero_pad)

        # concatenate heads and put through final linear layer
        concat = scores.transpose(1, 2).contiguous()\
            .view(bs, -1, self.d_model)

        output = self.out_proj(concat)

        return output


def attention(q, k, v, d_k, mask, dropout, zero_pad):
    """
    This is called by Multi-head atention object to find the values.
    """
    # d_k: 每一个头的dim
    scores = torch.matmul(q, k.transpose(-2, -1)) / \
        math.sqrt(d_k)  # BS, 8, seqlen, seqlen
    bs, head, seqlen = scores.size(0), scores.size(1), scores.size(2)

    scores.masked_fill_(mask == 0, -1e32)
    scores = F.softmax(scores, dim=-1)  # BS,8,seqlen,seqlen
    # print(f"before zero pad scores: {scores.shape}")
    # print(zero_pad)
    if zero_pad:
        pad_zero = torch.zeros(bs, head, 1, seqlen).to(device)
        scores = torch.cat([pad_zero, scores[:, :, 1:, :]], dim=2) # 第一行score置0
    # print(f"after zero pad scores: {scores}")
    scores = dropout(scores)
    output = torch.matmul(scores, v)
    # import sys
    # sys.exit()
    return output


class LearnablePositionalEmbedding(nn.Module):
    def __init__(self, d_model, max_len=512):
        super().__init__()
        # Compute the positional encodings once in log space.
        pe = 0.1 * torch.randn(max_len, d_model)
        pe = pe.unsqueeze(0)
        self.weight = nn.Parameter(pe, requires_grad=True)

    def forward(self, x):
        return self.weight[:, :x.size(Dim.seq), :]  # ( 1,seq,  Feature)


class CosinePositionalEmbedding(nn.Module):
    def __init__(self, d_model, max_len=512):
        super().__init__()
        # Compute the positional encodings once in log space.
        pe = 0.1 * torch.randn(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() *
                             -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.weight = nn.Parameter(pe, requires_grad=False)

    def forward(self, x):
        return self.weight[:, :x.size(Dim.seq), :]  # ( 1,seq,  Feature)
