import json
import os
import torch
from torch.nn import Module, Embedding, LSTM, Linear, Dropout, ModuleList, Parameter

class DKTPlus(Module):
    def __init__(
        self,
        num_c,
        emb_size,
        lambda_r,
        lambda_w1,
        lambda_w2,
        dropout=0.1,
        emb_type="qid",
        emb_path="",
        pretrain_dim=768,
        num_c_fmkc=None,
        dpath="",
        kc_tree_path="",
    ):
        super().__init__()
        self.model_name = "dkt+"
        self.num_c = num_c
        self.emb_size = emb_size
        self.hidden_size = emb_size
        self.lambda_r = lambda_r
        self.lambda_w1 = lambda_w1
        self.lambda_w2 = lambda_w2
        self.emb_type = emb_type

        if emb_type == "qid_fmkc":
            if num_c_fmkc is None or len(num_c_fmkc) < 1:
                raise ValueError("emb_type qid_fmkc requires num_c_fmkc")
            self.num_c_fmkc = [int(n) for n in num_c_fmkc]
            self.num_fmkc_fields = len(self.num_c_fmkc)
            self.kc_emb = ModuleList([Embedding(n, self.emb_size) for n in self.num_c_fmkc])
            self.kcr_emb = ModuleList([Embedding(n * 2, self.emb_size) for n in self.num_c_fmkc])
            self.kc_residual_emb = Embedding(self.num_c, self.emb_size)
            self.kc_residual_scale = Parameter(torch.tensor(0.1))
            self.alpha_fm1 = Parameter(torch.ones(self.num_fmkc_fields, self.emb_size))
            self.alpha_kcr_fm1 = Parameter(torch.ones(self.num_fmkc_fields, self.emb_size))
            self.fm2_scale = Parameter(torch.tensor(0.1))
            self.fm_out_scale = Parameter(torch.tensor(1.0))
            self.kcr_fm2_scale = Parameter(torch.tensor(0.1))
            self.kcr_fm_out_scale = Parameter(torch.tensor(1.0))
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
            self.fmkc_residual_collision_risk = self.fmkc_tuple_space > int(self.num_c)
        elif emb_type == "qid_tree":
            self.residual_kc_emb = Embedding(self.num_c, self.emb_size)
            self.response_emb = Embedding(2, self.emb_size)
            self.tree_mlp = torch.nn.Sequential(
                Linear(self.emb_size, self.emb_size),
                torch.nn.ReLU(),
            )
            self.edge_alpha = Parameter(torch.zeros(self.num_c))
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
            self.interaction_emb = Embedding(self.num_c * 2, self.emb_size)
        else:
            raise ValueError(f"DKTPlus unsupported emb_type: {emb_type}")
        self.lstm_layer = LSTM(self.emb_size, self.hidden_size, batch_first=True)
        self.dropout_layer = Dropout(dropout)
        if emb_type == "qid_fmkc":
            self.out_layer = Linear(self.hidden_size * 3, 1)
        else:
            self.out_layer = Linear(self.hidden_size, self.num_c)
        
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
        parent_index = [-1] * self.num_c
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
            if 0 <= cidx < self.num_c and 0 <= pidx < self.num_c:
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
        node_embs = [None] * self.num_c
        for idx in self.topo_index.tolist():
            base = residual[idx]
            pidx = int(self.parent_index[idx].item())
            if 0 <= pidx < self.num_c:
                parent_emb = node_embs[pidx]
                parent_proj = self.tree_mlp(parent_emb)
                a = torch.sigmoid(self.edge_alpha[idx])
                cur = a * parent_proj + (1.0 - a) * base
            else:
                cur = base
            node_embs[idx] = cur
        return torch.stack(node_embs, dim=0)

    def tree_kc_embed(self, q):
        q_valid = (q >= 0) & (q < self.num_c)
        safe_q = q.long().clamp(0, self.num_c - 1)
        kc_table = self._tree_kc_table()
        q_embed = kc_table[safe_q]
        return q_embed * q_valid.unsqueeze(-1).float()


    def _check_fmkc_shape(self, c_multi):
        if c_multi.dim() != 3:
            raise ValueError(f"qid_fmkc expects q shape [B, L, F], got {tuple(c_multi.shape)}")
        if c_multi.size(-1) != self.num_fmkc_fields:
            raise ValueError(f"Expected {self.num_fmkc_fields} KC fields, got {c_multi.size(-1)}")

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
            dense_valid = (dense_kc_ids >= 0) & (dense_kc_ids < self.num_c)
            valid = token_valid & dense_valid
            residual_ids = dense_kc_ids.clamp(min=0, max=self.num_c - 1)
            residual_ids = residual_ids.masked_fill(~valid, 0)
            return residual_ids, valid
        if self.fmkc_tuple_space > self.num_c:
            raise ValueError(
                "qid_fmkc residual requires q_dense when tuple space exceeds num_c. "
                f"Got tuple_space={self.fmkc_tuple_space}, num_c={self.num_c}."
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
        es = torch.stack(embs, dim=2)
        valid_count = field_valid.sum(dim=-1, keepdim=True)
        valid_count_safe = valid_count.clamp(min=1.0)
        token_valid = (valid_count > 0).float()
        alpha = alpha.view(1, 1, self.num_fmkc_fields, self.emb_size)
        first = (alpha * es).sum(dim=2) / torch.sqrt(valid_count_safe)
        s = es.sum(dim=2)
        sum_sq = (es * es).sum(dim=2)
        fm2 = 0.5 * (s * s - sum_sq)
        pair_count = valid_count_safe * (valid_count_safe - 1.0) / 2.0
        pair_count_safe = pair_count.clamp(min=1.0)
        fm2 = fm2 / torch.sqrt(pair_count_safe)
        out = fm_out_scale * (first + fm2_scale * fm2)
        return out * token_valid

    def fm_kc_embed(self, c_multi, dense_kc_ids=None):
        fm_emb = self._fm_embed(
            c_multi=c_multi,
            emb_tables=self.kc_emb,
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
            emb_tables=self.kcr_emb,
            alpha=self.alpha_kcr_fm1,
            fm2_scale=self.kcr_fm2_scale,
            fm_out_scale=self.kcr_fm_out_scale,
            r=r,
        )

    def forward(self, q, r, q_dense=None):
        emb_type = self.emb_type
        if emb_type == "qid_fmkc":
            if q_dense is None:
                raise ValueError("qid_fmkc requires q_dense to avoid residual id collision.")
            self._check_fmkc_shape(q)
            B, L, _ = q.shape
            if q_dense.dim() != 2 or q_dense.shape != q.shape[:2]:
                raise ValueError(
                    f"q_dense must have shape [B, L]={tuple(q.shape[:2])}, got {tuple(q_dense.shape)}"
                )
            xemb = self.fm_kcr_embed(q, r)
            h, _ = self.lstm_layer(xemb)
            h = self.dropout_layer(h)
            if L <= 1:
                return torch.full((B, L), 0.5, dtype=h.dtype, device=h.device)
            history = h[:, :-1, :]
            target_emb = self.fm_kc_embed(q[:, 1:, :], dense_kc_ids=q_dense[:, 1:])
            pred_features = torch.cat([history, target_emb, history * target_emb], dim=-1)
            y_next = torch.sigmoid(self.out_layer(pred_features).squeeze(-1))
            y = torch.full((B, L), 0.5, dtype=y_next.dtype, device=y_next.device)
            y[:, 1:] = y_next
            return y
        elif emb_type == "qid_tree":
            q_valid = (q >= 0) & (q < self.num_c)
            r_valid = (r >= 0) & (r <= 1)
            safe_r = r.long().clamp(0, 1)
            xemb = self.tree_kc_embed(q) + self.response_emb(safe_r)
            xemb = xemb * (q_valid & r_valid).unsqueeze(-1).float()
        elif emb_type == "qid":
            x = q + self.num_c * r
            xemb = self.interaction_emb(x)
        else:
            raise ValueError(f"DKTPlus unsupported emb_type: {emb_type}")

        h, _ = self.lstm_layer(xemb)
        h = self.dropout_layer(h)
        y = self.out_layer(h)
        y = torch.sigmoid(y)

        return y