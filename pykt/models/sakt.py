import json
import os
import torch

# # qid_fmkc( random residual+FMKC) + qid_tree；
from torch.nn import (
    Module,
    Embedding,
    Linear,
    MultiheadAttention,
    LayerNorm,
    Dropout,
    ModuleList,
    Parameter,
    ReLU,
    Sequential,
)
from .utils import transformer_FFN, pos_encode, ut_mask, get_clones


# qid_fmkc(random residual + FMKC) + qid_tree
class SAKT(Module):
    def __init__(
        self,
        num_c,
        seq_len,
        emb_size,
        num_attn_heads,
        dropout,
        num_en=2,
        emb_type="qid",
        emb_path="",
        pretrain_dim=768,
        num_c_fmkc=None,
        dpath="",
        kc_tree_path="",
    ):
        super().__init__()
        self.model_name = "sakt"
        self.emb_type = emb_type

        self.num_c = num_c
        self.seq_len = seq_len
        self.emb_size = emb_size
        self.num_attn_heads = num_attn_heads
        self.dropout = dropout
        self.num_en = num_en
        self.dpath = dpath

        if emb_type == "qid_fmkc":
            if num_c_fmkc is None or len(num_c_fmkc) < 1:
                raise ValueError("emb_type qid_fmkc requires num_c_fmkc")

            self.num_c_fmkc = [int(n) for n in num_c_fmkc]
            self.num_fmkc_fields = len(self.num_c_fmkc)

            # ---------------------------------------------------------
            # 1) Target KC embedding:
            #    Used as the query embedding in SAKT.
            #    This represents q_t only, without response.
            # ---------------------------------------------------------
            self.exercise_emb_fmkc = ModuleList(
                [Embedding(n, emb_size) for n in self.num_c_fmkc]
            )

            # ---------------------------------------------------------
            # Random residual embedding for each full KC tuple.
            #
            # Final target KC embedding becomes:
            #     target_emb(q) = FM(field_embs(q))
            #                     + residual_scale * residual_emb(dense_id(q))
            #
            # This residual is only used for target KC/query embedding.
            # The response-specific interaction embedding FMKC(q, r) is
            # left unchanged.
            # ---------------------------------------------------------
            self.kc_residual_emb = Embedding(self.num_c, self.emb_size)
            self.kc_residual_scale = Parameter(torch.tensor(0.1))

            # Deterministic mixed-radix strides for converting a field tuple
            # [field_1, ..., field_F] into one tuple id when dense ids are
            # not available and tuple_space <= num_c.
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

            # ---------------------------------------------------------
            # 2) Response-specific interaction embedding:
            #    Used as key/value input in SAKT.
            #
            #    Instead of:
            #        FMKC(q) + global r_emb(r)
            #
            #    We use:
            #        FMKC(q, r)
            #
            #    For each field:
            #        id_correct = field_id + n * 1
            #        id_wrong   = field_id + n * 0
            # ---------------------------------------------------------
            self.interaction_emb_fmkc = ModuleList(
                [Embedding(n * 2, emb_size) for n in self.num_c_fmkc]
            )

            # FM first-order weights for target KC/query embedding.
            self.alpha_q_fm1 = Parameter(torch.ones(self.num_fmkc_fields, emb_size))

            # FM first-order weights for response-specific interaction embedding.
            self.alpha_qr_fm1 = Parameter(torch.ones(self.num_fmkc_fields, emb_size))

            # Target KC/query FM scales.
            self.q_fm2_scale = Parameter(torch.tensor(0.1))
            self.q_fm_out_scale = Parameter(torch.tensor(1.0))

            # Input interaction FM scales.
            self.qr_fm2_scale = Parameter(torch.tensor(0.1))
            self.qr_fm_out_scale = Parameter(torch.tensor(1.0))

        elif emb_type == "qid_tree":
            self.residual_kc_emb = Embedding(self.num_c, self.emb_size)
            self.response_emb = Embedding(2, self.emb_size)
            self.tree_mlp = Sequential(
                Linear(self.emb_size, self.emb_size),
                ReLU(),
            )
            # Per child node edge scalar a; use sigmoid(a) in forward.
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
            self.interaction_emb = Embedding(num_c * 2, emb_size)
            self.exercise_emb = Embedding(num_c, emb_size)
        else:
            raise ValueError(f"SAKT unsupported emb_type: {emb_type}")

        self.position_emb = Embedding(seq_len, emb_size)

        self.blocks = get_clones(
            Blocks(emb_size, num_attn_heads, dropout),
            self.num_en,
        )

        self.dropout_layer = Dropout(dropout)
        self.pred = Linear(self.emb_size, 1)

    # ------------------------------------------------------------------
    # Utilities for qid_tree
    # ------------------------------------------------------------------

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
        qemb = kc_table[safe_q]
        qemb = qemb * q_valid.unsqueeze(-1).float()
        return qemb

    def tree_kcr_embed(self, q, r):
        q_valid = (q >= 0) & (q < self.num_c)
        r_valid = (r >= 0) & (r <= 1)
        interaction_valid = q_valid & r_valid

        safe_q = q.long().clamp(0, self.num_c - 1)
        safe_r = r.long().clamp(0, 1)

        kc_table = self._tree_kc_table()
        qemb = kc_table[safe_q]
        remb = self.response_emb(safe_r)
        xemb = qemb + remb
        xemb = xemb * interaction_valid.unsqueeze(-1).float()
        return xemb

    # ------------------------------------------------------------------
    # Utilities for FMKC
    # ------------------------------------------------------------------

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
        """
        A token is valid if at least one field id is inside its valid range.

        c_multi:
            [B, L, F]

        return:
            [B, L] bool
        """
        return self._fmkc_field_valid_bool(c_multi).any(dim=-1)

    def fmkc_interaction_valid(self, c_multi, r):
        """
        An interaction is valid only if:
            1. q has at least one valid field
            2. response is 0 or 1

        c_multi:
            [B, L, F]

        r:
            [B, L]

        return:
            [B, L] bool
        """
        token_valid = self.fmkc_token_valid(c_multi)
        response_valid = (r >= 0) & (r <= 1)
        return token_valid & response_valid

    def _fm_embed(
        self,
        c_multi,
        emb_tables,
        alpha,
        fm2_scale,
        fm_out_scale,
        r=None,
    ):
        """
        General FM-style embedding for multi-field KC.

        If r is None:
            build target KC/query embedding:
                field_id -> embedding

        If r is not None:
            build response-specific interaction embedding:
                field_id + num_field_values * response -> embedding

        c_multi:
            [B, L, F]

        r:
            None or [B, L]

        return:
            [B, L, D]
        """
        self._check_fmkc_shape(c_multi)

        c_multi = c_multi.long()

        # field_valid_bool: [B, L, F]
        field_valid_bool = self._fmkc_field_valid_bool(c_multi)

        if r is not None:
            response_valid_bool = (r >= 0) & (r <= 1)
            field_valid_bool = field_valid_bool & response_valid_bool.unsqueeze(-1)
            ri = r.long().clamp(0, 1)
        else:
            ri = None

        field_valid = field_valid_bool.float()

        # Prevent negative or out-of-range ids from entering Embedding.
        # Invalid positions will be masked out immediately after lookup.
        safe_ids = self._fmkc_safe_ids(c_multi)

        embs = []
        for i in range(self.num_fmkc_fields):
            ids_i = safe_ids[..., i]

            if r is not None:
                # Response-specific id:
                # wrong:   field_id + n_i * 0
                # correct: field_id + n_i * 1
                ids_i = ids_i + self.num_c_fmkc[i] * ri

            e_i = emb_tables[i](ids_i)

            # Mask invalid field values.
            e_i = e_i * field_valid[..., i].unsqueeze(-1)
            embs.append(e_i)

        # es: [B, L, F, D]
        es = torch.stack(embs, dim=2)

        # valid_count: [B, L, 1]
        valid_count = field_valid.sum(dim=-1, keepdim=True)
        valid_count_safe = valid_count.clamp(min=1.0)

        # token_valid: [B, L, 1]
        token_valid = (valid_count > 0).float()

        # First-order FM term: sum_i alpha_i * e_i.
        alpha = alpha.view(1, 1, self.num_fmkc_fields, self.emb_size)
        first = (alpha * es).sum(dim=2) / torch.sqrt(valid_count_safe)

        # Second-order FM term:
        #   1/2 * [ (sum_i e_i)^2 - sum_i(e_i^2) ]
        s = es.sum(dim=2)
        sum_sq = (es * es).sum(dim=2)
        fm2 = 0.5 * (s * s - sum_sq)

        pair_count = valid_count_safe * (valid_count_safe - 1.0) / 2.0
        pair_count_safe = pair_count.clamp(min=1.0)
        fm2 = fm2 / torch.sqrt(pair_count_safe)

        out = fm_out_scale * (first + fm2_scale * fm2)

        # Fully mask invalid tokens.
        out = out * token_valid
        return out

    def _fmkc_tuple_residual_ids(self, c_multi, dense_kc_ids=None):
        """
        Convert a multi-field KC tuple into a deterministic residual id.

        c_multi:
            [B, L, F]

        dense_kc_ids:
            None or [B, L]

        return:
            residual_ids: [B, L] long
            token_valid:  [B, L] bool
            used_dense_ids: bool
        """
        self._check_fmkc_shape(c_multi)

        c_multi = c_multi.long()
        token_valid = self.fmkc_token_valid(c_multi)

        # Preferred path: use external dense KC ids.
        # This is collision-free if dense ids are valid full-KC ids.
        if dense_kc_ids is not None:
            if dense_kc_ids.dim() != 2:
                raise ValueError(
                    f"dense_kc_ids must have shape [B, L], got {tuple(dense_kc_ids.shape)}"
                )
            if dense_kc_ids.shape != c_multi.shape[:2]:
                raise ValueError(
                    f"dense_kc_ids shape {tuple(dense_kc_ids.shape)} must match c_multi [B, L] {tuple(c_multi.shape[:2])}"
                )

            dense_kc_ids = dense_kc_ids.long()
            dense_valid = (dense_kc_ids >= 0) & (dense_kc_ids < self.num_c)
            valid = token_valid & dense_valid
            residual_ids = dense_kc_ids.clamp(min=0, max=self.num_c - 1)
            residual_ids = residual_ids.masked_fill(~valid, 0)
            return residual_ids, valid, True

        if self.fmkc_tuple_space > self.num_c:
            raise ValueError(
                "qid_fmkc residual requires qry_dense when tuple space exceeds num_c. "
                f"Got tuple_space={self.fmkc_tuple_space}, num_c={self.num_c}."
            )

        # Fallback is allowed only when tuple_space <= num_c.
        # No modulo fallback is used, so tuple ids do not collide in this case.
        safe_ids = self._fmkc_safe_ids(c_multi)
        strides = self.fmkc_strides.view(1, 1, self.num_fmkc_fields)
        residual_ids = (safe_ids * strides).sum(dim=-1)
        residual_ids = residual_ids.masked_fill(~token_valid, 0)
        return residual_ids, token_valid, False

    def fm_kc_embed(self, c_multi, dense_kc_ids=None):
        """
        Target KC/query embedding.

        Used as the query embedding in SAKT.

        c_multi:
            [B, L, F]

        dense_kc_ids:
            None or [B, L]

        return:
            [B, L, D]
        """
        fm_emb = self._fm_embed(
            c_multi=c_multi,
            emb_tables=self.exercise_emb_fmkc,
            alpha=self.alpha_q_fm1,
            fm2_scale=self.q_fm2_scale,
            fm_out_scale=self.q_fm_out_scale,
            r=None,
        )

        residual_ids, token_valid, _ = self._fmkc_tuple_residual_ids(
            c_multi,
            dense_kc_ids=dense_kc_ids,
        )
        residual_emb = self.kc_residual_emb(residual_ids)
        residual_emb = residual_emb * token_valid.unsqueeze(-1).float()
        return fm_emb + self.kc_residual_scale * residual_emb

    def fm_kcr_embed(self, c_multi, r):
        """
        Response-specific FMKC interaction embedding.

        Used as the key/value input in SAKT.

        c_multi:
            [B, L, F]

        r:
            [B, L]

        return:
            [B, L, D]
        """
        return self._fm_embed(
            c_multi=c_multi,
            emb_tables=self.interaction_emb_fmkc,
            alpha=self.alpha_qr_fm1,
            fm2_scale=self.qr_fm2_scale,
            fm_out_scale=self.qr_fm_out_scale,
            r=r,
        )

    # ------------------------------------------------------------------
    # Base embeddings
    # ------------------------------------------------------------------

    def base_emb(self, q, r, qry, qry_dense=None):
        if self.emb_type == "qid_fmkc":
            if qry_dense is None:
                raise ValueError("qid_fmkc requires qry_dense to avoid residual id collision.")
            if qry_dense.dim() != 2:
                raise ValueError(
                    f"qid_fmkc expects qry_dense shape [B, L], got {tuple(qry_dense.shape)}"
                )
            if qry_dense.shape != qry.shape[:2]:
                raise ValueError(
                    f"qid_fmkc expects qry_dense shape {tuple(qry.shape[:2])}, got {tuple(qry_dense.shape)}"
                )

            qshftemb = self.fm_kc_embed(qry, dense_kc_ids=qry_dense)
            xemb = self.fm_kcr_embed(q, r)

            interaction_valid = self.fmkc_interaction_valid(q, r)
            xemb = xemb * interaction_valid.unsqueeze(-1).float()

        elif self.emb_type == "qid_tree":
            qshftemb = self.tree_kc_embed(qry)
            xemb = self.tree_kcr_embed(q, r)

        else:
            # Original qid SAKT, with padding safety.
            q_valid = (q >= 0) & (q < self.num_c)
            r_valid = (r >= 0) & (r <= 1)
            interaction_valid = q_valid & r_valid
            qry_valid = (qry >= 0) & (qry < self.num_c)

            safe_q = q.long().clamp(0, self.num_c - 1)
            safe_r = r.long().clamp(0, 1)
            safe_qry = qry.long().clamp(0, self.num_c - 1)

            x = safe_q + self.num_c * safe_r
            qshftemb = self.exercise_emb(safe_qry)
            xemb = self.interaction_emb(x)

            qshftemb = qshftemb * qry_valid.unsqueeze(-1).float()
            xemb = xemb * interaction_valid.unsqueeze(-1).float()

        pos_ids = pos_encode(xemb.shape[1])
        if torch.is_tensor(pos_ids):
            pos_ids = pos_ids.to(xemb.device)
        posemb = self.position_emb(pos_ids)
        xemb = xemb + posemb
        return qshftemb, xemb

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(self, q, r, qry, qtest=False, q_dense=None, qry_dense=None):
        """
        qid mode:
            q:   [B, L]
            r:   [B, L]
            qry: [B, L]

        qid_fmkc mode:
            q:         [B, L, F]
            r:         [B, L]
            qry:       [B, L, F]
            qry_dense: [B, L]

            qry_dense should be the dense full-KC id corresponding to qry.
            It is required so that the random residual embedding has no
            tuple-id collision.

        qid_tree mode:
            q:   [B, L]
            r:   [B, L]
            qry: [B, L]
        """

        # Backward-compatible convenience:
        # if the 4th positional argument is a tensor, treat it as qry_dense
        # rather than qtest.
        if torch.is_tensor(qtest):
            if qry_dense is not None:
                raise ValueError("Do not pass both positional qry_dense and keyword qry_dense.")
            qry_dense = qtest
            qtest = False

        # Alias for callers that use q_dense as the dense id of target query KCs.
        if qry_dense is None and q_dense is not None:
            qry_dense = q_dense

        emb_type = self.emb_type
        qshftemb, xemb = None, None

        if emb_type in ["qid", "qid_fmkc", "qid_tree"]:
            qshftemb, xemb = self.base_emb(q, r, qry, qry_dense=qry_dense)
        else:
            raise ValueError(f"SAKT forward unsupported emb_type: {emb_type}")

        for i in range(self.num_en):
            xemb = self.blocks[i](qshftemb, xemb, xemb)

        p = torch.sigmoid(self.pred(self.dropout_layer(xemb))).squeeze(-1)
        if not qtest:
            return p
        else:
            return p, xemb


class Blocks(Module):
    def __init__(self, emb_size, num_attn_heads, dropout) -> None:
        super().__init__()

        self.attn = MultiheadAttention(emb_size, num_attn_heads, dropout=dropout)
        self.attn_dropout = Dropout(dropout)
        self.attn_layer_norm = LayerNorm(emb_size)

        self.FFN = transformer_FFN(emb_size, dropout)
        self.FFN_dropout = Dropout(dropout)
        self.FFN_layer_norm = LayerNorm(emb_size)

    def forward(self, q=None, k=None, v=None):
        q, k, v = q.permute(1, 0, 2), k.permute(1, 0, 2), v.permute(1, 0, 2)

        # attn -> drop -> skip -> norm
        causal_mask = ut_mask(seq_len=k.shape[0])
        if torch.is_tensor(causal_mask):
            causal_mask = causal_mask.to(q.device)

        attn_emb, _ = self.attn(q, k, v, attn_mask=causal_mask)

        attn_emb = self.attn_dropout(attn_emb)
        attn_emb, q = attn_emb.permute(1, 0, 2), q.permute(1, 0, 2)

        attn_emb = self.attn_layer_norm(q + attn_emb)

        emb = self.FFN(attn_emb)
        emb = self.FFN_dropout(emb)
        emb = self.FFN_layer_norm(attn_emb + emb)
        return emb