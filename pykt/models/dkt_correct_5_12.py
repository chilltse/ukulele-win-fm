import json
import os
import torch
from torch.nn import Dropout, Embedding, LSTM, Linear, Module, ModuleList, Parameter, ReLU, Sequential

# # qid_fmkc( random residual+FMKC) + qid_tree；
class DKT(Module):
    def __init__(
        self,
        num_c,
        emb_size,
        dropout=0.1,
        emb_type="qid",
        emb_path="",
        pretrain_dim=768,
        num_c_fmkc=None,
        dpath="",
        kc_tree_path="",
    ):
        super().__init__()

        self.model_name = "dkt"
        self.num_c = num_c
        self.emb_size = emb_size
        self.hidden_size = emb_size
        self.emb_type = emb_type
        self.dpath = dpath

        if emb_type == "qid_fmkc":
            if num_c_fmkc is None or len(num_c_fmkc) < 1:
                raise ValueError("emb_type qid_fmkc requires num_c_fmkc")

            self.num_c_fmkc = [int(n) for n in num_c_fmkc]
            self.num_fmkc_fields = len(self.num_c_fmkc)

            # ---------------------------------------------------------
            # 1) Target KC embedding:
            #    Used for target-conditioned prediction.
            #    This represents q_t only, without response.
            # ---------------------------------------------------------
            self.kc_emb = ModuleList(
                [
                    Embedding(n, self.emb_size)
                    for n in self.num_c_fmkc
                ]
            )

            # ---------------------------------------------------------
            # Random residual embedding for each full KC tuple.
            #
            # Final target KC embedding becomes:
            #     target_emb(q) = FM(field_embs(q))
            #                     + residual_scale * residual_emb(tuple_id(q))
            #
            # This residual is only used for target KC embedding.
            # The response-specific LSTM input FMKC(q, r) is left unchanged.
            # ---------------------------------------------------------
            self.kc_residual_emb = Embedding(self.num_c, self.emb_size)
            self.kc_residual_scale = Parameter(torch.tensor(0.1))

            # Deterministic mixed-radix strides for converting a field tuple
            # [field_1, ..., field_F] into one tuple id.
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
            # Total Cartesian tuple space size for FMKC fields.
            fmkc_tuple_space = 1
            for n in self.num_c_fmkc:
                fmkc_tuple_space *= int(n)
            self.fmkc_tuple_space = int(fmkc_tuple_space)
            self.fmkc_residual_collision_risk = self.fmkc_tuple_space > int(self.num_c)

            # ---------------------------------------------------------
            # 2) Response-specific interaction embedding:
            #    Used as LSTM input.
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
            #
            #    This is closer to original DKT:
            #        Embedding(q + num_c * r)
            # ---------------------------------------------------------
            self.kcr_emb = ModuleList(
                [
                    Embedding(n * 2, self.emb_size)
                    for n in self.num_c_fmkc
                ]
            )

            # FM first-order weights for target KC embedding
            self.alpha_fm1 = Parameter(
                torch.ones(self.num_fmkc_fields, self.emb_size)
            )

            # FM first-order weights for response-specific interaction embedding
            self.alpha_kcr_fm1 = Parameter(
                torch.ones(self.num_fmkc_fields, self.emb_size)
            )

            # Target KC FM scales
            self.fm2_scale = Parameter(torch.tensor(0.1))
            self.fm_out_scale = Parameter(torch.tensor(1.0))

            # Input interaction FM scales
            self.kcr_fm2_scale = Parameter(torch.tensor(0.1))
            self.kcr_fm_out_scale = Parameter(torch.tensor(1.0))

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
            self.register_buffer("parent_index", torch.tensor(parent_index, dtype=torch.long))
            topo_index = self._build_topo_order(parent_index)
            self.register_buffer("topo_index", torch.tensor(topo_index, dtype=torch.long))
        elif emb_type.startswith("qid"):
            # Original DKT interaction embedding:
            # each (q, r) pair has an independent embedding.
            self.interaction_emb = Embedding(self.num_c * 2, self.emb_size)

        else:
            raise ValueError(f"DKT unsupported emb_type: {emb_type}")

        self.lstm_layer = LSTM(
            self.emb_size,
            self.hidden_size,
            batch_first=True,
        )
        self.dropout_layer = Dropout(dropout)

        if emb_type == "qid_fmkc":
            # ---------------------------------------------------------
            # Target-conditioned prediction head.
            #
            # We predict:
            #   p(r_t = 1 | history up to t-1, target KC q_t)
            #
            # Input:
            #   [h_{t-1}, target_kc_emb(q_t), h_{t-1} * target_kc_emb(q_t)]
            #
            # Shape:
            #   hidden_size * 3 -> 1
            # ---------------------------------------------------------
            self.out_layer = Linear(self.hidden_size * 3, 1)
        else:
            # Original DKT output:
            # produce probability for every KC.
            self.out_layer = Linear(self.hidden_size, self.num_c)

    def _resolve_tree_path(self, kc_tree_path, dpath):
        if not dpath:
            raise ValueError("qid_tree requires non-empty dpath.")

        expected_tree_path = os.path.abspath(os.path.join(dpath, "kc_knowledge_tree.json"))

        # Root-cause guard: stale/contaminated config may pass a tree file from another dataset.
        if kc_tree_path:
            given_tree_path = os.path.abspath(kc_tree_path)
            if os.path.normcase(given_tree_path) != os.path.normcase(expected_tree_path):
                raise ValueError(
                    "kc_tree_path points outside current dataset directory. "
                    f"Expected: {expected_tree_path}, got: {given_tree_path}"
                )

        if not os.path.exists(expected_tree_path):
            raise FileNotFoundError(
                "Standardized qid_tree path not found. "
                f"Expected file: {expected_tree_path}"
            )

        return expected_tree_path

    # def _build_tree_parent_index(self, kc_tree_path, dpath):
    #     tree_path = self._resolve_tree_path(kc_tree_path, dpath)
    #     if not tree_path:
    #         raise FileNotFoundError(
    #             "emb_type qid_tree requires kc_tree_path or default tree json under dpath."
    #         )
    #     keyid2idx_path = os.path.join(dpath, "keyid2idx.json")
    #     if not os.path.exists(keyid2idx_path):
    #         raise FileNotFoundError(
    #             f"emb_type qid_tree requires keyid2idx.json under dpath, missing: {keyid2idx_path}"
    #         )

    #     with open(tree_path, "r", encoding="utf-8") as f:
    #         tree_data = json.load(f)
    #     with open(keyid2idx_path, "r", encoding="utf-8") as f:
    #         keyid2idx = json.load(f)

    #     concepts_map = keyid2idx.get("concepts", {})
    #     if not concepts_map:
    #         raise ValueError("keyid2idx.json has no `concepts` mapping for qid_tree.")

    #     kc_name_to_id = {}
    #     for item in tree_data.get("kc_index", []):
    #         name = str(item.get("name", "")).strip()
    #         kc_id = item.get("kc_id", None)
    #         if name and kc_id is not None:
    #             kc_name_to_id[name] = int(kc_id)

    #     child_to_parent_kc = {}
    #     for edge in tree_data.get("non_tree_prerequisite_edges", []):
    #         parent_name = str(edge.get("from", "")).strip()
    #         child_name = str(edge.get("to", "")).strip()
    #         if parent_name in kc_name_to_id and child_name in kc_name_to_id:
    #             child_to_parent_kc[kc_name_to_id[child_name]] = kc_name_to_id[parent_name]

    #     parent_index = [-1] * self.num_c
    #     for raw_kc, mapped_idx in concepts_map.items():
    #         try:
    #             child_kc_id = int(raw_kc)
    #         except Exception:
    #             continue
    #         parent_kc_id = child_to_parent_kc.get(child_kc_id, None)
    #         if parent_kc_id is None:
    #             continue
    #         parent_raw = str(parent_kc_id)
    #         if parent_raw not in concepts_map:
    #             continue
    #         cidx = int(mapped_idx)
    #         pidx = int(concepts_map[parent_raw])
    #         if 0 <= cidx < self.num_c and 0 <= pidx < self.num_c:
    #             parent_index[cidx] = pidx
    #     return parent_index
#################
    def _build_tree_parent_index(self, kc_tree_path, dpath):
        tree_path = self._resolve_tree_path(kc_tree_path, dpath)

        keyid2idx_tree_path = os.path.join(dpath, "keyid2idx_tree.json")
        if not os.path.exists(keyid2idx_tree_path):
            raise FileNotFoundError(
                "Standardized qid_tree index mapping not found. "
                f"Expected file: {keyid2idx_tree_path}"
            )

        with open(tree_path, "r", encoding="utf-8") as f:
            tree_data = json.load(f)

        with open(keyid2idx_tree_path, "r", encoding="utf-8") as f:
            keyid2idx_tree = json.load(f)

        concepts_map = keyid2idx_tree.get("concepts", {})
        tree_num_c = int(keyid2idx_tree.get("num_c", len(concepts_map)))

        if not concepts_map:
            raise ValueError("keyid2idx_tree.json has no `concepts` mapping for qid_tree.")

        if tree_num_c <= 0:
            raise ValueError("keyid2idx_tree.json has invalid `num_c` for qid_tree.")

        # Do not mask config bugs: qid_tree should initialize with tree num_c.
        if self.num_c != tree_num_c:
            raise ValueError(
                "num_c mismatch for qid_tree. "
                f"Model num_c={self.num_c}, keyid2idx_tree num_c={tree_num_c}. "
                "Use num_c_tree when initializing qid_tree model."
            )

        # ---------------------------------------------------------
        # 1) Recursively collect all nodes from nested JSON.
        # ---------------------------------------------------------
        nodes = []

        def collect_nodes(obj):
            if isinstance(obj, list):
                for x in obj:
                    collect_nodes(x)
                return

            if not isinstance(obj, dict):
                return

            nodes.append(obj)

            for child in obj.get("children", []) or []:
                collect_nodes(child)

        collect_nodes(tree_data)

        # ---------------------------------------------------------
        # 2) Directly build parent_index using:
        #
        #    child_node_id  -> concepts[child_node_id]
        #    parent_node_id -> concepts[parent_node_id]
        #
        # Your keyid2idx_tree.json already contains both internal
        # nodes and leaf KCs, so we do not need kc_id conversion here.
        # ---------------------------------------------------------
        parent_index = [-1] * tree_num_c

        raw_edges = 0
        mapped_edges = 0
        skipped_child_missing = 0
        skipped_parent_missing = 0
        skipped_out_of_range = 0

        for node in nodes:
            child_node_id = node.get("node_id", None)
            parent_node_id = node.get("parent_id", None)

            # Root node has no parent.
            if child_node_id is None or parent_node_id is None:
                continue

            child_raw = str(child_node_id)
            parent_raw = str(parent_node_id)

            raw_edges += 1

            if child_raw not in concepts_map:
                skipped_child_missing += 1
                continue

            if parent_raw not in concepts_map:
                skipped_parent_missing += 1
                continue

            child_idx = int(concepts_map[child_raw])
            parent_idx = int(concepts_map[parent_raw])

            if not (0 <= child_idx < tree_num_c and 0 <= parent_idx < tree_num_c):
                skipped_out_of_range += 1
                continue

            parent_index[child_idx] = parent_idx
            mapped_edges += 1

        # ---------------------------------------------------------
        # 3) Debug log.
        # ---------------------------------------------------------
        leaf_count = sum(1 for n in nodes if n.get("type") == "kc_leaf")
        internal_count = sum(1 for n in nodes if n.get("type") == "internal")
        parent_edges = sum(1 for p in parent_index if p >= 0)
        root_or_no_parent_nodes = sum(1 for p in parent_index if p < 0)

        print("\n========== qid_tree debug ==========")
        print(f"tree_path = {tree_path}")
        print(f"keyid2idx_tree_path = {keyid2idx_tree_path}")
        print(f"input num_c = {self.num_c}")
        print(f"tree_num_c = {tree_num_c}")
        print(f"concepts in keyid2idx_tree = {len(concepts_map)}")
        print(f"json total nodes = {len(nodes)}")
        print(f"json internal nodes = {internal_count}")
        print(f"json leaf nodes = {leaf_count}")
        print(f"raw json edges = {raw_edges}")
        print(f"mapped parent edges = {mapped_edges}")
        print(f"parent_edges in parent_index = {parent_edges}")
        print(f"root_or_no_parent_nodes = {root_or_no_parent_nodes}")
        print(f"skipped_child_missing = {skipped_child_missing}")
        print(f"skipped_parent_missing = {skipped_parent_missing}")
        print(f"skipped_out_of_range = {skipped_out_of_range}")

        if parent_edges > 0:
            print("tree structure status = USED")
            print("sample mapped parent edges: child_idx -> parent_idx")
            shown = 0
            for cidx, pidx in enumerate(parent_index):
                if pidx >= 0:
                    print(f"  {cidx} -> {pidx}")
                    shown += 1
                    if shown >= 10:
                        break
        else:
            print("tree structure status = NOT USED / NO VALID PARENT EDGES MAPPED")
            print("warning: qid_tree may have degraded to independent residual KC embeddings.")

        print("====================================\n")

        return parent_index
#################

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
                f"Expected {self.num_fmkc_fields} KC fields, "
                f"got {c_multi.size(-1)}"
            )

    def fmkc_token_valid(self, c_multi):
        """
        A token is valid if at least one field id is >= 0.

        c_multi:
            [B, L, F]

        return:
            [B, L] bool
        """
        self._check_fmkc_shape(c_multi)
        return (c_multi >= 0).any(dim=-1)

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
            build target KC embedding:
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

        # field_valid: [B, L, F]
        field_valid_bool = c_multi >= 0

        if r is not None:
            response_valid_bool = (r >= 0) & (r <= 1)
            field_valid_bool = field_valid_bool & response_valid_bool.unsqueeze(-1)
            ri = r.long().clamp(0, 1)
        else:
            ri = None

        field_valid = field_valid_bool.float()

        # Prevent negative ids from entering Embedding.
        # Invalid positions will be masked out immediately after lookup.
        safe_ids = c_multi.clamp(min=0)

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

        # -------------------------------------------------------------
        # First-order FM term:
        #   sum_i alpha_i * e_i
        # -------------------------------------------------------------
        alpha = alpha.view(1, 1, self.num_fmkc_fields, self.emb_size)
        first = (alpha * es).sum(dim=2) / torch.sqrt(valid_count_safe)

        # -------------------------------------------------------------
        # Second-order FM term:
        #
        #   1/2 * [ (sum_i e_i)^2 - sum_i(e_i^2) ]
        #
        # This equals:
        #   sum_{i < j} e_i * e_j
        #
        # where * is element-wise product.
        # -------------------------------------------------------------
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

        return:
            residual_ids: [B, L] long
            token_valid:  [B, L] bool
            used_dense_ids: bool
        """
        self._check_fmkc_shape(c_multi)

        c_multi = c_multi.long()
        token_valid = (c_multi >= 0).any(dim=-1)

        # Preferred path: use external dense KC ids (collision-free if valid).
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
                "qid_fmkc residual requires q_dense when tuple space exceeds num_c. "
                f"Got tuple_space={self.fmkc_tuple_space}, num_c={self.num_c}."
            )

        # Prevent invalid / padding fields from entering id computation.
        safe_ids = c_multi.clamp(min=0)

        strides = self.fmkc_strides.view(1, 1, self.num_fmkc_fields)
        residual_ids = (safe_ids * strides).sum(dim=-1)

        # No modulo fallback is allowed here:
        # when tuple_space <= num_c, mixed-radix ids are already in range.
        # Padding tokens use row 0 but are masked out later.
        residual_ids = residual_ids.masked_fill(~token_valid, 0)

        return residual_ids, token_valid, False

    def fm_kc_embed(self, c_multi, dense_kc_ids=None):
        """
        Target KC embedding.

        Used for target-conditioned prediction.

        c_multi:
            [B, L, F]

        return:
            [B, L, D]
        """
        fm_emb = self._fm_embed(
            c_multi=c_multi,
            emb_tables=self.kc_emb,
            alpha=self.alpha_fm1,
            fm2_scale=self.fm2_scale,
            fm_out_scale=self.fm_out_scale,
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

        Used as LSTM input.

        c_multi:
            [B, L, F]

        r:
            [B, L]

        return:
            [B, L, D]
        """
        return self._fm_embed(
            c_multi=c_multi,
            emb_tables=self.kcr_emb,
            alpha=self.alpha_kcr_fm1,
            fm2_scale=self.kcr_fm2_scale,
            fm_out_scale=self.kcr_fm_out_scale,
            r=r,
        )

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(self, q, r, q_dense):
        """
        qid mode:
            q: [B, L]
            r: [B, L]
            return:
                y: [B, L, num_c]

        qid_fmkc mode:
            q: [B, L, F]
            r: [B, L]
            return:
                y: [B, L]

            Important:
                y[:, t] predicts r[:, t] conditioned on:
                    history up to t-1
                    target KC q[:, t]

                y[:, 0] is filled with 0.5 because there is no previous history.
                In loss calculation, you should ignore position 0.
        """

        if self.emb_type == "qid_fmkc":
            self._check_fmkc_shape(q)
            if q_dense is None:
                raise ValueError("qid_fmkc requires q_dense to avoid residual id collision.")
            if q_dense.dim() != 2:
                raise ValueError(
                    f"qid_fmkc expects q_dense shape [B, L], got {tuple(q_dense.shape)}"
                )
            if q_dense.shape != q.shape[:2]:
                raise ValueError(
                    f"qid_fmkc expects q_dense shape {tuple(q.shape[:2])}, got {tuple(q_dense.shape)}"
                )

            B, L, _ = q.shape

            # ---------------------------------------------------------
            # Build response-specific interaction embedding:
            #
            # input at time t:
            #   FMKC(q_t, r_t)
            #
            # Invalid/padding interactions are fully masked to zero.
            # ---------------------------------------------------------
            interaction_valid = self.fmkc_interaction_valid(q, r)
            xemb = self.fm_kcr_embed(q, r)

            # Extra safety mask.
            xemb = xemb * interaction_valid.unsqueeze(-1).float()

            h, _ = self.lstm_layer(xemb)
            h = self.dropout_layer(h)

            # If sequence length is 1, there is no valid target-conditioned
            # prediction because no previous history exists.
            if L <= 1:
                return torch.full(
                    (B, L),
                    0.5,
                    dtype=h.dtype,
                    device=h.device,
                )

            # ---------------------------------------------------------
            # Target-conditioned prediction.
            #
            # h[:, :-1, :] represents history after observing interaction t-1.
            # It is used to predict response at time t.
            #
            # target_emb[:, t-1, :] corresponds to q[:, t, :].
            # ---------------------------------------------------------
            history = h[:, :-1, :]
            target_q = q[:, 1:, :]
            target_q_dense = q_dense[:, 1:]
            target_emb = self.fm_kc_embed(target_q, dense_kc_ids=target_q_dense)

            pred_features = torch.cat(
                [
                    history,
                    target_emb,
                    history * target_emb,
                ],
                dim=-1,
            )

            logits = self.out_layer(pred_features).squeeze(-1)
            y_next = torch.sigmoid(logits)

            # ---------------------------------------------------------
            # Valid prediction requires:
            #   1. previous interaction is valid
            #   2. current target KC is valid
            #
            # y[:, 1:] predicts r[:, 1:].
            # y[:, 0] is neutral 0.5.
            # ---------------------------------------------------------
            prev_valid = interaction_valid[:, :-1]
            target_valid = self.fmkc_token_valid(target_q)
            pred_valid = prev_valid & target_valid

            y = torch.full(
                (B, L),
                0.5,
                dtype=y_next.dtype,
                device=y_next.device,
            )

            y[:, 1:] = torch.where(
                pred_valid,
                y_next,
                torch.full_like(y_next, 0.5),
            )

            return y

        elif self.emb_type == "qid_tree":
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

            h, _ = self.lstm_layer(xemb)
            h = self.dropout_layer(h)

            y = self.out_layer(h)
            y = torch.sigmoid(y)
            return y
        elif self.emb_type == "qid":
            # Original DKT mode.
            #
            # I added basic padding safety here:
            # invalid q/r positions are masked to zero input.
            q_valid = (q >= 0) & (q < self.num_c)
            r_valid = (r >= 0) & (r <= 1)
            interaction_valid = q_valid & r_valid

            safe_q = q.long().clamp(0, self.num_c - 1)
            safe_r = r.long().clamp(0, 1)

            x = safe_q + self.num_c * safe_r
            xemb = self.interaction_emb(x)

            # Mask invalid/padding positions.
            xemb = xemb * interaction_valid.unsqueeze(-1).float()

            h, _ = self.lstm_layer(xemb)
            h = self.dropout_layer(h)

            y = self.out_layer(h)
            y = torch.sigmoid(y)

            return y

        else:
            raise ValueError(f"DKT forward unsupported emb_type: {self.emb_type}")