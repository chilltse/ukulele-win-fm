import json
import os
import torch
from torch.nn import Dropout, Embedding, GELU, LayerNorm, LSTM, Linear, Module, ModuleList, Parameter, ReLU, Sequential

# 改进版的FMKC结构

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

            # ---------------------------------------------------------
            # Residual FMKC improvements
            #
            # Instead of directly using:
            #   first + fm2_scale * fm2
            #
            # We use:
            #   LayerNorm(first + bias + gate * fm2_scale * MLP(fm2))
            #
            # This keeps the first-order field composition as the main
            # representation and lets the second-order FM interaction act
            # as a learnable residual branch.
            # ---------------------------------------------------------

            # Bias for first-order target KC embedding and interaction embedding.
            self.fmkc_bias = Parameter(torch.zeros(self.emb_size))
            self.kcr_fmkc_bias = Parameter(torch.zeros(self.emb_size))

            # Residual gates for second-order FM branches.
            # Initialize with -3.0 so sigmoid(gate) is small at the beginning.
            # This makes the model start close to a stable first-order model.
            self.fmkc_fm2_gate = Parameter(torch.tensor(-3.0))
            self.kcr_fm2_gate = Parameter(torch.tensor(-3.0))

            # Transform the raw second-order FM vector before residual addition.
            self.fmkc_fm2_mlp = Sequential(
                Linear(self.emb_size, self.emb_size),
                GELU(),
                Linear(self.emb_size, self.emb_size),
            )
            self.kcr_fm2_mlp = Sequential(
                Linear(self.emb_size, self.emb_size),
                GELU(),
                Linear(self.emb_size, self.emb_size),
            )

            # Normalize after residual addition.
            self.fmkc_norm = LayerNorm(self.emb_size)
            self.kcr_fmkc_norm = LayerNorm(self.emb_size)

            # Dropout only on the residual interaction branch.
            self.fmkc_branch_dropout = Dropout(dropout)
            self.kcr_fmkc_branch_dropout = Dropout(dropout)

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

            # Non-leaf knowledge-state output is derived from leaf descendants.
            # This avoids directly trusting raw non-leaf output neurons, which may
            # receive weak or no direct supervision when training mainly uses leaves.
            leaf_descendant_distance, non_leaf_mask = self._build_leaf_descendant_distance(parent_index)
            self.register_buffer("leaf_descendant_distance", leaf_descendant_distance)
            self.register_buffer("non_leaf_mask", non_leaf_mask)
            self.tree_desc_alpha = Parameter(torch.tensor(0.1))
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
        if kc_tree_path and os.path.exists(kc_tree_path):
            return kc_tree_path
        if dpath:
            default_path = os.path.join(
                dpath,
                "2_DBE_KT22_datafiles_100102_csv",
                "kc_knowledge_tree_original.json",
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
        keyid2idx_path = os.path.join(dpath, "keyid2idx.json")
        if not os.path.exists(keyid2idx_path):
            raise FileNotFoundError(
                f"emb_type qid_tree requires keyid2idx.json under dpath, missing: {keyid2idx_path}"
            )

        with open(tree_path, "r", encoding="utf-8") as f:
            tree_data = json.load(f)
        with open(keyid2idx_path, "r", encoding="utf-8") as f:
            keyid2idx = json.load(f)

        concepts_map = keyid2idx.get("concepts", {})
        if not concepts_map:
            raise ValueError("keyid2idx.json has no `concepts` mapping for qid_tree.")

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

    def _build_leaf_descendant_distance(self, parent_index):
        """
        Build leaf-descendant distances for tree-aware non-leaf output.

        leaf_descendant_distance[node, leaf] = distance from node to leaf
        if leaf is inside node's subtree; otherwise -1.

        Later in forward, non-leaf states are computed by an exponential
        weighted average over descendant leaf predictions:
            weight = exp(-alpha * distance)
        where alpha is learnable.
        """
        n = len(parent_index)
        children = [[] for _ in range(n)]
        for child, parent in enumerate(parent_index):
            if 0 <= parent < n:
                children[parent].append(child)

        non_leaf_mask = torch.tensor(
            [len(child_list) > 0 for child_list in children],
            dtype=torch.bool,
        )

        leaf_descendant_distance = torch.full((n, n), -1.0, dtype=torch.float)

        for start in range(n):
            stack = [(start, 0)]
            visited = set()

            while stack:
                node, dist = stack.pop()
                if node in visited:
                    raise ValueError("Cycle detected while building leaf descendant distance.")
                visited.add(node)

                if len(children[node]) == 0:
                    leaf_descendant_distance[start, node] = float(dist)
                else:
                    for child in children[node]:
                        stack.append((child, dist + 1))

        # Safety fallback: every node should at least point to itself if no leaf was found.
        for node in range(n):
            if (leaf_descendant_distance[node] >= 0).sum() == 0:
                leaf_descendant_distance[node, node] = 0.0

        return leaf_descendant_distance, non_leaf_mask

    def _aggregate_tree_output(self, y_raw):
        """
        Make non-leaf knowledge states depend on descendant leaf predictions.

        Leaf nodes keep their own raw prediction.
        Non-leaf nodes use an exponential distance-weighted average of their
        descendant leaf predictions.
        """
        valid = (self.leaf_descendant_distance >= 0).float()
        distance = self.leaf_descendant_distance.clamp(min=0.0)

        alpha = torch.nn.functional.softplus(self.tree_desc_alpha)
        weights = torch.exp(-alpha * distance) * valid
        weights = weights / weights.sum(dim=1, keepdim=True).clamp(min=1e-8)

        y_agg = torch.matmul(y_raw, weights.t())
        non_leaf_mask = self.non_leaf_mask.view(1, 1, -1)

        return torch.where(non_leaf_mask, y_agg, y_raw)

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
        bias,
        fm2_gate,
        fm2_mlp,
        norm_layer,
        branch_dropout,
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
        # First-order FM term with softmax-normalized field weights.
        #
        # Original version:
        #   sum_i alpha_i * e_i / sqrt(valid_count)
        #
        # Improved version:
        #   sum_i softmax(alpha_i) * e_i
        #
        # For every embedding dimension, the valid field weights sum to 1.
        # This makes the first-order composition more stable and more
        # interpretable than unconstrained alpha weights.
        # -------------------------------------------------------------
        alpha_logits = alpha.view(1, 1, self.num_fmkc_fields, self.emb_size)

        # field_mask: [B, L, F, 1]
        field_mask = field_valid_bool.unsqueeze(-1)

        # Invalid fields should not receive first-order weight.
        alpha_logits = alpha_logits.masked_fill(~field_mask, -1e9)

        # weights: [B, L, F, D]
        weights = torch.softmax(alpha_logits, dim=2)
        weights = weights * field_mask.float()

        # Safety renormalization for fully padded / partially missing tokens.
        weights = weights / weights.sum(dim=2, keepdim=True).clamp(min=1e-8)

        first = (weights * es).sum(dim=2)

        # Add a vector bias to the first-order base representation.
        base = first + bias.view(1, 1, self.emb_size)

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

        # If a token has fewer than two valid fields, it has no true pairwise
        # interaction. Force its second-order term to zero.
        has_pair = (valid_count >= 2).float()
        fm2 = fm2 * has_pair

        # -------------------------------------------------------------
        # Residual second-order branch:
        #
        #   out = LayerNorm(base + gate * fm2_scale * MLP(fm2))
        #
        # The small initial gate prevents the FM2 branch from disturbing the
        # first-order KC semantics too early in training.
        # -------------------------------------------------------------
        fm2_branch = fm2_mlp(fm2)
        fm2_branch = branch_dropout(fm2_branch)

        gate = torch.sigmoid(fm2_gate)
        out = base + gate * fm2_scale * fm2_branch

        out = norm_layer(out)
        out = fm_out_scale * out

        # Fully mask invalid tokens.
        out = out * token_valid

        return out

    def fm_kc_embed(self, c_multi):
        """
        Target KC embedding.

        Used for target-conditioned prediction.

        c_multi:
            [B, L, F]

        return:
            [B, L, D]
        """
        return self._fm_embed(
            c_multi=c_multi,
            emb_tables=self.kc_emb,
            alpha=self.alpha_fm1,
            fm2_scale=self.fm2_scale,
            fm_out_scale=self.fm_out_scale,
            bias=self.fmkc_bias,
            fm2_gate=self.fmkc_fm2_gate,
            fm2_mlp=self.fmkc_fm2_mlp,
            norm_layer=self.fmkc_norm,
            branch_dropout=self.fmkc_branch_dropout,
            r=None,
        )

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
            bias=self.kcr_fmkc_bias,
            fm2_gate=self.kcr_fm2_gate,
            fm2_mlp=self.kcr_fm2_mlp,
            norm_layer=self.kcr_fmkc_norm,
            branch_dropout=self.kcr_fmkc_branch_dropout,
            r=r,
        )

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(self, q, r):
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
            target_emb = self.fm_kc_embed(target_q)

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

            y_raw = self.out_layer(h)
            y_raw = torch.sigmoid(y_raw)
            y = self._aggregate_tree_output(y_raw)
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