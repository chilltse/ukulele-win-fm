import json
import os
import torch
import torch.nn.functional as F
from torch.nn import Dropout, Embedding, LSTM, Linear, Module, ModuleList, Parameter, ReLU, Sequential

# # qid_fmkc( random residual+FMKC) + qid_tree（parent loss+parent prediction）；
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
        tree_aux_loss_weight=0.4, # qid_tree: must be explicitly set by training entry.
        tree_aux_decay=0.3, # qid_tree: must be explicitly set by training entry.
        tree_aux_max_depth=2, # qid_tree: must be explicitly set by training entry.
        tree_aux_negative_scale=None, # qid_tree: must be explicitly set by training entry.
        
        tree_pred_fusion_mode=None, # qid_tree: must be explicitly set by training entry.
        tree_pred_fusion_max_weight=1, # qid_tree: must be explicitly set by training entry.
        tree_pred_fusion_fixed_weight=None, # qid_tree: must be explicitly set by training entry.
        tree_pred_fusion_count_tau=None, # qid_tree: must be explicitly set by training entry.
        tree_pred_fusion_depth_decay=0.1, # ancestor prediction 融合时的深度衰减；None 表示复用 tree_aux_decay。
        tree_pred_fusion_counts_path=None, # qid_tree optional: explicit path or empty.
        # Advanced decoupling controls.
        # shared: current behavior; ancestor loss updates shared LSTM/output parameters.
        # detached: ancestor loss uses a separate aux head on h.detach(), so it
        #           does not update the shared LSTM/output head.
        tree_aux_gradient_mode=None, # qid_tree: must be explicitly set by training entry.
        # main: prediction fusion uses ancestor predictions from the main DKT output head.
        # aux:  prediction fusion uses ancestor predictions from the separate aux head.
        tree_pred_fusion_source=None, # qid_tree: must be explicitly set by training entry.
        # If False, configured prediction fusion is applied only during eval/inference.
        # Training loss remains leaf-only unless you explicitly enable this.
        tree_pred_fusion_apply_train=None, # qid_tree: must be explicitly set by training entry.
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
            # ---------------------------------------------------------
            # New qid_tree design:
            #   1) KC embeddings are learned independently.
            #      No parent embedding is added into the child embedding.
            #   2) Tree structure is used only for auxiliary supervision:
            #      leaf loss + exponentially decayed ancestor losses.
            #
            # This avoids over-smoothing sibling KC embeddings while still
            # letting parent/grandparent nodes receive weak prediction signals.
            # ---------------------------------------------------------
            # Leaf branch: original DKT path.
            self.residual_kc_emb = Embedding(self.num_c, self.emb_size)
            self.response_emb = Embedding(2, self.emb_size)

            # Non-leaf branch: completely independent forward path.
            # It has its own KC/response embeddings, LSTM, dropout, and output head.
            # It does not reuse leaf hidden states or the leaf output layer.
            self.nonleaf_kc_emb = Embedding(self.num_c, self.emb_size)
            self.nonleaf_response_emb = Embedding(2, self.emb_size)
            self.nonleaf_lstm_layer = LSTM(
                self.emb_size,
                self.hidden_size,
                batch_first=True,
            )
            self.nonleaf_dropout_layer = Dropout(dropout)
            self.nonleaf_out_layer = Linear(self.hidden_size, self.num_c)
            self._last_qid_tree_nonleaf_y = None

            required_tree_args = {
                "tree_aux_loss_weight": tree_aux_loss_weight,
                "tree_aux_decay": tree_aux_decay,
                "tree_aux_max_depth": tree_aux_max_depth,
                "tree_aux_negative_scale": tree_aux_negative_scale,
                "tree_pred_fusion_mode": tree_pred_fusion_mode,
                "tree_pred_fusion_max_weight": tree_pred_fusion_max_weight,
                "tree_pred_fusion_fixed_weight": tree_pred_fusion_fixed_weight,
                "tree_pred_fusion_count_tau": tree_pred_fusion_count_tau,
                "tree_aux_gradient_mode": tree_aux_gradient_mode,
                "tree_pred_fusion_source": tree_pred_fusion_source,
                "tree_pred_fusion_apply_train": tree_pred_fusion_apply_train,
            }
            missing_tree_args = [k for k, v in required_tree_args.items() if v is None]
            if missing_tree_args:
                raise ValueError(
                    "qid_tree requires explicit tree hyperparameters from training entry; "
                    f"missing: {missing_tree_args}. "
                    "Please set them in examples/wandb_dkt_train.py CLI."
                )

            self.tree_aux_loss_weight = float(tree_aux_loss_weight)
            self.tree_aux_decay = float(tree_aux_decay)
            self.tree_aux_max_depth = int(tree_aux_max_depth)
            self.tree_aux_negative_scale = float(tree_aux_negative_scale)

            # ---------------------------------------------------------
            # Optional inference-time prediction fusion.
            #
            # Training objective:
            #   leaf loss + ancestor auxiliary loss
            #
            # Prediction fusion:
            #   final_pred = (1 - gate) * leaf_pred + gate * ancestor_prior
            #
            # Use mode="none" to keep the original leaf-only prediction.
            # Use mode="frequency" to let sparse KCs rely more on ancestor
            # predictions and frequent KCs rely mostly on themselves.
            # ---------------------------------------------------------
            self.tree_pred_fusion_mode = str(tree_pred_fusion_mode).lower().strip()
            self.tree_pred_fusion_max_weight = float(tree_pred_fusion_max_weight)
            self.tree_pred_fusion_fixed_weight = float(tree_pred_fusion_fixed_weight)
            self.tree_pred_fusion_count_tau = float(tree_pred_fusion_count_tau)
            # ---------------------------------------------------------
            # Learnable depth-decay parameter for ancestor prediction fusion.
            #
            # We store an unconstrained raw parameter and use sigmoid(raw)
            # during forward computation, so the effective decay is always
            # in (0, 1).
            #
            # If tree_pred_fusion_depth_decay is None, reuse tree_aux_decay
            # as the initial value. Otherwise, use the explicitly provided
            # initial value. Default CLI/config value should be 0.1.
            # ---------------------------------------------------------
            init_tree_pred_fusion_depth_decay = (
                float(tree_aux_decay)
                if tree_pred_fusion_depth_decay is None
                else float(tree_pred_fusion_depth_decay)
            )
            if init_tree_pred_fusion_depth_decay < 0 or init_tree_pred_fusion_depth_decay > 1:
                raise ValueError(
                    "tree_pred_fusion_depth_decay must be in [0, 1] for depth_decay fusion"
                )

            # Avoid logit(0) / logit(1). This keeps the parameter trainable
            # even if someone passes exactly 0 or 1.
            init_tree_pred_fusion_depth_decay = min(
                max(init_tree_pred_fusion_depth_decay, 1e-6),
                1.0 - 1e-6,
            )
            self.tree_pred_fusion_depth_decay_init = init_tree_pred_fusion_depth_decay
            self.tree_pred_fusion_depth_decay_logit = Parameter(
                torch.logit(
                    torch.tensor(
                        init_tree_pred_fusion_depth_decay,
                        dtype=torch.float,
                    )
                )
            )

            # Debug counter + backward hook.
            # If this hook never prints during training, the learnable depth-decay
            # parameter is not connected to the loss graph.
            self._tree_pred_decay_backward_count = 0
            self.tree_pred_fusion_depth_decay_logit.register_hook(
                self._debug_tree_pred_fusion_depth_decay_grad
            )

            self.tree_pred_fusion_counts_path = str(tree_pred_fusion_counts_path or "")

            self.tree_aux_gradient_mode = str(tree_aux_gradient_mode).lower().strip()
            self.tree_pred_fusion_source = str(tree_pred_fusion_source).lower().strip()
            if isinstance(tree_pred_fusion_apply_train, str):
                self.tree_pred_fusion_apply_train = tree_pred_fusion_apply_train.lower().strip() in {"1", "true", "yes", "y"}
            else:
                self.tree_pred_fusion_apply_train = bool(tree_pred_fusion_apply_train)

            if self.tree_aux_max_depth < 0:
                raise ValueError("tree_aux_max_depth must be >= 0")
            if self.tree_aux_loss_weight < 0:
                raise ValueError("tree_aux_loss_weight must be >= 0")
            if self.tree_aux_decay < 0:
                raise ValueError("tree_aux_decay must be >= 0")
            if self.tree_aux_negative_scale < 0:
                raise ValueError("tree_aux_negative_scale must be >= 0")

            # Clean decoupled design: prediction fusion is either disabled,
            # or it uses only depth-decayed non-leaf predictions.
            valid_fusion_modes = {"none", "depth_decay"}
            if self.tree_pred_fusion_mode not in valid_fusion_modes:
                raise ValueError(
                    f"tree_pred_fusion_mode must be one of {sorted(valid_fusion_modes)}, "
                    f"got {self.tree_pred_fusion_mode!r}"
                )

            # These legacy arguments are still accepted by the constructor for
            # training-script compatibility, but the new implementation always
            # uses a separate non-leaf forward branch.
            self.tree_aux_gradient_mode = "separate_forward"
            self.tree_pred_fusion_source = "nonleaf"
            if self.tree_pred_fusion_max_weight < 0 or self.tree_pred_fusion_max_weight > 1:
                raise ValueError("tree_pred_fusion_max_weight must be in [0, 1]")
            if self.tree_pred_fusion_fixed_weight < 0 or self.tree_pred_fusion_fixed_weight > 1:
                raise ValueError("tree_pred_fusion_fixed_weight must be in [0, 1]")
            if self.tree_pred_fusion_count_tau <= 0:
                raise ValueError("tree_pred_fusion_count_tau must be > 0")
            # No direct range check is needed for tree_pred_fusion_depth_decay_logit:
            # the effective decay used in forward is sigmoid(logit), hence always in (0, 1).

            parent_index = self._build_tree_parent_index(kc_tree_path, dpath)
            self.register_buffer("parent_index", torch.tensor(parent_index, dtype=torch.long))

            # Keep topo_index for compatibility/debugging, although the new
            # design no longer recursively mixes parent embeddings.
            topo_index = self._build_topo_order(parent_index)
            self.register_buffer("topo_index", torch.tensor(topo_index, dtype=torch.long))

            ancestor_index, ancestor_depth = self._build_ancestor_index(
                parent_index,
                self.tree_aux_max_depth,
            )
            self.register_buffer(
                "ancestor_index",
                torch.tensor(ancestor_index, dtype=torch.long),
            )
            self.register_buffer(
                "ancestor_depth",
                torch.tensor(ancestor_depth, dtype=torch.float),
            )

            kc_train_counts = self._load_qid_tree_train_counts(
                self.tree_pred_fusion_counts_path,
                dpath,
            )
            self.register_buffer("kc_train_counts", kc_train_counts)

            print("========== qid_tree auxiliary loss config ==========")
            print("KC input embedding = independent self embedding only")
            print("parent embedding inheritance = DISABLED")
            print(f"tree_aux_loss_weight = {self.tree_aux_loss_weight}")
            print(f"tree_aux_decay = {self.tree_aux_decay}")
            print(f"tree_aux_max_depth = {self.tree_aux_max_depth}")
            print(f"tree_aux_negative_scale = {self.tree_aux_negative_scale}")
            print(f"tree_pred_fusion_mode = {self.tree_pred_fusion_mode}")
            print(f"tree_pred_fusion_max_weight = {self.tree_pred_fusion_max_weight}")
            print(f"tree_pred_fusion_fixed_weight = {self.tree_pred_fusion_fixed_weight}")
            print(f"tree_pred_fusion_count_tau = {self.tree_pred_fusion_count_tau}")
            print(f"tree_pred_fusion_depth_decay_init = {self.tree_pred_fusion_depth_decay_init}")
            print(
                "tree_pred_fusion_depth_decay_effective = "
                f"{self.get_tree_pred_fusion_depth_decay().detach().item()}"
            )
            print("tree_pred_fusion_depth_decay_learnable = True")
            print("nonleaf forward = separate embeddings + separate LSTM + separate output head")
            print("ancestor loss updates nonleaf branch only")
            print("ancestor-to-leaf contribution = depth_decay fusion only")
            print(f"tree_pred_fusion_apply_train = {self.tree_pred_fusion_apply_train}")
            print("prediction fusion = ENABLED only when tree_pred_fusion_mode == 'depth_decay'")
            print("====================================================\n")
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

        # qid_tree non-leaf prediction head is defined inside the qid_tree branch
        # above so that its forward path is independent of the leaf branch.

    def get_tree_pred_fusion_depth_decay(self):
        """
        Return the effective learnable depth-decay value for qid_tree prediction fusion.

        Raw parameter:
            self.tree_pred_fusion_depth_decay_logit

        Effective value:
            sigmoid(raw) in (0, 1)

        This method is intentionally small so logging, debugging, and forward
        computation all use the same constrained value.
        """
        if not hasattr(self, "tree_pred_fusion_depth_decay_logit"):
            raise AttributeError(
                "tree_pred_fusion_depth_decay_logit only exists for emb_type == 'qid_tree'."
            )
        return torch.sigmoid(self.tree_pred_fusion_depth_decay_logit)

    def _debug_tree_pred_fusion_depth_decay_grad(self, grad):
        """
        Backward hook for the learnable tree_pred_fusion_depth_decay_logit.

        This function is called during loss.backward() if and only if
        tree_pred_fusion_depth_decay_logit participates in the computation graph.
        If this never prints, prediction fusion is not affecting the training loss.
        """
        if not hasattr(self, "_tree_pred_decay_backward_count"):
            self._tree_pred_decay_backward_count = 0

        self._tree_pred_decay_backward_count += 1

        # Print only early steps and then every 100 backward calls to avoid flooding logs.
        if self._tree_pred_decay_backward_count <= 10 or self._tree_pred_decay_backward_count % 100 == 0:
            effective = self.get_tree_pred_fusion_depth_decay().detach().item()
            raw = self.tree_pred_fusion_depth_decay_logit.detach().item()
            grad_value = grad.detach().item()

            print(
                "[DEBUG depth_decay backward] "
                f"count={self._tree_pred_decay_backward_count}, "
                f"effective={effective:.10f}, "
                f"raw_logit={raw:.10f}, "
                f"grad={grad_value:.10e}",
                flush=True,
            )

        return grad

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
        """
        Return the independent KC embedding table for qid_tree.

        In the previous tree-inheritance version, this method recursively mixed
        parent embeddings into child embeddings. In the new auxiliary-loss
        version, KC embeddings stay fully independent; the hierarchy is used
        only in get_qid_tree_loss(...).
        """
        return self.residual_kc_emb.weight

    def _build_ancestor_index(self, parent_index, max_depth):
        """
        Build a dense ancestor lookup table.

        ancestor_index[i, d-1] gives the d-hop ancestor of node i.
        If the ancestor does not exist, it is -1.

        ancestor_depth[i, d-1] is d for valid ancestors, 0 otherwise.
        """
        n = len(parent_index)
        max_depth = int(max_depth)

        ancestor_index = [[-1 for _ in range(max_depth)] for _ in range(n)]
        ancestor_depth = [[0 for _ in range(max_depth)] for _ in range(n)]

        if max_depth <= 0:
            print("========== qid_tree ancestor debug ==========")
            print("tree_aux_max_depth = 0, ancestor auxiliary loss is disabled")
            print("============================================\n")
            return ancestor_index, ancestor_depth

        total_valid = 0
        depth_counts = [0 for _ in range(max_depth)]

        for node_idx in range(n):
            cur = node_idx
            visited = {node_idx}

            for depth in range(1, max_depth + 1):
                pidx = parent_index[cur]

                if not (0 <= pidx < n):
                    break

                # Guard against accidental cycles in the JSON tree.
                if pidx in visited:
                    break

                ancestor_index[node_idx][depth - 1] = int(pidx)
                ancestor_depth[node_idx][depth - 1] = int(depth)

                total_valid += 1
                depth_counts[depth - 1] += 1

                visited.add(pidx)
                cur = pidx

        print("========== qid_tree ancestor debug ==========")
        print(f"tree_aux_max_depth = {max_depth}")
        print(f"valid ancestor links used for aux loss = {total_valid}")
        for depth, count in enumerate(depth_counts, start=1):
            print(f"depth {depth} ancestor count = {count}")
        print("sample ancestor routes: node_idx -> [parent, grandparent, ...]")
        shown = 0
        for node_idx, ancestors in enumerate(ancestor_index):
            if any(a >= 0 for a in ancestors):
                print(f"  {node_idx} -> {ancestors}")
                shown += 1
                if shown >= 10:
                    break
        print("============================================\n")

        return ancestor_index, ancestor_depth

    def _load_qid_tree_train_counts(self, counts_path, dpath):
        """
        Load per-KC training interaction counts for frequency-aware prediction fusion.

        Accepted formats:
            1) list/tuple of length num_c: [count_0, count_1, ...]
            2) dict with key "counts" storing such a list
            3) dict mapping model node index string -> count, e.g. {"0": 12, "1": 5}
            4) dict mapping raw node_id string -> count; if keyid2idx_tree.json exists,
               raw node ids are converted to model indices.

        If no valid path is provided, return all-zero counts. This keeps the model
        usable and lets you call set_qid_tree_train_counts(...) later.
        """
        counts = torch.zeros(self.num_c, dtype=torch.float)

        candidate_paths = []
        if counts_path:
            candidate_paths.append(counts_path)
        if dpath:
            candidate_paths.extend([
                os.path.join(dpath, "kc_train_counts.json"),
                os.path.join(dpath, "train_kc_counts.json"),
                os.path.join(dpath, "kc_counts.json"),
                os.path.join(dpath, "concept_train_counts.json"),
            ])

        selected_path = None
        for p in candidate_paths:
            if p and os.path.exists(p):
                selected_path = p
                break

        if selected_path is None:
            print("========== qid_tree frequency fusion count debug ==========")
            print("No KC train-count file found; kc_train_counts initialized as all zeros.")
            print("For frequency fusion, call model.set_qid_tree_train_counts(counts) after initialization")
            print("or provide tree_pred_fusion_counts_path.")
            print("==========================================================\n")
            return counts

        try:
            with open(selected_path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception as e:
            print("========== qid_tree frequency fusion count debug ==========")
            print(f"Failed to load KC train counts from: {selected_path}")
            print(f"Error: {e}")
            print("kc_train_counts initialized as all zeros.")
            print("==========================================================\n")
            return counts

        raw_to_idx = {}
        if dpath:
            keyid2idx_tree_path = os.path.join(dpath, "keyid2idx_tree.json")
            if os.path.exists(keyid2idx_tree_path):
                try:
                    with open(keyid2idx_tree_path, "r", encoding="utf-8") as f:
                        keyid2idx_tree = json.load(f)
                    raw_to_idx = {
                        str(k): int(v)
                        for k, v in keyid2idx_tree.get("concepts", {}).items()
                    }
                except Exception:
                    raw_to_idx = {}

        loaded = 0

        if isinstance(data, dict) and "counts" in data:
            data = data["counts"]

        if isinstance(data, list):
            n = min(len(data), self.num_c)
            for i in range(n):
                try:
                    counts[i] = float(data[i])
                    loaded += 1
                except Exception:
                    continue

        elif isinstance(data, dict):
            for k, v in data.items():
                idx = None
                sk = str(k)

                # First try model index directly.
                try:
                    direct_idx = int(sk)
                    if 0 <= direct_idx < self.num_c:
                        idx = direct_idx
                except Exception:
                    idx = None

                # Then try raw node_id -> model index mapping.
                if idx is None and sk in raw_to_idx:
                    idx = raw_to_idx[sk]

                if idx is None or not (0 <= idx < self.num_c):
                    continue

                try:
                    counts[idx] = float(v)
                    loaded += 1
                except Exception:
                    continue

        print("========== qid_tree frequency fusion count debug ==========")
        print(f"counts_path = {selected_path}")
        print(f"loaded count entries = {loaded}")
        print(f"nonzero count entries = {int((counts > 0).sum().item())}")
        print(f"total interactions counted = {float(counts.sum().item())}")
        print("==========================================================\n")

        return counts

    def set_qid_tree_train_counts(self, counts):
        """
        Set per-KC training interaction counts after model initialization.

        Usage:
            model.set_qid_tree_train_counts(kc_counts)

        counts can be:
            - list / tuple / torch.Tensor of length num_c
            - dict mapping model index -> count

        This is only used by tree_pred_fusion_mode="frequency".
        """
        if self.emb_type != "qid_tree":
            raise ValueError("set_qid_tree_train_counts is only valid for emb_type == 'qid_tree'.")

        new_counts = torch.zeros(self.num_c, dtype=torch.float, device=self.parent_index.device)

        if isinstance(counts, torch.Tensor):
            flat = counts.detach().float().view(-1).to(new_counts.device)
            n = min(flat.numel(), self.num_c)
            new_counts[:n] = flat[:n]
        elif isinstance(counts, (list, tuple)):
            n = min(len(counts), self.num_c)
            for i in range(n):
                new_counts[i] = float(counts[i])
        elif isinstance(counts, dict):
            for k, v in counts.items():
                idx = int(k)
                if 0 <= idx < self.num_c:
                    new_counts[idx] = float(v)
        else:
            raise TypeError("counts must be a tensor, list, tuple, or dict.")

        self.kc_train_counts.data.copy_(new_counts)

    # ------------------------------------------------------------------
    # qid_tree non-leaf branch
    # ------------------------------------------------------------------

    def _qid_tree_nonleaf_forward(self, q, r):
        """
        Independent non-leaf DKT forward branch.

        The leaf branch consumes the original leaf/tree node sequence.
        The non-leaf branch consumes the direct-parent sequence, so its hidden
        states are built from coarse-grained interactions only. It does not
        reuse the leaf branch hidden state or output head.

        Return:
            nonleaf_y: [B, L, num_c], probabilities from the non-leaf branch.
        """
        q_valid = (q >= 0) & (q < self.num_c)
        r_valid = (r >= 0) & (r <= 1)
        interaction_valid = q_valid & r_valid

        safe_q = q.long().clamp(0, self.num_c - 1)
        safe_r = r.long().clamp(0, 1)

        parent_q = self.parent_index[safe_q]
        parent_valid = interaction_valid & (parent_q >= 0)
        safe_parent_q = parent_q.clamp(0, self.num_c - 1)

        qemb = self.nonleaf_kc_emb(safe_parent_q)
        remb = self.nonleaf_response_emb(safe_r)
        xemb = qemb + remb
        xemb = xemb * parent_valid.unsqueeze(-1).float()

        h, _ = self.nonleaf_lstm_layer(xemb)
        h = self.nonleaf_dropout_layer(h)
        return torch.sigmoid(self.nonleaf_out_layer(h))

    # ------------------------------------------------------------------
    # qid_tree inference-time ancestor prediction fusion
    # ------------------------------------------------------------------

    def _qid_tree_prediction_valid_mask(self, q, r=None):
        """
        Build the valid mask for next-step prediction positions.

        DKT convention:
            y[:, :-1, :] predicts q[:, 1:], r[:, 1:].
        """
        if q.dim() != 2:
            raise ValueError(f"qid_tree expects q shape [B, L], got {tuple(q.shape)}")

        if q.size(1) <= 1:
            return torch.zeros(q.shape[0], 0, dtype=torch.bool, device=q.device)

        prev_q = q[:, :-1]
        target_q = q[:, 1:]
        valid = (
            (prev_q >= 0)
            & (prev_q < self.num_c)
            & (target_q >= 0)
            & (target_q < self.num_c)
        )

        if r is not None:
            if r.shape != q.shape:
                raise ValueError(f"r shape {tuple(r.shape)} must match q shape {tuple(q.shape)}")
            prev_r = r[:, :-1]
            target_r = r[:, 1:]
            valid = valid & (prev_r >= 0) & (prev_r <= 1) & (target_r >= 0) & (target_r <= 1)

        return valid

    def _qid_tree_ancestor_prior(self, pred_all, target_q):
        """
        Compute depth-decayed non-leaf contribution for each target leaf.

        This function does NOT return a normalized ancestor average. Instead,
        it returns the direct weighted ancestor contribution used in final
        fusion:

            fused = (1 - total_weight) * leaf + weighted_ancestor_sum

        Depth weights use tree_pred_fusion_depth_decay as the base:

            parent      depth=1 -> decay
            grandparent depth=2 -> decay^2
            ...

        Therefore the only path from non-leaf predictions to the final leaf
        prediction is the depth-decay weighted sum.
        """
        dtype = pred_all.dtype
        device = pred_all.device

        if self.tree_aux_max_depth <= 0 or self.ancestor_index.numel() == 0:
            shape = target_q.shape
            return (
                torch.zeros(shape, dtype=dtype, device=device),
                torch.zeros(shape, dtype=torch.bool, device=device),
                torch.zeros(shape, dtype=dtype, device=device),
            )

        safe_target_q = target_q.long().clamp(0, self.num_c - 1)

        anc_idx = self.ancestor_index[safe_target_q]
        anc_depth = self.ancestor_depth[safe_target_q].to(device=device, dtype=dtype)
        anc_valid = anc_idx >= 0

        if not bool(anc_valid.any().item()):
            shape = target_q.shape
            return (
                torch.zeros(shape, dtype=dtype, device=device),
                torch.zeros(shape, dtype=torch.bool, device=device),
                torch.zeros(shape, dtype=dtype, device=device),
            )

        D = anc_idx.size(-1)
        safe_anc_idx = anc_idx.clamp(min=0, max=self.num_c - 1)

        anc_pred = pred_all.unsqueeze(2).expand(-1, -1, D, -1).gather(
            dim=3,
            index=safe_anc_idx.unsqueeze(-1),
        ).squeeze(-1)

        # Learnable depth decay. Do NOT convert this to float, otherwise the
        # gradient path to tree_pred_fusion_depth_decay_logit will be broken.
        base = self.get_tree_pred_fusion_depth_decay().to(device=device, dtype=dtype)

        # parent: decay^1, grandparent: decay^2, ...
        raw_weight = torch.pow(
            base,
            anc_depth.clamp(min=1.0),
        )
        raw_weight = raw_weight * anc_valid.float()

        total_weight = raw_weight.sum(dim=-1)
        has_ancestor = total_weight > 0

        # Safety: keep leaf coefficient non-negative.
        scale = torch.where(
            total_weight > 1.0,
            1.0 / total_weight.clamp(min=1e-12),
            torch.ones_like(total_weight),
        )
        raw_weight = raw_weight * scale.unsqueeze(-1)
        total_weight = raw_weight.sum(dim=-1)

        weighted_ancestor_sum = (anc_pred * raw_weight).sum(dim=-1)
        weighted_ancestor_sum = torch.where(
            has_ancestor,
            weighted_ancestor_sum,
            torch.zeros_like(weighted_ancestor_sum),
        )

        return weighted_ancestor_sum, has_ancestor, total_weight

    def _qid_tree_fusion_gate(self, target_q, valid, has_ancestor):
        """
        Compute the ancestor-prior gate for each prediction position.

        fixed mode:
            gate = tree_pred_fusion_fixed_weight

        frequency mode:
            gate = tree_pred_fusion_max_weight * exp(-count(q) / count_tau)

        Therefore sparse KCs receive more ancestor prior, while frequent KCs
        rely mostly on their own leaf predictions.
        """
        dtype = valid.float().dtype
        device = target_q.device
        safe_target_q = target_q.long().clamp(0, self.num_c - 1)

        if self.tree_pred_fusion_mode == "none":
            gate = torch.zeros_like(target_q, dtype=torch.float, device=device)
        elif self.tree_pred_fusion_mode == "fixed":
            gate = torch.full_like(
                target_q,
                fill_value=self.tree_pred_fusion_fixed_weight,
                dtype=torch.float,
                device=device,
            )
            gate = gate.clamp(min=0.0, max=self.tree_pred_fusion_max_weight)
        elif self.tree_pred_fusion_mode == "frequency":
            counts = self.kc_train_counts[safe_target_q].to(device=device, dtype=torch.float)
            gate = self.tree_pred_fusion_max_weight * torch.exp(
                -counts / self.tree_pred_fusion_count_tau
            )
            gate = gate.clamp(min=0.0, max=self.tree_pred_fusion_max_weight)
        else:
            raise ValueError(f"Unsupported tree_pred_fusion_mode: {self.tree_pred_fusion_mode}")

        gate = torch.where(valid & has_ancestor, gate, torch.zeros_like(gate))
        return gate.to(dtype=dtype)

    def get_qid_tree_target_prediction(self, y, q, r=None, use_fusion=True, return_details=False, fusion_source_y=None):
        """
        Return next-step target predictions for qid_tree.

        This is useful if your evaluator does not want to modify y directly.

        Returns:
            pred: [B, L-1]
                If fusion is enabled, this is the fused leaf+ancestor prediction.
                Otherwise, this is the leaf-only prediction.
            valid: [B, L-1]
            details: optional diagnostic dictionary
        """
        if self.emb_type != "qid_tree":
            raise ValueError("get_qid_tree_target_prediction is only valid for emb_type == 'qid_tree'.")
        if y.dim() != 3 or y.size(-1) != self.num_c:
            raise ValueError(f"Expected y shape [B, L, {self.num_c}], got {tuple(y.shape)}")
        if q.dim() != 2 or y.shape[:2] != q.shape:
            raise ValueError(f"Shape mismatch: y[:2]={tuple(y.shape[:2])}, q={tuple(q.shape)}")

        if q.size(1) <= 1:
            pred = torch.zeros(q.shape[0], 0, dtype=y.dtype, device=y.device)
            valid = torch.zeros(q.shape[0], 0, dtype=torch.bool, device=y.device)
            if return_details:
                return pred, valid, {}
            return pred, valid

        pred_all = y[:, :-1, :].clamp(min=1e-7, max=1.0 - 1e-7)
        target_q = q[:, 1:].long()
        valid = self._qid_tree_prediction_valid_mask(q, r=r)
        safe_target_q = target_q.clamp(0, self.num_c - 1)

        leaf_pred = pred_all.gather(
            dim=2,
            index=safe_target_q.unsqueeze(-1),
        ).squeeze(-1)

        if (not use_fusion) or self.tree_pred_fusion_mode == "none":
            if return_details:
                details = {
                    "leaf_pred": leaf_pred.detach(),
                    "ancestor_prior": torch.zeros_like(leaf_pred).detach(),
                    "fusion_gate": torch.zeros_like(leaf_pred).detach(),
                    "fused_pred": leaf_pred.detach(),
                    "valid": valid.detach(),
                }
                return leaf_pred, valid, details
            return leaf_pred, valid

        if fusion_source_y is None:
            fusion_source_y = self._last_qid_tree_nonleaf_y
        if fusion_source_y is None:
            raise RuntimeError(
                "depth_decay fusion requires non-leaf predictions. "
                "Call forward() first so _last_qid_tree_nonleaf_y is set."
            )
        if fusion_source_y.shape != y.shape:
            raise ValueError(
                f"fusion_source_y shape {tuple(fusion_source_y.shape)} must match y shape {tuple(y.shape)}"
            )

        source_pred_all = fusion_source_y[:, :-1, :].clamp(min=1e-7, max=1.0 - 1e-7)

        ancestor_prior, has_ancestor, gate = self._qid_tree_ancestor_prior(source_pred_all, target_q)
        gate = torch.where(valid & has_ancestor, gate, torch.zeros_like(gate))
        fused_pred = (1.0 - gate) * leaf_pred + ancestor_prior

        if return_details:
            active = valid & has_ancestor & (gate > 0)
            details = {
                "leaf_pred": leaf_pred.detach(),
                "ancestor_prior": ancestor_prior.detach(),
                "fusion_gate": gate.detach(),
                "fused_pred": fused_pred.detach(),
                "valid": valid.detach(),
                "active_fusion_count": active.float().sum().detach(),
                "mean_fusion_gate": gate[active].mean().detach() if bool(active.any().item()) else torch.tensor(0.0, device=y.device),
                "mean_leaf_pred": leaf_pred[valid].mean().detach() if bool(valid.any().item()) else torch.tensor(0.0, device=y.device),
                "mean_ancestor_prior": ancestor_prior[active].mean().detach() if bool(active.any().item()) else torch.tensor(0.0, device=y.device),
                "tree_pred_fusion_depth_decay_effective": self.get_tree_pred_fusion_depth_decay().detach(),
            }
            return fused_pred, valid, details

        return fused_pred, valid

    def apply_qid_tree_prediction_fusion(self, y, q, r=None, return_details=False, fusion_source_y=None):
        """
        Return a copy of y where the next-step target leaf entries are replaced
        by frequency-aware fused predictions.

        This makes old evaluation code still work if it does:
            pred = y[:, :-1, :].gather(dim=2, index=q[:, 1:].unsqueeze(-1))

        Only the target leaf entries y[:, :-1, q[:, 1:]] are replaced. Other
        output dimensions stay unchanged.
        """
        if self.emb_type != "qid_tree":
            raise ValueError("apply_qid_tree_prediction_fusion is only valid for emb_type == 'qid_tree'.")

        if self.tree_pred_fusion_mode == "none" or q.size(1) <= 1:
            if return_details:
                return y, {"fusion_enabled": False}
            return y

        fused_pred, valid, details = self.get_qid_tree_target_prediction(
            y,
            q,
            r=r,
            use_fusion=True,
            return_details=True,
            fusion_source_y=fusion_source_y,
        )

        target_q = q[:, 1:].long()
        safe_target_q = target_q.clamp(0, self.num_c - 1)

        pred_block = y[:, :-1, :]

        # Keep original leaf prediction on invalid positions.
        original_leaf = pred_block.gather(
            dim=2,
            index=safe_target_q.unsqueeze(-1),
        ).squeeze(-1)
        fused_pred = torch.where(valid, fused_pred, original_leaf)

        # Avoid in-place modification because y participates in autograd.
        # We rebuild the output tensor with torch.cat so backward remains valid.
        pred_block_fused = pred_block.scatter(
            dim=2,
            index=safe_target_q.unsqueeze(-1),
            src=fused_pred.unsqueeze(-1),
        )
        y_fused = torch.cat([pred_block_fused, y[:, -1:, :]], dim=1)

        if return_details:
            details = dict(details)
            details["fusion_enabled"] = True
            return y_fused, details

        return y_fused

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
    # qid_tree hierarchical auxiliary loss
    # ------------------------------------------------------------------

    def _masked_mean(self, values, mask):
        """Mean over valid positions only."""
        mask = mask.float()
        denom = mask.sum().clamp(min=1.0)
        return (values * mask).sum() / denom

    def get_qid_tree_loss(self, y, q, r, return_details=False):
        """
        Compute DKT loss for the new qid_tree design.

        Main idea:
            - leaf/current target KC prediction is the main loss.
            - ancestors of the target KC receive weaker auxiliary losses.
            - ancestor weights decay exponentially by tree distance:

                weight(depth) = tree_aux_loss_weight * exp(-tree_aux_decay * depth)

        Important:
            forward(...) still returns y with shape [B, L, num_c].
            Your trainer must call this method for qid_tree if you want the
            ancestor auxiliary loss to actually participate in training.

        Args:
            y: [B, L, num_c], model output probabilities.
            q: [B, L], KC/tree node ids.
            r: [B, L], binary responses.
            return_details: if True, return (loss, details_dict).

        Returns:
            loss scalar, or (loss, details_dict).
        """
        if self.emb_type != "qid_tree":
            raise ValueError("get_qid_tree_loss is only valid when emb_type == 'qid_tree'.")

        if y.dim() != 3:
            raise ValueError(f"qid_tree loss expects y shape [B, L, num_c], got {tuple(y.shape)}")

        if q.dim() != 2 or r.dim() != 2:
            raise ValueError(
                f"qid_tree loss expects q and r shape [B, L], got q={tuple(q.shape)}, r={tuple(r.shape)}"
            )

        if y.shape[:2] != q.shape or q.shape != r.shape:
            raise ValueError(
                f"Shape mismatch: y[:2]={tuple(y.shape[:2])}, q={tuple(q.shape)}, r={tuple(r.shape)}"
            )

        if y.size(-1) != self.num_c:
            raise ValueError(f"Expected y last dim {self.num_c}, got {y.size(-1)}")

        B, L = q.shape
        device = y.device
        dtype = y.dtype

        if L <= 1:
            zero = y.sum() * 0.0
            if return_details:
                return zero, {
                    "leaf_loss": zero.detach(),
                    "ancestor_loss": zero.detach(),
                    "total_loss": zero.detach(),
                    "valid_leaf_count": torch.tensor(0.0, device=device),
                    "valid_ancestor_count": torch.tensor(0.0, device=device),
                }
            return zero

        # DKT convention:
        #   y[:, t, :] is produced after observing interaction t.
        #   It is used to predict response at t+1.
        pred_all = y[:, :-1, :]
        target_q = q[:, 1:].long()
        target_r_raw = r[:, 1:].float()
        # BCE requires all target values to be in [0, 1], even positions that
        # will be masked out later. Therefore we compute validity from the raw
        # response, then clamp the tensor used by BCE.
        target_r = target_r_raw.clamp(0.0, 1.0)

        prev_q = q[:, :-1]
        prev_r = r[:, :-1]

        prev_valid = (prev_q >= 0) & (prev_q < self.num_c) & (prev_r >= 0) & (prev_r <= 1)
        target_valid = (target_q >= 0) & (target_q < self.num_c) & (target_r_raw >= 0) & (target_r_raw <= 1)
        valid = prev_valid & target_valid

        safe_target_q = target_q.clamp(0, self.num_c - 1)

        eps = 1e-7
        pred_all = pred_all.clamp(min=eps, max=1.0 - eps)

        # Ancestor auxiliary loss is fully decoupled from the leaf branch.
        # It always uses predictions from the independent non-leaf forward.
        nonleaf_y = self._last_qid_tree_nonleaf_y
        if nonleaf_y is None:
            raise RuntimeError(
                "qid_tree ancestor loss requires non-leaf predictions. "
                "Call forward() before get_qid_tree_loss()."
            )
        if nonleaf_y.shape != y.shape:
            raise ValueError(
                f"Stored nonleaf_y shape {tuple(nonleaf_y.shape)} does not match y shape {tuple(y.shape)}"
            )
        aux_pred_all = nonleaf_y[:, :-1, :].clamp(min=eps, max=1.0 - eps)

        # -------------------------------------------------------------
        # 1) Main leaf loss
        # -------------------------------------------------------------
        leaf_pred = pred_all.gather(
            dim=2,
            index=safe_target_q.unsqueeze(-1),
        ).squeeze(-1)

        leaf_bce = F.binary_cross_entropy(
            leaf_pred,
            target_r,
            reduction="none",
        )
        leaf_loss = self._masked_mean(leaf_bce, valid)

        # -------------------------------------------------------------
        # 2) Ancestor auxiliary loss
        # -------------------------------------------------------------
        ancestor_loss = y.sum() * 0.0
        valid_ancestor_count = torch.tensor(0.0, device=device, dtype=dtype)

        if (
            self.tree_aux_loss_weight > 0
            and self.tree_aux_max_depth > 0
            and self.ancestor_index.numel() > 0
        ):
            # [B, L-1, D]
            anc_idx = self.ancestor_index[safe_target_q]
            anc_depth = self.ancestor_depth[safe_target_q].to(device=device, dtype=dtype)

            anc_valid = (anc_idx >= 0) & valid.unsqueeze(-1)
            valid_ancestor_count = anc_valid.float().sum().to(dtype=dtype)

            if bool(anc_valid.any().item()):
                D = anc_idx.size(-1)
                safe_anc_idx = anc_idx.clamp(min=0, max=self.num_c - 1)

                # Gather predictions for each ancestor target.
                # aux_pred_all: [B, L-1, num_c]
                # safe_anc_idx: [B, L-1, D]
                anc_pred = aux_pred_all.unsqueeze(2).expand(-1, -1, D, -1).gather(
                    dim=3,
                    index=safe_anc_idx.unsqueeze(-1),
                ).squeeze(-1)

                anc_target = target_r.unsqueeze(-1).expand_as(anc_pred)
                anc_bce = F.binary_cross_entropy(
                    anc_pred,
                    anc_target,
                    reduction="none",
                )

                # Optional asymmetric supervision:
                # wrong answers can be made weaker because one wrong leaf
                # answer does not necessarily mean the whole parent concept
                # is unmastered. Default = 1.0, i.e. symmetric.
                if self.tree_aux_negative_scale != 1.0:
                    response_scale = torch.where(
                        anc_target > 0.5,
                        torch.ones_like(anc_target),
                        torch.full_like(anc_target, self.tree_aux_negative_scale),
                    )
                    anc_bce = anc_bce * response_scale

                # exp-decayed ancestor weights. Invalid depths are masked later.
                anc_weight = self.tree_aux_loss_weight * torch.exp(
                    -self.tree_aux_decay * anc_depth
                )

                # Normalize by number of valid leaf prediction positions, not by
                # the sum of weights. This keeps tree_aux_loss_weight as a true
                # auxiliary strength coefficient.
                leaf_denom = valid.float().sum().clamp(min=1.0).to(dtype=dtype)
                ancestor_loss = (
                    anc_bce * anc_weight * anc_valid.float()
                ).sum() / leaf_denom

        total_loss = leaf_loss + ancestor_loss

        if return_details:
            details = {
                "leaf_loss": leaf_loss.detach(),
                "ancestor_loss": ancestor_loss.detach(),
                "total_loss": total_loss.detach(),
                "valid_leaf_count": valid.float().sum().detach(),
                "valid_ancestor_count": valid_ancestor_count.detach(),
            }
            return total_loss, details

        return total_loss

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(self, q, r, q_dense, apply_tree_pred_fusion=None):
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

            # Independent non-leaf forward. This does not reuse h, out_layer,
            # residual_kc_emb, or response_emb from the leaf branch.
            nonleaf_y = self._qid_tree_nonleaf_forward(q, r)
            self._last_qid_tree_nonleaf_y = nonleaf_y

            # Optional inference-time prediction fusion.
            # If self.tree_pred_fusion_mode == "none", this is a no-op.
            # If enabled, y[:, :-1, q[:, 1:]] is replaced by a selective
            # leaf+ancestor prediction, so existing gather-based evaluators
            # can use the fused prediction without further code changes.
            if apply_tree_pred_fusion is None:
                apply_tree_pred_fusion = (
                    self.tree_pred_fusion_mode == "depth_decay"
                    and ((not self.training) or self.tree_pred_fusion_apply_train)
                )

            # Forward-side debug: confirms whether fusion is actually applied during training.
            if self.training and self.tree_pred_fusion_mode == "depth_decay":
                if not hasattr(self, "_tree_pred_fusion_forward_debug_count"):
                    self._tree_pred_fusion_forward_debug_count = 0

                self._tree_pred_fusion_forward_debug_count += 1

                if (
                    self._tree_pred_fusion_forward_debug_count <= 5
                    or self._tree_pred_fusion_forward_debug_count % 100 == 0
                ):
                    print(
                        "[DEBUG depth_decay forward] "
                        f"count={self._tree_pred_fusion_forward_debug_count}, "
                        f"training={self.training}, "
                        f"tree_pred_fusion_mode={self.tree_pred_fusion_mode}, "
                        f"tree_pred_fusion_apply_train={self.tree_pred_fusion_apply_train}, "
                        f"apply_tree_pred_fusion={apply_tree_pred_fusion}, "
                        f"effective_decay={self.get_tree_pred_fusion_depth_decay().detach().item():.10f}, "
                        f"requires_grad={self.tree_pred_fusion_depth_decay_logit.requires_grad}",
                        flush=True,
                    )

            if apply_tree_pred_fusion:
                if not hasattr(self, "_tree_pred_fusion_detail_debug_count"):
                    self._tree_pred_fusion_detail_debug_count = 0

                self._tree_pred_fusion_detail_debug_count += 1

                debug_fusion = (
                    self.training
                    and self.tree_pred_fusion_mode == "depth_decay"
                    and (
                        self._tree_pred_fusion_detail_debug_count <= 5
                        or self._tree_pred_fusion_detail_debug_count % 100 == 0
                    )
                )

                if debug_fusion:
                    y, fusion_details = self.apply_qid_tree_prediction_fusion(
                        y,
                        q,
                        r=r,
                        return_details=True,
                        fusion_source_y=nonleaf_y,
                    )

                    print(
                        "[DEBUG depth_decay fusion details] "
                        f"count={self._tree_pred_fusion_detail_debug_count}, "
                        f"active_fusion_count={fusion_details.get('active_fusion_count', 'NA')}, "
                        f"mean_fusion_gate={fusion_details.get('mean_fusion_gate', 'NA')}, "
                        f"mean_leaf_pred={fusion_details.get('mean_leaf_pred', 'NA')}, "
                        f"mean_ancestor_prior={fusion_details.get('mean_ancestor_prior', 'NA')}, "
                        f"effective_decay={self.get_tree_pred_fusion_depth_decay().detach().item():.10f}",
                        flush=True,
                    )
                else:
                    y = self.apply_qid_tree_prediction_fusion(
                        y,
                        q,
                        r=r,
                        return_details=False,
                        fusion_source_y=nonleaf_y,
                    )

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