import json
import os
import math
from enum import IntEnum

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.nn.init import constant_, xavier_uniform_


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# qid_fmkc( random residual+FMKC) + qid_tree；
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
        num_c_fmkc=None,
        dpath="",
        kc_tree_path="",
        # qid_tree controls. Defaults are conservative: no parent loss and no
        # parent prediction fusion unless explicitly enabled in config.
        tree_embed_mode=None,# "independent", "inherit"
        tree_aux_loss_weight=0.3,
        tree_aux_loss_mode="depth", #"none" "direct" "depth" "specified" "all"
        tree_aux_max_depth=2,
        tree_aux_depths=None, # 只在tree_aux_loss_mode="specified"时使用。
        tree_pred_fusion_mode="depth_decay", # "none", "direct_parent", "ancestor_mean", "depth_mean", "specified_mean", "ancestor_decay", "depth_decay", "specified_decay", "gated_direct_parent"
        tree_pred_fusion_weight=0.8,
        tree_pred_max_depth=2,
        tree_pred_depths=None, # 只在"specified_mean" "specified_decay"时使用。
        tree_aux_ignore_first=None,
        tree_max_ancestor_depth=2,
        tree_pred_occurrence_decay=0.01,
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
        self.num_c_fmkc = num_c_fmkc
        self.dpath = dpath

        # ------------------------------------------------------------------
        # qid_tree behavior controls
        # ------------------------------------------------------------------
        # Keep these controls explicit at training entry (wandb_akt_train.py),
        # not hidden as model-internal defaults.
        if emb_type == "qid_tree":
            # ------------------------------------------------------------------
            # Decoupled qid_tree setting
            # ------------------------------------------------------------------
            # Design principle:
            #   1) Training loss is leaf-only. Parent/ancestor nodes do NOT
            #      contribute auxiliary loss and do NOT change the training logits.
            #   2) Parent/ancestor nodes participate only at prediction/evaluation
            #      time through depth-decayed ancestor prior.
            #   3) The AKT encoder remains leaf-centered and unchanged.
            required_tree_args = {
                "tree_embed_mode": tree_embed_mode,
                "tree_pred_fusion_mode": tree_pred_fusion_mode,
                "tree_pred_fusion_weight": tree_pred_fusion_weight,
                "tree_pred_max_depth": tree_pred_max_depth,
                "tree_max_ancestor_depth": tree_max_ancestor_depth,
            }
            missing_tree_args = [k for k, v in required_tree_args.items() if v is None]
            if missing_tree_args:
                raise ValueError(
                    "AKT qid_tree requires explicit tree prediction hyperparameters from training entry; "
                    f"missing: {missing_tree_args}. "
                    "Please set them in examples/wandb_akt_train.py CLI."
                )

            # tree_embed_mode:
            #   independent: child/node embedding is its own embedding only.
            #   inherit:     child embedding is a gated combination of own + parent.
            self.tree_embed_mode = str(tree_embed_mode).lower()

            # IMPORTANT:
            # Training-time ancestor auxiliary supervision is intentionally disabled.
            # This prevents coarse-grained parent nodes from backpropagating into
            # the leaf-centered AKT encoder/output head through an auxiliary loss.
            # We keep these attributes only for compatibility with old config files.
            self.tree_aux_loss_weight = 0.0
            self.tree_aux_loss_mode = "none"
            self.tree_aux_max_depth = 1 if tree_aux_max_depth is None else int(tree_aux_max_depth)
            self.tree_aux_depths = []

            # Only depth_decay is allowed as the parent/ancestor prediction mechanism
            # in the decoupled design. Use "none" for ablation.
            self.tree_pred_fusion_mode = self._normalize_tree_pred_mode(tree_pred_fusion_mode)
            self.tree_pred_fusion_weight = float(tree_pred_fusion_weight)
            self.tree_pred_occurrence_decay = (
                self.tree_pred_fusion_weight
                if tree_pred_occurrence_decay is None
                else float(tree_pred_occurrence_decay)
            )
            self.tree_pred_max_depth = int(tree_pred_max_depth)
            self.tree_pred_depths = self._parse_tree_depths_arg(tree_pred_depths)
            self.tree_aux_ignore_first = True if tree_aux_ignore_first is None else bool(tree_aux_ignore_first)
            self.tree_max_ancestor_depth = int(tree_max_ancestor_depth)

            valid_tree_embed_modes = {"independent"}
            valid_tree_pred_modes = {"none", "depth_decay"}

            if self.tree_embed_mode not in valid_tree_embed_modes:
                raise ValueError(f"Invalid tree_embed_mode={tree_embed_mode}. Valid: {valid_tree_embed_modes}")
            if self.tree_pred_fusion_mode not in valid_tree_pred_modes:
                raise ValueError(
                    f"Invalid tree_pred_fusion_mode={tree_pred_fusion_mode}. "
                    f"For the decoupled design, valid modes are: {valid_tree_pred_modes}."
                )
            if not (0.0 <= self.tree_pred_fusion_weight <= 1.0):
                raise ValueError("tree_pred_fusion_weight must be in [0, 1].")
            if self.tree_pred_occurrence_decay < 0.0:
                raise ValueError("tree_pred_occurrence_decay must be non-negative.")
            if self.tree_max_ancestor_depth <= 0:
                raise ValueError("tree_max_ancestor_depth must be positive.")
            if self.tree_pred_fusion_mode == "depth_decay" and self.tree_pred_max_depth <= 0:
                raise ValueError("tree_pred_max_depth must be positive when tree_pred_fusion_mode='depth_decay'.")

            max_requested_depth = 1
            if self.tree_pred_fusion_mode == "depth_decay":
                max_requested_depth = max(max_requested_depth, self.tree_pred_max_depth)

            # Do not silently ignore a requested depth.
            if max_requested_depth > self.tree_max_ancestor_depth:
                raise ValueError(
                    f"Requested ancestor depth {max_requested_depth}, but "
                    f"tree_max_ancestor_depth={self.tree_max_ancestor_depth}. "
                    "Increase tree_max_ancestor_depth or reduce tree_pred_max_depth."
                )
        else:
            # Non-qid_tree modes never read these controls; keep placeholders.
            self.tree_embed_mode = "independent"
            self.tree_aux_loss_weight = 0.0
            self.tree_aux_loss_mode = "none"
            self.tree_aux_max_depth = 1
            self.tree_aux_depths = []
            self.tree_pred_fusion_mode = "none"
            self.tree_pred_fusion_weight = 0.0
            self.tree_pred_occurrence_decay = 0.0
            self.tree_pred_max_depth = 1
            self.tree_pred_depths = []
            self.tree_aux_ignore_first = True
            self.tree_max_ancestor_depth = 1

        embed_l = d_model

        # ------------------------------------------------------------------
        # qid_fmkc: random residual + FMKC branch
        # ------------------------------------------------------------------
        if emb_type == "qid_fmkc":
            if num_c_fmkc is None or len(num_c_fmkc) < 1:
                raise ValueError(
                    "emb_type qid_fmkc requires num_c_fmkc with at least one field"
                )

            if separate_qa:
                raise ValueError("qid_fmkc does not support separate_qa")

            self.num_c_fmkc = [int(n) for n in num_c_fmkc]
            self.num_fm_fields = len(self.num_c_fmkc)

            # Field-wise KC embedding tables.
            self.kc_emb = nn.ModuleList(
                [nn.Embedding(int(n), embed_l) for n in self.num_c_fmkc]
            )

            # Random residual embedding for each full KC id.
            # Final KC embedding:
            #     FM(field embeddings) + residual_scale * residual_emb(q_dense)
            self.kc_residual_emb = nn.Embedding(self.n_question, embed_l)
            self.kc_residual_scale = nn.Parameter(torch.tensor(0.1))

            # Deterministic mixed-radix strides. These are only used as a safe
            # fallback when tuple_space <= n_question. If tuple_space > n_question,
            # q_dense must be supplied to avoid residual-id collision.
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

            # Field-wise and dimension-wise learnable weights for first-order terms.
            # Shape: [num_fields, d_model]
            self.alpha_fm1 = nn.Parameter(torch.ones(self.num_fm_fields, embed_l))

            # Start second-order FM interaction softly to avoid unstable logits.
            self.fm2_scale = nn.Parameter(torch.tensor(0.1))

            # Learnable global output scale for the FMKC embedding.
            self.fm_out_scale = nn.Parameter(torch.tensor(1.0))

            # Response embedding only. In AKT, qa_embed_data is still built as:
            #     qa_embed(response) + q_embed_data
            self.qa_embed = nn.Embedding(2, embed_l)

        # ------------------------------------------------------------------
        # qid_tree: leaf branch + fully decoupled non-leaf branch
        # ------------------------------------------------------------------
        elif emb_type == "qid_tree":
            if separate_qa:
                raise ValueError("qid_tree does not support separate_qa")

            # Leaf AKT branch: unchanged AKT-style KC/response embeddings.
            # Parent/internal nodes never construct leaf embeddings.
            self.residual_kc_emb = nn.Embedding(self.n_question, embed_l)
            self.response_emb = nn.Embedding(2, embed_l)

            parent_index = self._build_tree_parent_index(kc_tree_path, dpath)
            self.register_buffer("parent_index", torch.tensor(parent_index, dtype=torch.long))

            ancestor_index, ancestor_mask = self._build_ancestor_index(
                parent_index,
                max_depth=self.tree_max_ancestor_depth,
            )
            self.register_buffer("ancestor_index", torch.tensor(ancestor_index, dtype=torch.long))
            self.register_buffer("ancestor_mask", torch.tensor(ancestor_mask, dtype=torch.bool))

            # Non-leaf branch: completely separate embedding, encoder, and output head.
            # It never reuses leaf d_output or self.out.
            self.ancestor_kc_emb = nn.Embedding(self.n_question, embed_l)
            self.ancestor_response_emb = nn.Embedding(2, embed_l)
            self.ancestor_model = Architecture(
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
            self.ancestor_out = nn.Sequential(
                nn.Linear(d_model + embed_l, final_fc_dim),
                nn.ReLU(),
                nn.Dropout(self.dropout),
                nn.Linear(final_fc_dim, 256),
                nn.ReLU(),
                nn.Dropout(self.dropout),
                nn.Linear(256, 1),
            )

            # Optional independent training loss for the non-leaf branch.
            # This trains only ancestor_* parameters because the branch is separate.
            self.tree_branch_loss_weight = 0.0 if tree_aux_loss_weight is None else float(tree_aux_loss_weight)

            print("\n========== qid_tree behavior ==========")
            print(f"tree_embed_mode = {self.tree_embed_mode}")
            print(f"tree_pred_fusion_mode = {self.tree_pred_fusion_mode}")
            print(f"tree_pred_fusion_weight = {self.tree_pred_fusion_weight}")
            print(f"tree_pred_occurrence_decay = {self.tree_pred_occurrence_decay}")
            print(f"tree_pred_max_depth = {self.tree_pred_max_depth}")
            print(f"tree_branch_loss_weight = {self.tree_branch_loss_weight}")
            print("leaf_and_nonleaf_forward_decoupled = True")
            print("nonleaf_contribution_to_leaf = depth_decay only")
            print(f"tree_max_ancestor_depth(table capacity) = {self.tree_max_ancestor_depth}")
            print("=======================================\n")

        # ------------------------------------------------------------------
        # Original qid branch
        # ------------------------------------------------------------------
        elif emb_type.startswith("qid"):
            self.q_embed = nn.Embedding(self.n_question, embed_l)

            if self.separate_qa:
                self.qa_embed = nn.Embedding(2 * self.n_question + 1, embed_l)
            else:
                self.qa_embed = nn.Embedding(2, embed_l)

        else:
            raise ValueError(f"AKT unsupported emb_type: {emb_type}")

        # ------------------------------------------------------------------
        # Rasch / problem difficulty branch
        # ------------------------------------------------------------------
        if self.n_pid > 0:
            # problem difficulty scalar: u_q
            self.difficult_param = nn.Embedding(self.n_pid + 1, 1)

            # For qid_fmkc, q_embed_diff is indexed by pid_data because q_data
            # is multi-field and cannot directly index a standard embedding table.
            # For qid/qid_tree, q_embed_diff is indexed by q_data.
            qdiff_rows = (
                self.n_pid + 1
                if emb_type == "qid_fmkc"
                else self.n_question + 1
            )

            self.q_embed_diff = nn.Embedding(qdiff_rows, embed_l)

            # Keep the original AKT design here.
            self.qa_embed_diff = nn.Embedding(2 * self.n_question + 1, embed_l)

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
        In qid_fmkc, q_embed_diff also has n_pid + 1 rows.
        If q_embed_diff is also zeroed, the Rasch / problem difficulty branch
        can become too weak or nearly dead at initialization.
        """
        if self.n_pid > 0:
            torch.nn.init.constant_(self.difficult_param.weight, 0.0)

    # ------------------------------------------------------------------
    # qid_tree depth-selection helpers
    # ------------------------------------------------------------------

    def _normalize_tree_aux_mode(self, mode):
        mode = str(mode).lower().strip()
        aliases = {
            "direct_parent": "direct",
            "parent": "direct",
            "parents": "direct",
            "ancestors": "depth",
            "upto_depth": "depth",
            "max_depth": "depth",
            "specified_ancestors": "specified",
            "selected": "specified",
            "all_ancestors": "all",
        }
        return aliases.get(mode, mode)

    def _normalize_tree_pred_mode(self, mode):
        mode = str(mode).lower().strip()
        aliases = {
            "direct": "direct_parent",
            "parent": "direct_parent",
            "parents": "direct_parent",
            "all": "ancestor_mean",
            "all_ancestors": "ancestor_mean",
            "ancestors_mean": "ancestor_mean",
            "ancestor_depth_mean": "depth_decay",
            "depth": "depth_decay",
            "decay": "depth_decay",
            "depth_decayed": "depth_decay",
            "specified": "specified_mean",
            "specified_ancestors": "specified_mean",
            "selected": "specified_mean",
            "all_decay": "ancestor_decay",
            "ancestor_exp_decay": "ancestor_decay",
            "ancestors_decay": "ancestor_decay",
            "depth_exp_decay": "depth_decay",
            "specified_exp_decay": "specified_decay",
            "gated_direct": "gated_direct_parent",
        }
        return aliases.get(mode, mode)

    def _parse_tree_depths_arg(self, depths):
        if depths is None:
            return []
        if isinstance(depths, int):
            vals = [depths]
        elif isinstance(depths, str):
            text = depths.strip()
            if not text:
                vals = []
            else:
                vals = [int(x.strip()) for x in text.split(",") if x.strip()]
        elif isinstance(depths, (list, tuple, set)):
            vals = [int(x) for x in depths]
        else:
            raise ValueError(f"Unsupported depth argument type: {type(depths)}")

        vals = sorted(set(vals))
        if any(d <= 0 for d in vals):
            raise ValueError(f"Ancestor depths must be positive 1-based integers, got {vals}")
        return vals

    def _depth_keep_mask(self, adepth, mode, max_depth=None, depths=None, device=None):
        """Return a boolean mask over ancestor dimension [A]. Depth is 1-based."""
        device = device if device is not None else self.parent_index.device
        if mode == "all":
            return torch.ones(adepth, dtype=torch.bool, device=device)
        if mode == "direct":
            depths = [1]
        elif mode == "depth":
            if max_depth is None or int(max_depth) <= 0:
                raise ValueError("max_depth must be positive for mode='depth'.")
            depths = list(range(1, int(max_depth) + 1))
        elif mode == "specified":
            if not depths:
                raise ValueError("depths must be non-empty for mode='specified'.")
            depths = [int(d) for d in depths]
        else:
            raise ValueError(f"Unknown depth selection mode: {mode}")

        keep = torch.zeros(adepth, dtype=torch.bool, device=device)
        for d in depths:
            if 1 <= int(d) <= adepth:
                keep[int(d) - 1] = True
        return keep

    def _select_aux_ancestor_valid(self, ancestor_valid):
        """Select which ancestor depths participate in auxiliary loss."""
        adepth = ancestor_valid.size(-1)
        if self.tree_aux_loss_mode == "all":
            keep = self._depth_keep_mask(adepth, "all", device=ancestor_valid.device)
        elif self.tree_aux_loss_mode == "depth":
            keep = self._depth_keep_mask(
                adepth,
                "depth",
                max_depth=self.tree_aux_max_depth,
                device=ancestor_valid.device,
            )
        elif self.tree_aux_loss_mode == "specified":
            keep = self._depth_keep_mask(
                adepth,
                "specified",
                depths=self.tree_aux_depths,
                device=ancestor_valid.device,
            )
        elif self.tree_aux_loss_mode == "direct":
            keep = self._depth_keep_mask(adepth, "direct", device=ancestor_valid.device)
        else:
            raise ValueError(f"Cannot select aux ancestors for mode={self.tree_aux_loss_mode}")
        return ancestor_valid & keep.view(1, 1, -1)

    def _select_pred_ancestor_valid(self, ancestor_valid):
        """Select which ancestor depths participate in prediction fusion."""
        adepth = ancestor_valid.size(-1)
        if self.tree_pred_fusion_mode in {"ancestor_mean", "ancestor_decay"}:
            keep = self._depth_keep_mask(adepth, "all", device=ancestor_valid.device)
        elif self.tree_pred_fusion_mode in {"depth_mean", "depth_decay"}:
            keep = self._depth_keep_mask(
                adepth,
                "depth",
                max_depth=self.tree_pred_max_depth,
                device=ancestor_valid.device,
            )
        elif self.tree_pred_fusion_mode in {"specified_mean", "specified_decay"}:
            keep = self._depth_keep_mask(
                adepth,
                "specified",
                depths=self.tree_pred_depths,
                device=ancestor_valid.device,
            )
        else:
            raise ValueError(f"Cannot select prediction ancestors for mode={self.tree_pred_fusion_mode}")
        return ancestor_valid & keep.view(1, 1, -1)

    def _tree_count_previous_occurrences(self, q_data):
        """
        Count how many times each leaf/tree node id has appeared earlier in
        the same sequence. This is used as a cold-start confidence signal:
        ancestors help most when count_before is small.
        """
        if q_data.dim() != 2:
            raise ValueError(
                f"qid_tree occurrence decay expects q_data shape [B, L], got {tuple(q_data.shape)}"
            )

        with torch.no_grad():
            q_valid = (q_data >= 0) & (q_data < self.n_question)
            safe_q = q_data.long().clamp(0, self.n_question - 1)
            seqlen = safe_q.size(1)

            # previous_mask[i, j] is True when position j is before position i.
            previous_mask = torch.tril(
                torch.ones(seqlen, seqlen, dtype=torch.bool, device=q_data.device),
                diagonal=-1,
            )

            same_kc = safe_q.unsqueeze(2).eq(safe_q.unsqueeze(1))
            valid_pair = q_valid.unsqueeze(2) & q_valid.unsqueeze(1)
            count_before = (same_kc & valid_pair & previous_mask.view(1, seqlen, seqlen)).sum(dim=-1)

        return count_before.float()

    def _tree_occurrence_gate(self, q_data):
        """
        Exponential cold-start gate:
            gate = exp(-lambda * count_before)

        count_before=0 -> gate=1, so ancestors can help cold-start KCs.
        As the leaf KC appears more often, gate approaches 0 and prediction
        relies increasingly on the leaf-level logit.
        """
        count_before = self._tree_count_previous_occurrences(q_data)
        return torch.exp(-float(self.tree_pred_occurrence_decay) * count_before)

    def _tree_depth_decay_weights(self, adepth, device):
        """
        Depth-decay weights over ancestors. Depth is 1-based:
            parent      depth=1 -> w
            grandparent depth=2 -> w^2
            ...
        """
        depths = torch.arange(1, adepth + 1, dtype=torch.float32, device=device)
        base = torch.tensor(float(self.tree_pred_fusion_weight), dtype=torch.float32, device=device)
        return torch.pow(base, depths).view(1, 1, adepth)

    # ------------------------------------------------------------------
    # Utilities for qid_tree
    # ------------------------------------------------------------------

    # def _resolve_tree_path(self, kc_tree_path, dpath):
    #     if kc_tree_path and os.path.exists(kc_tree_path):
    #         return kc_tree_path
    #     if dpath:
    #         default_path = os.path.join(
    #             dpath,
    #             "2_DBE_KT22_datafiles_100102_csv",
    #             "kc_knowledge_tree_original.json",
    #         )
    #         if os.path.exists(default_path):
    #             return default_path
    #     return ""

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

    #     parent_index = [-1] * self.n_question
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
    #         if 0 <= cidx < self.n_question and 0 <= pidx < self.n_question:
    #             parent_index[cidx] = pidx

    #     return parent_index

    def _build_tree_parent_index(self, kc_tree_path, dpath):
        if not dpath:
            if kc_tree_path:
                dpath = os.path.dirname(kc_tree_path)
            else:
                dpath, kc_tree_path = self._infer_tree_paths_from_config()

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
        if self.n_question != tree_num_c:
            raise ValueError(
                "num_c mismatch for qid_tree. "
                f"Model num_c={self.n_question}, keyid2idx_tree num_c={tree_num_c}. "
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
        print(f"input num_c = {self.n_question}")
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

    def _infer_tree_paths_from_config(self):
        """Infer tree dpath/tree file when caller forgot to pass them."""
        repo_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        cfg_path = os.path.join(repo_root, "configs", "data_config.json")
        if not os.path.exists(cfg_path):
            raise ValueError(
                "qid_tree requires dpath/kc_tree_path, and config file not found for fallback: "
                f"{cfg_path}"
            )

        with open(cfg_path, "r", encoding="utf-8") as f:
            cfg = json.load(f)

        hits = []
        for dname, item in cfg.items():
            if not item.get("kc_tree", False):
                continue
            tree_num_c = item.get("num_c_tree", item.get("num_c", -1))
            if int(tree_num_c) != int(self.n_question):
                continue
            dpath = item.get("dpath", "")
            if not dpath:
                continue
            hits.append((dname, dpath))

        if len(hits) != 1:
            raise ValueError(
                "qid_tree requires explicit dpath/kc_tree_path in AKT init. "
                f"Auto-infer candidates by num_c={self.n_question}: {hits}"
            )

        _, dpath = hits[0]
        return dpath, os.path.join(dpath, "kc_knowledge_tree.json")

    def _resolve_tree_path(self, kc_tree_path, dpath):
        if not dpath:
            if kc_tree_path:
                dpath = os.path.dirname(kc_tree_path)
            else:
                dpath, kc_tree_path = self._infer_tree_paths_from_config()

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

    def _build_topo_order(self, parent_index):
        """
        Return parent-before-child order.

        This is only needed for tree_embed_mode="inherit". We still build it
        for debugging and for safe backward compatibility. Unlike the previous
        implementation, this version raises an error if a cycle is detected,
        because a cycle means the JSON is not a valid tree/DAG for inheritance.
        """
        n = len(parent_index)
        state = [0] * n
        order = []

        def dfs(u):
            if state[u] == 2:
                return
            if state[u] == 1:
                raise ValueError(
                    "Cycle detected in parent_index while building qid_tree topology. "
                    "Please check kc_knowledge_tree.json / keyid2idx_tree.json."
                )
            state[u] = 1
            p = parent_index[u]
            if 0 <= p < n:
                dfs(p)
            state[u] = 2
            order.append(u)

        for i in range(n):
            dfs(i)

        return order

    def _build_ancestor_index(self, parent_index, max_depth=16):
        """
        Build fixed-shape ancestor lookup tables.

        ancestor_index[c, d] is the d-th ancestor of node c, starting from
        direct parent. Invalid slots are 0 and masked out by ancestor_mask.
        """
        n = len(parent_index)
        ancestor_index = [[0 for _ in range(max_depth)] for _ in range(n)]
        ancestor_mask = [[False for _ in range(max_depth)] for _ in range(n)]

        truncated_nodes = 0
        for c in range(n):
            seen = set([c])
            p = parent_index[c]
            depth = 0
            while 0 <= p < n:
                if p in seen:
                    raise ValueError(
                        f"Cycle detected when collecting ancestors for node {c}."
                    )
                if depth >= max_depth:
                    truncated_nodes += 1
                    break
                ancestor_index[c][depth] = int(p)
                ancestor_mask[c][depth] = True
                seen.add(p)
                p = parent_index[p]
                depth += 1

        if truncated_nodes > 0:
            print(
                f"warning: {truncated_nodes} qid_tree nodes have ancestors deeper "
                f"than tree_max_ancestor_depth={max_depth}; deeper ancestors are ignored."
            )

        return ancestor_index, ancestor_mask

    def _tree_kc_table(self):
        """
        Return embedding table for all tree nodes.

        independent mode:
            node embedding = its own residual embedding.
            No parent-to-child embedding inheritance.

        inherit mode:
            old behavior. child embedding = gated combination of projected
            parent embedding and own residual embedding.
        """
        residual = self.residual_kc_emb.weight

        if self.tree_embed_mode == "independent":
            return residual

        if self.tree_embed_mode != "inherit":
            raise ValueError(f"Unknown tree_embed_mode: {self.tree_embed_mode}")

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
        """
        Embed tree node ids. Supports q_data with shape [B, L] or [B, L, A].
        Invalid ids are embedded as zero vectors.
        """
        q_valid = (q_data >= 0) & (q_data < self.n_question)
        safe_q = q_data.long().clamp(0, self.n_question - 1)

        kc_table = self._tree_kc_table()
        q_embed_data = kc_table[safe_q]
        q_embed_data = q_embed_data * q_valid.unsqueeze(-1).float()
        return q_embed_data

    def _direct_parent_ids(self, q_data):
        """Return direct parent id and valid mask for each q_data token."""
        q_valid = (q_data >= 0) & (q_data < self.n_question)
        safe_q = q_data.long().clamp(0, self.n_question - 1)
        parent_ids = self.parent_index[safe_q]
        parent_valid = q_valid & (parent_ids >= 0)
        parent_ids = parent_ids.masked_fill(~parent_valid, 0)
        return parent_ids.long(), parent_valid

    def _ancestor_ids(self, q_data):
        """Return all ancestor ids and valid mask for each q_data token."""
        q_valid = (q_data >= 0) & (q_data < self.n_question)
        safe_q = q_data.long().clamp(0, self.n_question - 1)
        ancestor_ids = self.ancestor_index[safe_q]
        ancestor_valid = self.ancestor_mask[safe_q] & q_valid.unsqueeze(-1)
        ancestor_ids = ancestor_ids.masked_fill(~ancestor_valid, 0)
        return ancestor_ids.long(), ancestor_valid

    def _tree_query_embed_with_difficulty(self, node_ids, pid_embed_data=None):
        """
        Build query embedding for tree node ids, optionally with AKT's Rasch
        difficulty branch added. node_ids may be [B, L] or [B, L, A].
        """
        node_valid = (node_ids >= 0) & (node_ids < self.n_question)
        safe_ids = node_ids.long().clamp(0, self.n_question - 1)
        emb = self.tree_kc_embed(safe_ids)

        if self.n_pid > 0 and pid_embed_data is not None:
            diff = self.q_embed_diff(safe_ids.clamp(min=0, max=self.n_question))
            pdiff = pid_embed_data
            # pid_embed_data is [B, L, 1]. For ancestor embeddings [B, L, A, D],
            # insert an ancestor dimension before the feature dimension.
            while pdiff.dim() < diff.dim():
                pdiff = pdiff.unsqueeze(-2)
            emb = emb + pdiff * diff

        emb = emb * node_valid.unsqueeze(-1).float()
        return emb, node_valid

    def _masked_bce_with_logits(self, logits, labels, mask):
        """Stable BCE with an explicit boolean mask."""
        mask = mask.bool()
        if mask.sum().item() == 0:
            return logits.new_tensor(0.0)
        return F.binary_cross_entropy_with_logits(
            logits[mask],
            labels.float()[mask],
            reduction="mean",
        )

    def _ancestor_base_emb(self, q_data, target):
        """
        Independent non-leaf branch embedding.

        q_data is an ancestor-id sequence with shape [B*, L]. This branch has
        its own KC embedding and response embedding, so it is fully decoupled
        from the leaf AKT branch.
        """
        q_valid = (q_data >= 0) & (q_data < self.n_question)
        target_valid = (target >= 0) & (target <= 1)

        safe_q = q_data.long().clamp(0, self.n_question - 1)
        safe_target = target.long().clamp(0, 1)

        q_embed = self.ancestor_kc_emb(safe_q)
        q_embed = q_embed * q_valid.unsqueeze(-1).float()

        qa_embed = self.ancestor_response_emb(safe_target) + q_embed
        qa_embed = qa_embed * (q_valid & target_valid).unsqueeze(-1).float()
        return q_embed, qa_embed, q_valid & target_valid

    def _ancestor_forward(self, q_data, target):
        """
        Run a separate AKT-style forward pass for ancestors.

        Returns:
            ancestor_logits: [B, L, A]
            ancestor_valid:  [B, L, A]
        """
        ancestor_ids, ancestor_valid = self._ancestor_ids(q_data)
        bsz, seqlen, adepth = ancestor_ids.shape

        # Treat each depth as a separate sequence in the batch.
        flat_q = ancestor_ids.permute(0, 2, 1).reshape(bsz * adepth, seqlen)
        flat_valid = ancestor_valid.permute(0, 2, 1).reshape(bsz * adepth, seqlen)
        flat_target = target.unsqueeze(1).expand(-1, adepth, -1).reshape(bsz * adepth, seqlen)

        q_embed, qa_embed, target_valid = self._ancestor_base_emb(flat_q, flat_target)
        branch_valid = flat_valid & target_valid

        # Fully independent encoder. No leaf d_output is reused here.
        d_output = self.ancestor_model(q_embed, qa_embed, None)
        logits = self.ancestor_out(torch.cat([d_output, q_embed], dim=-1)).squeeze(-1)
        logits = logits.masked_fill(~branch_valid, 0.0)

        logits = logits.view(bsz, adepth, seqlen).permute(0, 2, 1).contiguous()
        valid = branch_valid.view(bsz, adepth, seqlen).permute(0, 2, 1).contiguous()
        return logits, valid

    def _apply_depth_decay_fusion(self, leaf_logits, q_data, target):
        """
        Fuse leaf logits with independent ancestor logits using depth decay only.

        parent      depth=1 -> w
        grandparent depth=2 -> w^2
        depth-d             -> w^d

        Then multiply all ancestor weights by occurrence gate:
            exp(-lambda * count_before)
        """
        if self.tree_pred_fusion_mode == "none":
            return leaf_logits
        if self.tree_pred_fusion_mode != "depth_decay":
            raise ValueError("Decoupled tree prediction supports only 'none' or 'depth_decay'.")

        ancestor_logits, ancestor_valid = self._ancestor_forward(q_data, target)
        selected_valid = self._select_pred_ancestor_valid(ancestor_valid)
        has_ancestor = selected_valid.any(dim=-1)

        valid_float = selected_valid.float()
        adepth = ancestor_logits.size(-1)
        depth_weights = self._tree_depth_decay_weights(adepth, ancestor_logits.device)
        ancestor_weights = valid_float * depth_weights

        total_depth_weight = ancestor_weights.sum(dim=-1)
        scale = torch.where(
            total_depth_weight > 1.0,
            1.0 / total_depth_weight.clamp(min=1e-8),
            torch.ones_like(total_depth_weight),
        )
        ancestor_weights = ancestor_weights * scale.unsqueeze(-1)
        total_depth_weight = ancestor_weights.sum(dim=-1)

        occ_gate = self._tree_occurrence_gate(q_data)
        ancestor_weights = ancestor_weights * occ_gate.unsqueeze(-1)
        total_weight = total_depth_weight * occ_gate

        ancestor_prior = (ancestor_logits * ancestor_weights).sum(dim=-1)
        fused_logits = (1.0 - total_weight) * leaf_logits + ancestor_prior
        return torch.where(has_ancestor, fused_logits, leaf_logits)

    def _independent_ancestor_loss(self, q_data, target):
        """
        Optional independent loss for the non-leaf branch.

        This loss trains only ancestor_kc_emb, ancestor_response_emb,
        ancestor_model, and ancestor_out. It does not share the leaf encoder or
        leaf prediction head, so it cannot backpropagate into the leaf branch.
        """
        if self.tree_branch_loss_weight <= 0.0:
            return target.new_tensor(0.0, dtype=torch.float32)

        ancestor_logits, ancestor_valid = self._ancestor_forward(q_data, target)
        selected_valid = self._select_pred_ancestor_valid(ancestor_valid)
        labels = target.float().unsqueeze(-1).expand_as(ancestor_logits)
        loss = self._masked_bce_with_logits(ancestor_logits, labels, selected_valid)
        return self.tree_branch_loss_weight * loss

    # ------------------------------------------------------------------
    # Utilities for FMKC
    # ------------------------------------------------------------------
    # Utilities for FMKC
    # ------------------------------------------------------------------

    def _check_fmkc_shape(self, c_multi):
        if c_multi.dim() != 3:
            raise ValueError(
                f"qid_fmkc expects q_data shape [B, L, F], got {tuple(c_multi.shape)}"
            )
        if c_multi.size(-1) != self.num_fm_fields:
            raise ValueError(
                f"Expected {self.num_fm_fields} KC fields, but got {c_multi.size(-1)}"
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
        """
        return self._fmkc_field_valid_bool(c_multi).any(dim=-1)

    def _fmkc_tuple_residual_ids(self, c_multi, dense_kc_ids=None):
        """
        Convert a multi-field KC tuple into a residual id.

        Preferred path:
            use dense_kc_ids, which should be a collision-free full-KC id.

        Fallback path:
            only allowed when tuple_space <= n_question. No modulo fallback is
            used, so residual-id collision is not silently introduced.
        """
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
            return residual_ids, valid, True

        if self.fmkc_tuple_space > self.n_question:
            raise ValueError(
                "qid_fmkc residual requires q_dense when tuple space exceeds n_question. "
                f"Got tuple_space={self.fmkc_tuple_space}, n_question={self.n_question}."
            )

        safe_ids = self._fmkc_safe_ids(c_multi)
        strides = self.fmkc_strides.view(1, 1, self.num_fm_fields)
        residual_ids = (safe_ids * strides).sum(dim=-1)
        residual_ids = residual_ids.masked_fill(~token_valid, 0)
        return residual_ids, token_valid, False

    def fm_kc_embed(self, c_multi, dense_kc_ids=None):
        """
        Factorized KC embedding with random residual.

        Output:
            FM(field embeddings) + residual_scale * residual_emb(dense_kc_ids)
        """
        self._check_fmkc_shape(c_multi)

        c_multi = c_multi.long()
        field_valid_bool = self._fmkc_field_valid_bool(c_multi)
        field_valid = field_valid_bool.float()
        safe_ids = self._fmkc_safe_ids(c_multi)

        embs = []
        for i in range(self.num_fm_fields):
            e_i = self.kc_emb[i](safe_ids[..., i])
            e_i = e_i * field_valid[..., i].unsqueeze(-1)
            embs.append(e_i)

        # [B, L, F, D]
        es = torch.stack(embs, dim=2)

        # [B, L, 1]
        valid_count = field_valid.sum(dim=-1, keepdim=True)
        valid_count_safe = valid_count.clamp(min=1.0)
        token_valid = (valid_count > 0).float()

        # First-order FM term.
        alpha = self.alpha_fm1.view(1, 1, self.num_fm_fields, -1)
        first = (alpha * es).sum(dim=2)
        first = first / torch.sqrt(valid_count_safe)

        # Second-order FM term.
        s = es.sum(dim=2)
        sum_sq = (es * es).sum(dim=2)
        fm2 = 0.5 * (s * s - sum_sq)

        pair_count = valid_count_safe * (valid_count_safe - 1.0) / 2.0
        pair_count_safe = pair_count.clamp(min=1.0)
        fm2 = fm2 / torch.sqrt(pair_count_safe)

        fm_emb = self.fm_out_scale * (first + self.fm2_scale * fm2)
        fm_emb = fm_emb * token_valid

        residual_ids, residual_valid, _ = self._fmkc_tuple_residual_ids(
            c_multi,
            dense_kc_ids=dense_kc_ids,
        )
        residual_emb = self.kc_residual_emb(residual_ids)
        residual_emb = residual_emb * residual_valid.unsqueeze(-1).float()

        return fm_emb + self.kc_residual_scale * residual_emb

    # ------------------------------------------------------------------
    # Base embeddings
    # ------------------------------------------------------------------

    def base_emb(self, q_data, target, q_dense=None):
        target_valid = (target >= 0) & (target <= 1)
        safe_target = target.long().clamp(0, 1)

        if self.emb_type == "qid_fmkc":
            if q_dense is None:
                raise ValueError("qid_fmkc requires q_dense to avoid residual id collision.")
            q_embed_data = self.fm_kc_embed(q_data, dense_kc_ids=q_dense)
            q_valid = self.fmkc_token_valid(q_data)
            qa_embed_data = self.qa_embed(safe_target) + q_embed_data
            qa_embed_data = qa_embed_data * (q_valid & target_valid).unsqueeze(-1).float()

        elif self.emb_type == "qid_tree":
            q_valid = (q_data >= 0) & (q_data < self.n_question)
            q_embed_data = self.tree_kc_embed(q_data)
            qa_embed_data = self.response_emb(safe_target) + q_embed_data
            qa_embed_data = qa_embed_data * (q_valid & target_valid).unsqueeze(-1).float()

        else:
            q_valid = (q_data >= 0) & (q_data < self.n_question)
            safe_q = q_data.long().clamp(0, self.n_question - 1)
            q_embed_data = self.q_embed(safe_q)
            q_embed_data = q_embed_data * q_valid.unsqueeze(-1).float()

            if self.separate_qa:
                qa_data = safe_q + self.n_question * safe_target
                qa_embed_data = self.qa_embed(qa_data)
                qa_embed_data = qa_embed_data * (q_valid & target_valid).unsqueeze(-1).float()
            else:
                qa_embed_data = self.qa_embed(safe_target) + q_embed_data
                qa_embed_data = qa_embed_data * (q_valid & target_valid).unsqueeze(-1).float()

        return q_embed_data, qa_embed_data

    def forward(self, q_data, target, pid_data=None, qtest=False, q_dense=None):
        emb_type = self.emb_type

        # Backward-compatible convenience:
        # if the 4th positional argument is a tensor, treat it as q_dense rather than qtest.
        if torch.is_tensor(qtest):
            if q_dense is not None:
                raise ValueError("Do not pass both positional q_dense and keyword q_dense.")
            q_dense = qtest
            qtest = False

        # ------------------------------------------------------------------
        # Base question/KC embedding and QA embedding
        # ------------------------------------------------------------------
        if emb_type.startswith("qid"):
            q_embed_data, qa_embed_data = self.base_emb(
                q_data,
                target,
                q_dense=q_dense,
            )
        else:
            raise ValueError(f"AKT forward unsupported emb_type: {emb_type}")

        pid_embed_data = None

        # ------------------------------------------------------------------
        # Problem difficulty / Rasch branch
        # ------------------------------------------------------------------
        if self.n_pid > 0:
            if pid_data is None:
                raise ValueError("pid_data must be provided when n_pid > 0")

            pid_ids = pid_data.long().clamp(min=0, max=self.n_pid)

            if self.emb_type == "qid_fmkc":
                # qid_fmkc uses pid ids for q_embed_diff.
                q_embed_diff_data = self.q_embed_diff(pid_ids)
            else:
                # Original AKT qid/qid_tree mode uses q_data for q_embed_diff.
                safe_q_for_diff = q_data.long().clamp(min=0, max=self.n_question)
                q_embed_diff_data = self.q_embed_diff(safe_q_for_diff)

            # u_q: problem difficulty scalar
            pid_embed_data = self.difficult_param(pid_ids)

            # question encoder:
            # c_ct + u_q * d_ct
            q_embed_data = q_embed_data + pid_embed_data * q_embed_diff_data

            safe_target = target.long().clamp(0, 1)
            qa_embed_diff_data = self.qa_embed_diff(safe_target)

            if self.separate_qa:
                qa_embed_data = qa_embed_data + pid_embed_data * qa_embed_diff_data
            else:
                qa_embed_data = qa_embed_data + pid_embed_data * (
                    qa_embed_diff_data + q_embed_diff_data
                )

            c_reg_loss = (pid_embed_data ** 2.0).sum() * self.l2
        else:
            c_reg_loss = q_embed_data.new_tensor(0.0)

        # ------------------------------------------------------------------
        # AKT architecture
        # ------------------------------------------------------------------
        d_output = self.model(q_embed_data, qa_embed_data, pid_embed_data)

        concat_q = torch.cat([d_output, q_embed_data], dim=-1)

        leaf_logits = self.out(concat_q).squeeze(-1)

        # Optional qid_tree parent/ancestor prediction fusion.
        if self.emb_type == "qid_tree":
            # Leaf training is always leaf-only. The non-leaf branch is separate:
            # it can have its own loss, but it never reuses leaf d_output or self.out.
            if self.training and not qtest:
                c_reg_loss = c_reg_loss + self._independent_ancestor_loss(q_data, target)
                output = leaf_logits
            elif self.tree_pred_fusion_mode != "none":
                output = self._apply_depth_decay_fusion(leaf_logits, q_data, target)
            else:
                output = leaf_logits
        else:
            output = leaf_logits

        preds = torch.sigmoid(output)

        if not qtest:
            return preds, c_reg_loss
        else:
            # concat_q is still the leaf-level representation, which is useful
            # for compatibility with existing pyKT prediction utilities.
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
        qa_pos_embed = qa_embed_data
        q_pos_embed = q_embed_data

        y = qa_pos_embed
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
            pad_zero = torch.zeros(bs, 1, dim, device=scores.device)
            scores = torch.cat([pad_zero, scores[:, 0:-1, :]], dim=1)

        return scores


def attention(q, k, v, d_k, mask, dropout, zero_pad, gamma=None, pdiff=None):
    """
    Multi-head attention with AKT distance decay.
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
        pad_zero = torch.zeros(bs, head, 1, seqlen, device=scores.device)
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
