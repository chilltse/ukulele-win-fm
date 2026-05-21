import json
import os
from typing import List, Optional, Sequence, Tuple, Union

import torch
import torch.nn as nn


def load_aegiskc_item_fields(path: str, num_c: int) -> Tuple[List[str], List[int], torch.LongTensor]:
    """Load ontology field metadata for AegisKC.

    JSON format:
    {
      "field_names": ["prev_pitches", "prev_strings", "pitches", "strings"],
      "field_dims": [100, 20, 100, 20],
      "item_fields": [[0, 1, 2, 3], ...]
    }

    len(item_fields) must equal num_c. Row i must describe concept/item id i.
    Use -1 in item_fields for missing values; internally it is mapped to padding id 0.
    """
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    field_names = data["field_names"]
    field_dims = [int(x) for x in data["field_dims"]]
    item_fields = data["item_fields"]

    if len(field_names) != len(field_dims):
        raise ValueError(f"field_names length {len(field_names)} != field_dims length {len(field_dims)}")
    if len(item_fields) != num_c:
        raise ValueError(f"len(item_fields)={len(item_fields)} must equal num_c={num_c}; path={path}")

    num_fields = len(field_dims)
    for i, row in enumerate(item_fields):
        if len(row) != num_fields:
            raise ValueError(f"item_fields[{i}] length {len(row)} != num_fields={num_fields}")

    return field_names, field_dims, torch.tensor(item_fields, dtype=torch.long)


class AegisKCEncoder(nn.Module):
    """AegisKC ontology encoder.

    Supports two modes:

    1. Static mode, context=None:
       alpha_i and beta_ij depend only on item ontology fields.

    2. Dynamic mode, context is not None:
       alpha_i(t) and beta_ij(t) are conditioned on student state h_t.
       This lets the same item/concept have different field/pair weights for
       different students and different time steps.

    Given an item/concept id c, ontology fields are looked up, such as:
        prev_pitches | prev_strings | pitches | strings

    Then:
        z_main = Σ_i α_i e_i
        z_pair = Σ_{i<j} β_ij (e_i ⊙ e_j)
        z_c    = MLP([z_main ; z_pair])
    """

    def __init__(
        self,
        num_c: int,
        field_dims: Sequence[int],
        item_fields: torch.LongTensor,
        emb_size: int = 200,
        attn_dim: Optional[int] = None,
        dropout: float = 0.1,
        qid_residual: bool = False,
        qid_residual_scale: float = 0.1,
        context_size: Optional[int] = None,
    ):
        super().__init__()
        self.num_c = int(num_c)
        self.field_dims = [int(x) for x in field_dims]
        self.num_fields = len(self.field_dims)
        self.emb_size = int(emb_size)
        self.context_size = int(context_size) if context_size is not None and int(context_size) > 0 else self.emb_size
        self.qid_residual = bool(qid_residual)
        self.qid_residual_scale = float(qid_residual_scale)

        if self.num_fields <= 0:
            raise ValueError("AegisKCEncoder requires at least one ontology field.")
        if tuple(item_fields.shape) != (self.num_c, self.num_fields):
            raise ValueError(
                f"item_fields shape must be ({self.num_c}, {self.num_fields}), got {tuple(item_fields.shape)}"
            )

        self.register_buffer("item_fields", item_fields.long())

        # Padding id 0 is reserved for missing field values; raw ontology value v is shifted to v+1.
        self.field_embs = nn.ModuleList([
            nn.Embedding(dim + 1, self.emb_size, padding_idx=0)
            for dim in self.field_dims
        ])
        self.field_type_emb = nn.Embedding(self.num_fields, self.emb_size)

        self.pair_indices = [(i, j) for i in range(self.num_fields) for j in range(i + 1, self.num_fields)]
        self.num_pairs = len(self.pair_indices)
        self.pair_type_emb = nn.Embedding(max(self.num_pairs, 1), self.emb_size)

        if attn_dim is None or int(attn_dim) <= 0:
            attn_dim = max(16, self.emb_size // 2)
        self.attn_dim = int(attn_dim)

        # Static attention networks.
        self.main_attn = nn.Sequential(
            nn.Linear(self.emb_size * 2, self.attn_dim),
            nn.Tanh(),
            nn.Linear(self.attn_dim, 1),
        )
        self.pair_attn = nn.Sequential(
            nn.Linear(self.emb_size * 2, self.attn_dim),
            nn.Tanh(),
            nn.Linear(self.attn_dim, 1),
        )

        # Dynamic attention networks. These include context h_t.
        self.context_proj = nn.Linear(self.context_size, self.emb_size)
        self.dynamic_main_attn = nn.Sequential(
            nn.Linear(self.emb_size * 3, self.attn_dim),
            nn.Tanh(),
            nn.Linear(self.attn_dim, 1),
        )
        self.dynamic_pair_attn = nn.Sequential(
            nn.Linear(self.emb_size * 3, self.attn_dim),
            nn.Tanh(),
            nn.Linear(self.attn_dim, 1),
        )

        self.fusion = nn.Sequential(
            nn.Linear(self.emb_size * 2, self.emb_size),
            nn.Tanh(),
            nn.Dropout(dropout),
        )

        self.id_residual_emb = nn.Embedding(self.num_c, self.emb_size) if self.qid_residual else None

    def _lookup_field_embs(self, c: torch.LongTensor):
        safe_c = c.clamp(min=0, max=self.num_c - 1)
        fields = self.item_fields[safe_c]  # [..., F]
        embs = []
        for i, emb in enumerate(self.field_embs):
            vals = (fields[..., i] + 1).clamp(min=0, max=self.field_dims[i])
            embs.append(emb(vals))
        return embs

    def _expand_context(self, context: torch.Tensor, target_like: torch.Tensor) -> torch.Tensor:
        """Project and expand context to match [..., D] item field tensors."""
        ctx = torch.tanh(self.context_proj(context))  # context.shape[:-1] + [D]
        # If c has shape [B,T] and ctx [B,T,D], this is already aligned.
        # If c has shape [C] in static all-concept encoding, dynamic context should not be used.
        if ctx.shape[:-1] != target_like.shape[:-1]:
            raise ValueError(
                f"Dynamic AegisKC context shape {tuple(ctx.shape)} is not aligned with target shape {tuple(target_like.shape)}. "
                "Use dynamic mode only for target concepts, not all-concept scoring."
            )
        return ctx

    def forward(
        self,
        c: torch.LongTensor,
        return_info: bool = False,
        context: Optional[torch.Tensor] = None,
    ):
        c = c.long()
        field_embs = self._lookup_field_embs(c)
        E = torch.stack(field_embs, dim=-2)  # [..., F, D]
        dynamic = context is not None

        # Main-effect attention.
        field_ids = torch.arange(self.num_fields, device=E.device)
        field_type = self.field_type_emb(field_ids)
        while field_type.dim() < E.dim():
            field_type = field_type.unsqueeze(0)
        field_type = field_type.expand_as(E)

        if dynamic:
            ctx = self._expand_context(context, E[..., 0, :])  # [..., D]
            ctx_field = ctx.unsqueeze(-2).expand_as(E)
            main_logits = self.dynamic_main_attn(torch.cat([E, field_type, ctx_field], dim=-1)).squeeze(-1)
        else:
            main_logits = self.main_attn(torch.cat([E, field_type], dim=-1)).squeeze(-1)

        alpha = torch.softmax(main_logits, dim=-1)
        z_main = (alpha.unsqueeze(-1) * E).sum(dim=-2)

        # Pairwise-interaction attention.
        if self.num_pairs > 0:
            pair_vecs = torch.stack([field_embs[i] * field_embs[j] for i, j in self.pair_indices], dim=-2)
            pair_ids = torch.arange(self.num_pairs, device=E.device)
            pair_type = self.pair_type_emb(pair_ids)
            while pair_type.dim() < pair_vecs.dim():
                pair_type = pair_type.unsqueeze(0)
            pair_type = pair_type.expand_as(pair_vecs)

            if dynamic:
                ctx_pair = ctx.unsqueeze(-2).expand_as(pair_vecs)
                pair_logits = self.dynamic_pair_attn(torch.cat([pair_vecs, pair_type, ctx_pair], dim=-1)).squeeze(-1)
            else:
                pair_logits = self.pair_attn(torch.cat([pair_vecs, pair_type], dim=-1)).squeeze(-1)

            beta = torch.softmax(pair_logits, dim=-1)
            z_pair = (beta.unsqueeze(-1) * pair_vecs).sum(dim=-2)
        else:
            beta = None
            z_pair = torch.zeros_like(z_main)

        z = self.fusion(torch.cat([z_main, z_pair], dim=-1))

        if self.id_residual_emb is not None:
            safe_c = c.clamp(min=0, max=self.num_c - 1)
            z = z + self.qid_residual_scale * self.id_residual_emb(safe_c)

        if return_info:
            return z, {
                "field_attention": alpha,
                "pair_attention": beta,
                "pair_indices": self.pair_indices,
                "dynamic": dynamic,
            }
        return z


class AegisKC(nn.Module):
    """AegisKC-DKT / AegisKC-Sentinel.

    Two emb_type branches:

    1. Static branch:
       emb_type in {"aegiskc", "aegiskc_static", "static"}
       forward(c, r) returns [B, T, num_c].
       This is drop-in compatible with pyKT DKT gather-style loss.

    2. Dynamic branch:
       emb_type in {"aegiskc_dynamic", "aegiskc_sentinel", "sentinel", "dynamic"}
       forward(c, r) returns [B, T].
       Attention depends on student state h_t, so the same concept can have
       different field/pair weights for different students/time steps.

       For pyKT training, use a special branch:
           y = model(cseqs, rseqs)              # [B,T]
           loss = BCE(y[:, 1:][smasks], rshft[smasks])
       or adapt according to your existing shifted-mask convention.
    """

    STATIC_EMB_TYPES = {"aegiskc", "aegiskc_static", "static"}
    DYNAMIC_EMB_TYPES = {"aegiskc_dynamic", "aegiskc_sentinel", "sentinel", "dynamic"}

    def __init__(
        self,
        num_c: int,
        emb_size: int,
        field_dims: Sequence[int],
        item_fields: torch.LongTensor,
        dropout: float = 0.1,
        emb_type: str = "aegiskc",
        hidden_size: Optional[int] = None,
        attn_dim: Optional[int] = None,
        qid_residual: bool = False,
        qid_residual_scale: float = 0.1,
        emb_path: str = "",
        pretrain_dim: int = 768,
    ):
        super().__init__()
        self.model_name = "aegiskc"
        self.num_c = int(num_c)
        self.emb_size = int(emb_size)
        self.hidden_size = int(hidden_size) if hidden_size is not None and int(hidden_size) > 0 else int(emb_size)
        self.emb_type = str(emb_type)
        self.is_dynamic = self.emb_type in self.DYNAMIC_EMB_TYPES
        if self.emb_type not in self.STATIC_EMB_TYPES and self.emb_type not in self.DYNAMIC_EMB_TYPES:
            raise ValueError(
                f"Unsupported AegisKC emb_type={self.emb_type!r}. "
                f"Use one of static={sorted(self.STATIC_EMB_TYPES)} or dynamic={sorted(self.DYNAMIC_EMB_TYPES)}."
            )

        self.encoder = AegisKCEncoder(
            num_c=self.num_c,
            field_dims=field_dims,
            item_fields=item_fields,
            emb_size=self.emb_size,
            attn_dim=attn_dim,
            dropout=dropout,
            qid_residual=qid_residual,
            qid_residual_scale=qid_residual_scale,
            context_size=self.hidden_size,
        )
        self.response_emb = nn.Embedding(2, self.emb_size)
        self.start_token = nn.Parameter(torch.zeros(1, 1, self.emb_size))

        self.lstm_layer = nn.LSTM(self.emb_size, self.hidden_size, batch_first=True)
        self.dropout_layer = nn.Dropout(dropout)

        # Static all-concept scoring.
        self.h_proj = nn.Linear(self.hidden_size, self.emb_size)
        self.c_proj = nn.Linear(self.emb_size, self.emb_size)
        self.item_bias = nn.Parameter(torch.zeros(self.num_c))

        # Dynamic target-aware scoring.
        self.dynamic_h_proj = nn.Linear(self.hidden_size, self.emb_size)
        self.dynamic_pred = nn.Sequential(
            nn.Linear(self.emb_size * 3, self.emb_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(self.emb_size, 1),
        )

    def encode_all_concepts(self):
        ids = torch.arange(self.num_c, device=self.item_bias.device)
        return self.encoder(ids)

    def _build_student_state(self, c: torch.LongTensor, r: torch.Tensor):
        """Use static concept representation for history encoding."""
        c = c.long()
        r = r.long().clamp(min=0, max=1)
        B, T = c.shape

        z_c_static = self.encoder(c)            # [B, T, D], static history representation
        z_r = self.response_emb(r)              # [B, T, D]
        x = self.dropout_layer(z_c_static + z_r)

        # Previous interaction predicts the current / shifted target, matching pyKT DKT usage.
        prev_x = torch.cat([self.start_token.expand(B, 1, self.emb_size), x[:, :-1, :]], dim=1)
        h, _ = self.lstm_layer(prev_x)
        h = self.dropout_layer(h)               # [B, T, H]
        return h

    def _forward_static(self, c: torch.LongTensor, r: torch.Tensor):
        h = self._build_student_state(c, r)
        all_z = self.encode_all_concepts()      # [C, D]
        h2 = self.h_proj(h)                     # [B, T, D]
        c2 = self.c_proj(all_z)                 # [C, D]
        logits = torch.matmul(h2, c2.t()) + self.item_bias
        return torch.sigmoid(logits)            # [B, T, C]

    def _forward_dynamic(self, c: torch.LongTensor, r: torch.Tensor, return_info: bool = False):
        h = self._build_student_state(c, r)     # [B, T, H]
        z_dyn, info = self.encoder(c, return_info=True, context=h)  # [B, T, D]
        h_dyn = self.dynamic_h_proj(h)          # [B, T, D]
        logits = self.dynamic_pred(torch.cat([h_dyn, z_dyn, h_dyn * z_dyn], dim=-1)).squeeze(-1)
        y = torch.sigmoid(logits)               # [B, T]
        if return_info:
            return y, info
        return y

    def forward(self, c: torch.LongTensor, r: torch.Tensor, return_info: bool = False):
        c = c.long()
        r = r.long().clamp(min=0, max=1)
        if self.is_dynamic:
            return self._forward_dynamic(c, r, return_info=return_info)
        if return_info:
            y = self._forward_static(c, r)
            return y, {"dynamic": False}
        return self._forward_static(c, r)

    def get_attention_for_concepts(self, concept_ids: Union[List[int], torch.LongTensor]):
        """Static attention inspection for selected concepts."""
        if not torch.is_tensor(concept_ids):
            concept_ids = torch.tensor(concept_ids, dtype=torch.long, device=self.item_bias.device)
        else:
            concept_ids = concept_ids.to(self.item_bias.device).long()
        self.eval()
        with torch.no_grad():
            _, info = self.encoder(concept_ids, return_info=True)
        return info

    def get_dynamic_attention_for_batch(self, c: torch.LongTensor, r: torch.Tensor):
        """Dynamic attention inspection for a batch sequence.

        Returns field/pair attention conditioned on the generated student states.
        """
        if not self.is_dynamic:
            raise RuntimeError("Dynamic attention is only available for emb_type='aegiskc_dynamic' or 'aegiskc_sentinel'.")
        self.eval()
        with torch.no_grad():
            _, info = self._forward_dynamic(c, r, return_info=True)
        return info
