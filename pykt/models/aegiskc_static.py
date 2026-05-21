
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

    AegisKC = Main-effect + Pairwise-interaction ontology encoder.

    Given an item/concept id c, we look up its ontology fields, such as:
        prev_pitches | prev_strings | pitches | strings

    Then we build:
        z_main = Σ_i α_i e_i
        z_pair = Σ_{i<j} β_ij (e_i ⊙ e_j)
        z_c    = MLP([z_main ; z_pair])

    This avoids blindly concatenating all fields into one sparse id while still
    preserving compositional ontology structure.
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
    ):
        super().__init__()
        self.num_c = int(num_c)
        self.field_dims = [int(x) for x in field_dims]
        self.num_fields = len(self.field_dims)
        self.emb_size = int(emb_size)
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

    def forward(self, c: torch.LongTensor, return_info: bool = False):
        c = c.long()
        field_embs = self._lookup_field_embs(c)
        E = torch.stack(field_embs, dim=-2)  # [..., F, D]

        # Main-effect attention.
        field_ids = torch.arange(self.num_fields, device=E.device)
        field_type = self.field_type_emb(field_ids)
        while field_type.dim() < E.dim():
            field_type = field_type.unsqueeze(0)
        field_type = field_type.expand_as(E)
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
            }
        return z


class AegisKC(nn.Module):
    """AegisKC-DKT.

    Drop-in DKT-style model for pyKT.

    forward(c, r) returns [batch, seq_len, num_c].
    pyKT can then gather predictions at cshft just like standard DKT.
    """

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
        self.emb_type = emb_type

        self.encoder = AegisKCEncoder(
            num_c=self.num_c,
            field_dims=field_dims,
            item_fields=item_fields,
            emb_size=self.emb_size,
            attn_dim=attn_dim,
            dropout=dropout,
            qid_residual=qid_residual,
            qid_residual_scale=qid_residual_scale,
        )
        self.response_emb = nn.Embedding(2, self.emb_size)
        self.start_token = nn.Parameter(torch.zeros(1, 1, self.emb_size))

        self.lstm_layer = nn.LSTM(self.emb_size, self.hidden_size, batch_first=True)
        self.dropout_layer = nn.Dropout(dropout)
        self.h_proj = nn.Linear(self.hidden_size, self.emb_size)
        self.c_proj = nn.Linear(self.emb_size, self.emb_size)
        self.item_bias = nn.Parameter(torch.zeros(self.num_c))

    def encode_all_concepts(self):
        ids = torch.arange(self.num_c, device=self.item_bias.device)
        return self.encoder(ids)

    def forward(self, c: torch.LongTensor, r: torch.Tensor):
        c = c.long()
        r = r.long().clamp(min=0, max=1)
        B, T = c.shape

        z_c = self.encoder(c)                  # [B, T, D]
        z_r = self.response_emb(r)             # [B, T, D]
        x = self.dropout_layer(z_c + z_r)

        # Previous interaction predicts the current / shifted target, matching pyKT DKT usage.
        prev_x = torch.cat([self.start_token.expand(B, 1, self.emb_size), x[:, :-1, :]], dim=1)
        h, _ = self.lstm_layer(prev_x)
        h = self.dropout_layer(h)

        all_z = self.encode_all_concepts()     # [C, D]
        h2 = self.h_proj(h)                    # [B, T, D]
        c2 = self.c_proj(all_z)                # [C, D]
        logits = torch.matmul(h2, c2.t()) + self.item_bias
        return torch.sigmoid(logits)

    def get_attention_for_concepts(self, concept_ids: Union[List[int], torch.LongTensor]):
        if not torch.is_tensor(concept_ids):
            concept_ids = torch.tensor(concept_ids, dtype=torch.long, device=self.item_bias.device)
        else:
            concept_ids = concept_ids.to(self.item_bias.device).long()
        self.eval()
        with torch.no_grad():
            _, info = self.encoder(concept_ids, return_info=True)
        return info
