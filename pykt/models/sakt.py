import torch

from torch.nn import Module, Embedding, Linear, MultiheadAttention, LayerNorm, Dropout, ModuleList, Parameter
from .utils import transformer_FFN, pos_encode, ut_mask, get_clones

class SAKT(Module):
    def __init__(self, num_c, seq_len, emb_size, num_attn_heads, dropout, num_en=2, emb_type="qid", emb_path="", pretrain_dim=768, num_c_fmkc=None):
        super().__init__()
        self.model_name = "sakt"
        self.emb_type = emb_type

        self.num_c = num_c
        self.seq_len = seq_len
        self.emb_size = emb_size
        self.num_attn_heads = num_attn_heads
        self.dropout = dropout
        self.num_en = num_en

        if emb_type == "qid_fmkc":
            if num_c_fmkc is None or len(num_c_fmkc) < 1:
                raise ValueError("emb_type qid_fmkc requires num_c_fmkc")
            self.num_c_fmkc = [int(n) for n in num_c_fmkc]
            self.num_fmkc_fields = len(self.num_c_fmkc)
            self.exercise_emb_fmkc = ModuleList(
                [Embedding(n, emb_size) for n in self.num_c_fmkc]
            )
            self.interaction_emb_fmkc = ModuleList(
                [Embedding(n * 2, emb_size) for n in self.num_c_fmkc]
            )
            self.alpha_q_fm1 = Parameter(torch.ones(self.num_fmkc_fields, emb_size))
            self.alpha_qr_fm1 = Parameter(torch.ones(self.num_fmkc_fields, emb_size))
            self.q_fm2_scale = Parameter(torch.tensor(0.1))
            self.q_fm_out_scale = Parameter(torch.tensor(1.0))
            self.qr_fm2_scale = Parameter(torch.tensor(0.1))
            self.qr_fm_out_scale = Parameter(torch.tensor(1.0))
        elif emb_type.startswith("qid"):
            # num_c, seq_len, emb_size, num_attn_heads, dropout, emb_path="")
            self.interaction_emb = Embedding(num_c * 2, emb_size)
            self.exercise_emb = Embedding(num_c, emb_size)
            # self.P = Parameter(torch.Tensor(self.seq_len, self.emb_size))
        self.position_emb = Embedding(seq_len, emb_size)

        self.blocks = get_clones(Blocks(emb_size, num_attn_heads, dropout), self.num_en)

        self.dropout_layer = Dropout(dropout)
        self.pred = Linear(self.emb_size, 1)

    def _check_fmkc_shape(self, c_multi):
        if c_multi.dim() != 3:
            raise ValueError(
                f"qid_fmkc expects q shape [B, L, F], got {tuple(c_multi.shape)}"
            )
        if c_multi.size(-1) != self.num_fmkc_fields:
            raise ValueError(
                f"Expected {self.num_fmkc_fields} KC fields, got {c_multi.size(-1)}"
            )

    def _fm_embed(self, c_multi, emb_tables, alpha, fm2_scale, fm_out_scale, r=None):
        self._check_fmkc_shape(c_multi)
        c_multi = c_multi.long()
        field_valid_bool = c_multi >= 0
        if r is not None:
            response_valid_bool = (r >= 0) & (r <= 1)
            field_valid_bool = field_valid_bool & response_valid_bool.unsqueeze(-1)
            ri = r.long().clamp(0, 1)
        else:
            ri = None
        field_valid = field_valid_bool.float()
        safe_ids = c_multi.clamp(min=0)
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
        alpha = alpha.view(1, 1, self.num_fmkc_fields, self.emb_size)
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

    def fm_kc_embed(self, c_multi):
        return self._fm_embed(
            c_multi=c_multi,
            emb_tables=self.exercise_emb_fmkc,
            alpha=self.alpha_q_fm1,
            fm2_scale=self.q_fm2_scale,
            fm_out_scale=self.q_fm_out_scale,
            r=None,
        )

    def fm_kcr_embed(self, c_multi, r):
        return self._fm_embed(
            c_multi=c_multi,
            emb_tables=self.interaction_emb_fmkc,
            alpha=self.alpha_qr_fm1,
            fm2_scale=self.qr_fm2_scale,
            fm_out_scale=self.qr_fm_out_scale,
            r=r,
        )

    def base_emb(self, q, r, qry):
        if self.emb_type == "qid_fmkc":
            qshftemb = self.fm_kc_embed(qry)
            xemb = self.fm_kcr_embed(q, r)
        else:
            x = q + self.num_c * r
            qshftemb, xemb = self.exercise_emb(qry), self.interaction_emb(x)
    
        posemb = self.position_emb(pos_encode(xemb.shape[1]))
        xemb = xemb + posemb
        return qshftemb, xemb

    def forward(self, q, r, qry, qtest=False):
        emb_type = self.emb_type
        qemb, qshftemb, xemb = None, None, None
        if emb_type in ["qid", "qid_fmkc"]:
            qshftemb, xemb = self.base_emb(q, r, qry)
        # print(f"qemb: {qemb.shape}, xemb: {xemb.shape}, qshftemb: {qshftemb.shape}")
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
        # transformer: attn -> drop -> skip -> norm transformer default
        causal_mask = ut_mask(seq_len = k.shape[0])
        attn_emb, _ = self.attn(q, k, v, attn_mask=causal_mask)

        attn_emb = self.attn_dropout(attn_emb)
        attn_emb, q = attn_emb.permute(1, 0, 2), q.permute(1, 0, 2)

        attn_emb = self.attn_layer_norm(q + attn_emb)

        emb = self.FFN(attn_emb)
        emb = self.FFN_dropout(emb)
        emb = self.FFN_layer_norm(attn_emb + emb)
        return emb