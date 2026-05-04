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
            self.alpha_fm1 = Parameter(torch.ones(self.num_fmkc_fields, self.emb_size))
            self.alpha_kcr_fm1 = Parameter(torch.ones(self.num_fmkc_fields, self.emb_size))
            self.fm2_scale = Parameter(torch.tensor(0.1))
            self.fm_out_scale = Parameter(torch.tensor(1.0))
            self.kcr_fm2_scale = Parameter(torch.tensor(0.1))
            self.kcr_fm_out_scale = Parameter(torch.tensor(1.0))
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
        

    def _check_fmkc_shape(self, c_multi):
        if c_multi.dim() != 3:
            raise ValueError(f"qid_fmkc expects q shape [B, L, F], got {tuple(c_multi.shape)}")
        if c_multi.size(-1) != self.num_fmkc_fields:
            raise ValueError(f"Expected {self.num_fmkc_fields} KC fields, got {c_multi.size(-1)}")

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

    def fm_kc_embed(self, c_multi):
        return self._fm_embed(
            c_multi=c_multi,
            emb_tables=self.kc_emb,
            alpha=self.alpha_fm1,
            fm2_scale=self.fm2_scale,
            fm_out_scale=self.fm_out_scale,
            r=None,
        )

    def fm_kcr_embed(self, c_multi, r):
        return self._fm_embed(
            c_multi=c_multi,
            emb_tables=self.kcr_emb,
            alpha=self.alpha_kcr_fm1,
            fm2_scale=self.kcr_fm2_scale,
            fm_out_scale=self.kcr_fm_out_scale,
            r=r,
        )

    def forward(self, q, r):
        emb_type = self.emb_type
        if emb_type == "qid_fmkc":
            self._check_fmkc_shape(q)
            B, L, _ = q.shape
            xemb = self.fm_kcr_embed(q, r)
            h, _ = self.lstm_layer(xemb)
            h = self.dropout_layer(h)
            if L <= 1:
                return torch.full((B, L), 0.5, dtype=h.dtype, device=h.device)
            history = h[:, :-1, :]
            target_emb = self.fm_kc_embed(q[:, 1:, :])
            pred_features = torch.cat([history, target_emb, history * target_emb], dim=-1)
            y_next = torch.sigmoid(self.out_layer(pred_features).squeeze(-1))
            y = torch.full((B, L), 0.5, dtype=y_next.dtype, device=y_next.device)
            y[:, 1:] = y_next
            return y
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