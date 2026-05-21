import torch
from torch.nn import Module, Embedding, LSTM, Linear, Dropout


class DKT(Module):
    def __init__(
        self,
        num_c,
        emb_size,
        dropout=0.1,
        emb_type="qid",
        emb_path="",
        pretrain_dim=768,
        num_q=None,
    ):
        super().__init__()

        self.model_name = "dkt"
        self.num_c = num_c
        self.num_q = num_q
        self.emb_size = emb_size
        self.hidden_size = emb_size
        self.emb_type = emb_type

        if emb_type == "pure_question":
            if num_q is None:
                raise ValueError(
                    "emb_type='pure_question' requires num_q. "
                    "Please pass the number of questions as num_q."
                )

            self.num_input = num_q
            self.num_output = num_q

        elif emb_type.startswith("qid"):
            self.num_input = num_c
            self.num_output = num_c

        else:
            raise NotImplementedError(
                f"emb_type={emb_type} is not supported in DKT."
            )

        # interaction embedding:
        # KC-based:
        #   x = c + num_c * r
        #
        # question-based:
        #   x = q + num_q * r
        self.interaction_emb = Embedding(
            self.num_input * 2,
            self.emb_size
        )

        self.lstm_layer = LSTM(
            self.emb_size,
            self.hidden_size,
            batch_first=True
        )

        self.dropout_layer = Dropout(dropout)

        # KC-based: predict all KCs
        # question-based: predict all questions
        self.out_layer = Linear(
            self.hidden_size,
            self.num_output
        )

    def forward(self, q, r):
        """
        Default KC-based mode:
            q means KC id
            q range: 0 ~ num_c - 1

        pure_question mode:
            q means question id
            q range: 0 ~ num_q - 1

        q: [batch_size, seq_len]
        r: [batch_size, seq_len]

        return:
            y: [batch_size, seq_len, num_output]
        """

        q = q.long()
        r = r.long()

        if q.min() < 0 or q.max() >= self.num_input:
            raise ValueError(
                f"Input id out of range. "
                f"q.min={q.min().item()}, q.max={q.max().item()}, "
                f"num_input={self.num_input}, emb_type={self.emb_type}"
            )

        if r.min() < 0 or r.max() > 1:
            raise ValueError(
                f"r must be 0/1. "
                f"r.min={r.min().item()}, r.max={r.max().item()}"
            )

        x = q + self.num_input * r

        xemb = self.interaction_emb(x)

        h, _ = self.lstm_layer(xemb)
        h = self.dropout_layer(h)

        y = self.out_layer(h)
        y = torch.sigmoid(y)

        return y