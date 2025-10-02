import torch, torch.nn as nn
from .layers import SinusoidalPE

class TinySeq2Seq(nn.Module):
    def __init__(self, vocab_size: int, d_model=384, nhead=6,
                 num_encoder_layers=3, num_decoder_layers=3,
                 dim_feedforward=1536, dropout=0.1,
                 pad_id=1, tie_embeddings=True):
        super().__init__()
        self.pad_id = pad_id
        self.emb = nn.Embedding(vocab_size, d_model, padding_idx=pad_id)
        self.pe = SinusoidalPE(d_model)
        self.tfm = nn.Transformer(
            d_model=d_model, nhead=nhead,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            dim_feedforward=dim_feedforward, dropout=dropout,
            batch_first=True
        )
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        if tie_embeddings: self.lm_head.weight = self.emb.weight

    @staticmethod
    def _subsequent_mask(sz: int, device):
        return torch.triu(torch.full((sz, sz), float('-inf'), device=device), 1)

    def forward(self, input_ids, attention_mask, labels=None):
        device = input_ids.device
        enc = self.pe(self.emb(input_ids))
        src_kpm = (attention_mask == 0)

        if labels is not None:
            y_inp = labels[:, :-1]
            y_tgt = labels[:, 1:]
        else:
            y_inp, y_tgt = input_ids, None

        dec = self.pe(self.emb(y_inp))
        tgt_mask = self._subsequent_mask(y_inp.size(1), device)
        tgt_kpm = (y_inp == self.pad_id)

        mem = self.tfm.encoder(enc, src_key_padding_mask=src_kpm)
        out = self.tfm.decoder(dec, mem, tgt_mask=tgt_mask,
                               tgt_key_padding_mask=tgt_kpm,
                               memory_key_padding_mask=src_kpm)
        logits = self.lm_head(out)

        loss = None
        if y_tgt is not None:
            loss_fct = nn.CrossEntropyLoss(ignore_index=self.pad_id)
            loss = loss_fct(logits.reshape(-1, logits.size(-1)), y_tgt.reshape(-1))
        return {"logits": logits, "loss": loss}
