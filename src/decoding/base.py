"""Funzioni di decoding semplici ma documentate per NanoSocrates.

Il progetto richiede soltanto un'inferenza **greedy** per verificare gli
output dei checkpoint in fase di revisione. Invece di mantenere più file
vuoti o feature non usate, qui racchiudo in poche funzioni ben commentate
il necessario per:

* scegliere in modo robusto il token di inizio sequenza;
* generare token uno alla volta finché non incontro ``<EOT>`` o raggiungo
  un limite massimo;
* riconvertire gli ID in testo usando il wrapper del tokenizer.

Queste utility sono usate sia dagli script (`src/run.py` e
`scripts/predict_example.py`) sia dalle notebook note personali.
"""

from __future__ import annotations

from typing import Optional

import logging

import torch


LOGGER = logging.getLogger(__name__)


def _select_start_token_id(tok, eot_id: Optional[int] = None) -> int:
    """Restituisce il token di avvio più adatto per il decoder.

    Nell'ordine provo ``<SOT>``, ``<EOT>`` e infine il token di padding.
    In questo modo non vincolo il progetto a una particolare convenzione
    del tokenizer e intercetto subito configurazioni incoerenti.
    """

    start_id = tok.token_to_id("<SOT>")
    if start_id is None:
        if eot_id is None:
            eot_id = tok.token_to_id("<EOT>")
        start_id = eot_id if eot_id is not None else tok.pad_id
    if start_id is None:
        raise ValueError(
            "Tokenizer must define at least one start token among <SOT>, <EOT> or the pad token."
        )
    return int(start_id)


@torch.no_grad()
def greedy_decode(
    model,
    tok,
    input_text: str,
    max_new_tokens: int = 128,
    device: str = "cpu",
    *,
    min_new_tokens: int = 1,
    debug: bool = False,
):
    """Esegue il decoding greedy restituendo gli ID generati.

    Il modello viene posto in eval mode, preparo encoder/attention mask e
    genero un token per volta scegliendo l'argmax dei logit.
    """

    model.eval()
    pad_id = tok.pad_id
    eot_id = tok.token_to_id("<EOT>")

    # encoder input
    inp = torch.tensor([tok.encode(input_text)], dtype=torch.long, device=device)
    att = inp != pad_id

    # seed decoder: usa <SOT> se esiste, altrimenti <EOT> o <pad>
    start_id = _select_start_token_id(tok, eot_id)
    y = torch.tensor([[start_id]], dtype=torch.long, device=device)

    for step in range(max_new_tokens):
        out = model(inp, att, decoder_input_ids=y)
        logits_step = out["logits"][:, -1, :]
        generated_len = y.size(1) - 1
        if eot_id is not None and generated_len < int(max(0, min_new_tokens)):
            logits_step = logits_step.clone()
            min_val = torch.finfo(logits_step.dtype).min
            logits_step[..., int(eot_id)] = min_val
        next_id = logits_step.argmax(-1, keepdim=True)
        y = torch.cat([y, next_id], dim=1)
        token_id = int(next_id.item())
        if debug:
            LOGGER.debug("[decode] step=%d token_id=%d", step, token_id)
        if eot_id is not None and token_id == int(eot_id):
            break
    ids = y[0].tolist()
    if debug:
        LOGGER.debug("[decode] generated ids=%s", ids)
    return ids


@torch.no_grad()
def decode_to_text(model, tok, input_text: str, **kwargs) -> str:
    """Wrapper pratico che nasconde la rimozione del token iniziale."""

    return_ids = bool(kwargs.pop("return_ids", False))
    debug = bool(kwargs.pop("debug", False))
    min_new_tokens = int(kwargs.pop("min_new_tokens", 1))
    ids = greedy_decode(
        model,
        tok,
        input_text,
        min_new_tokens=min_new_tokens,
        debug=debug,
        **kwargs,
    )
    start_id = _select_start_token_id(tok)
    if ids and ids[0] == start_id:
        ids = ids[1:]
    text = tok.tk.decode(ids)
    if debug:
        LOGGER.debug("[decode] decoded text='%s'", text)
    if return_ids:
        return text, ids
    return text
