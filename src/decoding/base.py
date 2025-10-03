# src/decoding/base.py
import torch


def _select_start_token_id(tok, eot_id=None):
    start_id = tok.token_to_id("<SOT>")
    if start_id is None:
        if eot_id is None:
            eot_id = tok.token_to_id("<EOT>")
        start_id = eot_id if eot_id is not None else tok.pad_id
    if start_id is None:
        raise ValueError(
            "Tokenizer must define at least one start token among <SOT>, <EOT> or the pad token."
        )
    return start_id

@torch.no_grad()
def greedy_decode(model, tok, input_text: str, max_new_tokens=128, device="cpu"):
    """
    Decoding greedy autoregressivo. Stop su <EOT> se presente.
    """
    model.eval()
    pad_id = tok.pad_id
    eot_id = tok.token_to_id("<EOT>")
    # encoder input
    inp = torch.tensor([tok.encode(input_text)], dtype=torch.long, device=device)
    att = (inp != pad_id)
    # seed decoder: usa <SOT> se esiste, altrimenti <EOT> o <pad>
    start_id = _select_start_token_id(tok, eot_id)
    y = torch.tensor([[start_id]], dtype=torch.long, device=device)

    for _ in range(max_new_tokens):
        out = model(inp, att, decoder_input_ids=y)
        next_id = out["logits"][:, -1, :].argmax(-1, keepdim=True)  # greedy
        y = torch.cat([y, next_id], dim=1)
        if eot_id is not None and next_id.item() == eot_id:
            break
    return y[0].tolist()

@torch.no_grad()
def decode_to_text(model, tok, input_text, **kw):
    ids = greedy_decode(model, tok, input_text, **kw)
    start_id = _select_start_token_id(tok)
    if ids and ids[0] == start_id:
        ids = ids[1:]
    return tok.tk.decode(ids)
