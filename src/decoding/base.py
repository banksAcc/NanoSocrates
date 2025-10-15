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

from typing import Callable, Optional, Sequence

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


class RDFGrammarLogitsProcessor:
    """Mask invalid structural transitions for the RDF linearisation grammar."""

    def __init__(self, tok) -> None:
        tokenizer = getattr(tok, "tk", tok)
        self._tokenizer = tokenizer
        self._id_to_token = getattr(tokenizer, "id_to_token", None)
        if not callable(self._id_to_token):
            self._id_to_token = None

        self.sot = "<SOT>"
        self.eot = "<EOT>"
        self.subj = "<SUBJ>"
        self.pred = "<PRED>"
        self.obj = "<OBJ>"

        self.sot_id = self._lookup_id(tok, self.sot)
        self.eot_id = self._lookup_id(tok, self.eot)
        self.subj_id = self._lookup_id(tok, self.subj)
        self.pred_id = self._lookup_id(tok, self.pred)
        self.obj_id = self._lookup_id(tok, self.obj)
        self.pad_id = getattr(tok, "pad_id", None)
        if self.pad_id is None:
            self.pad_id = self._lookup_id(tok, "<pad>")

        self._special_ids = {
            tok_id
            for tok_id in (
                self.sot_id,
                self.eot_id,
                self.subj_id,
                self.pred_id,
                self.obj_id,
                self.pad_id,
            )
            if tok_id is not None
        }

    @staticmethod
    def _lookup_id(tok, name: str) -> Optional[int]:
        func = getattr(tok, "token_to_id", None)
        if callable(func):
            token_id = func(name)
            if token_id is not None:
                return int(token_id)
        tokenizer = getattr(tok, "tk", None)
        func = getattr(tokenizer, "token_to_id", None)
        if callable(func):
            token_id = func(name)
            if token_id is not None:
                return int(token_id)
        return None

    def __call__(self, input_ids: torch.LongTensor, scores: torch.Tensor) -> torch.Tensor:
        if self._id_to_token is None:
            return scores

        batch_size = input_ids.size(0)
        mask_value = torch.finfo(scores.dtype).min
        processed = scores.clone()
        for row_idx in range(batch_size):
            allowed, block_special = self._allowed_next_tokens(input_ids[row_idx])
            row = processed[row_idx]
            if allowed is not None:
                valid_ids = [idx for idx in allowed if 0 <= idx < row.size(-1)]
                if not valid_ids:
                    continue
                mask = torch.ones_like(row, dtype=torch.bool)
                mask[valid_ids] = False
                row = row.clone()
                row[mask] = mask_value
                processed[row_idx] = row
                continue
            if block_special:
                row = row.clone()
                for tok_id in block_special:
                    if 0 <= tok_id < row.size(-1):
                        row[tok_id] = mask_value
                processed[row_idx] = row
        return processed

    def _allowed_next_tokens(
        self, sequence: torch.LongTensor
    ) -> tuple[Optional[set[int]], set[int]]:
        tokens = [self._id_to_token(int(idx)) for idx in sequence.tolist()]
        if any(tok is None for tok in tokens):
            return None, set()

        stage = "expect_sot"
        pad_token = self._id_to_token(self.pad_id) if self.pad_id is not None else None
        for tok in tokens:
            if stage == "expect_sot":
                if tok == self.sot:
                    stage = "expect_subj"
                else:
                    return None, set()
            elif stage == "expect_subj":
                if tok == self.subj:
                    stage = "expect_subj_value"
                else:
                    return None, set()
            elif stage == "expect_subj_value":
                if tok in {self.sot, self.subj, self.pred, self.obj, self.eot}:
                    return None, set()
                stage = "expect_pred"
            elif stage == "expect_pred":
                if tok == self.pred:
                    stage = "expect_pred_value"
                else:
                    return None, set()
            elif stage == "expect_pred_value":
                if tok in {self.sot, self.subj, self.pred, self.obj, self.eot}:
                    return None, set()
                stage = "expect_obj"
            elif stage == "expect_obj":
                if tok == self.obj:
                    stage = "expect_obj_value"
                else:
                    return None, set()
            elif stage == "expect_obj_value":
                if tok in {self.sot, self.subj, self.pred, self.obj, self.eot}:
                    return None, set()
                stage = "expect_eot"
            elif stage == "expect_eot":
                if tok == self.eot:
                    stage = "expect_sot_or_end"
                else:
                    return None, set()
            elif stage == "expect_sot_or_end":
                if tok == self.sot:
                    stage = "expect_subj"
                elif tok == self.eot:
                    stage = "expect_sot_or_end"
                elif pad_token is not None and tok == pad_token:
                    stage = "expect_sot_or_end"
                else:
                    return None, set()
            else:
                return None, set()

        if stage == "expect_sot":
            if self.sot_id is None:
                return None, set()
            return {self.sot_id}, set()
        if stage == "expect_subj":
            if self.subj_id is None:
                return None, set()
            return {self.subj_id}, set()
        if stage == "expect_subj_value":
            return None, self._special_ids
        if stage == "expect_pred":
            if self.pred_id is None:
                return None, set()
            return {self.pred_id}, set()
        if stage == "expect_pred_value":
            return None, self._special_ids
        if stage == "expect_obj":
            if self.obj_id is None:
                return None, set()
            return {self.obj_id}, set()
        if stage == "expect_obj_value":
            return None, self._special_ids
        if stage == "expect_eot":
            if self.eot_id is None:
                return None, set()
            return {self.eot_id}, set()
        if stage == "expect_sot_or_end":
            allowed: set[int] = set()
            for token_id in (self.sot_id, self.eot_id, self.pad_id):
                if token_id is not None:
                    allowed.add(int(token_id))
            return allowed, set()
        return None, set()


def _apply_repetition_penalty(
    logits: torch.Tensor, sequences: Sequence[torch.Tensor], penalty: float
) -> torch.Tensor:
    if penalty is None or penalty <= 1.0:
        return logits
    adjusted = logits.clone()
    for row_idx, seq in enumerate(sequences):
        row = adjusted[row_idx]
        unique_tokens = {int(token) for token in seq.tolist()}
        for token_id in unique_tokens:
            if not (0 <= token_id < row.size(-1)):
                continue
            value = row[token_id]
            if value < 0:
                row[token_id] = value * penalty
            else:
                row[token_id] = value / penalty
        adjusted[row_idx] = row
    return adjusted


def _apply_no_repeat_ngram(
    logits: torch.Tensor, sequences: Sequence[torch.Tensor], ngram_size: int
) -> torch.Tensor:
    if ngram_size is None or ngram_size <= 0:
        return logits
    adjusted = logits.clone()
    mask_value = torch.finfo(logits.dtype).min
    for row_idx, seq in enumerate(sequences):
        tokens = seq.tolist()[1:]
        if len(tokens) < ngram_size - 1 or ngram_size == 1:
            continue
        generated: dict[tuple[int, ...], set[int]] = {}
        for i in range(len(tokens) - ngram_size + 1):
            prefix = tuple(tokens[i : i + ngram_size - 1])
            next_token = tokens[i + ngram_size - 1]
            generated.setdefault(prefix, set()).add(int(next_token))
        current_prefix = tuple(tokens[-(ngram_size - 1) :])
        banned = generated.get(current_prefix, set())
        if not banned:
            continue
        row = adjusted[row_idx]
        row = row.clone()
        for token_id in banned:
            if 0 <= token_id < row.size(-1):
                row[token_id] = mask_value
        adjusted[row_idx] = row
    return adjusted


def _length_penalty(sequence_length: int, penalty: float) -> float:
    if penalty is None or penalty == 0:
        return 1.0
    length = max(sequence_length, 1)
    return ((5 + length) / 6) ** penalty


@torch.no_grad()
def beam_search_decode(
    model,
    tok,
    input_text: str,
    max_new_tokens: int = 128,
    device: str = "cpu",
    *,
    min_new_tokens: int = 1,
    debug: bool = False,
    forbidden_token_ids: Optional[Sequence[int]] = None,
    logits_processors: Optional[Sequence[Callable[[torch.LongTensor, torch.Tensor], torch.Tensor]]] = None,
    num_beams: int = 4,
    length_penalty: float = 1.0,
    early_stopping: bool = True,
    no_repeat_ngram_size: int = 3,
    repetition_penalty: float = 1.1,
    **unused_generation_kwargs,
):
    """Run a minimal beam-search decoder compatible with greedy decoding."""

    if num_beams <= 1:
        raise ValueError("beam_search_decode requires num_beams > 1")

    model.eval()
    pad_id = tok.pad_id
    eot_id = tok.token_to_id("<EOT>")

    inp = torch.tensor([tok.encode(input_text)], dtype=torch.long, device=device)
    att = inp != pad_id

    start_id = _select_start_token_id(tok, eot_id)
    start_token = torch.tensor([start_id], dtype=torch.long, device=device)

    processors: tuple[Callable[[torch.LongTensor, torch.Tensor], torch.Tensor], ...] = tuple(
        logits_processors or ()
    )

    extra_mask_ids: tuple[int, ...] = tuple(
        int(token_id)
        for token_id in (forbidden_token_ids or [])
        if token_id is not None
    )

    active_sequences: list[torch.Tensor] = [start_token]
    active_scores = torch.zeros(1, device=device)
    completed: list[tuple[float, torch.Tensor]] = []

    if unused_generation_kwargs:
        LOGGER.debug(
            "[beam] ignoring unsupported generation kwargs: %s",
            sorted(unused_generation_kwargs),
        )

    for step in range(max_new_tokens):
        if not active_sequences:
            break

        decoder_input = torch.stack(active_sequences)
        encoder_inp = inp.expand(decoder_input.size(0), -1)
        encoder_att = att.expand(decoder_input.size(0), -1)
        out = model(encoder_inp, encoder_att, decoder_input_ids=decoder_input)
        logits_step = out["logits"][:, -1, :]
        for processor in processors:
            logits_step = processor(decoder_input, logits_step)

        mask_ids: list[int] = []
        generated_len = decoder_input.size(1) - 1
        if eot_id is not None and generated_len < int(max(0, min_new_tokens)):
            mask_ids.append(int(eot_id))
        if start_id is not None and int(start_id) != int(eot_id or -1):
            mask_ids.append(int(start_id))
        if extra_mask_ids:
            mask_ids.extend(extra_mask_ids)
        if mask_ids:
            logits_step = logits_step.clone()
            min_val = torch.finfo(logits_step.dtype).min
            for token_id in {int(tok_id) for tok_id in mask_ids}:
                if 0 <= token_id < logits_step.size(-1):
                    logits_step[:, token_id] = min_val

        logits_step = _apply_repetition_penalty(
            logits_step, decoder_input, repetition_penalty
        )
        logits_step = _apply_no_repeat_ngram(
            logits_step, decoder_input, no_repeat_ngram_size
        )

        log_probs = torch.nn.functional.log_softmax(logits_step, dim=-1)
        vocab_size = log_probs.size(-1)

        next_scores = (log_probs + active_scores.unsqueeze(1)).view(-1)
        k = min(num_beams * 2, next_scores.size(0))
        top_scores, top_indices = torch.topk(next_scores, k=k)

        new_active_sequences: list[torch.Tensor] = []
        new_active_scores: list[float] = []

        for idx in range(top_indices.size(0)):
            beam_idx = int(top_indices[idx].item() // vocab_size)
            token_id = int(top_indices[idx].item() % vocab_size)
            prev_seq = active_sequences[beam_idx]
            new_seq = torch.cat(
                [prev_seq, torch.tensor([token_id], device=device, dtype=torch.long)],
                dim=0,
            )
            score = float(top_scores[idx].item())

            if eot_id is not None and token_id == int(eot_id):
                length = new_seq.size(0) - 1
                if length >= int(min_new_tokens):
                    adjusted = score / _length_penalty(length, length_penalty)
                    completed.append((adjusted, new_seq))
                    if debug:
                        LOGGER.debug(
                            "[beam] completed beam=%d score=%.4f", beam_idx, adjusted
                        )
                continue

            new_active_sequences.append(new_seq)
            new_active_scores.append(score)
            if len(new_active_sequences) == num_beams:
                break

        active_sequences = new_active_sequences
        if new_active_scores:
            active_scores = torch.tensor(new_active_scores, device=device)
        else:
            active_scores = torch.zeros(0, device=device)

        if early_stopping and len(completed) >= num_beams:
            break

    if not completed and active_sequences:
        for score, seq in zip(active_scores.tolist(), active_sequences):
            length = seq.size(0) - 1
            adjusted = float(score) / _length_penalty(length, length_penalty)
            completed.append((adjusted, seq))

    if not completed:
        return [int(token) for token in start_token.tolist()]

    best_score, best_sequence = max(completed, key=lambda item: item[0])
    if debug:
        LOGGER.debug("[beam] best score %.4f sequence=%s", best_score, best_sequence.tolist())
    return [int(token) for token in best_sequence.tolist()]


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
    forbidden_token_ids: Optional[Sequence[int]] = None,
    logits_processors: Optional[Sequence[Callable[[torch.LongTensor, torch.Tensor], torch.Tensor]]] = None,
    **unused_generation_kwargs,
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

    extra_mask_ids: tuple[int, ...] = tuple(
        int(token_id)
        for token_id in (forbidden_token_ids or [])
        if token_id is not None
    )

    if unused_generation_kwargs:
        LOGGER.debug(
            "[decode] ignoring unsupported generation kwargs: %s",
            sorted(unused_generation_kwargs),
        )

    processors: tuple[Callable[[torch.LongTensor, torch.Tensor], torch.Tensor], ...] = tuple(
        logits_processors or ()
    )

    for step in range(max_new_tokens):
        out = model(inp, att, decoder_input_ids=y)
        logits_step = out["logits"][:, -1, :]
        for processor in processors:
            logits_step = processor(y, logits_step)
        generated_len = y.size(1) - 1
        mask_ids = []
        if eot_id is not None and generated_len < int(max(0, min_new_tokens)):
            mask_ids.append(int(eot_id))
        if start_id is not None and int(start_id) != int(eot_id or -1):
            mask_ids.append(int(start_id))
        if extra_mask_ids:
            mask_ids.extend(extra_mask_ids)
        if mask_ids:
            logits_step = logits_step.clone()
            min_val = torch.finfo(logits_step.dtype).min
            for token_id in {int(tok_id) for tok_id in mask_ids}:
                if 0 <= token_id < logits_step.size(-1):
                    logits_step[..., token_id] = min_val
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
    max_new_tokens = int(kwargs.pop("max_new_tokens", 128))
    device = kwargs.pop("device", "cpu")
    forbidden_token_ids = kwargs.pop("forbidden_token_ids", None)
    extra_mask_ids: tuple[int, ...] = tuple(
        int(token_id)
        for token_id in (forbidden_token_ids or [])
        if token_id is not None
    )
    use_beam_search = bool(kwargs.pop("use_beam_search", False))
    beam_size = int(kwargs.pop("beam_size", kwargs.pop("num_beams", 4)))
    length_penalty = float(kwargs.pop("length_penalty", 1.0))
    early_stopping = bool(kwargs.pop("early_stopping", True))
    no_repeat_ngram_size = int(kwargs.pop("no_repeat_ngram_size", 3))
    repetition_penalty = float(kwargs.pop("repetition_penalty", 1.1))
    enforce_rdf_grammar = bool(kwargs.pop("enforce_rdf_grammar", False))
    logits_processors: list[
        Callable[[torch.LongTensor, torch.Tensor], torch.Tensor]
    ] = list(kwargs.pop("logits_processors", ()))

    if enforce_rdf_grammar:
        logits_processors.append(RDFGrammarLogitsProcessor(tok))

    if use_beam_search and beam_size > 1:
        ids = beam_search_decode(
            model,
            tok,
            input_text,
            max_new_tokens=max_new_tokens,
            device=device,
            min_new_tokens=min_new_tokens,
            debug=debug,
            forbidden_token_ids=extra_mask_ids,
            logits_processors=logits_processors,
            num_beams=beam_size,
            length_penalty=length_penalty,
            early_stopping=early_stopping,
            no_repeat_ngram_size=no_repeat_ngram_size,
            repetition_penalty=repetition_penalty,
            **kwargs,
        )
    else:
        ids = greedy_decode(
            model,
            tok,
            input_text,
            max_new_tokens=max_new_tokens,
            device=device,
            min_new_tokens=min_new_tokens,
            debug=debug,
            forbidden_token_ids=extra_mask_ids,
            logits_processors=logits_processors,
            **kwargs,
        )
    start_id = _select_start_token_id(tok)
    if ids and ids[0] == start_id:
        ids = ids[1:]
    # Keep special tokens in the decoded string so that structural markers
    # (e.g., <SUBJ>, <PRED>, <OBJ>, <EOT>) are preserved for non-textual tasks.
    # Textual tasks explicitly forbid structural tokens during decoding.
    try:
        text = tok.tk.decode(ids, skip_special_tokens=False)
    except TypeError:
        # Alcuni tokenizer di test espongono una API semplificata senza
        # l'argomento keyword; in tal caso richiamiamo la versione posizionale.
        text = tok.tk.decode(ids)
    if debug:
        LOGGER.debug("[decode] decoded text='%s'", text)
    if return_ids:
        return text, ids
    return text
