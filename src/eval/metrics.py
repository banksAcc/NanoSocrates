"""Utility per il calcolo delle metriche di valutazione."""
from __future__ import annotations

import math
from collections import Counter
from typing import Iterable, List, Sequence, Tuple

from src.data.serialization import parse_rdf

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _tokenize(text: str) -> List[str]:
    if text is None:
        return []
    return [tok for tok in text.strip().split() if tok]


def _ngram_counts(tokens: Sequence[str], n: int) -> Counter:
    if n <= 0:
        return Counter()
    return Counter(tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1))


# ---------------------------------------------------------------------------
# Testo ↔ testo: BLEU / ROUGE-L / METEOR
# ---------------------------------------------------------------------------


def corpus_bleu(
    predictions: Sequence[str],
    references: Sequence[Sequence[str] | str],
    max_order: int = 4,
    smooth: float = 1e-9,
) -> float:
    """Calcola il BLEU corpus-level (0–100)."""

    if len(predictions) != len(references):
        raise ValueError("predictions e references devono avere la stessa lunghezza")
    matches_by_order = [0] * max_order
    possible_matches_by_order = [0] * max_order
    ref_length = 0
    pred_length = 0

    for pred, ref in zip(predictions, references):
        pred_tokens = _tokenize(pred)
        ref_list = [ref] if isinstance(ref, str) else list(ref)
        ref_tokens_list = [_tokenize(r) for r in ref_list]
        pred_length += len(pred_tokens)

        if ref_tokens_list:
            ref_lengths = [len(rt) for rt in ref_tokens_list]
            closest_ref_len = min(
                ref_lengths,
                key=lambda rl: (abs(rl - len(pred_tokens)), rl),
            )
            ref_length += closest_ref_len
        else:
            ref_length += 0

        for n in range(1, max_order + 1):
            pred_ngrams = _ngram_counts(pred_tokens, n)
            max_ref_counts: Counter = Counter()
            for ref_tokens in ref_tokens_list:
                ref_ngrams = _ngram_counts(ref_tokens, n)
                for ngram, count in ref_ngrams.items():
                    if count > max_ref_counts[ngram]:
                        max_ref_counts[ngram] = count
            overlap = {
                ngram: min(count, max_ref_counts.get(ngram, 0))
                for ngram, count in pred_ngrams.items()
            }
            matches_by_order[n - 1] += sum(overlap.values())
            possible_matches_by_order[n - 1] += max(0, len(pred_tokens) - n + 1)

    if pred_length == 0:
        return 0.0

    precisions: List[float] = []
    for i in range(max_order):
        if possible_matches_by_order[i] == 0:
            precisions.append(0.0)
        else:
            precisions.append(
                (matches_by_order[i] + smooth) / (possible_matches_by_order[i] + smooth)
            )

    if min(precisions) <= 0:
        geo_mean = 0.0
    else:
        geo_mean = math.exp(sum(math.log(p) for p in precisions) / max_order)

    if pred_length < ref_length and pred_length > 0:
        bp = math.exp(1 - ref_length / pred_length)
    else:
        bp = 1.0

    bleu = geo_mean * bp
    return float(bleu * 100)


def rouge_l_score(predictions: Sequence[str], references: Sequence[str]) -> float:
    """ROUGE-L medio (F1 con beta=1.2) restituito in percentuale."""

    if len(predictions) != len(references):
        raise ValueError("predictions e references devono avere la stessa lunghezza")

    def lcs(a: Sequence[str], b: Sequence[str]) -> int:
        m, n = len(a), len(b)
        if m == 0 or n == 0:
            return 0
        dp = [[0] * (n + 1) for _ in range(m + 1)]
        for i in range(m):
            for j in range(n):
                if a[i] == b[j]:
                    dp[i + 1][j + 1] = dp[i][j] + 1
                else:
                    dp[i + 1][j + 1] = max(dp[i][j + 1], dp[i + 1][j])
        return dp[m][n]

    scores: List[float] = []
    beta = 1.2
    beta_sq = beta ** 2

    for pred, ref in zip(predictions, references):
        pred_tokens = _tokenize(pred)
        ref_tokens = _tokenize(ref)
        if not pred_tokens or not ref_tokens:
            scores.append(0.0)
            continue
        lcs_len = lcs(pred_tokens, ref_tokens)
        prec = lcs_len / len(pred_tokens)
        rec = lcs_len / len(ref_tokens)
        if prec == 0 or rec == 0:
            scores.append(0.0)
        else:
            f_score = ((1 + beta_sq) * prec * rec) / (rec + beta_sq * prec)
            scores.append(f_score)
    if not scores:
        return 0.0
    return float(sum(scores) / len(scores) * 100)


def meteor_score(predictions: Sequence[str], references: Sequence[str]) -> float:
    """Implementazione semplificata di METEOR (precisione/recall unigrammi)."""

    if len(predictions) != len(references):
        raise ValueError("predictions e references devono avere la stessa lunghezza")

    alpha = 0.9
    beta = 3.0
    gamma = 0.5

    scores: List[float] = []
    for pred, ref in zip(predictions, references):
        hyp = _tokenize(pred.lower())
        ref_tokens = _tokenize(ref.lower())
        if not hyp or not ref_tokens:
            scores.append(0.0)
            continue

        matched_ref_idx: List[int] = []
        matched_mask = [False] * len(ref_tokens)
        matches = 0
        for tok in hyp:
            found = False
            for idx, ref_tok in enumerate(ref_tokens):
                if not matched_mask[idx] and ref_tok == tok:
                    matched_mask[idx] = True
                    matched_ref_idx.append(idx)
                    matches += 1
                    found = True
                    break
            if not found:
                continue

        if matches == 0:
            scores.append(0.0)
            continue

        matched_ref_idx.sort()
        chunks = 1
        for i in range(1, len(matched_ref_idx)):
            if matched_ref_idx[i] != matched_ref_idx[i - 1] + 1:
                chunks += 1

        precision = matches / len(hyp)
        recall = matches / len(ref_tokens)
        f_mean = (precision * recall) / (
            (alpha * precision) + ((1 - alpha) * recall)
        )
        frag = chunks / matches
        penalty = gamma * (frag ** beta)
        score = (1 - penalty) * f_mean
        scores.append(score)

    if not scores:
        return 0.0
    return float(sum(scores) / len(scores) * 100)


# ---------------------------------------------------------------------------
# Triple RDF e accuracy
# ---------------------------------------------------------------------------


def _coerce_triples(obj: Iterable | str) -> List[Tuple[str, str, str]]:
    if isinstance(obj, str):
        return parse_rdf(obj)
    triples: List[Tuple[str, str, str]] = []
    for item in obj or []:
        if isinstance(item, (list, tuple)) and len(item) == 3:
            triples.append(tuple(map(str, item)))
    return triples


def triple_precision_recall_f1(
    predictions: Sequence[Iterable | str], references: Sequence[Iterable | str]
) -> dict:
    if len(predictions) != len(references):
        raise ValueError("predictions e references devono avere la stessa lunghezza")

    tp = fp = fn = 0
    for pred, ref in zip(predictions, references):
        pred_set = set(_coerce_triples(pred))
        ref_set = set(_coerce_triples(ref))
        inter = pred_set & ref_set
        tp += len(inter)
        fp += len(pred_set - inter)
        fn += len(ref_set - inter)

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    if precision + recall == 0:
        f1 = 0.0
    else:
        f1 = 2 * precision * recall / (precision + recall)
    return {
        "precision": precision * 100,
        "recall": recall * 100,
        "f1": f1 * 100,
    }


def accuracy_score(predictions: Sequence[str], references: Sequence[str]) -> float:
    if len(predictions) != len(references):
        raise ValueError("predictions e references devono avere la stessa lunghezza")
    total = len(predictions)
    if total == 0:
        return 0.0
    correct = 0
    for pred, ref in zip(predictions, references):
        if (pred or "").strip() == (ref or "").strip():
            correct += 1
    return float(correct / total * 100)


# ---------------------------------------------------------------------------
# Interfacce comode per blocchi di valutazione
# ---------------------------------------------------------------------------


def compute_text_generation_metrics(
    predictions: Sequence[str], references: Sequence[str]
) -> dict:
    return {
        "bleu": corpus_bleu(predictions, references),
        "rouge_l": rouge_l_score(predictions, references),
        "meteor": meteor_score(predictions, references),
    }


def compute_triple_metrics(
    predictions: Sequence[Iterable | str], references: Sequence[Iterable | str]
) -> dict:
    return triple_precision_recall_f1(predictions, references)


def compute_accuracy(predictions: Sequence[str], references: Sequence[str]) -> dict:
    return {"accuracy": accuracy_score(predictions, references)}
