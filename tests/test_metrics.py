import pytest

from src.eval.metrics import (
    accuracy_score,
    compute_accuracy,
    compute_text_generation_metrics,
    compute_triple_metrics,
    corpus_bleu,
    meteor_score,
    rouge_l_score,
    triple_precision_recall_f1,
)


def test_bleu_perfect_match():
    pred = ["the cat is on the mat"]
    ref = ["the cat is on the mat"]
    assert corpus_bleu(pred, ref) == pytest.approx(100.0)


def test_bleu_partial_match():
    pred = ["the cat sat on the mat"]
    ref = ["the cat is on the mat"]
    score = corpus_bleu(pred, ref)
    assert 0.0 < score < 100.0


def test_rouge_l_score():
    pred = ["the cat sat on the mat"]
    ref = ["the cat is on the mat"]
    score = rouge_l_score(pred, ref)
    assert 0.0 < score < 100.0


def test_meteor_score():
    pred = ["The quick brown fox"]
    ref = ["the quick fox"]
    score = meteor_score(pred, ref)
    assert 0.0 < score <= 100.0


def test_triple_f1_from_strings():
    pred = ["<SOT> <SUBJ> a <PRED> b <OBJ> c <EOT> <SOT> <SUBJ> a <PRED> d <OBJ> e <EOT>"]
    ref = ["<SOT> <SUBJ> a <PRED> b <OBJ> c <EOT>"]
    metrics = triple_precision_recall_f1(pred, ref)
    assert pytest.approx(metrics["precision"], rel=1e-6) == 50.0
    assert pytest.approx(metrics["recall"], rel=1e-6) == 100.0
    assert metrics["f1"] > 0.0


def test_accuracy_score():
    preds = ["entity", "wrong"]
    refs = ["entity", "gold"]
    score = accuracy_score(preds, refs)
    assert score == pytest.approx(50.0)


def test_compute_helpers_return_expected_keys():
    preds = ["the cat is on the mat"]
    refs = ["the cat is on the mat"]
    text_metrics = compute_text_generation_metrics(preds, refs)
    assert set(text_metrics.keys()) == {"bleu", "rouge_l", "meteor"}

    triple_metrics = compute_triple_metrics(
        ["<SOT> <SUBJ> x <PRED> y <OBJ> z <EOT>"],
        ["<SOT> <SUBJ> x <PRED> y <OBJ> z <EOT>"],
    )
    assert triple_metrics["f1"] == pytest.approx(100.0)

    acc_metrics = compute_accuracy(["foo"], ["foo"])
    assert acc_metrics == {"accuracy": pytest.approx(100.0)}
