from src.eval import evaluate


def test_summarise_lengths_empty_sequence():
    stats = evaluate._summarise_lengths([])
    assert stats == {"count": 0, "min": 0, "max": 0, "mean": 0.0, "median": 0.0, "zeros": 0}


def test_summarise_lengths_basic_statistics():
    stats = evaluate._summarise_lengths([0, 2, 4, 4])
    assert stats["count"] == 4
    assert stats["zeros"] == 1
    assert stats["min"] == 0
    assert stats["max"] == 4
    assert stats["mean"] == 2.5
    assert stats["median"] == 3.0
