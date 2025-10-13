import pathlib
import sys

PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.pairing import pair_and_filter


def test_pair_and_filter_normalises_incoming_triples() -> None:
    triples_stream = [
        {"film": "film_1", "p": "hasDirector", "o": "Person:A", "dir": "out"},
        {"film": "film_1", "p": "starring", "o": "Person:B", "dir": "in"},
    ]
    texts_stream = [
        {"film": "film_1", "text": "Film intro."},
    ]

    paired = list(pair_and_filter(triples_stream, texts_stream, min_triples=1))

    assert len(paired) == 1
    record = paired[0]
    assert record["film"] == "film_1"
    assert record["triples"] == [
        ("film_1", "hasDirector", "Person:A"),
        ("film_1", "starring", "Person:B"),
    ]
