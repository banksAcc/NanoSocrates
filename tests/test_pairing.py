import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

from data.pairing import pair_and_filter


def test_pair_and_filter_normalizes_incoming_triples():
    triples_stream = [
        {"film": "Film A", "p": "hasGenre", "o": "Drama", "dir": "out"},
        {"film": "Film A", "p": "won", "o": "Festival", "dir": "in"},
    ]
    texts_stream = [{"film": "Film A", "text": "Some intro."}]

    result = list(pair_and_filter(triples_stream, texts_stream, min_triples=1))

    assert result[0]["triples"][1] == ("Festival", "won", "Film A")