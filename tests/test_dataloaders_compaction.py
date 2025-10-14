from src.training import dataloaders


def test_compact_rdf_input_groups_repeated_objects():
    original = (
        "<SOT> <SUBJ> dbr:Film <PRED> dbo:genre <OBJ> dbr:Doc <EOT> "
        "<SOT> <SUBJ> dbr:Film <PRED> dbo:genre <OBJ> dbr:Drama <EOT> <RDF2Text>"
    )
    compacted = dataloaders.compact_rdf(original)
    assert compacted.count("<SOT>") == 1
    assert "<OBJ_LIST>" in compacted
    assert "dbr:Doc | dbr:Drama" in compacted
    assert compacted.strip().endswith("<RDF2Text>")


def test_compact_rdf_input_preserves_singletons():
    original = "<SOT> <SUBJ> dbr:Film <PRED> dbo:runtime <OBJ> 5400.0 <EOT> <RDF2Text>"
    compacted = dataloaders.compact_rdf(original)
    assert "<OBJ_LIST>" not in compacted
    assert compacted == original


def test_compact_rdf_input_keeps_mask_token():
    original = (
        "<SOT> <SUBJ> dbr:Film <PRED> dbo:starring <OBJ> dbr:Actor <EOT> "
        "<SOT> <SUBJ> dbr:Film <PRED> dbo:starring <OBJ> <MASK> <EOT> <MASK>"
    )
    compacted = dataloaders.compact_rdf(original)
    assert "<OBJ_LIST>" in compacted
    assert "dbr:Actor | <MASK>" in compacted
    assert compacted.strip().endswith("<MASK>")
