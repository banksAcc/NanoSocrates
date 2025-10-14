from src import run


def test_prepare_predict_input_adds_marker():
    result = run._prepare_predict_input("Hello", "text2rdf")
    assert result.endswith(run.TASK_MARKERS["text2rdf"])


def test_prepare_predict_input_keeps_existing_marker():
    marker = run.TASK_MARKERS["rdf2text"]
    text = f"Sample {marker}"
    result = run._prepare_predict_input(text, "rdf2text")
    assert result == text


def test_prepare_predict_input_adds_mask_for_rdfcomp1():
    result = run._prepare_predict_input("Subject predicate object", "rdfcomp1")
    assert result.endswith("<MASK>")


def test_prepare_predict_input_handles_none_task():
    result = run._prepare_predict_input("  Hello world  ", None)
    assert result == "Hello world"
