import pathlib
import sys

from tokenizers import Tokenizer

PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.training.datamodule import JSONLSeq2SeqDataModule


def test_jsonl_datamodule_compacts_and_tokenizes():
    tokenizer = Tokenizer.from_file(str(PROJECT_ROOT / "data" / "vocab" / "bpe.json"))
    data_files = {
        "train": str(PROJECT_ROOT / "tests" / "fixtures" / "toy_data_train.jsonl"),
        "validation": str(PROJECT_ROOT / "tests" / "fixtures" / "toy_data_val.jsonl"),
    }

    module = JSONLSeq2SeqDataModule(
        tokenizer=tokenizer,
        data_files=data_files,
        batch_size=2,
        max_input_length=64,
        max_target_length=32,
        shuffle_train=False,
    )
    module.setup()

    assert module.train_dataset is not None
    assert module.val_dataset is not None
    assert len(module.train_dataset) == 2
    assert len(module.val_dataset) == 1

    batch = next(iter(module.train_dataloader()))
    assert batch["input_ids"].shape[0] == 2
    assert any("<OBJ_LIST>" in text for text in batch["raw_input"])
    assert all(task == "rdf2text" for task in batch["tasks"])
    assert batch["labels"].shape[1] <= 32
