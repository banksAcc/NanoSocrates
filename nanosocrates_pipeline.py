"""Unified NanoSocrates pipeline.

This module condenses the previous multi-file project into a single, self-contained
script that covers the full workflow:

1. Load the four multitask datasets (Text2RDF, RDF2Text, RDF Completion 1, RDF Completion 2).
2. Train or reuse a Byte-Pair Encoding (BPE) tokenizer with the project-specific
   special tokens.
3. Train a lightweight Transformer encoder–decoder model implemented with PyTorch.
4. Evaluate the model computing an exact-match metric across tasks.
5. Provide utilities for sanity-check/overfit experiments and free-form generation.

The goal is to keep everything simple and transparent without reinventing building
blocks that are readily available in the Python ecosystem.  The script only requires
libraries already listed in ``requirements.txt``.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import torch
from torch import Tensor
from torch import nn
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset

from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.pre_tokenizers import Sequence as PreTokenizerSequence
from tokenizers.pre_tokenizers import Whitespace, Punctuation
from tokenizers.trainers import BpeTrainer

# ---------------------------------------------------------------------------
# Configuration and constants
# ---------------------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parent
DATA_DIR = REPO_ROOT / "data"
PROCESSED_DIR = DATA_DIR / "processed"
VOCAB_DIR = DATA_DIR / "vocab"
CHECKPOINTS_DIR = REPO_ROOT / "checkpoints"
DEFAULT_TOKENIZER_PATH = VOCAB_DIR / "bpe.json"
DEFAULT_MODEL_PATH = CHECKPOINTS_DIR / "nanosocrates_transformer.pt"

SPECIAL_TOKENS: Sequence[str] = (
    "<SOT>",
    "<EOT>",
    "<SUBJ>",
    "<PRED>",
    "<OBJ>",
    "<OBJ_LIST>",
    "|",
    "<RDF2Text>",
    "<Text2RDF>",
    "<CONTINUERDF>",
    "<MASK>",
    "dbr:",
    "dbo:",
)

TASKS = ("text2rdf", "rdf2text", "rdfcomp1", "rdfcomp2")
SPLITS = ("train", "val", "test")


# ---------------------------------------------------------------------------
# Utility dataclasses
# ---------------------------------------------------------------------------

@dataclass
class TokenizerConfig:
    vocab_size: int = 16000
    min_frequency: int = 2
    special_tokens: Sequence[str] = SPECIAL_TOKENS
    save_path: Path = DEFAULT_TOKENIZER_PATH


@dataclass
class TrainingConfig:
    epochs: int = 10
    batch_size: int = 16
    lr: float = 3e-4
    d_model: int = 256
    nhead: int = 4
    num_encoder_layers: int = 4
    num_decoder_layers: int = 4
    dim_feedforward: int = 512
    dropout: float = 0.1
    max_seq_len: int = 512
    gradient_clip: float = 1.0
    device: Optional[str] = None
    save_path: Path = DEFAULT_MODEL_PATH


@dataclass
class OverfitConfig:
    steps: int = 200
    batch_size: int = 8
    lr: float = 5e-4
    max_seq_len: int = 256


# ---------------------------------------------------------------------------
# Data loading helpers
# ---------------------------------------------------------------------------

class MultitaskExample(Dataset):
    """Torch Dataset that reads the JSONL files lazily and encodes them on demand."""

    def __init__(
        self,
        samples: Sequence[Tuple[str, str]],
        tokenizer: Tokenizer,
        max_length: int,
        add_boundary_tokens: bool = True,
    ) -> None:
        self.samples = list(samples)
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.add_boundary_tokens = add_boundary_tokens
        pad_token = "<EOT>"
        pad_id = tokenizer.token_to_id(pad_token)
        if pad_id is None:
            raise ValueError(
                "The tokenizer is expected to contain the <EOT> token which we use for padding."
            )
        self.pad_id = pad_id

    def __len__(self) -> int:
        return len(self.samples)

    def _prepare_text(self, text: str, *, is_target: bool) -> str:
        cleaned = text.strip()
        if not self.add_boundary_tokens:
            return cleaned
        has_sot = cleaned.startswith("<SOT>")
        has_eot = cleaned.endswith("<EOT>")
        if is_target:
            if not has_sot:
                cleaned = f"<SOT> {cleaned}"
            if not has_eot:
                cleaned = f"{cleaned} <EOT>"
        return cleaned

    def __getitem__(self, idx: int) -> Tuple[Tensor, Tensor]:
        source, target = self.samples[idx]
        source_text = self._prepare_text(source, is_target=False)
        target_text = self._prepare_text(target, is_target=True)
        source_ids = self.tokenizer.encode(source_text).ids[: self.max_length]
        target_ids = self.tokenizer.encode(target_text).ids[: self.max_length]
        if len(target_ids) < 2:
            target_ids = target_ids + [self.pad_id]
        source_tensor = torch.tensor(source_ids, dtype=torch.long)
        target_tensor = torch.tensor(target_ids, dtype=torch.long)
        return source_tensor, target_tensor

    def collate(self, batch: Sequence[Tuple[Tensor, Tensor]]):
        src_batch, tgt_batch = zip(*batch)
        src_padded = pad_sequence(src_batch, batch_first=True, padding_value=self.pad_id)
        tgt_padded = pad_sequence(tgt_batch, batch_first=True, padding_value=self.pad_id)
        tgt_input = tgt_padded[:, :-1]
        tgt_output = tgt_padded[:, 1:]
        return {
            "src": src_padded,
            "tgt_in": tgt_input,
            "tgt_out": tgt_output,
            "src_padding_mask": src_padded.eq(self.pad_id),
            "tgt_padding_mask": tgt_input.eq(self.pad_id),
        }


def load_samples(task: str, split: str) -> List[Tuple[str, str]]:
    path = PROCESSED_DIR / f"{task}.{split}.jsonl"
    if not path.exists():
        raise FileNotFoundError(f"Missing dataset file: {path}")
    records: List[Tuple[str, str]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            payload = json.loads(line)
            source = payload["input"]
            target = payload["target"]
            records.append((source, target))
    return records


def load_all_samples(split: str) -> List[Tuple[str, str]]:
    combined: List[Tuple[str, str]] = []
    for task in TASKS:
        combined.extend(load_samples(task, split))
    random.shuffle(combined)
    return combined


# ---------------------------------------------------------------------------
# Tokenizer helpers
# ---------------------------------------------------------------------------

def train_tokenizer(cfg: TokenizerConfig) -> Tokenizer:
    VOCAB_DIR.mkdir(parents=True, exist_ok=True)
    files: List[str] = []
    for task in TASKS:
        for split in ("train",):
            path = PROCESSED_DIR / f"{task}.{split}.jsonl"
            if path.exists():
                files.append(str(path))
    if not files:
        raise RuntimeError("No training data found for tokenizer.")

    tokenizer = Tokenizer(BPE(unk_token="<MASK>"))
    tokenizer.pre_tokenizer = PreTokenizerSequence([Whitespace(), Punctuation()])
    trainer = BpeTrainer(
        vocab_size=cfg.vocab_size,
        min_frequency=cfg.min_frequency,
        special_tokens=list(cfg.special_tokens),
    )
    tokenizer.train(files, trainer=trainer)
    tokenizer.enable_padding(pad_id=tokenizer.token_to_id("<EOT>"), pad_token="<EOT>")
    cfg.save_path.parent.mkdir(parents=True, exist_ok=True)
    tokenizer.save(str(cfg.save_path))
    print(f"Tokenizer saved to {cfg.save_path}")
    return tokenizer


def load_tokenizer(path: Path = DEFAULT_TOKENIZER_PATH) -> Tokenizer:
    if not path.exists():
        raise FileNotFoundError(
            f"Tokenizer file not found at {path}. Run the pipeline with --retrain-tokenizer first."
        )
    tokenizer = Tokenizer.from_file(str(path))
    tokenizer.enable_padding(pad_id=tokenizer.token_to_id("<EOT>"), pad_token="<EOT>")
    return tokenizer


# ---------------------------------------------------------------------------
# Transformer model definition
# ---------------------------------------------------------------------------

class Seq2SeqTransformer(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        cfg: TrainingConfig,
        pad_id: int,
    ) -> None:
        super().__init__()
        self.model_type = "Transformer"
        self.src_tok_emb = nn.Embedding(vocab_size, cfg.d_model)
        self.tgt_tok_emb = nn.Embedding(vocab_size, cfg.d_model)
        self.pos_encoder = PositionalEncoding(cfg.d_model, cfg.dropout)
        self.pos_decoder = PositionalEncoding(cfg.d_model, cfg.dropout)
        self.transformer = nn.Transformer(
            d_model=cfg.d_model,
            nhead=cfg.nhead,
            num_encoder_layers=cfg.num_encoder_layers,
            num_decoder_layers=cfg.num_decoder_layers,
            dim_feedforward=cfg.dim_feedforward,
            dropout=cfg.dropout,
            batch_first=True,
        )
        self.generator = nn.Linear(cfg.d_model, vocab_size)
        self.pad_id = pad_id
        self.cfg = cfg
        self._reset_parameters()

    def _reset_parameters(self) -> None:
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(
        self,
        src: Tensor,
        tgt_in: Tensor,
        src_padding_mask: Tensor,
        tgt_padding_mask: Tensor,
    ) -> Tensor:
        src_emb = self.pos_encoder(self.src_tok_emb(src) * math.sqrt(self.cfg.d_model))
        tgt_emb = self.pos_decoder(self.tgt_tok_emb(tgt_in) * math.sqrt(self.cfg.d_model))
        tgt_mask = nn.Transformer.generate_square_subsequent_mask(tgt_in.size(1)).to(tgt_in.device)
        memory = self.transformer(
            src_emb,
            tgt_emb,
            tgt_mask=tgt_mask,
            src_key_padding_mask=src_padding_mask,
            tgt_key_padding_mask=tgt_padding_mask,
            memory_key_padding_mask=src_padding_mask,
        )
        logits = self.generator(memory)
        return logits

    @torch.no_grad()
    def greedy_decode(
        self,
        src: Tensor,
        src_padding_mask: Tensor,
        max_len: int,
        start_token_id: int,
    ) -> Tensor:
        self.eval()
        memory = self.encode(src, src_padding_mask)
        ys = torch.full((src.size(0), 1), start_token_id, dtype=torch.long, device=src.device)
        for _ in range(max_len - 1):
            tgt_mask = nn.Transformer.generate_square_subsequent_mask(ys.size(1)).to(src.device)
            tgt_emb = self.pos_decoder(self.tgt_tok_emb(ys) * math.sqrt(self.cfg.d_model))
            out = self.transformer.decoder(
                tgt_emb,
                memory,
                tgt_mask=tgt_mask,
                tgt_key_padding_mask=ys.eq(self.pad_id),
                memory_key_padding_mask=src_padding_mask,
            )
            prob = self.generator(out[:, -1])
            next_word = prob.argmax(dim=-1, keepdim=True)
            ys = torch.cat([ys, next_word], dim=1)
        return ys

    def encode(self, src: Tensor, src_padding_mask: Tensor) -> Tensor:
        src_emb = self.pos_encoder(self.src_tok_emb(src) * math.sqrt(self.cfg.d_model))
        return self.transformer.encoder(src_emb, src_key_padding_mask=src_padding_mask)


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000) -> None:
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x: Tensor) -> Tensor:
        x = x + self.pe[:, : x.size(1)]
        return self.dropout(x)


# ---------------------------------------------------------------------------
# Training, evaluation, overfit routines
# ---------------------------------------------------------------------------

def prepare_dataloaders(
    tokenizer: Tokenizer, cfg: TrainingConfig
) -> Tuple[DataLoader, DataLoader, MultitaskExample]:
    train_samples = load_all_samples("train")
    val_samples = load_all_samples("val")
    dataset = MultitaskExample(train_samples, tokenizer, cfg.max_seq_len)
    val_dataset = MultitaskExample(val_samples, tokenizer, cfg.max_seq_len)
    train_loader = DataLoader(
        dataset,
        batch_size=cfg.batch_size,
        shuffle=True,
        collate_fn=dataset.collate,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=cfg.batch_size,
        shuffle=False,
        collate_fn=val_dataset.collate,
    )
    return train_loader, val_loader, dataset


def build_model(tokenizer: Tokenizer, cfg: TrainingConfig) -> Seq2SeqTransformer:
    vocab_size = tokenizer.get_vocab_size()
    pad_id = tokenizer.token_to_id("<EOT>")
    if pad_id is None:
        raise RuntimeError("Tokenizer must provide <EOT> token for padding.")
    model = Seq2SeqTransformer(vocab_size, cfg, pad_id)
    return model


def run_training(
    tokenizer: Tokenizer,
    cfg: TrainingConfig,
    *,
    resume: bool = False,
) -> Seq2SeqTransformer:
    CHECKPOINTS_DIR.mkdir(parents=True, exist_ok=True)
    train_loader, val_loader, dataset = prepare_dataloaders(tokenizer, cfg)
    device = torch.device(cfg.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    model = build_model(tokenizer, cfg).to(device)
    criterion = nn.CrossEntropyLoss(ignore_index=dataset.pad_id)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr)

    start_epoch = 1
    best_val = float("inf")
    if resume and cfg.save_path.exists():
        state = torch.load(cfg.save_path, map_location=device)
        model.load_state_dict(state["model_state"])
        optimizer.load_state_dict(state["optim_state"])
        start_epoch = state.get("epoch", 1)
        best_val = state.get("best_val", float("inf"))
        print(f"Resumed training from {cfg.save_path} at epoch {start_epoch}")

    for epoch in range(start_epoch, cfg.epochs + 1):
        model.train()
        total_loss = 0.0
        for step, batch in enumerate(train_loader, start=1):
            src = batch["src"].to(device)
            tgt_in = batch["tgt_in"].to(device)
            tgt_out = batch["tgt_out"].to(device)
            src_mask = batch["src_padding_mask"].to(device)
            tgt_mask = batch["tgt_padding_mask"].to(device)

            optimizer.zero_grad()
            logits = model(src, tgt_in, src_mask, tgt_mask)
            loss = criterion(logits.reshape(-1, logits.size(-1)), tgt_out.reshape(-1))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.gradient_clip)
            optimizer.step()
            total_loss += loss.item()

            if step % 50 == 0:
                avg_loss = total_loss / step
                print(f"Epoch {epoch} Step {step} | Loss {avg_loss:.4f}")

        val_loss = evaluate_loss(model, val_loader, criterion, device)
        print(f"Epoch {epoch} completed. Validation loss: {val_loss:.4f}")

        if val_loss < best_val:
            best_val = val_loss
            save_checkpoint(cfg.save_path, model, optimizer, epoch, best_val)
            print(f"New best model saved to {cfg.save_path}")

    return model


def evaluate_loss(
    model: Seq2SeqTransformer,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> float:
    model.eval()
    total_loss = 0.0
    total_items = 0
    with torch.no_grad():
        for batch in dataloader:
            src = batch["src"].to(device)
            tgt_in = batch["tgt_in"].to(device)
            tgt_out = batch["tgt_out"].to(device)
            src_mask = batch["src_padding_mask"].to(device)
            tgt_mask = batch["tgt_padding_mask"].to(device)

            logits = model(src, tgt_in, src_mask, tgt_mask)
            loss = criterion(logits.reshape(-1, logits.size(-1)), tgt_out.reshape(-1))
            batch_items = tgt_out.numel()
            total_loss += loss.item() * batch_items
            total_items += batch_items
    return total_loss / max(total_items, 1)


def exact_match(
    model: Seq2SeqTransformer,
    dataloader: DataLoader,
    tokenizer: Tokenizer,
    device: torch.device,
    max_len: int,
) -> float:
    model.eval()
    start_token = tokenizer.token_to_id("<SOT>")
    if start_token is None:
        raise RuntimeError("Tokenizer must include the <SOT> token for generation.")
    matches = 0
    total = 0
    with torch.no_grad():
        for batch in dataloader:
            src = batch["src"].to(device)
            src_mask = batch["src_padding_mask"].to(device)
            tgt_out = batch["tgt_out"].to(device)
            predictions = model.greedy_decode(src, src_mask, max_len, start_token)
            pred_ids = predictions[:, 1:]  # drop the initial <SOT>
            for pred, target in zip(pred_ids, tgt_out):
                pred_text = detokenize_sequence(pred, tokenizer)
                target_text = detokenize_sequence(target, tokenizer)
                if pred_text == target_text:
                    matches += 1
                total += 1
    if total == 0:
        return 0.0
    return matches / total


def detokenize_sequence(ids: Tensor, tokenizer: Tokenizer) -> str:
    filtered = [i.item() for i in ids if i.item() != tokenizer.token_to_id("<EOT>")]
    text = tokenizer.decode(filtered, skip_special_tokens=False)
    text = text.replace("<SOT>", "").strip()
    if text.endswith("<EOT>"):
        text = text[: -len("<EOT>")].strip()
    return text


def save_checkpoint(
    path: Path,
    model: Seq2SeqTransformer,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    best_val: float,
) -> None:
    CHECKPOINTS_DIR.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state": model.state_dict(),
            "optim_state": optimizer.state_dict(),
            "epoch": epoch,
            "best_val": best_val,
        },
        path,
    )


def load_model(tokenizer: Tokenizer, cfg: TrainingConfig) -> Seq2SeqTransformer:
    device = torch.device(cfg.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    model = build_model(tokenizer, cfg).to(device)
    if not cfg.save_path.exists():
        raise FileNotFoundError(f"Checkpoint not found at {cfg.save_path}")
    state = torch.load(cfg.save_path, map_location=device)
    model.load_state_dict(state["model_state"])
    model.eval()
    return model


# ---------------------------------------------------------------------------
# High level orchestration
# ---------------------------------------------------------------------------

def run_full_pipeline(args: argparse.Namespace) -> None:
    tokenizer = load_or_train_tokenizer(force_retrain=args.retrain_tokenizer)
    cfg = TrainingConfig(
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        d_model=args.d_model,
        nhead=args.nhead,
        num_encoder_layers=args.encoder_layers,
        num_decoder_layers=args.decoder_layers,
        dim_feedforward=args.ff_dim,
        dropout=args.dropout,
        max_seq_len=args.max_seq_len,
        device=args.device,
    )
    model = run_training(tokenizer, cfg, resume=args.resume)
    evaluate_and_report(model, tokenizer, cfg)
    showcase_generation(model, tokenizer, cfg)


def evaluate_and_report(
    model: Seq2SeqTransformer,
    tokenizer: Tokenizer,
    cfg: TrainingConfig,
) -> None:
    device = next(model.parameters()).device
    _, val_loader, _ = prepare_dataloaders(tokenizer, cfg)
    em_score = exact_match(model, val_loader, tokenizer, device, cfg.max_seq_len)
    print(f"Exact match on validation set: {em_score * 100:.2f}%")


def showcase_generation(
    model: Seq2SeqTransformer,
    tokenizer: Tokenizer,
    cfg: TrainingConfig,
    num_examples: int = 3,
) -> None:
    device = next(model.parameters()).device
    samples = load_all_samples("test")
    random.shuffle(samples)
    start_token = tokenizer.token_to_id("<SOT>")
    if start_token is None:
        print("Tokenizer missing <SOT>; skipping showcase generation.")
        return
    print("\nSample generations:")
    for source, reference in samples[:num_examples]:
        batch = MultitaskExample([(source, reference)], tokenizer, cfg.max_seq_len)
        data = batch.collate([batch[0]])
        src = data["src"].to(device)
        src_mask = data["src_padding_mask"].to(device)
        generated = model.greedy_decode(src, src_mask, cfg.max_seq_len, start_token)
        decoded = detokenize_sequence(generated[0, 1:], tokenizer)
        print("---")
        print(f"Input: {source[:200]}...")
        print(f"Reference: {reference}")
        print(f"Generated: {decoded}")


def run_evaluation_only(args: argparse.Namespace) -> None:
    tokenizer = load_tokenizer()
    cfg = TrainingConfig(
        batch_size=args.batch_size,
        max_seq_len=args.max_seq_len,
        device=args.device,
        save_path=Path(args.checkpoint),
    )
    model = load_model(tokenizer, cfg)
    evaluate_and_report(model, tokenizer, cfg)
    showcase_generation(model, tokenizer, cfg)


def run_overfit(args: argparse.Namespace) -> None:
    tokenizer = load_or_train_tokenizer(force_retrain=args.retrain_tokenizer)
    overfit_cfg = OverfitConfig(
        steps=args.steps,
        batch_size=args.batch_size,
        lr=args.lr,
        max_seq_len=args.max_seq_len,
    )
    dataset = MultitaskExample(load_all_samples("train"), tokenizer, overfit_cfg.max_seq_len)
    subset = torch.utils.data.Subset(dataset, range(overfit_cfg.batch_size))
    loader = DataLoader(subset, batch_size=overfit_cfg.batch_size, shuffle=True, collate_fn=dataset.collate)
    training_cfg = TrainingConfig(
        epochs=1,
        batch_size=overfit_cfg.batch_size,
        lr=overfit_cfg.lr,
        max_seq_len=overfit_cfg.max_seq_len,
    )
    model = build_model(tokenizer, training_cfg)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    criterion = nn.CrossEntropyLoss(ignore_index=dataset.pad_id)
    optimizer = torch.optim.AdamW(model.parameters(), lr=overfit_cfg.lr)
    model.train()
    step = 0
    print("Starting overfit sanity check...")
    while step < overfit_cfg.steps:
        for batch in loader:
            step += 1
            src = batch["src"].to(device)
            tgt_in = batch["tgt_in"].to(device)
            tgt_out = batch["tgt_out"].to(device)
            src_mask = batch["src_padding_mask"].to(device)
            tgt_mask = batch["tgt_padding_mask"].to(device)
            optimizer.zero_grad()
            logits = model(src, tgt_in, src_mask, tgt_mask)
            loss = criterion(logits.reshape(-1, logits.size(-1)), tgt_out.reshape(-1))
            loss.backward()
            optimizer.step()
            if step % 10 == 0 or step == 1:
                print(f"Step {step}/{overfit_cfg.steps} | Loss {loss.item():.4f}")
            if step >= overfit_cfg.steps:
                break
    print("Overfit routine completed.")
    showcase_generation(model, tokenizer, training_cfg)


def load_or_train_tokenizer(force_retrain: bool = False) -> Tokenizer:
    if not DEFAULT_TOKENIZER_PATH.exists() or force_retrain:
        print("Training tokenizer from scratch...")
        cfg = TokenizerConfig()
        return train_tokenizer(cfg)
    print(f"Reusing existing tokenizer at {DEFAULT_TOKENIZER_PATH}")
    return load_tokenizer()


# ---------------------------------------------------------------------------
# Command-line interface
# ---------------------------------------------------------------------------

def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="NanoSocrates unified pipeline")
    subparsers = parser.add_subparsers(dest="command", required=True)

    def add_shared_training_args(train_parser: argparse.ArgumentParser) -> None:
        train_parser.add_argument("--epochs", type=int, default=10)
        train_parser.add_argument("--batch-size", type=int, default=16)
        train_parser.add_argument("--lr", type=float, default=3e-4)
        train_parser.add_argument("--d-model", type=int, default=256)
        train_parser.add_argument("--nhead", type=int, default=4)
        train_parser.add_argument("--encoder-layers", type=int, default=4)
        train_parser.add_argument("--decoder-layers", type=int, default=4)
        train_parser.add_argument("--ff-dim", type=int, default=512)
        train_parser.add_argument("--dropout", type=float, default=0.1)
        train_parser.add_argument("--max-seq-len", type=int, default=512)
        train_parser.add_argument("--device", type=str, default=None)
        train_parser.add_argument("--resume", action="store_true")
        train_parser.add_argument("--retrain-tokenizer", action="store_true")

    pipeline_parser = subparsers.add_parser("pipeline", help="Train tokenizer, train model, evaluate, showcase")
    add_shared_training_args(pipeline_parser)

    train_parser = subparsers.add_parser("train", help="Train model only")
    add_shared_training_args(train_parser)

    eval_parser = subparsers.add_parser("evaluate", help="Evaluate a checkpoint")
    eval_parser.add_argument("--checkpoint", type=str, default=str(DEFAULT_MODEL_PATH))
    eval_parser.add_argument("--batch-size", type=int, default=16)
    eval_parser.add_argument("--max-seq-len", type=int, default=512)
    eval_parser.add_argument("--device", type=str, default=None)

    overfit_parser = subparsers.add_parser("overfit", help="Run a sanity-check overfitting loop on a small batch")
    overfit_parser.add_argument("--steps", type=int, default=200)
    overfit_parser.add_argument("--batch-size", type=int, default=8)
    overfit_parser.add_argument("--lr", type=float, default=5e-4)
    overfit_parser.add_argument("--max-seq-len", type=int, default=256)
    overfit_parser.add_argument("--retrain-tokenizer", action="store_true")

    gen_parser = subparsers.add_parser("generate", help="Generate text given a custom input")
    gen_parser.add_argument("prompt", type=str, help="Input string with task token, e.g. '<SOT> ... <Text2RDF>'")
    gen_parser.add_argument("--max-seq-len", type=int, default=128)
    gen_parser.add_argument("--checkpoint", type=str, default=str(DEFAULT_MODEL_PATH))

    return parser


def cli_generate(args: argparse.Namespace) -> None:
    tokenizer = load_tokenizer()
    cfg = TrainingConfig(max_seq_len=args.max_seq_len, save_path=Path(args.checkpoint))
    model = load_model(tokenizer, cfg)
    device = next(model.parameters()).device
    start_token = tokenizer.token_to_id("<SOT>")
    if start_token is None:
        print("Tokenizer missing <SOT>; unable to generate.")
        return
    dataset = MultitaskExample([(args.prompt, "")], tokenizer, args.max_seq_len)
    batch = dataset.collate([dataset[0]])
    src = batch["src"].to(device)
    src_mask = batch["src_padding_mask"].to(device)
    generated = model.greedy_decode(src, src_mask, args.max_seq_len, start_token)
    decoded = detokenize_sequence(generated[0, 1:], tokenizer)
    print(decoded)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = build_argument_parser()
    args = parser.parse_args(argv)

    if args.command == "pipeline":
        run_full_pipeline(args)
    elif args.command == "train":
        tokenizer = load_or_train_tokenizer(force_retrain=args.retrain_tokenizer)
        cfg = TrainingConfig(
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            d_model=args.d_model,
            nhead=args.nhead,
            num_encoder_layers=args.encoder_layers,
            num_decoder_layers=args.decoder_layers,
            dim_feedforward=args.ff_dim,
            dropout=args.dropout,
            max_seq_len=args.max_seq_len,
            device=args.device,
        )
        run_training(tokenizer, cfg, resume=args.resume)
    elif args.command == "evaluate":
        run_evaluation_only(args)
    elif args.command == "overfit":
        run_overfit(args)
    elif args.command == "generate":
        cli_generate(args)
    else:
        parser.error(f"Unknown command: {args.command}")


if __name__ == "__main__":
    main(sys.argv[1:])
