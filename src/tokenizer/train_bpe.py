import glob, json
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.processors import TemplateProcessing

def _iter_text(files):
    for fp in files:
        with open(fp, "r", encoding="utf-8") as f:
            for line in f:
                r = json.loads(line)
                yield r.get("input", "")
                yield r.get("target", "")

def train_bpe(glob_pat: str, out_path: str, vocab_size: int, min_freq: int, special_tokens):
    files = glob.glob(glob_pat)
    tok = Tokenizer(BPE(unk_token="<unk>"))
    tok.pre_tokenizer = Whitespace()
    trainer = BpeTrainer(
        vocab_size=vocab_size,
        min_frequency=min_freq,
        special_tokens=["<unk>", "<pad>"] + list(special_tokens),
        show_progress=True,
    )
    tok.train_from_iterator(_iter_text(files), trainer=trainer)
    tok.post_processor = TemplateProcessing(single="$A", special_tokens=[("<pad>", tok.token_to_id("<pad>"))])
    tok.save(out_path)
    print(f"[tokenizer] saved -> {out_path}")
