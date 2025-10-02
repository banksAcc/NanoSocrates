from tokenizers import Tokenizer

class TokWrapper:
    def __init__(self, path: str):
        self.tk = Tokenizer.from_file(path)
        self._pad_id = self.tk.token_to_id("<pad>")
        if self._pad_id is None:
            raise ValueError("Tokenizer privo di <pad>: rigenera il BPE includendo <pad>.")

    def encode(self, text: str):
        return self.tk.encode(text).ids

    def token_to_id(self, tok: str):
        return self.tk.token_to_id(tok)

    @property
    def pad_id(self): return self._pad_id

    def vocab_size(self): return self.tk.get_vocab_size()
