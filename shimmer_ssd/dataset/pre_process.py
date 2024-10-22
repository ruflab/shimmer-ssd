import torch
from simple_shapes_dataset import Text
from tokenizers.implementations import ByteLevelBPETokenizer


class TokenizeCaptions:
    def __init__(self, vocab: str, merges: str, pad_length: int):
        self._pad_length = pad_length
        self.tokenizer = ByteLevelBPETokenizer(vocab, merges)
        self.tokenizer.enable_padding(pad_token="<pad>", length=self._pad_length)

    def __call__(self, x: Text) -> dict[str, torch.Tensor]:
        text: dict[str, torch.Tensor] = {"bert": x.bert}
        text["tokens"] = torch.tensor(
            self.tokenizer.encode(x.caption).ids, dtype=torch.long
        )
        return text
