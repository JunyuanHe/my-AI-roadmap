import torch
from typing import Optional, Tuple


class GPTConfig:
    def __init__(
        self,
        vocab_size: int = 1000,
        seq_len: int = 256,
        d_model: int = 384,
        n_heads: int = 6,
        n_layers: int = 8,
        dropout: float = 0.15,
        device: Optional[torch.device] = None,
    ):
        assert d_model % n_heads == 0
        self.vocab_size = vocab_size
        self.seq_len = seq_len
        self.d_model = d_model
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.dropout = dropout
        self.device = device if device is not None else (torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))
