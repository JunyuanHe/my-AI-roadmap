import torch
from typing import Optional, Tuple


class GPTConfig:
    


    def __init__(
        self,
        vocab_size: int = 1000,
        seq_len: int = 256,
        d_model: int = 384,
        n_heads: int = 6,
        n_layers: int = 6,
        dropout: float = 0.1,
        max_steps = 15001,
        warmup_steps = 300,
        max_lr = 3e-4,
        min_lr = 3e-5,
        grad_clip = 1.0,
        model_dir = "AI-generation/my-small-gpt/small-gpt-1/model",
        data_dir = "AI-generation/my-small-gpt/small-gpt-1/data",
        device: Optional[torch.device] = None,
    ):
        assert d_model % n_heads == 0
        self.vocab_size = vocab_size
        self.seq_len = seq_len
        self.d_model = d_model
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.dropout = dropout
        self.max_steps = max_steps
        self.warmup_steps = warmup_steps
        self.max_lr = max_lr
        self.min_lr = min_lr
        self.grad_clip = grad_clip
        self.model_dir = model_dir
        self.data_dir = data_dir
        self.device = device if device is not None else (torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))
