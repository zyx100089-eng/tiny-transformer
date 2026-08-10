import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class Head(nn.Module):
    """Single head of scaled dot-product self-attention."""

    def __init__(self, embed_dim: int, head_size: int, block_size: int, dropout: float = 0.0):
        super().__init__()
        self.query = nn.Linear(embed_dim, head_size, bias=False)
        self.key = nn.Linear(embed_dim, head_size, bias=False)
        self.value = nn.Linear(embed_dim, head_size, bias=False)
        self.register_buffer("mask", torch.tril(torch.ones(block_size, block_size)))
        self.scale = math.sqrt(head_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape
        q = self.query(x)  # (B, T, head_size)
        k = self.key(x)
        v = self.value(x)

        scores = (q @ k.transpose(-2, -1)) / self.scale  # (B, T, T)
        scores = scores.masked_fill(self.mask[:T, :T] == 0, float("-inf"))
        weights = F.softmax(scores, dim=-1)
        weights = self.dropout(weights)
        return weights @ v  # (B, T, head_size)


class MultiHeadAttention(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int, block_size: int, dropout: float = 0.0):
        super().__init__()
        head_size = embed_dim // num_heads
        self.heads = nn.ModuleList(
            [Head(embed_dim, head_size, block_size, dropout) for _ in range(num_heads)]
        )
        self.proj = nn.Linear(embed_dim, embed_dim)
        self.resid_dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        return self.resid_dropout(self.proj(out))


class FeedForward(nn.Module):
    def __init__(self, embed_dim: int, dropout: float = 0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(embed_dim, 4 * embed_dim),
            nn.GELU(),
            nn.Linear(4 * embed_dim, embed_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class TransformerBlock(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int, block_size: int, dropout: float = 0.0):
        super().__init__()
        self.ln1 = nn.LayerNorm(embed_dim)
        self.attn = MultiHeadAttention(embed_dim, num_heads, block_size, dropout)
        self.ln2 = nn.LayerNorm(embed_dim)
        self.ff = FeedForward(embed_dim, dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln1(x))
        x = x + self.ff(self.ln2(x))
        return x


class TinyTransformer(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        embed_dim: int = 64,
        num_heads: int = 4,
        num_layers: int = 2,
        block_size: int = 64,
        use_pos_embed: bool = True,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.block_size = block_size
        self.use_pos_embed = use_pos_embed

        self.token_embed = nn.Embedding(vocab_size, embed_dim)
        if use_pos_embed:
            self.pos_embed = nn.Embedding(block_size, embed_dim)
        self.embed_dropout = nn.Dropout(dropout)

        self.blocks = nn.Sequential(
            *[TransformerBlock(embed_dim, num_heads, block_size, dropout) for _ in range(num_layers)]
        )
        self.ln_f = nn.LayerNorm(embed_dim)
        self.lm_head = nn.Linear(embed_dim, vocab_size)

        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        B, T = idx.shape
        x = self.token_embed(idx)  # (B, T, embed_dim)
        if self.use_pos_embed:
            positions = torch.arange(T, device=idx.device)
            x = x + self.pos_embed(positions)
        x = self.embed_dropout(x)
        x = self.blocks(x)
        x = self.ln_f(x)
        return self.lm_head(x)  # (B, T, vocab_size)

    @torch.no_grad()
    def generate(
        self,
        idx: torch.Tensor,
        max_new_tokens: int,
        temperature: float = 1.0,
    ) -> torch.Tensor:
        was_training = self.training
        self.eval()
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.block_size :]
            logits = self(idx_cond)  # (B, T, vocab_size)
            logits = logits[:, -1, :] / temperature  # last token
            probs = F.softmax(logits, dim=-1)
            next_id = torch.multinomial(probs, num_samples=1)
            idx = torch.cat([idx, next_id], dim=1)
        if was_training:
            self.train()
        return idx
