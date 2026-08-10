import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
from model import Head, MultiHeadAttention


def test_head_output_shape():
    head = Head(embed_dim=32, head_size=8, block_size=16)
    x = torch.randn(2, 10, 32)
    out = head(x)
    assert out.shape == (2, 10, 8)


def test_causal_mask_blocks_future():
    torch.manual_seed(42)
    head = Head(embed_dim=32, head_size=8, block_size=16)
    x = torch.randn(1, 5, 32, requires_grad=True)
    out = head(x)

    out[0, 2, 0].backward()
    grad = x.grad[0]  # (5, 32)
    assert grad[0].abs().sum() > 0, "Position 2 should attend to position 0"
    assert grad[2].abs().sum() > 0, "Position 2 should attend to itself"
    assert grad[3].abs().sum() == 0, "Position 2 must NOT attend to position 3"
    assert grad[4].abs().sum() == 0, "Position 2 must NOT attend to position 4"


def test_multi_head_output_shape():
    mha = MultiHeadAttention(embed_dim=32, num_heads=4, block_size=16)
    x = torch.randn(2, 10, 32)
    out = mha(x)
    assert out.shape == (2, 10, 32)


def test_head_with_dropout():
    head = Head(embed_dim=32, head_size=8, block_size=16, dropout=0.5)
    x = torch.randn(2, 10, 32)
    out = head(x)
    assert out.shape == (2, 10, 8)