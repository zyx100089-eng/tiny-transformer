import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
from model import TinyTransformer


def test_forward_output_shape():
    model = TinyTransformer(vocab_size=50, embed_dim=32, num_heads=4, num_layers=2, block_size=16)
    idx = torch.randint(0, 50, (2, 10))
    logits = model(idx)
    assert logits.shape == (2, 10, 50)


def test_generate_produces_tokens():
    model = TinyTransformer(vocab_size=50, embed_dim=32, num_heads=4, num_layers=2, block_size=16)
    idx = torch.zeros((1, 1), dtype=torch.long)
    out = model.generate(idx, max_new_tokens=20)
    assert out.shape == (1, 21)  # 1 initial + 20 generated
    assert (out >= 0).all() and (out < 50).all()


def test_no_positional_embedding():
    model = TinyTransformer(vocab_size=50, embed_dim=32, num_heads=4, num_layers=2, block_size=16, use_pos_embed=False)
    idx = torch.randint(0, 50, (2, 10))
    logits = model(idx)
    assert logits.shape == (2, 10, 50)


def test_single_head():
    model = TinyTransformer(vocab_size=50, embed_dim=32, num_heads=1, num_layers=1, block_size=16)
    idx = torch.randint(0, 50, (1, 8))
    logits = model(idx)
    assert logits.shape == (1, 8, 50)


def test_with_dropout():
    model = TinyTransformer(vocab_size=50, embed_dim=32, num_heads=4, num_layers=2, block_size=16, dropout=0.1)
    idx = torch.randint(0, 50, (2, 10))
    logits = model(idx)
    assert logits.shape == (2, 10, 50)


def test_dropout_changes_output_in_train_mode():
    """With dropout, two forward passes should differ in train mode."""
    torch.manual_seed(0)
    model = TinyTransformer(vocab_size=50, embed_dim=32, num_heads=4, num_layers=2, block_size=16, dropout=0.5)
    model.train()
    idx = torch.randint(0, 50, (1, 8))
    out1 = model(idx)
    out2 = model(idx)
    assert not torch.allclose(out1, out2), "Dropout should make outputs differ in train mode"


def test_dropout_eval_mode_deterministic():
    """In eval mode, dropout is disabled so outputs should be identical."""
    torch.manual_seed(0)
    model = TinyTransformer(vocab_size=50, embed_dim=32, num_heads=4, num_layers=2, block_size=16, dropout=0.5)
    model.eval()
    idx = torch.randint(0, 50, (1, 8))
    with torch.no_grad():
        out1 = model(idx)
        out2 = model(idx)
    assert torch.allclose(out1, out2), "Eval mode dropout should be deterministic"