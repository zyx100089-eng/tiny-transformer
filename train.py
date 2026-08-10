import torch
import csv
import math
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from dataset import TextDataset
from model import TinyTransformer


def get_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


@torch.no_grad()
def estimate_loss(
    model: TinyTransformer,
    dataset: TextDataset,
    batch_size: int,
    block_size: int,
    device: str,
    eval_iters: int = 50,
) -> dict[str, float]:
    model.eval()
    losses = {}
    for split in ("train", "val"):
        total = 0.0
        for _ in range(eval_iters):
            x, y = dataset.get_batch(split, batch_size, block_size, device)
            logits = model(x)
            loss = torch.nn.functional.cross_entropy(
                logits.view(-1, logits.size(-1)), y.view(-1)
            )
            total += loss.item()
        losses[split] = total / eval_iters
    model.train()
    return losses


def train(
    data_path: str = "data/input.txt",
    embed_dim: int = 64,
    num_heads: int = 4,
    num_layers: int = 2,
    block_size: int = 64,
    batch_size: int = 32,
    lr: float = 3e-4,
    max_iters: int = 5000,
    eval_interval: int = 250,
    use_pos_embed: bool = True,
    dropout: float = 0.0,
    lr_schedule: str = "constant",
    warmup_iters: int = 200,
    min_lr: float = 3e-5,
    seed: int = 42,
    save_path: str = "model.pth",
    results_dir: str = "results",
) -> dict:
    torch.manual_seed(seed)
    device = str(get_device())
    print(f"Using device: {device}")

    dataset = TextDataset(data_path)
    print(f"Vocab size: {dataset.vocab_size}, Train tokens: {len(dataset.train_data):,}")

    model = TinyTransformer(
        vocab_size=dataset.vocab_size,
        embed_dim=embed_dim,
        num_heads=num_heads,
        num_layers=num_layers,
        block_size=block_size,
        use_pos_embed=use_pos_embed,
        dropout=dropout,
    ).to(device)

    param_count = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {param_count:,}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    def get_lr(step: int) -> float:
        if lr_schedule == "constant":
            return lr
        elif lr_schedule == "cosine":
            if step < warmup_iters:
                return lr * (step / warmup_iters)
            decay_ratio = (step - warmup_iters) / (max_iters - warmup_iters)
            coeff = 0.5 * (1 + math.cos(math.pi * decay_ratio))
            return min_lr + coeff * (lr - min_lr)
        return lr

    history = []

    for step in range(max_iters):
        lr_now = get_lr(step)
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr_now

        x, y = dataset.get_batch("train", batch_size, block_size, device)
        logits = model(x)
        loss = torch.nn.functional.cross_entropy(
            logits.view(-1, logits.size(-1)), y.view(-1)
        )
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if step % eval_interval == 0 or step == max_iters - 1:
            losses = estimate_loss(model, dataset, batch_size, block_size, device)
            history.append({"step": step, "train_loss": losses["train"], "val_loss": losses["val"], "lr": lr_now})
            print(f"Step {step:5d} | train {losses['train']:.4f} | val {losses['val']:.4f} | lr {lr_now:.2e}")

    results_path = Path(results_dir)
    results_path.mkdir(exist_ok=True)

    torch.save(
        {"model_state": model.state_dict(), "config": {
            "vocab_size": dataset.vocab_size, "embed_dim": embed_dim,
            "num_heads": num_heads, "num_layers": num_layers,
            "block_size": block_size, "use_pos_embed": use_pos_embed,
            "dropout": dropout,
        }},
        save_path,
    )

    dataset.tokenizer.save(str(results_path / "tokenizer.json"))

    csv_path = results_path / "training_log.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["step", "train_loss", "val_loss", "lr"])
        writer.writeheader()
        writer.writerows(history)

    fig, ax = plt.subplots(figsize=(8, 5))
    steps = [h["step"] for h in history]
    ax.plot(steps, [h["train_loss"] for h in history], label="Train")
    ax.plot(steps, [h["val_loss"] for h in history], label="Validation")
    ax.set_xlabel("Step")
    ax.set_ylabel("Loss")
    ax.set_title("Training Loss Curve")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(str(results_path / "loss_curves.png"), dpi=150)
    plt.close(fig)

    print(f"\nModel saved to {save_path}")
    print(f"Loss curve saved to {results_path / 'loss_curves.png'}")
    return {"history": history, "dataset": dataset, "model": model}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/input.txt")
    parser.add_argument("--embed_dim", type=int, default=64)
    parser.add_argument("--num_heads", type=int, default=4)
    parser.add_argument("--num_layers", type=int, default=2)
    parser.add_argument("--block_size", type=int, default=64)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--max_iters", type=int, default=5000)
    parser.add_argument("--no_pos_embed", action="store_true")
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--lr_schedule", choices=["constant", "cosine"], default="constant")
    parser.add_argument("--warmup_iters", type=int, default=200)
    parser.add_argument("--min_lr", type=float, default=3e-5)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    train(
        data_path=args.data,
        embed_dim=args.embed_dim,
        num_heads=args.num_heads,
        num_layers=args.num_layers,
        block_size=args.block_size,
        batch_size=args.batch_size,
        lr=args.lr,
        max_iters=args.max_iters,
        use_pos_embed=not args.no_pos_embed,
        dropout=args.dropout,
        lr_schedule=args.lr_schedule,
        warmup_iters=args.warmup_iters,
        min_lr=args.min_lr,
        seed=args.seed,
    )
