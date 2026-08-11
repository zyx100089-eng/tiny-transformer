import csv
import time
import math
import torch
import statistics
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from dataset import TextDataset
from model import TinyTransformer
from train import estimate_loss, get_device


def run_experiment(
    dataset: TextDataset,
    device: str,
    embed_dim: int = 64,
    num_heads: int = 4,
    num_layers: int = 2,
    block_size: int = 64,
    batch_size: int = 32,
    lr: float = 3e-4,
    max_iters: int = 2000,
    use_pos_embed: bool = True,
    dropout: float = 0.0,
    lr_schedule: str = "constant",
    seed: int = 42,
) -> dict:
    torch.manual_seed(seed)
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
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    warmup_iters = 200
    min_lr = 3e-5

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

    start = time.time()
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

    elapsed = time.time() - start
    losses = estimate_loss(model, dataset, batch_size, block_size, device)

    sample_ids = torch.zeros((1, 1), dtype=torch.long, device=device)
    generated = model.generate(sample_ids, max_new_tokens=200, temperature=0.8)
    sample_text = dataset.tokenizer.decode(generated[0].tolist())

    return {
        "train_loss": losses["train"],
        "val_loss": losses["val"],
        "params": param_count,
        "time_s": round(elapsed, 1),
        "sample": sample_text[:200],
    }


def run_variant_multi_seed(dataset, device, seeds, **kwargs) -> dict:
    results = []
    for seed in seeds:
        r = run_experiment(dataset, device, seed=seed, **kwargs)
        results.append(r)

    train_losses = [r["train_loss"] for r in results]
    val_losses = [r["val_loss"] for r in results]
    times = [r["time_s"] for r in results]

    return {
        "train_mean": statistics.mean(train_losses),
        "train_std": statistics.stdev(train_losses) if len(train_losses) > 1 else 0.0,
        "val_mean": statistics.mean(val_losses),
        "val_std": statistics.stdev(val_losses) if len(val_losses) > 1 else 0.0,
        "params": results[0]["params"],
        "time_s": round(statistics.mean(times), 1),
        # sample text is from the first seed only (representative, not averaged)
        "sample": results[0]["sample"],
    }


def run_all_experiments(
    data_path: str = "data/input.txt",
    max_iters: int = 2000,
    seeds: tuple = (42, 123, 777),
) -> None:
    device = str(get_device())
    print(f"Using device: {device}")
    print(f"Seeds: {seeds}  |  Iters per experiment: {max_iters}")

    dataset = TextDataset(data_path)
    results_dir = Path("results")
    results_dir.mkdir(exist_ok=True)

    experiments = []

    # --- Context length ---
    for block_size in (16, 32, 64):
        label = f"context_{block_size}"
        print(f"\nRunning: {label} (seeds={seeds})")
        r = run_variant_multi_seed(dataset, device, seeds, block_size=block_size, max_iters=max_iters)
        experiments.append({"experiment": "context_length", "variant": str(block_size), **r})

    # --- Attention heads ---
    for num_heads in (1, 2, 4):
        label = f"heads_{num_heads}"
        print(f"\nRunning: {label} (seeds={seeds})")
        r = run_variant_multi_seed(dataset, device, seeds, num_heads=num_heads, max_iters=max_iters)
        experiments.append({"experiment": "num_heads", "variant": str(num_heads), **r})

    # --- Layers ---
    for num_layers in (1, 2, 4):
        label = f"layers_{num_layers}"
        print(f"\nRunning: {label} (seeds={seeds})")
        r = run_variant_multi_seed(dataset, device, seeds, num_layers=num_layers, max_iters=max_iters)
        experiments.append({"experiment": "num_layers", "variant": str(num_layers), **r})

    # --- Positional encoding ---
    for use_pos in (True, False):
        label = f"pos_{'on' if use_pos else 'off'}"
        print(f"\nRunning: {label} (seeds={seeds})")
        r = run_variant_multi_seed(dataset, device, seeds, use_pos_embed=use_pos, max_iters=max_iters)
        experiments.append({"experiment": "positional_encoding", "variant": str(use_pos), **r})

    # --- Regularisation (dropout + cosine LR) ---
    for dropout, schedule in [(0.0, "constant"), (0.1, "cosine"), (0.2, "cosine")]:
        label = f"dropout_{dropout}_{schedule}"
        print(f"\nRunning: {label} (seeds={seeds})")
        r = run_variant_multi_seed(dataset, device, seeds, dropout=dropout, lr_schedule=schedule, max_iters=max_iters)
        experiments.append({
            "experiment": "regularisation",
            "variant": f"d={dropout},lr={schedule}",
            **r,
        })

    # --- Tokeniser comparison (char vs BPE) ---
    for tok_type, num_merges in [("char", 0), ("bpe", 200), ("bpe", 500)]:
        label = f"tokenizer_{tok_type}_{num_merges}"
        print(f"\nRunning: {label} (seeds={seeds})")
        ds = TextDataset(data_path, tokenizer_type=tok_type, num_merges=num_merges)
        r = run_variant_multi_seed(ds, device, seeds, max_iters=max_iters)
        experiments.append({
            "experiment": "tokeniser",
            "variant": f"{tok_type}{'_'+str(num_merges) if tok_type=='bpe' else ''}",
            **r,
        })

    # Save CSV
    csv_path = results_dir / "ablation_results.csv"
    fields = [
        "experiment", "variant",
        "train_mean", "train_std", "val_mean", "val_std",
        "params", "time_s", "sample",
    ]
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for e in experiments:
            row = {k: e.get(k, "") for k in fields}
            writer.writerow(row)
    print(f"\nResults saved to {csv_path}")

    # Print summary table
    print(f"\n{'Experiment':<22} {'Variant':<18} {'Train±Std':<16} {'Val±Std':<16} {'Params':<10} {'Time':<8}")
    print("-" * 95)
    for e in experiments:
        print(
            f"{e['experiment']:<22} {e['variant']:<18} "
            f"{e['train_mean']:<.4f}±{e['train_std']:.4f}   "
            f"{e['val_mean']:<.4f}±{e['val_std']:.4f}   "
            f"{e['params']:<10,} {e['time_s']:<8.1f}"
        )

    _plot_ablations(experiments, results_dir)


def _plot_ablations(experiments: list[dict], results_dir: Path) -> None:
    exp_groups = {}
    for e in experiments:
        exp_groups.setdefault(e["experiment"], []).append(e)

    ncols = 3
    nrows = (len(exp_groups) + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(7 * ncols, 5 * nrows))
    axes = axes.flatten()

    for ax, (name, group) in zip(axes, exp_groups.items()):
        variants = [g["variant"] for g in group]
        val_means = [g["val_mean"] for g in group]
        val_stds = [g["val_std"] for g in group]

        x = range(len(variants))
        ax.bar(x, val_means, yerr=val_stds, capsize=5, color="steelblue", alpha=0.8, label="Val loss")
        ax.set_xticks(list(x))
        ax.set_xticklabels(variants, rotation=30, ha="right")
        ax.set_ylabel("Validation Loss")
        ax.set_title(name.replace("_", " ").title())
        ax.grid(True, alpha=0.3, axis="y")

    for i in range(len(exp_groups), len(axes)):
        axes[i].set_visible(False)

    fig.tight_layout()
    fig.savefig(str(results_dir / "ablation_plots.png"), dpi=150)
    plt.close(fig)
    print(f"Ablation plots saved to {results_dir / 'ablation_plots.png'}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/input.txt")
    parser.add_argument("--max_iters", type=int, default=2000)
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 123, 777])
    args = parser.parse_args()
    run_all_experiments(data_path=args.data, max_iters=args.max_iters, seeds=tuple(args.seeds))