import torch
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from model import TinyTransformer
from train import get_device


@torch.no_grad()
def extract_attention_weights(model: TinyTransformer, x: torch.Tensor) -> dict:
    """Run forward pass and extract attention weights from each head in each layer.

    Returns dict[layer_idx] -> tensor of shape (num_heads, T, T)
    """
    B, T = x.shape
    token_emb = model.token_embed(x)
    if model.use_pos_embed:
        positions = torch.arange(T, device=x.device)
        token_emb = token_emb + model.pos_embed(positions)

    attention_weights = {}
    h = token_emb
    for i, block in enumerate(model.blocks):
        ln_out = block.ln1(h)
        head_weights = []
        for head in block.attn.heads:
            q = head.query(ln_out)
            k = head.key(ln_out)
            v = head.value(ln_out)
            scores = (q @ k.transpose(-2, -1)) / head.scale
            scores = scores.masked_fill(head.mask[:T, :T] == 0, float("-inf"))
            weights = torch.softmax(scores, dim=-1)
            head_weights.append(weights[0])
        attention_weights[i] = torch.stack(head_weights)
        h = block(h)

    return attention_weights


def visualize_attention(
    model_path: str = "model.pth",
    output_dir: str = "results/attention",
    prompt: str = "ROMEO:\nTo be, or n",
    tokenizer_path: str = "results/tokenizer.json",
) -> None:
    device = str(get_device())
    checkpoint = torch.load(model_path, map_location=device, weights_only=True)
    config = checkpoint["config"]

    from tokenizer import CharTokenizer
    tokenizer = CharTokenizer().load(tokenizer_path)

    model = TinyTransformer(**config).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    ids = torch.tensor([tokenizer.encode(prompt)], dtype=torch.long, device=device)
    if ids.size(1) > model.block_size:
        ids = ids[:, -model.block_size:]

    attn = extract_attention_weights(model, ids)
    tokens = [tokenizer.decode([i]) for i in ids[0].tolist()]

    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    num_layers = len(attn)
    num_heads = attn[0].shape[0]
    T = len(tokens)

    # Full grid: layers × heads
    fig, axes = plt.subplots(num_layers, num_heads, figsize=(4 * num_heads, 4 * num_layers), squeeze=False)

    for layer in range(num_layers):
        for head in range(num_heads):
            ax = axes[layer, head]
            matrix = attn[layer][head].cpu().numpy()
            im = ax.imshow(matrix, cmap="viridis", vmin=0, vmax=1)
            ax.set_title(f"Layer {layer+1}, Head {head+1}", fontsize=9)
            ax.set_xticks(range(T))
            ax.set_xticklabels(tokens, rotation=90, fontsize=7)
            ax.set_yticks(range(T))
            ax.set_yticklabels(tokens, fontsize=7)
            plt.colorbar(im, ax=ax, fraction=0.046)

    fig.suptitle(f"Attention Weights (prompt: '{prompt}')", fontsize=12)
    fig.tight_layout()
    fig.savefig(str(out_path / "attention_heatmaps.png"), dpi=150)
    plt.close(fig)
    print(f"Attention heatmaps saved to {out_path / 'attention_heatmaps.png'}")

    # Average across heads for each layer
    fig2, axes2 = plt.subplots(1, num_layers, figsize=(5 * num_layers, 5), squeeze=False)
    for layer in range(num_layers):
        avg = attn[layer].mean(dim=0).cpu().numpy()
        im = axes2[0, layer].imshow(avg, cmap="viridis", vmin=0, vmax=1)
        axes2[0, layer].set_title(f"Layer {layer+1} (avg over heads)")
        axes2[0, layer].set_xticks(range(T))
        axes2[0, layer].set_xticklabels(tokens, rotation=90, fontsize=7)
        axes2[0, layer].set_yticks(range(T))
        axes2[0, layer].set_yticklabels(tokens, fontsize=7)
        plt.colorbar(im, ax=axes2[0, layer], fraction=0.046)
    fig2.tight_layout()
    fig2.savefig(str(out_path / "attention_avg_by_layer.png"), dpi=150)
    plt.close(fig2)
    print(f"Layer averages saved to {out_path / 'attention_avg_by_layer.png'}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="model.pth")
    parser.add_argument("--tokenizer", default="results/tokenizer.json")
    parser.add_argument("--prompt", default="ROMEO:\nTo be, or n")
    parser.add_argument("--output_dir", default="results/attention")
    args = parser.parse_args()

    visualize_attention(
        model_path=args.model,
        tokenizer_path=args.tokenizer,
        prompt=args.prompt,
        output_dir=args.output_dir,
    )