import torch
import json
import argparse
from pathlib import Path

from model import TinyTransformer
from tokenizer import CharTokenizer
from bpe_tokenizer import BPETokenizer


def generate(
    model_path: str = "model.pth",
    tokenizer_path: str = "results/tokenizer.json",
    prompt: str = "\n",
    max_tokens: int = 500,
    temperature: float = 0.8,
    num_samples: int = 3,
    output_path: str = "results/generated_samples.txt",
) -> list[str]:
    checkpoint = torch.load(model_path, map_location="cpu", weights_only=True)
    config = checkpoint["config"]

    data = json.loads(Path(tokenizer_path).read_text())
    if "merges" in data:
        tokenizer = BPETokenizer().load(tokenizer_path)
    else:
        tokenizer = CharTokenizer().load(tokenizer_path)

    model = TinyTransformer(**config)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    samples = []
    for i in range(num_samples):
        ids = tokenizer.encode(prompt)
        idx = torch.tensor([ids], dtype=torch.long)
        out = model.generate(idx, max_new_tokens=max_tokens, temperature=temperature)
        text = tokenizer.decode(out[0].tolist())
        samples.append(text)
        print(f"\n{'='*60}")
        print(f"Sample {i+1} (temperature={temperature}):")
        print(f"{'='*60}")
        print(text)

    Path(output_path).parent.mkdir(exist_ok=True)
    with open(output_path, "w") as f:
        for i, s in enumerate(samples):
            f.write(f"--- Sample {i+1} ---\n{s}\n\n")

    print(f"\nSaved {num_samples} samples to {output_path}")
    return samples


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="model.pth")
    parser.add_argument("--tokenizer", default="results/tokenizer.json")
    parser.add_argument("--prompt", default="\n")
    parser.add_argument("--max_tokens", type=int, default=500)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--num_samples", type=int, default=3)
    args = parser.parse_args()

    generate(
        model_path=args.model,
        tokenizer_path=args.tokenizer,
        prompt=args.prompt,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        num_samples=args.num_samples,
    )
