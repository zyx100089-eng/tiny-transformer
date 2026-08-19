# Tiny Transformer from Scratch

[![Tests](https://github.com/zyx100089-eng/tiny-transformer/actions/workflows/tests.yml/badge.svg)](https://github.com/zyx100089-eng/tiny-transformer/actions/workflows/tests.yml)

A small character-level Transformer language model trained on Tiny
Shakespeare, with multi-seed ablations on context length, heads,
positional encoding, regularisation, and tokenisation. The architecture
(attention, blocks, training loop, BPE tokeniser) is written from first
principles; PyTorch supplies only the tensor ops and autograd (which my
other projects deliberately avoid) — used here because the ablations
needed to run at speed.

## Results

### Training loss curve
![Training Loss Curve](results/loss_curves.png)

### Ablation study (3 seeds, mean ± std)
![Ablation Plots](results/ablation_plots.png)

### Attention weight heatmaps
![Attention Heatmaps](results/attention/attention_heatmaps.png)
![Attention Averaged by Layer](results/attention/attention_avg_by_layer.png)

### Generated text samples

```
BRUTUS:
My stall prace me my come, what I wall?
And him that is but cracial, and it the a the hath
Thou god Parcureds
Beautiof a gates all here.

SICINIUS:
By strun'd, to the light, to the long slie.

MAUMNIUS:
He as have you in honour's foot,
That what and should this roighter home a thy bone
Which and the to will frown in than thy laone.
```

*(See `results/generated_samples.txt` for full samples — the CAPS:
dialogue format and verse structure are captured, though character
names come out garbled (e.g. "QUEEN MVIUS:") at this model size.)*

## What it implements

Character-level tokenizer and a from-scratch byte-pair encoding (BPE)
tokenizer with greedy merge learning. Token/positional embeddings,
scaled dot-product self-attention with causal masking, multi-head
attention, GELU feed-forward, pre-norm residual blocks, dropout
throughout. Cosine LR schedule with warmup, temperature-sampled
generation, attention heatmaps, multi-seed ablations (mean ± std).

## Run it

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Download training data (Tiny Shakespeare)
mkdir -p data
curl -sL "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt" -o data/input.txt

# Train (default: no regularisation)
python train.py
python train.py --dropout 0.1 --lr_schedule cosine

python generate.py --prompt "ROMEO:" --temperature 0.8

# Visualise attention weights
python visualize.py --prompt "ROMEO:\nTo be, or n"

# Ablations (multi-seed; regularisation and tokeniser comparisons)
python experiments.py
python experiments.py --seeds 42 123 777 --max_iters 2000

python -m pytest tests/ -v
```

## Default Hyperparameters

| Parameter | Value |
|---|---|
| Embedding dimension | 64 |
| Attention heads | 4 |
| Transformer layers | 2 |
| Context length | 64 |
| Batch size | 32 |
| Learning rate | 3e-4 |
| Training iterations | 5000 |
| Dropout | 0.0 (configurable) |
| LR schedule | constant (or cosine with warmup) |

## Ablation Experiments

All experiments run across 3 random seeds and report mean ± standard deviation:

| Experiment | Variants |
|---|---|
| Baseline | bigram (0 parameters, exact validation loss) |
| Context length | 16 vs 32 vs 64 |
| Attention heads | 1 vs 2 vs 4 |
| Number of layers | 1 vs 2 vs 4 |
| Positional encoding | with vs without |
| Regularisation | none vs dropout=0.1+cosine vs dropout=0.2+cosine |
| Tokeniser | character vs BPE(200) vs BPE(500) |

The **bigram baseline** (`bigram.py`) is the standard reference point
for "did the model learn anything": it counts P(next | current) on
the training split and evaluates the exact cross-entropy on
validation — no parameters, no training. On Tiny Shakespeare the
bigram validation loss is **2.51**, while the default transformer
reaches **~2.01** (3-seed mean at 2000 iterations, see
`results/ablation_results.csv`) — the model genuinely learns beyond
token co-occurrence statistics, and every ablation variant can be
read against that floor.

Results are saved to `results/ablation_results.csv` and `results/ablation_plots.png`.
Attention heatmaps are saved to `results/attention/`.

Layout: `model.py` (architecture), `tokenizer.py`, `bpe_tokenizer.py`
(BPE from scratch), `bigram.py` (baseline), `dataset.py`, `train.py`,
`generate.py`, `experiments.py`, `visualize.py`, plus `tests/` and
`results/`.

## Known limits

- Small model (~112K parameters) — not comparable to production LLMs
- Single small dataset (1MB Shakespeare)
- BPE implementation does not pre-tokenise on whitespace
- No comparison with RNN/LSTM baselines (the bigram baseline covers the
  "did it learn anything" question; an LSTM comparison is a natural
  extension)

## References

- Vaswani et al., "Attention Is All You Need", 2017
- Karpathy, nanoGPT and minGPT implementations
- Ba et al., "Layer Normalization", 2016
