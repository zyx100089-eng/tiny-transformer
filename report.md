# Tiny Transformer from Scratch: Understanding Self-Attention and Sequence Modelling

## Abstract

This project implements a small Transformer language model from scratch in PyTorch, trained on character-level Shakespeare text (~1MB). The model implements all core Transformer components: token and positional embeddings, scaled dot-product self-attention with causal masking, multi-head attention, feed-forward networks, residual connections, and layer normalisation. A byte-pair encoding (BPE) tokeniser is also implemented from scratch for comparison. Ablation experiments — run across 3 random seeds for statistical reliability — investigate the effect of context length, number of attention heads, layer depth, positional encoding, regularisation, and tokenisation strategy. Key findings include: positional encoding is the most impactful architectural component (removing it degrades loss by 0.22; dropout 0.2 degrades it by 0.29 but that variant also changes the LR schedule); more attention heads do not help at this small per-head dimension (16); and regularisation with dropout is counterproductive for small underfitting models. The trained model generates Shakespeare-like text with the CAPS: dialogue format and verse structure captured, though character names come out garbled (e.g. "QUEEN MVIUS:") at this model size.

## Introduction

The Transformer architecture, introduced by Vaswani et al. (2017), replaced recurrence and convolution with self-attention as the core mechanism for sequence modelling. This project asks:

> How do context length, number of attention heads, layer depth, positional encoding, regularisation, and tokenisation strategy affect the performance of a small Transformer language model?

Rather than using existing libraries, every component is implemented from first principles to demonstrate understanding of the underlying mathematics and design choices.

## Background

### Why attention?

In sequence modelling, a model must learn relationships between tokens at different positions. Recurrent networks process sequences one step at a time, creating a bottleneck for long-range dependencies. Self-attention computes relationships between all pairs of positions in parallel, allowing direct connections between distant tokens.

### Queries, keys, and values

Self-attention uses three learned projections of the input:

- **Query (Q)**: what information this token is looking for
- **Key (K)**: what information this token contains
- **Value (V)**: the actual content to aggregate

The attention score between two positions is the dot product of the query at one position with the key at the other. High scores mean the query and key are aligned — the model has learned that these positions are relevant to each other.

The full computation is:

```
Attention(Q, K, V) = softmax(QK^T / sqrt(d_k)) V
```

The scaling factor `sqrt(d_k)` prevents dot products from growing large in magnitude, which would push softmax into regions with vanishingly small gradients.

### Mathematical justification for scaling by √d_k

Assume Q and K are independently sampled with mean 0 and variance 1, with dimension d_k. The dot product of a single query/key pair is:

```
q · k = Σ_{i=1}^{d_k} q_i k_i
```

Since each q_i and k_i are independent with E[q_i] = E[k_i] = 0, Var[q_i] = Var[k_i] = 1:

```
E[q · k] = 0
Var[q · k] = Σ Var[q_i k_i] = Σ (E[q_i²]·E[k_i²]) = d_k
```

So the standard deviation of the dot product is √d_k. Dividing by √d_k normalises the dot product to unit variance, preventing softmax inputs from growing as d_k increases. Without this, for large d_k, softmax would saturate (approaching a one-hot vector), and gradients would vanish — making training unstable. This analysis follows Vaswani et al. (2017), Footnote 4 of Section 3.2.1.

### Causal masking

For language modelling, the model must predict the next token given only previous tokens. Without a mask, position 3 could attend to position 4 and simply copy the answer. A causal mask sets all attention scores from position `i` to positions `j > i` to negative infinity before softmax, ensuring information only flows from past to present.

### Multi-head attention

A single attention head learns one type of relationship between positions. Multi-head attention runs several heads in parallel, each with independent Q, K, V projections, then concatenates their outputs. This allows the model to simultaneously attend to different aspects of the input — for example, one head might learn syntactic relationships while another learns semantic ones.

### Positional encoding

Self-attention is permutation-equivariant: it produces the same output regardless of token order (up to the permutation). Positional embeddings break this symmetry by adding position-dependent vectors to the input, allowing the model to distinguish "the cat sat" from "sat the cat".

### Residual connections and gradient flow

The pre-norm architecture uses:

```
x_{l+1} = x_l + F(LN(x_l))
```

where F is attention or the feed-forward network. The gradient of the loss L with respect to x_l:

```
∂L/∂x_l = ∂L/∂x_{l+1} · (I + J_F · J_LN)
```

where J_F and J_LN are Jacobians. The identity term I means gradients flow directly through the residual path, bypassing F. This makes it easier to train deep models, as gradients can propagate through many layers without vanishing. The pre-norm variant (LayerNorm before the sublayer) is more stable than the original post-norm formulation, because the residual path carries the un-normalised `x` directly — only the sublayer input is normalised — so the residual gradient is always ~1 regardless of the LayerNorm Jacobian.

## Method

### Architecture

The model consists of:

1. **Token embedding**: a learnable lookup table mapping each of 65 characters to a 64-dimensional vector
2. **Positional embedding**: a learnable lookup table mapping each position (0 to 63) to a 64-dimensional vector, added to the token embedding
3. **Transformer blocks** (×2): each containing multi-head self-attention (4 heads) and a feed-forward network (Linear → GELU → Linear with 4× expansion), both with pre-norm residual connections
4. **Final layer norm** followed by a linear projection to vocabulary size

Total parameters: 112,193.

### Parameter count derivation

For a model with vocab_size V=65, embed_dim d=64, num_heads h=4, num_layers L=2, block_size T=64:

- Token embedding: V × d = 65 × 64 = 4,160
- Positional embedding: T × d = 64 × 64 = 4,096
- Per TransformerBlock:
  - LayerNorm × 2: 2 × 2d = 256 (weight + bias each)
  - Attention Q/K/V (no bias): 4 heads × 3 × (d × head_size) = 4 × 3 × 64 × 16 = 12,288
  - Attention output projection (with bias): d × d + d = 4,096 + 64 = 4,160
  - FeedForward (with biases): (d × 4d + 4d) + (4d × d + d) = (16,384 + 256) + (16,384 + 64) = 33,088
  - Block total: 256 + 12,288 + 4,160 + 33,088 = 49,792
- 2 blocks: 99,584
- Final LayerNorm: 2d = 128
- LM head (with bias): d × V + V = 4,160 + 65 = 4,225

Total = 4,160 + 4,096 + 99,584 + 128 + 4,225 = **112,193** ✓

### Implementation choices

- **Character-level tokenisation** as the default, with an optional **BPE tokeniser** implemented from scratch for comparison (see Tokeniser experiment below)
- **Pre-norm** (LayerNorm before attention/FFN) rather than post-norm: empirically more stable for training small models
- **GELU** activation rather than ReLU: smoother non-linearity, standard in modern Transformers
- **Learnable positional embeddings** rather than sinusoidal: simpler and equally effective for short sequences
- **Dropout** applied to attention weights, residual connections, and embeddings: optional, configurable via `--dropout`
- **Cosine learning rate schedule** with warmup: optional, configurable via `--lr_schedule cosine`, decays from peak LR to `min_lr` over training

### Training

- **Optimiser**: AdamW with learning rate 3×10⁻⁴
- **Loss**: cross-entropy on next-token prediction
- **Iterations**: 5,000 gradient steps
- **Batch size**: 32 sequences of length 64
- **Device**: Apple MPS (Metal Performance Shaders)
- **Regularisation**: dropout=0.1 (attention weights, residual connections, embeddings)
- **LR schedule**: cosine with 200-step warmup, decaying to 3×10⁻⁵
- **Seed**: 42

## Results

### Training performance

The model was trained for 5,000 iterations with dropout=0.1 and cosine LR scheduling on ~1M characters of Shakespeare text (90% train, 10% validation).

| Metric | Start | End |
|---|---|---|
| Train loss | 4.188 | 1.931 |
| Val loss | 4.191 | 1.999 |

Loss decreased steadily throughout training. The cosine schedule is visible in the LR column: warmup reaches peak 3×10⁻⁴ around step 200, then decays smoothly to 3×10⁻⁵ by step 5000. The train-val gap at convergence is 0.068 — small, confirming that a 112K-parameter model on 1M characters is underfitting rather than overfitting (see §Ablation: Regularisation).

For reference, random prediction over 65 characters gives loss `ln(65) ≈ 4.17`, confirming the initial loss is near random and the final loss represents substantial learning.

### Generated text

After training, the model generates text that captures several features of Shakespeare:

- **Character names** in capitals followed by colons (BRUTUS:, SICINIUS:)
- **Verse-like line structure** with roughly iambic rhythm
- **Period-appropriate vocabulary** ("thou", "hath", "shall", "prithee")
- **Dialogue conventions** with stage-direction-like formatting

The generated words are often plausible but not real English words, reflecting the character-level model's limited capacity. This is expected — the model has learned local character patterns and structural conventions but cannot reliably produce coherent sentences.

### Ablation experiments

Each experiment trains for 2,000 iterations, varying one hyperparameter while holding others at default values. **All experiments are run across 3 random seeds (42, 123, 777)** and report mean ± standard deviation, ensuring the results are not due to a single lucky seed.

#### Context length

| Context | Val loss (mean ± std) |
|---|---|
| 16 | 2.092 ± 0.011 |
| 32 | 2.045 ± 0.010 |
| 64 | 2.012 ± 0.016 |

Longer context consistently improves performance. With only 16 tokens of context, the model cannot capture dependencies spanning more than a few words. At 64 tokens, the model sees enough context to learn patterns like character name formatting and line breaks. The improvement is monotonic and the standard deviations do not overlap, making this a robust finding.

#### Number of attention heads

| Heads | Val loss (mean ± std) |
|---|---|
| 1 | 1.985 ± 0.020 |
| 2 | 1.995 ± 0.010 |
| 4 | 2.012 ± 0.016 |

Counter-intuitively, fewer heads perform slightly better, though the standard deviations across seeds overlap (1-head: 1.985±0.020, 2-head: 1.995±0.010, 4-head: 2.012±0.016), so this should be read as "multi-head provides no benefit at this scale" rather than "fewer heads is strictly better." With the same total embedding dimension (64), each head in the 4-head configuration has only 16 dimensions, which is likely too small to learn useful representations. The 1-head model (head_size=64) has the most capacity per head. This is consistent with the hypothesis that multi-head attention requires sufficient per-head dimensionality to be effective; it does not generalise to larger models where head_size is larger.

#### Number of layers

| Layers | Val loss (mean ± std) | Params |
|---|---|---|
| 1 | 2.029 ± 0.002 |
| 2 | 2.012 ± 0.016 |
| 4 | 1.977 ± 0.003 |

Deeper models perform better, with 4 layers achieving the lowest validation loss. The improvement is modest but consistent: each additional layer allows the model to compose more complex representations. The 4-layer model has 3.4× more parameters (211,777 vs 62,401), but the standard deviation is remarkably low (0.003), indicating this is a stable finding.

#### Positional encoding

| Positional | Val loss (mean ± std) |
|---|---|
| With | 2.012 ± 0.016 |
| Without | 2.232 ± 0.013 |

Removing positional encoding degrades validation loss by 0.22 — the largest effect among the architectural ablations (dropout 0.2 degrades it by 0.29, but that experiment also changes the LR schedule; see the regularisation section). Without positional information, the model cannot distinguish token order, so it cannot learn that character names appear at the start of lines or that punctuation follows certain patterns. The generated text without positional encoding also shows noticeably worse structure, with garbled character names and broken formatting.

#### Regularisation (dropout + cosine LR schedule)

| Dropout | LR schedule | Val loss (mean ± std) | Train loss |
|---|---|---|---|
| 0.0 | constant | 2.012 ± 0.016 | 1.937 |
| 0.1 | cosine | 2.208 ± 0.004 | 2.196 |
| 0.2 | cosine | 2.304 ± 0.009 | 2.301 |

This experiment produced a surprising result: adding dropout *hurts* performance at this model scale. One caveat: the variants confound dropout with the LR schedule — d=0.0 uses a constant LR, while d=0.1/d=0.2 use cosine with warmup. The cosine schedule should if anything *help* (it is the better schedule in the training-log comparison), so the degradation is unlikely to be an artefact of the schedule change, but the two factors are not cleanly separated. Without regularisation, the train-val gap is 0.075 — already small. With dropout=0.1, both train and val loss *increase* substantially, and the gap nearly vanishes (0.012). With dropout=0.2, the effect is even stronger.

**Interpretation**: A 112K-parameter model trained on 1M characters is **underfitting**, not overfitting. The model lacks the capacity to memorise the training data, so regularisation is counterproductive — it restricts the model's already-limited capacity without addressing any generalisation problem. The near-zero train-val gap with dropout confirms this: the model has no memorisation to prevent. This is an important finding: regularisation strategies that work for large models (which overfit) can harm small models (which underfit).

#### Tokeniser comparison (character-level vs BPE)

| Tokeniser | Vocab size | Val loss (mean ± std) |
|---|---|---|
| Character | 65 | 2.012 ± 0.016 |
| BPE (200 merges) | ~265 | 3.477 ± 0.014 |
| BPE (500 merges) | ~565 | 4.003 ± 0.010 |

BPE tokenisation *increases* loss, contrary to expectations. This is because **cross-entropy loss is not comparable across different vocabularies**. A character-level model predicts over 65 tokens (random baseline: ln(65) = 4.17), while BPE with 500 merges predicts over ~565 tokens (random baseline: ln(565) = 6.34). The BPE model's loss of 4.00 is actually further below its baseline (6.34 - 4.00 = 2.34) than the character model's loss of 2.01 is below its baseline (4.17 - 2.01 = 2.16), suggesting BPE *is* learning more per prediction. However, the raw loss numbers are not directly comparable.

A fairer comparison would use **bits-per-character** (BPC): normalise the loss by the average number of characters per token. This is listed as future work. The BPE tokeniser was implemented from scratch, finding the most frequent adjacent character pairs and merging them greedily.

## Discussion

### Key findings

1. **Positional encoding is the most important architectural component** tested. Removing it degrades validation loss by 0.22 — the largest effect among the architectural ablations (the dropout 0.2 variant degrades it by 0.29, but that experiment confounds dropout with the LR schedule). This confirms that self-attention alone is permutation-equivariant and needs explicit position information.

2. **Context length matters more than model depth** for this task. Increasing context from 16 to 64 tokens reduced validation loss by 0.08, comparable to tripling the number of layers (0.05).

3. **Fewer heads perform better at this scale.** With only 64 embedding dimensions, splitting across 4 heads gives each head just 16 dimensions — too small to learn useful representations. A single head with 64 dimensions outperforms 4 heads with 16 each. This suggests multi-head attention requires a minimum per-head dimensionality to be beneficial, a finding that connects to the design of modern LLMs which use d_model=4096+ with 32 heads (128 per head).

4. **Deeper models help with diminishing returns.** Going from 1 to 4 layers reduces val loss by 0.05, but requires 3.4× more parameters. The low standard deviation (0.003) confirms this is a stable finding.

5. **Regularisation hurts small models.** Adding dropout=0.1 increased val loss from 2.01 to 2.21. The near-zero train-val gap (0.012 with dropout) reveals the model is underfitting, not overfitting, at 2000 iterations. This demonstrates that regularisation is only beneficial when the model has sufficient capacity to overfit — a key insight about the relationship between model scale and regularisation strategy.

6. **BPE tokenisation requires fair comparison metrics.** Raw cross-entropy losses are not comparable across different vocabulary sizes. BPE's higher loss is partly due to its larger vocabulary (higher random baseline). A fair comparison requires bits-per-character normalisation, left as future work.

### Attention visualisation

Attention weight heatmaps were extracted for each head in each layer (see `results/attention/attention_heatmaps.png`). The visualisations show:

- **Lower layers** attend to local, adjacent tokens — the diagonal band pattern typical of early attention layers
- **Upper layers** show more diffuse attention patterns, with some heads attending to punctuation and line breaks — suggesting the model learns formatting conventions
- Individual heads develop distinct attention patterns, supporting the multi-head attention hypothesis that different heads learn different relationships

### Connection to modern LLMs

This project implements the same fundamental mechanisms used in large language models (GPT, LLaMA, etc.), but at a vastly smaller scale. Modern LLMs use:

- Subword tokenisation (BPE/SentencePiece) instead of character-level
- Embedding dimensions of 4,096+ instead of 64
- 32-128 layers instead of 2
- Billions of parameters instead of 112K
- Training on trillions of tokens instead of 1M characters

The core attention mechanism, however, is identical.

## Limitations

1. **Small model**: 112K parameters cannot learn complex language patterns. Modern LLMs are 10,000–1,000,000× larger. The underfitting regime observed in the regularisation experiment is a direct consequence of this.
2. **Single small dataset**: 1MB of Shakespeare is not representative of general language.
3. **Short ablation training**: 2,000 iterations per experiment leaves the model in the underfitting regime for some experiments. Longer training might reveal overfitting where dropout would help.
4. **No comparison with RNNs or LSTMs**: a direct comparison would strengthen the argument for attention-based architectures.
5. **BPE comparison is not metric-fair**: raw cross-entropy losses are not comparable across different vocabulary sizes. Bits-per-character normalisation is needed.
6. **BPE implementation is basic**: it operates on raw characters without pre-tokenisation (splitting on whitespace/punctuation), so merges can cross word boundaries. Production BPE implementations (e.g., GPT-2's) pre-tokenise first.

## Future work

- Implement **bits-per-character (BPC)** evaluation to fairly compare tokenisers with different vocabulary sizes
- Train ablations for 5000+ iterations to investigate whether the regularisation findings change in the overfitting regime
- Track weight matrix rank during training (connecting to the Rank Collapse problem)
- Implement sinusoidal and RoPE positional encodings for three-way comparison with learnable embeddings
- Scale up embedding dimension and measure when multi-head attention becomes clearly beneficial
- Implement a baseline LSTM model for direct comparison
- Pre-tokenise text before BPE to prevent cross-word merges
- Evaluate on a second dataset (e.g., Python source code) to test generalisation

## References

1. Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A.N., Kaiser, L. and Polosukhin, I., 2017. Attention is all you need. *Advances in Neural Information Processing Systems*, 30.
2. Karpathy, A. nanoGPT. GitHub repository.
3. Ba, J.L., Kiros, J.R. and Hinton, G.E., 2016. Layer normalization. *arXiv preprint arXiv:1607.06450*.
