"""Bigram baseline: the simplest possible language model.

Predicts the next token from a count table P(next | current) built
from the training data — no parameters, no training loop. This is the
standard reference point for "did the model learn anything": any
neural model that cannot beat the bigram loss has learned nothing
beyond token co-occurrence statistics.

The loss is the cross-entropy of the bigram model on the validation
split, computed exactly (no sampling noise).
"""

from __future__ import annotations

import math

import torch

from dataset import TextDataset


def bigram_loss(dataset: TextDataset, block_size: int = 64) -> float:
    """Exact validation cross-entropy of the bigram model.

    Counts P(next | current) on the training split, then evaluates
    the negative log-likelihood on every adjacent pair of the
    validation split.
    """
    counts = torch.zeros(dataset.vocab_size, dataset.vocab_size)
    train = dataset.train_data
    for i in range(len(train) - 1):
        counts[train[i], train[i + 1]] += 1
    probs = counts / counts.sum(dim=1, keepdim=True).clamp(min=1.0)

    val = dataset.val_data
    nll = 0.0
    n = 0
    for i in range(len(val) - 1):
        p = probs[val[i], val[i + 1]].item()
        nll += -math.log(max(p, 1e-12))
        n += 1
    return nll / n
