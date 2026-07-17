# Bird Intelligence System

## Transformer Architecture

**Version:** 1.0
**Project:** Bird Intelligence System

---

# Overview

The Bird Intelligence System uses a **Vision Transformer (ViT)** adapted for audio classification. Rather than processing raw waveforms directly, the model operates on **mel spectrograms**, treating them as two-dimensional images.

This design enables the transformer to learn both temporal and frequency relationships through self-attention, allowing it to capture long-range dependencies that are difficult for conventional convolutional networks to model.

The architecture has been intentionally designed to be modular so that individual components can be replaced or extended without affecting the rest of the system.

---

# Design Goals

The architecture is designed to satisfy the following objectives:

* Learn global acoustic relationships.
* Maintain a modular object-oriented implementation.
* Support future self-supervised learning.
* Scale from development datasets to BirdCLEF.
* Remain compatible with modern transformer research.

---

# High-Level Architecture

```text id="h9y9k9"
Bird Audio
      │
      ▼
Mel Spectrogram
(128 × T)
      │
      ▼
Patch Embedding
      │
      ▼
CLS Token
      │
      ▼
Positional Encoding
      │
      ▼
Transformer Encoder
      │
      ▼
Layer Normalization
      │
      ▼
Classification Head
      │
      ▼
Species Prediction
```

---

# Complete Data Flow

```text id="3q24d0"
Audio Recording

↓

Mel Spectrogram

↓

Patch Embedding

↓

Linear Projection

↓

CLS Token

↓

Position Embedding

↓

Transformer Block 1

↓

Transformer Block 2

↓

...

↓

Transformer Block N

↓

CLS Representation

↓

Classification Head

↓

Species Probabilities
```

---

# Input Representation

The model receives mel spectrograms generated during offline preprocessing.

Each input has the form

```text id="6upfr8"
X ∈ ℝ^(128 × T)
```

where

* 128 = mel frequency bins
* T = patch-aligned number of time frames

Unlike images, spectrograms preserve temporal ordering while encoding frequency information vertically.

---

# Why Mel Spectrograms?

Mel spectrograms provide several advantages:

* Compact representation
* Reduced computational cost
* Human-inspired frequency scaling
* Robustness to recording variability
* Compatibility with vision architectures

They have become the standard representation for modern audio transformers.

---

# Patch Embedding

## Purpose

Transformers require sequences rather than images.

Patch embedding converts the spectrogram into a sequence of learnable vectors.

---

## Process

```text id="wbk0wm"
Mel Spectrogram

↓

Split into Patches

↓

Flatten

↓

Linear Projection

↓

Patch Tokens
```

Each patch becomes one transformer token.

---

## Mathematical Formulation

Given

```text id="c7mjlwm"
X ∈ ℝ^(128 × T)
```

the spectrogram is partitioned into

```text id="fj2kkn"
N patches
```

Each patch is flattened

```text id="0xwm9x"
Patch_i → ℝ^P
```

and projected into the embedding space

```text id="dxjlwm"
Embedding_i = W × Patch_i + b
```

producing

```text id="l71n5q"
Z ∈ ℝ^(N × D)
```

where

* N = number of patches
* D = embedding dimension

---

# CLS Token

A learnable classification token is prepended to every sequence.

```text id="ukm7wa"
CLS

Patch 1

Patch 2

...

Patch N
```

During training, the CLS token aggregates information from all patches through self-attention.

Only this token is passed to the classification head.

---

# Positional Encoding

Transformers do not inherently understand spatial or temporal order.

Learnable positional embeddings are added to every token.

```text id="7cb94o"
Input Token

+

Position Embedding

↓

Transformer Input
```

This allows the model to distinguish

* early versus late bird calls
* low versus high frequencies
* relationships between distant patches

---

# Transformer Encoder

The encoder is composed of multiple identical transformer blocks.

```text id="m1h34o"
Input

↓

Block 1

↓

Block 2

↓

...

↓

Block N
```

Each block processes the entire sequence simultaneously.

---

# Transformer Block

Each block follows the pre-normalization architecture.

```text id="mp2p2h"
Input

↓

LayerNorm

↓

Multi-Head Self Attention

↓

Residual Addition

↓

LayerNorm

↓

Feed Forward Network

↓

Residual Addition

↓

Output
```

This architecture improves optimization stability in deeper transformers.

---

# Multi-Head Self-Attention

Self-attention enables every patch to interact with every other patch.

Instead of processing local neighborhoods like CNNs, attention captures long-range dependencies across the entire spectrogram.

The attention operation computes

```text id="i7a45q"
Attention(Q,K,V)

=

Softmax(QKᵀ / √d)

V
```

where

* Q = Query
* K = Key
* V = Value

Multiple attention heads learn complementary acoustic relationships simultaneously.

---

# Feed Forward Network

Each transformer block contains an independent feed-forward network.

Typical structure

```text id="89gnok"
Linear

↓

GELU

↓

Dropout

↓

Linear
```

This network transforms token representations after self-attention.

---

# Residual Connections

Residual connections are used throughout the encoder.

Advantages

* Easier optimization
* Stable gradients
* Improved convergence
* Better deep network performance

---

# Layer Normalization

Layer normalization is applied before each major sub-layer.

Benefits

* Stable training
* Improved convergence
* Reduced internal covariate shift

---

# Classification Head

Only the CLS token is used for classification.

```text id="g3vw76"
CLS

↓

LayerNorm

↓

Linear

↓

GELU

↓

Dropout

↓

Linear

↓

Softmax
```

The output is a probability distribution across bird species.

---

# Loss Function

The supervised baseline minimizes categorical cross-entropy.

```text id="tr3nv6"
L = CrossEntropy(y, ŷ)
```

Future versions will investigate

* Focal Loss
* Label Smoothing
* Class-Balanced Loss

---

# Training Pipeline

```text id="y9np0j"
Input

↓

Forward Pass

↓

Cross Entropy

↓

Backpropagation

↓

AdamW

↓

Parameter Update
```

Mixed precision training may be enabled for improved performance.

---

# Why Vision Transformers?

Traditional CNNs learn local receptive fields.

Bird vocalizations often contain long-range temporal structures.

Examples include

* repeated phrases
* harmonic relationships
* delayed call patterns
* frequency interactions

Self-attention enables every patch to attend to every other patch, making transformers well suited for spectrogram analysis.

---

# Current Model Components

The implementation consists of the following modules:

```text id="jmxs2x"
patch_embedding.py

↓

audio_transformer_input.py

↓

self_attention.py

↓

transformer_block.py

↓

encoder.py

↓

bird_classifier.py
```

Each module has a single responsibility and can be modified independently.

---

# Planned Architecture Extensions

Future research will investigate

## Alternative Backbones

* Audio Spectrogram Transformer (AST)
* HTS-AT
* BEATs
* AudioMAE
* CNN baselines
* Hybrid CNN-Transformer architectures

---

## Self-Supervised Learning

The encoder has been designed so that the classification head can be removed during pre-training.

Future methods include

* SimCLR
* MoCo
* BYOL
* Masked Autoencoders
* DINO-style self-distillation

The pretrained encoder will later be fine-tuned for supervised bird classification.

---

# Design Decisions

| Decision                        | Rationale                                        |
| ------------------------------- | ------------------------------------------------ |
| Mel spectrogram input           | Standard representation for audio transformers   |
| Vision Transformer backbone     | Captures global temporal-frequency relationships |
| Learnable positional embeddings | Allows the model to encode temporal order        |
| CLS token                       | Standard transformer classification strategy     |
| Pre-normalization blocks        | Improves optimization stability                  |
| Offline preprocessing           | Faster and reproducible training                 |
| Modular implementation          | Easier experimentation and extension             |

---

# Future Improvements

Planned architectural enhancements include

* Relative positional embeddings
* Rotary Position Embeddings (RoPE)
* FlashAttention
* Stochastic depth
* LayerScale
* SwiGLU feed-forward networks
* Token pruning
* Dynamic patch sizes
* Multi-scale spectrogram processing
* Hierarchical transformer architectures

---

# Summary

The Bird Intelligence System adopts a modular Vision Transformer architecture tailored for bird audio classification. By representing bird vocalizations as mel spectrograms and processing them as sequences of image patches, the model can capture both local acoustic patterns and long-range temporal dependencies through self-attention.

The architecture is intentionally designed to serve as a strong supervised baseline while remaining flexible enough to support future research in contrastive learning, self-supervised pre-training, transfer learning, and next-generation audio transformer models.
