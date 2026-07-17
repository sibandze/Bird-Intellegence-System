# READING.md

# Bird Intelligence System

## Reading List & Research References

**Version:** 1.0
**Purpose:** Living reference document for papers, books, articles, blog posts, and educational resources that influenced the design of the Bird Intelligence System.

---

# Introduction

This document serves as the project's knowledge base.

Unlike the other documentation, which explains **what** the system does, this document explains **why** it was designed this way by collecting the research papers, textbooks, articles, blog posts, and talks that inspired the project's architecture and future direction.

This is a living document and should evolve as the project grows.

Whenever an important design decision is made, the supporting literature should be added here.

---

# How to Use This Document

Each resource should include:

* Citation
* Link
* Summary
* Key ideas
* Why it matters
* Where it influenced the project

Example:

---

## Resource

**Paper**

Author(s)

Publication

Link

### Summary

...

### Key Contributions

...

### Used In

* Data preprocessing
* Architecture
* Training
* Future work

---

# Reading Categories

---

# 1. Bird Audio Classification

## Goal

Understand the current state of bird sound recognition.

### Topics

* Bird sound classification
* Bioacoustics
* Environmental sound recognition
* Acoustic ecology

### Papers

* BirdCLEF challenge papers
* BirdNET
* Perch
* Merlin Sound ID
* Ecoacoustics surveys

---

# 2. Audio Deep Learning

## Goal

Understand modern audio representations.

Topics include

* Spectrogram learning
* Raw waveform learning
* Audio transformers
* Audio embeddings

Important papers

* AST
* HTS-AT
* BEATs
* AudioMAE
* PANNs

---

# 3. Vision Transformers

## Goal

Understand why transformers work on images.

Important papers

* Vision Transformer (ViT)
* DeiT
* Swin Transformer
* ConvNeXt
* MViT

Topics

* Patch embeddings
* Positional encoding
* Self-attention
* Hierarchical transformers

---

# 4. Self-Supervised Learning

## Goal

Understand representation learning without labels.

Core papers

* SimCLR
* MoCo
* BYOL
* DINO
* Barlow Twins
* VICReg
* SimSiam

Topics

* Contrastive learning
* Non-contrastive learning
* Representation collapse
* Projection heads

---

# 5. Audio Self-Supervised Learning

Important papers

* wav2vec 2.0
* HuBERT
* WavLM
* BEATs
* AudioMAE

Topics

* Audio masking
* Audio pretraining
* Fine tuning
* Transfer learning

---

# 6. Machine Learning Engineering

References for building production ML systems.

Topics

* reproducibility
* experiment tracking
* configuration management
* model deployment
* software engineering

Recommended books

* Designing Machine Learning Systems
* Machine Learning Engineering
* Building Machine Learning Powered Applications

---

# 7. Software Architecture

Resources that influenced the code structure.

Topics

* object-oriented design
* SOLID principles
* clean architecture
* dependency injection
* modular software

Recommended books

* Clean Architecture
* Clean Code
* Design Patterns
* Refactoring

---

# 8. PyTorch

Resources related to implementation.

Topics

* Dataset design
* DataLoader
* AMP
* Distributed Training
* TorchScript

Useful references

* Official PyTorch Documentation
* PyTorch Tutorials
* TorchVision examples

---

# 9. Mathematics

Resources for understanding the underlying mathematics.

Topics

* Linear algebra
* Probability
* Statistics
* Optimization
* Information theory

References

* Mathematics for Machine Learning
* Deep Learning (Goodfellow)
* Pattern Recognition and Machine Learning

---

# 10. Audio Processing

Topics

* FFT
* STFT
* Mel scale
* Spectrograms
* Signal processing

Recommended references

* librosa documentation
* DSP Guide
* Fundamentals of Digital Signal Processing

---

# 11. Optimization

Topics

* AdamW
* SGD
* Learning rate scheduling
* Gradient clipping
* Mixed precision

References

* Decoupled Weight Decay Regularization
* Attention Is All You Need
* PyTorch optimization tutorials

---

# 12. Evaluation

Topics

* Precision
* Recall
* F1
* Confusion matrices
* Statistical testing
* Calibration

References

* scikit-learn documentation
* Statistical Learning references

---

# 13. Experiment Management

Topics

* MLflow
* Weights & Biases
* DVC
* Sacred
* Hydra

Purpose

Design reproducible machine learning experiments.

---

# 14. Bioacoustics

Topics

* Bird ecology
* Bird vocalizations
* Acoustic monitoring
* Conservation

Purpose

Better understand the biological domain rather than only the machine learning.

---

# 15. Future Reading

Interesting topics to investigate.

* Foundation models
* Retrieval
* Multimodal learning
* Audio-language models
* Continual learning
* Active learning
* Open-set recognition
* Few-shot learning

---

# Paper Notes Template

Use the following template whenever reading a paper.

```markdown
# Paper

Title:

Authors:

Conference:

Year:

Link:

---

## Problem

What problem does the paper solve?

---

## Main Idea

Summarize the approach.

---

## Architecture

Describe the model.

---

## Training

How was it trained?

---

## Results

Important benchmarks.

---

## Strengths

-

---

## Weaknesses

-

---

## Ideas for Bird Intelligence System

-

---

## Questions

-

---

## Personal Notes

-
```

---

# Blog Notes Template

```markdown
# Article

Title:

Author:

Link:

---

## Summary

-

---

## Interesting Ideas

-

---

## Things to Try

-

---

## Impact on Project

-
```

---

# Design Decision Log

Whenever a significant design decision is made, record it here.

| Decision              | Reason                          | Supporting Resource    |
| --------------------- | ------------------------------- | ---------------------- |
| Vision Transformer    | Better global context           | ViT paper              |
| Mel Spectrograms      | Standard audio representation   | AST paper              |
| Offline preprocessing | Faster, reproducible training   | PyTorch best practices |
| AdamW optimizer       | Better transformer optimization | Decoupled Weight Decay |

This log helps connect implementation decisions to supporting evidence.

---

# Personal Learning Roadmap

This project is also a learning journey.

Suggested progression:

1. Digital Signal Processing
2. Audio Feature Extraction
3. Vision Transformers
4. Attention Mechanisms
5. Self-Supervised Learning
6. Audio Foundation Models
7. BirdCLEF literature
8. Production ML Systems

As understanding improves, revisit earlier papers to gain deeper insights.

---

# Final Notes

This document is intentionally **not** a bibliography. It is a curated research journal that captures the knowledge influencing the Bird Intelligence System.

Every major architectural decision should be traceable to one or more references, and every important paper should include personal notes describing what was learned, what ideas were adopted, and what future experiments it inspired.

Over time, this document should become both a reference library and a record of the project's intellectual evolution, making it easier to justify design decisions, reproduce research directions, and onboard future contributors.
