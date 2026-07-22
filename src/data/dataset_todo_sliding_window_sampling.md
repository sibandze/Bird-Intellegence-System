# TODO: Implement Epoch-Based Sliding Window Sampling Strategy

## Motivation

The current training pipeline randomly selects a crop from each spectrogram every epoch.

```text
Epoch 1
------------------------------
      [ Random Crop ]

Epoch 2
------------------------------
                [ Random Crop ]

Epoch 3
------------------------------
 [ Random Crop ]
```

While this introduces useful randomness, it has a significant limitation:

* Long recordings may never be fully observed.
* Informative bird vocalizations occurring late in recordings can be missed entirely.
* Training coverage depends on chance rather than guaranteeing exposure to the full recording.

For BirdCLEF recordings that can span several minutes, the model currently only sees a small fraction of each recording during training.

---

# Objective

Replace purely random cropping with an **epoch-aware sliding window sampling strategy** that gradually exposes the model to the entire recording over multiple epochs.

This should improve temporal coverage while preserving compatibility with the existing training pipeline.

---

# Design Goals

* Maintain the same input tensor shape expected by the model.
* Guarantee that long recordings are fully covered over time.
* Keep the implementation modular so different sampling strategies can be compared experimentally.
* Preserve the ability to use random sampling as the baseline.

---

# Phase 1 — Introduce Sampling Strategy Configuration

Add a configurable sampling section to the experiment configuration.

Example:

```yaml
sampling:
    strategy: random
```

Supported strategies should eventually include:

* random
* center
* sliding
* sliding_jitter (future)
* energy_based (future)
* birdcall_based (future)

The dataset should never hardcode a particular strategy.

---

# Phase 2 — Make the Dataset Epoch-Aware

Currently the dataset has no knowledge of which epoch is being trained.

Introduce a lightweight mechanism for tracking the current epoch.

Example interface:

```python
dataset.set_epoch(epoch)
```

The trainer should update the dataset at the beginning of every epoch.

Example flow:

```text
Trainer
    ↓
set_epoch(epoch)
    ↓
Dataset
    ↓
_crop_or_pad()
```

This enables deterministic sampling policies.

---

# Phase 3 — Implement Sliding Window Logic

Instead of

```python
random.randint(...)
```

compute the crop position from the current epoch.

Conceptually:

```text
Epoch 1

|------window------|

Epoch 2

      |------window------|

Epoch 3

            |------window------|

Epoch 4

                  |------window------|
```

The crop position should advance by a configurable stride.

---

# Phase 4 — Add Configurable Stride

Stride should be configurable rather than hardcoded.

Example:

```yaml
sampling:
    strategy: sliding
    stride: 256
```

Guidelines:

* stride < segment_size
* overlapping windows are preferred
* allow experimentation with different overlap amounts

---

# Phase 5 — Handle End-of-Recording

Decide how sampling behaves when the sliding window reaches the end of a recording.

Candidate strategies:

### Clamp

```text
|--------------audio--------------|
                     |------window------|
```

Window remains at the final valid position.

---

### Wrap Around (Future)

```text
|--------------audio--------------|
                     |----|
|----|
```

Window restarts from the beginning.

---

### Bounce (Future)

```text
→→→→→
←←←←←
```

Window alternates forward and backward.

Initially implement the simplest and most deterministic option (clamping).

---

# Phase 6 — Preserve Existing Behaviour

Random sampling should remain available.

This allows fair comparisons between:

* Random crop (baseline)
* Sliding window
* Future sampling methods

Avoid replacing the baseline implementation entirely.

---

# Phase 7 — Future Sliding + Jitter

After establishing the sliding baseline, investigate adding a small random offset.

Concept:

```text
Base position
      ↓

      [---------]

Random offset
        +12 frames
```

Benefits:

* Maintains full recording coverage.
* Introduces slight variation.
* Prevents the model from repeatedly seeing identical crop boundaries.

This should be implemented only after evaluating deterministic sliding.

---

# Phase 8 — Experimental Evaluation

Run controlled experiments comparing:

## Experiment A

Random cropping (current baseline)

---

## Experiment B

Sliding window

---

## Experiment C

Sliding window + jitter

---

Measure:

* Validation accuracy
* Validation loss
* Macro F1
* Per-class accuracy
* Confusion matrix
* Training speed
* GPU utilization
* Memory usage

---

# Phase 9 — Documentation

Update the project documentation to describe:

* Why random sampling was insufficient.
* The motivation for temporal coverage.
* The sliding window algorithm.
* Configuration options.
* Experimental comparisons between sampling strategies.

---

# Expected Benefits

* Every recording is eventually observed in full.
* Better utilization of long BirdCLEF recordings.
* Reduced probability of missing informative vocalizations.
* More reproducible sampling compared to purely random crops.
* Flexible infrastructure for future research on temporal sampling strategies.
