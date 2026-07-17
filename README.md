Bird Intelligence System

<div align="center">

Audio-based Bird Species Classification using Vision Transformers

https://img.shields.io/badge/Python-3.8+-blue.svg
https://img.shields.io/badge/PyTorch-1.12+-red.svg
https://img.shields.io/badge/License-MIT-green.svg

</div>

📋 Overview

The Bird Intelligence System is a deep learning pipeline for classifying bird species from audio recordings. The system converts bird calls into mel spectrograms and processes them through a Vision Transformer (ViT) architecture, treating the spectrograms as images.

The project follows a systematic research approach: first establishing strong supervised baselines, then exploring contrastive self-supervised learning to improve classification performance, especially for species with limited recordings. The end goal is to compare supervised learning, self-supervised contrastive learning, and fine-tuned models to identify the most effective approach for large-scale bird species classification.

Key Features

· 🎵 Audio Processing: Automatic download, preprocessing, and conversion of audio to mel spectrograms
· 🧠 Transformer Architecture: Custom Vision Transformer designed for spectrogram classification
· 🧪 Experiment Framework: Systematic hyperparameter sweeps and experiment tracking
· 📊 Comprehensive Evaluation: Per-class metrics, confusion matrices, and automated reporting
· 🔄 Modular Design: Clean separation between data, model, training, and evaluation components

🏗️ Architecture

```
Bird Audio → Mel Spectrogram → Patch Embedding → Transformer Encoder → Classification Head
```

Model Components

· SpectrogramPatchEmbedding: Converts spectrograms into patch tokens
· AudioTransformerInput: Adds positional encoding and CLS token
· Transformer Encoder: Multi-layer transformer with self-attention
· Classification Head: LayerNorm → Linear → GELU → Dropout → Linear

📁 Project Structure

```
Bird-Intellegence-System/
├── configs/
│   └── config.yaml                 # Main configuration file
├── data/
│   ├── birds_voices.csv            # Dataset metadata
│   ├── raw_audio/                  # Temporary audio storage
│   ├── processed_spectrograms/     # Preprocessed spectrograms (.npy)
│   └── metadata/                   # Processed dataset metadata CSVs
├── src/
│   ├── data/
│   │   ├── download.py             # Audio download utilities
│   │   ├── process_audio.py        # Spectrogram generation & I/O
│   │   ├── dataset.py              # PyTorch Dataset with SpecAugment
│   │   └── run_pipeline.py         # End-to-end data pipeline
│   ├── models/
│   │   ├── patch_embedding.py      # Spectrogram → patches
│   │   ├── positional_encoding.py  # Learnable position embeddings
│   │   ├── self_attention.py       # Multi-head self-attention
│   │   ├── transformer_block.py    # Transformer block with FFN
│   │   ├── audio_transformer_input.py  # Input preprocessing
│   │   ├── encoder.py              # Full transformer encoder
│   │   └── bird_classifier.py      # Complete classification model
│   ├── training/
│   │   ├── train.py                # Standard training loop
│   │   └── experiment_train.py     # Experiment-optimized trainer
│   ├── evaluation/
│   │   ├── evaluate.py             # Model evaluation
│   │   └── metrics_collector.py    # Metrics collection & visualization
│   └── utils/
│       └── configs.py              # Configuration loading & resolution
├── experiments/
│   ├── experiment_runner.py        # Hyperparameter sweep orchestrator
│   └── sweep_configs.py            # Sweep configuration definitions
├── scripts/
│   └── compare_experiments.py      # Compare baseline vs contrastive
├── results/                        # Experiment results & model checkpoints
├── main.py                         # Main orchestration script
└── requirements.txt                # Python dependencies
```

🚀 Quick Start

Prerequisites

· Python 3.8+
· CUDA-capable GPU (recommended for full-scale training)
· 50+ GB disk space for large-scale datasets

Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/Bird-Intellegence-System.git
cd Bird-Intellegence-System

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

Development Mode

For quick experimentation during development, the default config uses 5 species with 50 samples each. This allows for rapid iteration:

```bash
# Run with development settings (5 species × 50 samples)
python main.py --all
```

Production Mode (Coming Soon)

For large-scale training with the full dataset:

```bash
# Will support full BirdCLEF dataset and additional sources
python main.py --all --config configs/config_full.yaml
```

⚙️ Configuration

The system is configured via configs/config.yaml. For development, we use a small subset. For full-scale training, update the data section:

Development Configuration (Current Default)

```yaml
data:
  num_classes: 5
  num_samples_per_class: 50
```

Full-Scale Configuration (Future)

```yaml
data:
  # Remove num_samples_per_class to use all available data
  # Remove num_classes to use all species
  data_sources:
    - "birdclef2023"
    - "birdclef2024" 
    - "xeno_canto"
    - "additional_datasets"
  min_samples_per_class: 20  # Filter species with too few samples
```

📊 Available Sweep Suites

Suite Configurations Description
quick_baseline 5 Learning rate sweep only (for dev)
standard_baseline 45 LR, batch size, dropout sweeps
comprehensive 1,080 Full architecture + augmentation exploration
optimization_focus 18 Optimizer, scheduler, mixed precision tests

Note: The comprehensive suite with 1,080 configurations is intended for final baseline establishment on the full dataset. For development, use quick_baseline or standard_baseline.

📈 Experiment Tracking

Each experiment run generates:

· config.yaml: Exact hyperparameters used
· best_model.pth: Best model checkpoint
· training_metrics.json: Epoch-by-epoch logs
· evaluation_metrics.json: Final test metrics
· confusion_matrix.png: Confusion matrix visualization
· per_class_metrics.png: Per-class performance plots
· results.csv: Aggregate results across all runs

🔬 Current Status

✅ Completed

· Audio download and preprocessing pipeline
· Mel spectrogram generation and storage
· Vision Transformer model implementation
· Standard supervised training loop
· SpecAugment data augmentation
· Hyperparameter sweep framework
· Experiment tracking and logging
· Metrics collection and visualization
· Model evaluation suite
· Experiment comparison tools
· Development pipeline (5 species × 50 samples)

🚧 In Progress

· Fix configuration resolution order in data pipeline
· Add early stopping to prevent overfitting
· First baseline training runs with comprehensive sweeps

🗺️ Roadmap

Phase 1: Baseline Establishment (Current Focus)

Data Pipeline Scaling

· Integrate BirdCLEF Dataset
  · Download BirdCLEF 2023/2024 full datasets
  · Parse BirdCLEF metadata format
  · Handle BirdCLEF's background species labels
  · Create unified metadata format with Xeno-canto data
· Expand Data Sources
  · Integrate Xeno-canto API for additional recordings
  · Add Macaulay Library dataset
  · Support for user-provided custom datasets
  · Data quality filtering and validation
· Dataset Statistics & Balancing
  · Analyze class distribution across all data sources
  · Implement minimum samples per class thresholding
  · Create balanced sampling strategies
  · Handle long-tailed distribution with weighted sampling
  · Document dataset composition (species, samples, recording quality)
· Data Pipeline Optimization
  · Parallel download and processing for large datasets
  · Incremental processing (skip already processed files)
  · Data integrity verification (checksums, file validation)
  · Automatic cleanup of corrupted/incomplete downloads

Model Training & Baseline

· Development Baseline (Current)
  · Complete baseline sweeps on 5-species dev dataset
  · Profile model performance characteristics
  · Establish training time/memory benchmarks
  · Document learning dynamics and convergence patterns
· Full-Scale Baseline Training
  · Scale to full BirdCLEF dataset (hundreds of species)
  · Run comprehensive hyperparameter sweeps
  · Implement distributed training for large-scale experiments
  · Optimize batch size and gradient accumulation for GPU memory
  · Establish strong supervised baseline (target: >80% accuracy)
· Baseline Analysis & Documentation
  · Generate detailed performance reports
  · Analyze per-class and per-family accuracy distributions
  · Identify challenging species and acoustic similarities
  · Document data efficiency (samples per class vs accuracy curve)
  · Error analysis: confusion patterns and failure modes

Infrastructure Improvements

· Experiment Management
  · Fix configuration resolution order in data pipeline
  · Add proper random seed management per experiment
  · Implement early stopping based on validation metrics
  · Add gradient accumulation for larger effective batch sizes
· Performance Optimization
  · Cache preprocessed tensors for faster data loading
  · Profile and optimize data loading pipeline
  · Implement mixed-precision training (AMP)
  · GPU memory optimization for large-scale training

Phase 2: Contrastive Learning Implementation

Self-Supervised Pre-training Framework

· Contrastive Learning Architectures
  · Implement SimCLR-style contrastive learning
  · Add MoCo (Momentum Contrast) with memory bank
  · Implement BYOL (Bootstrap Your Own Latent)
  · Design audio-specific augmentations for contrastive pairs
· Augmentation Strategies for Contrastive Learning
  · Time-domain augmentations (pitch shift, time stretch, noise injection)
  · Frequency-domain augmentations (frequency masking, filtering)
  · Background noise mixing from environmental sounds
  · Study impact of augmentation strength on representation quality
· Pre-training Pipeline
  · Large-scale pre-training on full BirdCLEF + additional data
  · Support for unsupervised pre-training (no labels needed)
  · Multi-GPU distributed contrastive training
  · Representation quality evaluation (linear probing, kNN)

Evaluation Framework for Representations

· Representation Analysis Tools
  · t-SNE/UMAP visualization of learned embeddings
  · Silhouette score and clustering metrics
  · Retrieval evaluation (same-species audio retrieval)
  · Probing tasks for acoustic property understanding

Phase 3: Fine-tuning & Comprehensive Comparison

Fine-tuning Strategies

· Transfer Learning Approaches
  · Linear probing (frozen backbone + trainable classifier)
  · Full fine-tuning with differential learning rates
  · Layer-wise learning rate decay strategies
  · Progressive unfreezing experiments
· Low-Data Regime Evaluation
  · Few-shot learning: 1, 5, 10, 20, 50 samples per class
  · Zero-shot generalization to unseen species
  · Cross-dataset transfer performance
  · Sample efficiency curves for each method

Multi-Model Comparison Matrix

· Models to Compare
  · Supervised baseline (from Phase 1)
  · SimCLR pre-trained + fine-tuned
  · MoCo pre-trained + fine-tuned
  · BYOL pre-trained + fine-tuned
  · Ensemble of best models
· Evaluation Dimensions
  · Overall accuracy and macro F1 score
  · Per-class accuracy distribution
  · Performance vs training data available
  · Robustness to noise and environmental conditions
  · Inference speed and model size
  · Statistical significance testing
· Ablation Studies
  · Pre-training data scale vs downstream performance
  · Model capacity vs pre-training effectiveness
  · Augmentation strength vs representation quality
  · Projection head architecture importance

Final Report & Artifacts

· Comprehensive Comparison Report
  · Detailed methodology documentation
  · Statistical analysis with confidence intervals
  · Recommendations for different use cases
  · Guidelines for practitioners
· Model Release
  · Best model weights on Hugging Face Hub
  · Model cards with performance characteristics
  · Inference examples and API documentation
  · Pre-computed embeddings for the dataset

Phase 4: Production & Advanced Features

Model Optimization

· Inference Optimization
  · Model pruning and quantization (INT8, FP16)
  · ONNX export for cross-platform deployment
  · TensorRT optimization for NVIDIA GPUs
  · TorchScript compilation for production
· Mobile & Edge Deployment
  · Mobile-optimized model variants
  · Edge device benchmarking
  · Offline inference support

Advanced Applications

· Multi-Bird Detection
  · Multi-label classification for multiple species
  · Temporal localization of bird calls
  · Overlapping call separation
· Real-time Processing
  · Streaming audio processing pipeline
  · Sliding window inference with overlap
  · Confidence thresholding and alerting
· Conservation Tools
  · Population monitoring dashboards
  · Migration pattern analysis
  · Endangered species alert system

📊 Timeline & Milestones

Phase Timeline Key Milestone Success Metric
Phase 1a: Dev Baseline Weeks 1-2 Working pipeline on 5 species 85% accuracy on dev set
Phase 1b: Data Scaling Weeks 2-3 Full BirdCLEF integration All data processed and validated
Phase 1c: Full Baseline Weeks 3-5 Comprehensive sweeps complete 75% on full dataset
Phase 2: Contrastive Weeks 5-8 3 contrastive methods working Quality representations (kNN >60%)
Phase 3: Comparison Weeks 8-10 Complete comparison report 10-20% improvement over baseline
Phase 4: Production Weeks 10-12 Deployable models <100ms inference, >70% accuracy

🔧 Technical Debt & Improvements

Immediate Priorities

· Fix configuration resolution order (segment_size dependency)
· Add early stopping with patience parameter
· Implement proper random seed management per experiment
· Add validation for sweep config parameters
· Create comprehensive logging (replace print with logging module)

Medium-term Improvements

· Refactor to use PyTorch Lightning for cleaner training
· Add comprehensive unit tests (>80% coverage)
· Implement data versioning (DVC or similar)
· Add experiment tracking with MLflow or Weights & Biases
· Create API documentation with Sphinx/MkDocs
· Implement CI/CD for automated testing

Long-term Goals

· Benchmark against SOTA: Audio Spectrogram Transformer (AST), HTS-AT, BEATs
· Active learning for efficient data collection
· Federated learning for privacy-preserving multi-institution training
· Integration with citizen science platforms (eBird, Xeno-canto, iNaturalist)

📦 Dependencies

Core

· PyTorch ≥ 1.12
· torchaudio ≥ 0.12
· librosa ≥ 0.9
· NumPy, SciPy, Pandas
· scikit-learn

Experiments & Visualization

· matplotlib, seaborn
· tqdm
· PyYAML

Future

· PyTorch Lightning
· MLflow / Weights & Biases
· ONNX, TensorRT

📝 Citation

If you use this code in your research, please cite:

```bibtex
@software{bird_intelligence_system,
  title = {Bird Intelligence System: Audio-based Species Classification},
  author = {Nkosingiphile Sibandze},
  year = {2026},
  url = {https://github.com/sibandze/Bird-Intellegence-System}
}
```

📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

🤝 Contributing

Contributions are welcome! Areas where help is especially appreciated:

1. Data Integration: Help add support for new bird audio datasets
2. Model Implementations: Contribute new audio model architectures
3. Evaluation Tools: Improve metrics collection and visualization
4. Documentation: Improve tutorials and examples
5. Bug Fixes: Help resolve issues in the issue tracker

Please feel free to submit a Pull Request. For major changes, please open an issue first to discuss what you would like to change.

📧 Contact

For questions or collaboration, please open an issue on GitHub or contact the maintainers.

---

<div align="center">

Building tools for avian conservation through machine learning

🐦 🎵 🧠

</div>
