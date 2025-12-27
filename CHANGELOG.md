# Changelog

All notable changes to GDyNet-Ferro will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [2.0.0] - 2025-12-26

### Added
- PyTorch implementation (migrated from TensorFlow)
- Support for `torch.compile` (PyTorch 2.0+)
- Mixed precision training with GradScaler
- Distributed training support (DDP) for multi-GPU and multi-node
- Two model variants: `gdynet_vanilla` and `gdynet_ferro`
- Model registry for dynamic model-dataset pairing
- Comprehensive checkpointing with resume capability
- Both last-batch and average metrics tracking
- Koopman operator post-processing analysis
- Chapman-Kolmogorov validation tests
- Implied timescales visualization
- YAML-based configuration system
- WandB integration for experiment tracking
- Pre-commit hooks for code quality

### Changed
- Complete rewrite from TensorFlow to PyTorch
- PyTorch Geometric for graph operations
- Modern package structure with pyproject.toml
- Comprehensive documentation (README, TRAINING_GUIDE)

### Fixed
- Memory-efficient data loading with mmap for large trajectories
- Numerical stability improvements in VAMP loss computation

## [1.0.0] - 2023-05-01

### Added
- Initial release accompanying the Carbon Trends 2023 paper
- TensorFlow implementation of GDyNet-Ferro
- VAMP loss functions (VAMP1, VAMP2, VAMP_sym)
- Training and evaluation scripts for ferroelectric systems
