______________________________________________________________________

# GDyNet-Ferro - A Graph Dynamical Neural Network Approach for Decoding Dynamical States in Ferroelectrics.

[![PyTorch](https://img.shields.io/badge/PyTorch-ee4c2c?logo=pytorch&logoColor=white)](https://pytorch.org/get-started/locally/) [![PyG](https://img.shields.io/badge/PyG-3C2179?logo=pyg&logoColor=white)](https://www.pyg.org/) [![Python 3.8+](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/downloads/) [![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT) [![Paper](https://img.shields.io/badge/Paper-Carbon%20Trends-orange)](https://doi.org/10.1016/j.cartre.2023.100264)

*Scalable PyTorch implementation of **GDyNet-ferro** - Graph Dynamical Networks with VAMP loss for analyzing molecular dynamics trajectories*

[Features](#features) •
[Installation](#installation) •
[Quick Start](#quick-start) •
[Documentation](https://abhijeetdhakane.github.io/gdy_pl) •
[Citation](#citation)

---

## Overview

GDyNet-ferro is a graph neural network framework for identifying slow dynamical features and hidden states in molecular dynamics simulations. This implementation uses the **Variational Approach for Markov Processes (VAMP)** to learn meaningful collective variables from atomistic trajectories.

**Key Applications**:
- Ferroelectric materials dynamics
- Phase transition analysis
- Reaction coordinate identification
- Coarse-graining for MD data

**This repository** provides a production-ready, optimized implementation with:
- PyTorch 2.0+ support with `torch.compile`
- Distributed training and inference (DDP) on HPC clusters (NERSC Perlmutter, OLCF Frontier (Summit))
- Robust checkpoint/resume functionality
- Comprehensive metrics tracking
- Easy configuration via YAML files
- Post-processing tools for Koopman analysis

---

## Features

### Two Model Variants

| Model | Description | Use Case |
|-------|-------------|----------|
| **gdynet_vanilla** | Standard GDyNet without direction features | General molecular systems |
| **gdynet_ferro** | Enhanced with atom direction features | Ferroelectric materials, polarization |



### Post-processing & Analysis

- **Koopman Operator Analysis**: Relaxation timescales, eigenvalue decomposition
- **Chapman-Kolmogorov Tests**: Validate Markovian dynamics
- **3D Visualization**: State probability distributions in real space
- **Jupyter Notebooks**: Ready-to-use analysis examples

---

## Installation

### Prerequisites

- Python >= 3.8
- PyTorch >= 2.0.0 with CUDA support
- Git

### Quick Install

```bash
# Clone repository
git clone https://github.com/abhijeetdhakane/gdy_pl.git
cd gdy_pl

# Create conda environment (recommended)
conda create -n gdynet python=3.10  # Or 3.8+
conda activate gdynet

# Install PyTorch with CUDA support
# For CUDA 11.8:
pip install torch --index-url https://download.pytorch.org/whl/cu118

# For CUDA 12.1:
pip install torch --index-url https://download.pytorch.org/whl/cu121

# Install PyTorch Geometric
pip install torch-geometric torch-scatter
```
**OR**
```bash
# Option 1: Install with pip (editable mode, recommended)
pip install -e .

# Option 2: Install with setup.py
python setup.py install

# Option 3: Install dependencies only (no package install)
pip install -r requirements.txt
```

### Install with Optional Dependencies

```bash
# Install with development tools (testing, linting)
pip install -e ".[dev]"

# Install with documentation tools
pip install -e ".[docs]"

# Install with Weights & Biases support
pip install -e ".[wandb]"

# Install everything
pip install -e ".[all]"
```

### Dependencies

#### Core requirements (automatically installed)
- torch >= 2.0.0
- torch-geometric >= 2.3.0
- torch-scatter >= 2.1.0
- numpy >= 1.21.0
- pyyaml >= 6.0
- matplotlib >= 3.5.0
- seaborn >= 0.12.0

### HPC Setup

Please refer to the official documentation for configuring Python and PyTorch on the respective HPC systems:

- **NERSC Perlmutter** – [Python on Perlmutter](https://docs.nersc.gov/development/languages/python/using-python-perlmutter/)
- **OLCF Frontier** – [PyTorch on Frontier](https://docs.olcf.ornl.gov/software/analytics/pytorch_frontier.html)

An example batch submission script for **OLCF Frontier** is provided here:  
- [Example SLURM submission script](/frontier/example_submit.sbatch)




---

## Project Structure

```
gdy_pl/
├── asset/                     # Source figures
├── config/                    # Model registry and configuration
│   ├── __init__.py
│   └── registry.py            # Dynamic model-dataset pairing
├── configs/                   # YAML configuration files
│   ├── gdynet_vanilla.yaml
│   └── gdynet_ferro.yaml
├── data/                      # Data loading utilities
│   ├── __init__.py
│   └── gdynet_dataloader.py   # PyTorch Geometric dataloaders
├── docs/                      # Documentation (MkDocs)
│   ├── asset/                 # Docs images
│   ├── index.md
│   ├── readme.md              # Includes README.md
│   └── training-guide.md      # Includes TRAINING_GUIDE.md
├── frontier/                  # OLCF Frontier HPC job scripts
│   ├── example_submit.sbatch  # Example SLURM submission script
│   └── export_DDP_var.sh      # DDP environment variable helper
├── loss/                      # Loss functions
│   ├── __init__.py
│   └── vamploss.py            # VAMP1, VAMP2, VAMP_sym implementations
├── models/                    # Neural network architectures
│   ├── __init__.py
│   ├── gdynet_vanilla.py      # Standard GDyNet
│   └── gdynet_ferro.py        # GDyNet with direction features
├── notebooks/                 # Jupyter notebooks for analysis
│   ├── README.md
│   ├── analysis_example.ipynb
│   ├── tensorflow/            # TensorFlow examples (legacy)
│   └── torch/                 # PyTorch-specific examples
├── output/                    # Training outputs (generated)
├── postprocess/               # Post-processing and analysis
│   ├── __init__.py
│   ├── postprocess.py         # GDYNetAnalyzer class
│   └── koopman_postprocess.py # Koopman analysis, timescales, CK tests
├── utils/                     # Utility functions
│   ├── __init__.py
│   └── sampler.py             # Custom samplers for distributed inference
├── trainer.py                 # Main training script
├── pyproject.toml             # Package configuration
├── setup.py                   # Legacy setup script
├── requirements.txt           # Dependencies
├── mkdocs.yml                 # Documentation configuration
├── CHANGELOG.md               # Version history
├── CITATION.cff               # Citation metadata
├── LICENSE                    # MIT License
├── README.md                  # This file
└── TRAINING_GUIDE.md          # Detailed training guide
```

---

## Quick Start 
Follow [Training Guide](https://github.com/abhijeetdhakane/gdy_pl/blob/main/TRAINING_GUIDE.md) - Comprehensive training instructions

### 1. Prepare the Dataset from MD Trajectory

GDyNet-ferro uses a graph data structure similar to the original GDyNet implementation. The preprocessing can be performed using [ASE](https://wiki.fysik.dtu.dk/ase/), [MDTraj](https://www.mdtraj.org/), or similar libraries. For constructing `atom_directions` (*Local Polarization*) , please refer to the [paper](https://doi.org/10.1016/j.cartre.2023.100264) and accompanying code.

Save each array as a separate `.npy` file to avoid out-of-memory errors when loading large trajectories:

| File | Shape | Type | Description |
|------|-------|------|-------------|
| `traj_coords.npy` | (F, N, 3) | float | Cartesian coordinates of each atom in each frame |
| `atom_directions.npy` | (F, N, 3) | float | Local polarization vectors centered at Ti (zero for other atoms). **Required for gdynet_ferro only.** |
| `nbr_lists.npy` | (F, N, M) | int | M neighbor indices for each atom per frame, considering periodic boundary conditions |
| `nbr_dists.npy` | (F, N, M) | float | Distances to M neighbors for each atom per frame, considering periodic boundary conditions |
| `atom_types.npy` | (N,) | int | Atomic number of each atom in the simulation |
| `target_index.npy` | (n,) | int | 0-based indices of target atoms (n <= N). For BaTiO3, these are Ti atoms. |

Where:
- **F** = number of frames in the trajectory
- **N** = total number of atoms in the simulation
- **M** = number of neighbors per atom
- **n** = number of target atoms

**Example: Creating dataset files**
```python
import numpy as np

# After preprocessing your MD trajectory...
# Save each array separately to avoid OOM errors
np.save('train_traj_coords.npy', traj_coords)      # (F, N, 3)
np.save('train_atom_directions.npy', directions)   # (F, N, 3) - only for ferro model
np.save('train_nbr_lists.npy', neighbor_indices)   # (F, N, M)
np.save('train_nbr_dists.npy', neighbor_distances) # (F, N, M)
np.save('train_atom_types.npy', atomic_numbers)    # (N,)
np.save('train_target_index.npy', target_atoms)    # (n,)
```

### 2. Configure Training

Copy and edit a sample configuration:

```bash
cp configs/gdynet_vanilla.yaml configs/my_experiment.yaml
```

Update data paths in your config:
```yaml
data:
  train_fnames:
    - /path/to/train_atom_types.npy
    - /path/to/train_target_index.npy
    - /path/to/train_nbr_lists.npy
    - /path/to/train_nbr_dists.npy
  val_fnames:
  ....

  test_fnames:
  ....

model:
  tau: 10                # Time lag for pairs
  batch_size: 32
  state_len: 10          # Number of output states
  learning_rate: 0.001

training:
  epochs: 30
  loss_schedule: ['vamp2', 'vamp1', 'vamp2']  # 90 epochs total
```

### 3. Train

**Single GPU**:
```bash
python trainer.py --config configs/my_experiment.yaml --mode train
```

**Multi-GPU (DDP)**:
```bash
torchrun --nproc_per_node=4 trainer.py --config configs/my_experiment.yaml --mode train
```

**OLCF-Frontier**: 
```bash
# Training and evaluation commands
train_cmd="python trainer.py \
  --config configs/gdynet_ferro.yaml \
  --mode train"

eval_cmd="python trainer.py \
  --config configs/gdynet_ferro.yaml \
  --mode evaluate"

echo "Job started at $(date)"
echo "Step 1: TRAIN"
echo "-------------------------------------------"

# ---- SRUN: TRAIN ----
srun -l bash -lc "
    # Export standard DDP environment variables
    source frontier/export_DDP_var.sh

    echo 'Starting TRAIN at ' \$(date)
    ${train_cmd}
    echo 'TRAIN finished with status: ' \$? ' at ' \$(date)
"
```

**HPC (SLURM)**:
```bash
sbatch frontier/example_submit.sbatch
```

### 4. Resume Training

```bash
python trainer.py \
    --config configs/my_experiment.yaml \
    --mode train \
    --resume output/gdynet_vanilla/checkpoints/checkpoint_latest.pth
```

### 5. Evaluate

```bash
python trainer.py --config configs/my_experiment.yaml --mode evaluate
```

### 6. Analyze Results

Open the analysis notebook:
```bash
jupyter notebook notebooks/analysis_example.ipynb
```

Or use the post-processing API:
```python
from postprocess import GDYNetAnalyzer, KoopmanAnalysis
from postprocess.koopman_postprocess import plot_timescales, plot_ck_tests

# Load predictions
analyzer = GDYNetAnalyzer(
    predictions_path='output/gdynet_vanilla/predictions/gdynet_vanilla_predictions.npy',
    metrics_path='output/gdynet_vanilla/metrics/metrics.json'
)

# Koopman analysis
preds = analyzer.predictions
plot_timescales(preds, lags=range(1, 500, 10), time_unit_in_ns=1e-4)
plot_ck_tests(preds, tau_msm=100, steps=5, time_unit_in_ns=1e-4)
```

---

## Output Structure

```
output/my_experiment/
├── checkpoints/
│   ├── checkpoint_epoch_0030.pth
│   ├── checkpoint_latest.pth
│   └── checkpoint_best.pth
├── metrics/
│   ├── metrics.json              # All metrics (JSON)
│   ├── train_losses_avg.npy
│   ├── val_losses_avg.npy
│   └── ...
├── final_models/
│   ├── gdynet_vanilla_lobe_0_final.pth
│   └── gdynet_vanilla_lobe_t_final.pth
├── predictions/
│   └── gdynet_vanilla_predictions.npy
└── hyperparameters.yml
```

---

## Configuration Options

### Loss Schedule

```yaml
# Single phase (30 epochs)
loss_schedule: ['vamp2']

# Default multi-phase (90 epochs)
loss_schedule: ['vamp2', 'vamp1', 'vamp2']

# Extended training (120 epochs)
loss_schedule: ['vamp2', 'vamp1', 'vamp2', 'vamp1']
```

### Performance Optimization

```yaml
optimization:
  mixed_precision: true     # Enable AMP (2-3x speedup)
  torch_compile: true       # Enable compilation (1.2x speedup)
  compile_mode: 'default'   # 'default', 'reduce-overhead', 'max-autotune'
```

### Checkpointing

```yaml
checkpointing:
  frequency: 1              # Save every N epochs
  save_best_only: false     # Only save when validation improves
```

---

## Citation

If you use this code in your research, please cite our paper:

```bibtex
@article{dhakane2023graph,
  title={A Graph Dynamical Neural Network Approach for Decoding Dynamical States in Ferroelectrics},
  author={Dhakane, Abhijeet and Xie, Tian and Yilmaz, Dundar and van Duin, Adri and Sumpter, Bobby G and Ganesh, P},
  journal={Carbon Trends},
  volume={11},
  pages={100264},
  year={2023},
  publisher={Elsevier},
  doi={10.1016/j.cartre.2023.100264}
}
```

**Paper**: [A Graph Dynamical Neural Network Approach for Decoding Dynamical States in Ferroelectrics](https://doi.org/10.1016/j.cartre.2023.100264)

---

## Related Work

- **Original GDyNet**: [Xie et al., Nature Communications 2019](https://www.nature.com/articles/s41467-019-10663-6)
- **VAMPnets**: [Mardt et al., Nature Communications 2018](https://www.nature.com/articles/s41467-017-02388-1)
- **deeptime**: [deeptime-ml/deeptime](https://github.com/deeptime-ml/deeptime)

---

## Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Run pre-commit hooks (`pre-commit install && pre-commit run --all-files`)
4. Commit your changes (`git commit -m 'Add amazing feature'`)
5. Push to the branch (`git push origin feature/amazing-feature`)
6. Open a Pull Request

### Development Setup

```bash
# Install with dev dependencies
pip install -e ".[dev]"

# Set up pre-commit hooks
pre-commit install

# Format code
black .
isort .
ruff check .
```

---

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## Acknowledgments

- **Oak Ridge National Laboratory**: Research support
- **NERSC & OLCF**: Computational resources on Perlmutter and Frontier (Summit)
- **Original GDyNet Authors**: Tian Xie and collaborators
- **deeptime-ml Team**: VAMP loss implementations

---

## Contact

- **Author**: Abhijeet Dhakane
- **Email**: adhakane@vols.utk.edu
- **Issues**: [GitHub Issues](https://github.com/abhijeetdhakane/gdy_pl/issues)

---

**If you find this work useful, please consider giving it a star!**

[Report Bug](https://github.com/abhijeetdhakane/gdy_pl/issues) • [Request Feature](https://github.com/abhijeetdhakane/gdy_pl/issues)
