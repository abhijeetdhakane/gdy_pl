# GDyNet-ferro Training Guide

Comprehensive guide for training GDyNet-ferro models on molecular dynamics data.

## Table of Contents
1. [Understanding the Training Process](#understanding-the-training-process)
2. [Preparing Your Data](#preparing-your-data)
3. [Configuration](#configuration)
4. [Running Training](#running-training)
5. [Monitoring and Checkpoints](#monitoring-and-checkpoints)
6. [Resuming Training](#resuming-training)
7. [Best Practices](#best-practices)
8. [Troubleshooting](#troubleshooting)

---

## Understanding the Training Process

### VAMP Loss and Multi-Phase Training

GDyNet-ferro uses VAMP (Variational Approach for Markov Processes) loss to learn slow dynamical features from molecular dynamics trajectories.

#### Loss Types

- **VAMP2**: Based on Frobenius norm
    - More stable, commonly used for initial training
    - Loss value: negative eigenvalue sum
- **VAMP1** (VAMP_sym): Based on nuclear norm
    - Uses custom gradients for symmetric VAMP
    - Can refine features learned by VAMP2

#### Multi-Phase Training Schedule

The training cycles through different loss types:

```python
loss_schedule = ['vamp2', 'vamp1', 'vamp2']
epochs = 30
# Total training: 3 × 30 = 90 epochs
```

**Phase Breakdown**:
- **Phase 0 (Epochs 0-29)**: VAMP2 - Initial feature learning
- **Phase 1 (Epochs 30-59)**: VAMP1 - Feature refinement
- **Phase 2 (Epochs 60-89)**: VAMP2 - Final optimization

**Global Epoch Calculation**:
```python
global_epoch = phase_index * epochs_per_phase + epoch_in_phase
```

Example: Phase 1, Epoch 15 → Global Epoch 45

#### Why Multi-Phase?
- Different loss types emphasize different aspects of dynamics
- Cycling helps avoid local minima
- Provides robustness in learned representations

#### Hyperparameters for VAMP losses
- **`epsilon`** – Numerical stability parameter used during matrix inversion and eigenvalue computations. For stable and reliable training, this value should typically be set **below `1e-5`**.
- **`mode`** – Controls how eigenvalues are handled during **VAMP loss** computation to ensure numerical stability. The available options are:
    - **`trunc`** – Truncates eigenvalues below `epsilon`, effectively discarding near-zero modes that can cause numerical instabilities.
    - **`regularize`** – Adds `epsilon`-level regularization to the covariance matrices before eigenvalue decomposition, improving conditioning while preserving all modes.
    - **`clamp`** – Clamps eigenvalues to a minimum value of `epsilon`, preventing singularities without fully removing low-energy modes.


---

## Preparing Your Data

### Data Format Requirements

#### Vanilla Model (4 required files)

```python
# Directory structure
data/
  train/
    atom_types.npy       # (N,) int32 - atomic numbers
    target_index.npy     # (n,) int32 - indices of target atoms
    nbr_lists.npy        # (F, N, n_nbrs) int32 - neighbor indices
    nbr_dists.npy        # (F, N, n_nbrs) float32 - neighbor distances
```

#### Ferro Model (5 required files)

All vanilla files PLUS:
```python
    atom_directions.npy  # (F, N, 3) float32 - atom direction vectors
```

### Data Validation Checklist

The dataloader automatically validates:

1. **File Existence**:
   ```
   ✓ All 4 (or 5) required files exist
   ✗ Missing files cause immediate FileNotFoundError
   ```

2. **Target Index Validity**:
   ```python
   # All values in target_index must satisfy:
   0 <= target_index[i] < N (number of atoms)
   ```
   ```
   ✓ Valid: target_index = [0, 5, 10, 15] with N=100
   ✗ Invalid: target_index = [0, 5, 100, 150] with N=100
   ```

3. **Shape Consistency**:
   ```python
   # Must match:
   nbr_lists.shape == nbr_dists.shape  # (F, N, n_nbrs)

   # For ferro model:
   atom_directions.shape == (F, N, 3)
   ```

4. **Duplicate Warning**:
   - Warns if target_index contains duplicates
   - May be intentional, so only warns (doesn't error)

### Preparing Your Data

```python
import numpy as np

# Example: Create target_index for a system with 100 atoms
# Select atoms 0, 10, 20, ..., 90 as targets
target_index = np.arange(0, 100, 10)  # [0, 10, 20, ..., 90]
np.save('target_index.npy', target_index)

# Verify
print(f"Number of targets: {len(target_index)}")
print(f"Max index: {target_index.max()}")  # Must be < 100
```

---

## Configuration

### Basic Configuration

```yaml
# configs/my_experiment.yaml

# Model selection (automatically pairs with correct dataset)
model_type: 'gdynet_vanilla'  # or 'gdynet_ferro'

# Architecture
model:
  atom_fea_len: 64      # Embedding dimension
  n_conv: 3             # Number of graph conv layers
  state_len: 10         # Output dimension
  cutoff: 6.0           # Neighbor cutoff (Angstroms)
  num_gaussians: 50     # Distance encoding resolution
  tau: 10               # Time lag for dynamics
  epsilon: 1.0e-10      # Numerical stability
  batch_size: 32        # Batch size
  learning_rate: 0.001  # Adam learning rate

# Data paths
data:
  train_fnames: ['/path/to/train']
  val_fnames: ['/path/to/val']
  test_fnames: ['/path/to/test']

# Training schedule
training:
  epochs: 30                                  # Per loss type
  loss_schedule: ['vamp2', 'vamp1', 'vamp2']  # 90 total epochs
  seed: 1234

# Optimization (all optional)
optimization:
  torch_compile: false
  mixed_precision: false
  torchscript: false

# Checkpointing
checkpointing:
  frequency: 1              # Save every N epochs
  save_best_only: false     # Save all or only best

# Output
output:
  folder: './output/my_experiment'
```

### Advanced Configuration

#### Custom Loss Schedules

```yaml
# Short training (single loss, 30 epochs)
loss_schedule: ['vamp2']

# Balanced training (60 epochs)
loss_schedule: ['vamp1', 'vamp2']

# Extended training (120 epochs)
loss_schedule: ['vamp2', 'vamp1', 'vamp2', 'vamp1']

# Symmetri (180 epochs)
loss_schedule: ['vamp2', 'vamp1', 'vamp2', 'vamp1', 'vamp2', 'vamp1']
```

#### Performance Optimization

```yaml
optimization:
  # Enable for 2-3x speedup on A100 GPUs
  mixed_precision: true

  # Enable for additional 10-20% speedup (PyTorch 2.0+)
  torch_compile: true
  compile_mode: 'default'  # Options:
                           # - 'default': balanced
                           # - 'reduce-overhead': minimize per-op overhead
                           # - 'max-autotune': longest compile, best runtime

  # Export optimized model after training
  torchscript: true
```

**Performance Impact**:
- Mixed precision alone: ~2.5x faster
- torch.compile alone: ~1.2x faster
- Combined: ~3x faster

**Note**: First epoch with `torch.compile` is slower (compilation time).

#### WandB Integration

```yaml
wandb:
  enabled: true
  project: 'gdynet-production'
  entity: 'your_team'
  run_name: 'vanilla_batch32_lr0.001'
```

---

## Running Training

### Local Training (Development)

```bash
# Activate environment
conda activate <conda-env>

# Single GPU
python trainer.py \
    --config configs/my_experiment.yaml \
    --mode train

# Custom output folder
python trainer.py \
    --config configs/my_experiment.yaml \
    --mode train \
    --output ./custom_output_dir
```

### NERSC Perlmutter (Production)

#### 1. Prepare Batch Script

Edit `frontier/example_submit.sbatch`:

```bash
#!/bin/bash
#SBATCH -A <PROJECT NAME>
#SBATCH -J <JOB NAME>>
#SBATCH -o logs/<LOGSNAME>-%j.out
#SBATCH -e logs/<LOGSNAME>-%j.err
#SBATCH -t 02:00:00
#SBATCH -p batch
#SBATCH -N 2
#SBATCH --ntasks-per-node=8
#SBATCH --gpus-per-node=8
#SBATCH --cpus-per-task=7
#SBATCH --exclusive
#SBATCH --mem=0

# Update paths
cmd="python trainer.py --config configs/my_experiment.yaml --mode train"
```

#### 2. Submit Job

```bash
# Make sure you're in the repository root
cd /path/to/gdy_pl

# Create logs directory
mkdir -p logs

# Submit
sbatch frontier/example_submit.sbatch`

# Check status
squeue -u $USER

# View output (replace JOBID)
tail -f logs/gdynet_vanilla_JOBID.out
```

#### 3. Monitor Progress

```bash
# Live monitoring
watch -n 10 'tail -n 30 logs/gdynet_vanilla_JOBID.out'

# Check checkpoints
ls -lh output/my_experiment/checkpoints/

# View metrics
python -c "import json; print(json.load(open('output/my_experiment/metrics/metrics.json')))"
```

---

## Monitoring and Checkpoints

### Training Output

```
Phase 0: Training with loss type: vamp2
==================================================

World Rank : 0 || Train Epoch: 1, Loss Type: vamp2
Train Loss - Last: 1.456789, Avg: 1.478901
Train Metrics - VAMP1 (Last: 0.789012, Avg: 0.776543), VAMP2 (Last: 1.456789, Avg: 1.478901)
Val Loss - Last: 1.234567, Avg: 1.245678
Val Metrics - VAMP1 (Last: 0.856789, Avg: 0.851234), VAMP2 (Last: 1.234567, Avg: 1.245678)
--- 45.67 seconds ---

Saved checkpoint: output/my_experiment/checkpoints/checkpoint_epoch_0001.pth
```

### Understanding Metrics

#### Loss values
- **Last**: Loss from final batch of epoch.
- **Avg**: Average loss across all batches in epoch.

#### VAMP metrics
- **VAMP1**: Sum of nuclear norm eigenvalues (higher = better slow features).
- **VAMP2**: Sum of Frobenius norm eigenvalues (related to explained variance).

### Checkpoint Contents

```python
checkpoint = {
    'epoch': 45,                        # Global epoch
    'model_name': 'gdynet_vanilla',
    'lobe_0_state_dict': {...},        # Model weights
    'lobe_t_state_dict': {...},
    'optimizer_state_dict': {...},      # Optimizer state
    'scaler_state_dict': {...},         # GradScaler (if AMP)
    'train_loss_last': 1.23,
    'train_loss_avg': 1.25,
    'val_loss_last': 1.10,
    'val_loss_avg': 1.12,
    'best_metric': 1.08,                # Best validation loss seen
    'config': {...},                    # Full configuration
    'loss_schedule': ['vamp2', ...],
    # Full metric histories
    'train_losses_last': [...],
    'train_losses_avg': [...],
    # ... etc
}
```

### Best Model Tracking

The trainer automatically tracks the best model based on **average validation loss**:

```python
# Checkpoint saved as checkpoint_best.pth when validation improves
if val_loss_avg < best_metric:
    save_checkpoint(..., is_best=True)
```

---

## Resuming Training

### Why Resume?

- Training interrupted (time limit, node failure)
- Want to extend training (add more phases)
- Tune hyperparameters from checkpoint

### Resume Procedure

#### 1. Identify Checkpoint

```bash
ls output/my_experiment/checkpoints/
# checkpoint_epoch_0030.pth  (end of phase 0)
# checkpoint_epoch_0045.pth  (middle of phase 1)
# checkpoint_epoch_0060.pth  (end of phase 1)
# checkpoint_latest.pth      (most recent)
# checkpoint_best.pth        (best validation)
```

#### 2. Resume Training

```bash
# Local
python trainer.py \
    --config configs/my_experiment.yaml \
    --mode train \
    --resume output/my_experiment/checkpoints/checkpoint_epoch_0045.pth

# NERSC (edit nersc/submit_resume.sbatch first)
sbatch nersc/submit_resume.sbatch
```

#### 3. What Happens During Resume

```
Loading checkpoint from: checkpoint_epoch_0045.pth

Resume information:
  Checkpoint epoch: 45
  Best validation loss (avg): 1.089765
  Loaded 45 training epochs of history
Loaded model weights from checkpoint
Loaded optimizer state from checkpoint

Starting training:
  Total phases: 3
  Epochs per phase: 30
  Starting from phase 1, epoch 15
  Loss schedule: ['vamp2', 'vamp1', 'vamp2']

Phase 1: Training with loss type: vamp1
==================================================
```

**Resume Calculation**:
```python
checkpoint_epoch = 45
epochs_per_phase = 30

phase_index = 45 // 30 = 1  # Resume in phase 1 (vamp1)
epoch_in_phase = 45 % 30 = 15  # Start at epoch 15 of phase

# Training continues: epochs 15-29 of phase 1, then all of phase 2
```

### Extending Training

To add more training beyond original schedule:

1. **Modify config**:
   ```yaml
   training:
     epochs: 30
     loss_schedule: ['vamp2', 'vamp1', 'vamp2', 'vamp1']  # Added phase 3
   ```

2. **Resume from last checkpoint**:
   ```bash
   python trainer.py \
       --config configs/extended.yaml \
       --mode train \
       --resume output/my_experiment/checkpoints/checkpoint_epoch_0089.pth
   ```

3. **Training continues** with new phase 3 (epochs 90-119)

---

## Best Practices

### Data Preparation

1. **Validate data before training**:
   ```python
   # Quick check script
   import numpy as np

   atom_types = np.load('data/atom_types.npy')
   target_index = np.load('data/target_index.npy')

   N = atom_types.shape[0]
   assert target_index.max() < N, f"Invalid target index: {target_index.max()} >= {N}"
   assert target_index.min() >= 0, "Negative target indices"
   print(f"✓ Data valid: {N} atoms, {len(target_index)} targets")
   ```

2. **Use consistent train/val/test splits**
3. **Normalize trajectories** if needed (coordinate centering, etc.)

### Training Strategy

#### Start small
- Use `loss_schedule: ['vamp2']` for 30 epochs.
- Verify training is stable.
- Then extend to the full schedule.

#### Monitor validation
- Validation loss should decrease.
- If diverging, reduce learning rate.
- Also monitor the `vamp2` loss value; it should be close to `(state_len - 1)`.

#### Checkpointing
- Use `frequency: 1` to save every epoch (debugging).
- Use `frequency: 10` for production (saves disk space).
- Always keep `checkpoint_best.pth`.

#### Learning rate
- Default `0.001` works well for most systems.
- If unstable: try `0.0005`.
- If slow: try `0.002`.

### Performance Tuning

#### Batch size
- Larger values generally lead to faster training due to better GPU utilization.
- Upper limits are constrained by available GPU memory.
- Recommended values to try: **16, 32, 64** (powers of two are typically more efficient).
- When using `DistributedSampler` together with `DataLoader`, ensure the selected `batch_size` allows the model to see each data point exactly once per epoch across all processes.

#### Mixed precision
- Enable for large models.
- Test accuracy first (rarely an issue).

#### torch.compile
- Use `mode: 'default'` initially.
- Try `max-autotune` for production (longer compile, better runtime).

### Distributed Training

#### Multi-GPU (single node)
- Automatic with SLURM (4 GPUs = 4x speedup).

#### Multi-node
- Near-linear scaling up to 4-8 nodes.
- Watch for NCCL errors (network issues).

#### Batch size scaling
- **Effective batch size** is computed as: `batch_size × num_GPUs`.
- When increasing the effective batch size, the learning rate may need to be adjusted accordingly; however, this adjustment does **not** necessarily follow a strict or universal scaling law and should be tuned empirically.


---

## Troubleshooting

### Common Issues

#### 1. Out of Memory (OOM)

**Symptom**: `RuntimeError: CUDA out of memory`

**Solutions**:
- Reduce `batch_size` (try 16 → 8 → 4)
- Enable `mixed_precision: true`
- Reduce model size (`atom_fea_len: 64 → 32`)

#### 2. Loss is NaN

**Symptom**: `Train Loss: nan`

**Causes**:
- Learning rate too high
- Numerical instability in VAMP

**Solutions**:
- Reduce `learning_rate` (0.001 → 0.0005)
- Increase `epsilon` (1e-07 → 1e-05)
- Check data for NaN values

#### 3. Training Too Slow

**Symptom**: < 1 epoch per minute

**Solutions**:
- Enable `mixed_precision: true` (2x speedup)
- Enable `torch_compile: true` (1.2x speedup)
- Increase `batch_size` if memory allows
- Use more GPUs


#### 4. Checkpoint Resume Fails

**Symptom**: `ValueError: Loss schedule mismatch`

**Cause**: Config loss_schedule differs from checkpoint

**Solution**:
- Use same config as original training
- Or accept the warning and continue with new schedule

#### 5. NCCL Hangs (Multi-node)

**Symptom**: Training hangs at initialization

**Solutions**:
- Check environment variables in batch script
- Verify network connectivity (`srun hostname`)
- Use smaller job (single node) to debug

### Debugging Tips

1. **Enable debug mode**:
   ```yaml
   training:
     epochs: 2  # Quick test
   checkpointing:
     frequency: 1
   ```

2. **Check GPU usage**:
   ```bash
   nvidia-smi  # Local
   srun --pty nvidia-smi  # NERSC
   ```

3. **Profile training**:
   ```python
   # Add to trainer.py
   import time
   start = time.time()
   # ... training code ...
   print(f"Epoch time: {time.time() - start:.2f}s")
   ```

4. **Verify data loading**:
   ```python
   # Test script
   from data.gdynet_dataloader import PyGMDStackGen_vanilla
   dataset = PyGMDStackGen_vanilla(
       fnames=[...],
       tau=10,
       cutoff=6.0,
       num_gaussians=50
   )
   print(f"Dataset size: {len(dataset)}")
   data_0, data_t = dataset[0]
   print(f"Loaded sample: {data_0}")
   ```

---

## Advanced Topics

### Custom Loss Schedules

You can implement complex schedules:

```yaml
# Gradual refinement
loss_schedule: ['vamp2', 'vamp2', 'vamp1', 'vamp2']

# Heavy VAMP1 focus
loss_schedule: ['vamp2', 'vamp1', 'vamp1', 'vamp1', 'vamp2']

# Alternating
loss_schedule: ['vamp2', 'vamp1'] * 3  # Python: expand to 6 phases
```

### Transfer Learning

Resume from one model to train on new data:

```bash
# Train on dataset A
python trainer.py --config configA.yaml --mode train

# Fine-tune on dataset B (using A's checkpoint)
python trainer.py \
    --config configB.yaml \
    --mode train \
    --resume output/experimentA/checkpoints/checkpoint_best.pth
```

### Ensemble Models

Train multiple models with different seeds:

```yaml
# config1.yaml
training:
  seed: 1234

# config2.yaml
training:
  seed: 5678

# config3.yaml
training:
  seed: 9012
```

Combine predictions for better results.

---

## Summary Checklist

### Before training
- [ ] Data files validated (correct format, no missing values).
- [ ] Config file prepared (paths updated, schedule chosen).
- [ ] Output directory has enough space.
- [ ] Environment activated (`conda activate pyg_latest`).

### During training
- [ ] Monitor console output for errors.
- [ ] Check validation loss is decreasing.
- [ ] Verify checkpoints are being saved.

### After training
- [ ] Check final metrics (train vs validation).
- [ ] Save best model for inference.
- [ ] Document hyperparameters and results.

---

## Getting Help

If you encounter issues:

1. Check this guide's [Troubleshooting](#troubleshooting) section
2. Review console output for error messages
3. Verify data with validation script
4. Open an issue on GitHub with:
   - Config file
   - Error message
   - System info (GPU, PyTorch version)

Happy training!
