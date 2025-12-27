# GDyNet Analysis Notebooks

This directory contains Jupyter notebooks for analyzing GDyNet training results and predictions.

## Available Notebooks

### `analysis_example.ipynb`

Comprehensive analysis notebook that provides:

1. **Prediction Statistics**: Summary statistics for state predictions
2. **Training Metrics Visualization**: Interactive plots of training/validation loss and VAMP scores
   - Supports both **last batch** and **average** metrics
   - Generalization gap analysis
3. **State Population Analysis**: Track state probabilities over time
4. **State Distribution Analysis**: Compare state distributions across frames
5. **Temporal Evolution**: Visualize individual atom state transitions
6. **Transition Matrix**: Compute state transition probabilities
7. **Autocorrelation Analysis**: Measure decorrelation times
8. **State Switching Events**: Analyze dynamics of state changes

## Quick Start

### 1. Install Dependencies

```bash
# Install visualization dependencies
pip install matplotlib seaborn jupyter

# Or install all requirements
pip install -r ../requirements.txt
```

### 2. Launch Jupyter

```bash
# From the notebooks directory
jupyter notebook analysis_example.ipynb

# Or from project root
jupyter notebook notebooks/analysis_example.ipynb
```

### 3. Configure Paths

In the notebook's **Configuration** cell, update:

```python
# Model type: 'gdynet_vanilla' or 'gdynet_ferro'
MODEL_NAME = 'gdynet_vanilla'

# Base output directory (created by trainer_optimized.py)
OUTPUT_DIR = f'../output/{MODEL_NAME}'
```

The notebook will automatically construct paths to:
- Predictions: `output/{model_name}/predictions/{model_name}_predictions.npy`
- Metrics: `output/{model_name}/metrics/metrics.json`

### 4. Run Analysis

Execute cells sequentially (Shift+Enter) or run all cells (Cell → Run All)

## Output Structure

After running the complete analysis, the following files will be saved:

```
output/{model_name}/
├── analysis_training_metrics.png      # Loss and VAMP score curves
├── analysis_loss_comparison.png       # Last vs Avg metric comparison
├── analysis_state_populations.png     # State probabilities over time
├── analysis_state_distributions.png   # Histograms for first/last frame
├── analysis_temporal_evolution.png    # Individual atom trajectories
├── analysis_transition_matrix_lag1.png # State transition probabilities
├── analysis_autocorrelation.png       # Autocorrelation function
├── analysis_state_switches.png        # State switching dynamics
└── analysis_summary.json              # Comprehensive JSON summary
```

## Using the Analysis Tools

### Programmatic Access

You can use the `GDYNetAnalyzer` class directly in your own scripts:

```python
from postprocess import GDYNetAnalyzer

# Load results
analyzer = GDYNetAnalyzer(
    predictions_path='output/gdynet_vanilla/predictions/gdynet_vanilla_predictions.npy',
    metrics_path='output/gdynet_vanilla/metrics/metrics.json'
)

# Get summary statistics
stats = analyzer.get_summary_stats()
print(f"Predictions shape: {stats['shape']}")

# Access metrics (new format supports last and avg)
train_loss_avg = analyzer.get_metric('train_losses', version='avg')
train_loss_last = analyzer.get_metric('train_losses', version='last')
train_loss_both = analyzer.get_metric('train_losses', version='both')

# Generate training plots
fig = analyzer.plot_training_metrics(
    save_path='my_metrics.png',
    show_both=True  # Plot both last and avg
)

# Analyze predictions
state_pops = analyzer.get_state_populations()
transition_matrix = analyzer.compute_state_transitions(lag=1)
autocorr = analyzer.compute_autocorrelation(max_lag=100)

# Export summary
analyzer.export_summary('my_summary.json')
```

### Command Line Usage

Analyze predictions from the command line:

```bash
# Basic analysis
python -m postprocess.postprocess output/gdynet_vanilla/predictions/gdynet_vanilla_predictions.npy

# With metrics
python -m postprocess.postprocess \
    output/gdynet_vanilla/predictions/gdynet_vanilla_predictions.npy \
    --metrics output/gdynet_vanilla/metrics/metrics.json

# Export summary
python -m postprocess.postprocess \
    output/gdynet_vanilla/predictions/gdynet_vanilla_predictions.npy \
    --metrics output/gdynet_vanilla/metrics/metrics.json \
    --output output/gdynet_vanilla/
```

## Metrics Format

The analysis tools support the current training configuration which saves **both** last and average values:

### New Format (Current)
```json
{
  "train_losses_last": [0.123, 0.115, ...],
  "train_losses_avg": [0.125, 0.118, ...],
  "val_losses_last": [0.134, 0.128, ...],
  "val_losses_avg": [0.136, 0.130, ...],
  "train_vamp1_scores_last": [...],
  "train_vamp1_scores_avg": [...],
  ...
}
```

### Old Format (Legacy Support)
```json
{
  "train_losses": [0.125, 0.118, ...],
  "val_losses": [0.136, 0.130, ...],
  ...
}
```

The `GDYNetAnalyzer` automatically detects and handles both formats.

## Customization

### Add Custom Analysis

Use cell 13 in the notebook for custom analysis:

```python
# Access raw predictions
predictions = analyzer.predictions  # Shape: (n_frames, n_atoms, n_states)

# Access metrics dictionary
metrics = analyzer.metrics

# Example: Find most stable atoms
stability = predictions.std(axis=0).mean(axis=1)  # Variance over time
most_stable = stability.argmin()
print(f"Most stable atom: {most_stable}")
```

### Modify Visualizations

All matplotlib plots can be customized:

```python
# Example: Change plot style
import matplotlib.pyplot as plt
plt.style.use('seaborn-darkgrid')
plt.rcParams['figure.figsize'] = (16, 8)
plt.rcParams['font.size'] = 12
```

## Troubleshooting

### Import Errors

```bash
# If you get "ModuleNotFoundError: No module named 'utils'"
# Make sure you're running from the notebooks directory
# Or add parent to path:
import sys
sys.path.append('..')
```

### File Not Found

```
# Check that predictions and metrics exist:
ls -la ../output/gdynet_vanilla/predictions/
ls -la ../output/gdynet_vanilla/metrics/
```

### Memory Issues

For large datasets, analyze subsets:

```python
# Load only part of predictions
predictions = np.load('predictions.npy', mmap_mode='r')
subset = predictions[:1000, :, :]  # First 1000 frames
```

## Examples

### Example 1: Quick Summary

```bash
cd notebooks
jupyter notebook analysis_example.ipynb
# Run cells 1-3 for quick statistics
```

### Example 2: Training Curve Analysis

```python
analyzer = GDYNetAnalyzer(
    predictions_path='../output/gdynet_ferro/predictions/gdynet_ferro_predictions.npy',
    metrics_path='../output/gdynet_ferro/metrics/metrics.json'
)

# Plot training metrics
analyzer.plot_training_metrics(
    save_path='training_curves.png',
    show_both=True  # Shows both last and avg
)
```

### Example 3: Batch Processing

Analyze multiple experiments:

```python
experiments = ['exp1', 'exp2', 'exp3']

for exp in experiments:
    analyzer = GDYNetAnalyzer(
        predictions_path=f'../output/{exp}/predictions/gdynet_vanilla_predictions.npy',
        metrics_path=f'../output/{exp}/metrics/metrics.json'
    )
    analyzer.plot_training_metrics(save_path=f'{exp}_metrics.png')
    analyzer.export_summary(f'{exp}_summary.json')
```

## References

- **Main Documentation**: [../README.md](../README.md)
- **Training Guide**: [../TRAINING_GUIDE.md](../TRAINING_GUIDE.md)
- **Paper**: [Carbon Trends 2023](https://doi.org/10.1016/j.cartre.2023.100264)
- **Postprocess Module**: [../postprocess/postprocess.py](../postprocess/postprocess.py)
