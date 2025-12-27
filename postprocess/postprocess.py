"""
Post-processing utilities for GDYNet predictions

This module provides tools for analyzing and visualizing GDYNet outputs.
"""

import numpy as np
import torch
from typing import Tuple, Optional, Union
import json
from pathlib import Path


class GDYNetAnalyzer:
    """Analyzer for GDYNet prediction outputs."""

    def __init__(self, predictions_path: str, metrics_path: Optional[str] = None):
        """
        Initialize analyzer with predictions and optional metrics.

        Args:
            predictions_path: Path to predictions file (.npy or .pt)
            metrics_path: Optional path to metrics.json
        """
        self.predictions_path = Path(predictions_path)
        self.metrics_path = Path(metrics_path) if metrics_path else None

        # Load predictions
        self.predictions = self.load_predictions()

        # Load metrics if available
        self.metrics = self.load_metrics() if self.metrics_path else None

        print(f"Loaded predictions with shape: {self.predictions.shape}")
        if self.metrics:
            print(f"Loaded {len(self.metrics)} metric types")

    def load_predictions(self) -> np.ndarray:
        """Load predictions from file."""
        if self.predictions_path.suffix == '.npy':
            return np.load(self.predictions_path)
        elif self.predictions_path.suffix == '.pt':
            return torch.load(self.predictions_path).numpy()
        else:
            raise ValueError(f"Unsupported file format: {self.predictions_path.suffix}")

    def load_metrics(self) -> dict:
        """Load training metrics from JSON."""
        if self.metrics_path and self.metrics_path.exists():
            with open(self.metrics_path, 'r') as f:
                return json.load(f)
        return None

    def get_summary_stats(self) -> dict:
        """Get summary statistics of predictions."""
        stats = {
            'shape': self.predictions.shape,
            'n_frames': self.predictions.shape[0],
            'n_atoms': self.predictions.shape[1],
            'n_states': self.predictions.shape[2],
            'mean': float(self.predictions.mean()),
            'std': float(self.predictions.std()),
            'min': float(self.predictions.min()),
            'max': float(self.predictions.max()),
        }

        # Per-state statistics
        for i in range(self.predictions.shape[2]):
            stats[f'state_{i}_mean'] = float(self.predictions[:, :, i].mean())
            stats[f'state_{i}_std'] = float(self.predictions[:, :, i].std())

        return stats

    def get_temporal_evolution(self, atom_idx: int = 0) -> np.ndarray:
        """
        Get temporal evolution for a specific atom.

        Args:
            atom_idx: Index of atom to analyze

        Returns:
            Array of shape (n_frames, n_states)
        """
        return self.predictions[:, atom_idx, :]

    def get_state_populations(self) -> np.ndarray:
        """
        Calculate state populations over time.

        Returns:
            Array of shape (n_frames, n_states) with average populations
        """
        # Average over all atoms
        return self.predictions.mean(axis=1)

    def get_atom_state_probabilities(self, frame_idx: int = 0) -> np.ndarray:
        """
        Get state probabilities for all atoms at a specific frame.

        Args:
            frame_idx: Frame index

        Returns:
            Array of shape (n_atoms, n_states)
        """
        return self.predictions[frame_idx, :, :]

    def compute_state_transitions(self, lag: int = 1) -> np.ndarray:
        """
        Compute state transition probabilities.

        Args:
            lag: Time lag for transitions

        Returns:
            Transition matrix of shape (n_states, n_states)
        """
        n_states = self.predictions.shape[2]

        # Get dominant state for each atom at each time
        states_t0 = self.predictions[:-lag].argmax(axis=2)  # (n_frames-lag, n_atoms)
        states_t1 = self.predictions[lag:].argmax(axis=2)   # (n_frames-lag, n_atoms)

        # Count transitions
        transition_matrix = np.zeros((n_states, n_states))

        for i in range(states_t0.shape[0]):
            for j in range(states_t0.shape[1]):
                s0 = states_t0[i, j]
                s1 = states_t1[i, j]
                transition_matrix[s0, s1] += 1

        # Normalize
        row_sums = transition_matrix.sum(axis=1, keepdims=True)
        transition_matrix = transition_matrix / (row_sums + 1e-10)

        return transition_matrix

    def compute_autocorrelation(self, max_lag: int = 100) -> np.ndarray:
        """
        Compute autocorrelation function for state probabilities.

        Args:
            max_lag: Maximum lag to compute

        Returns:
            Autocorrelation array of shape (max_lag,)
        """
        # Use state 0 probabilities averaged over atoms
        signal = self.predictions[:, :, 0].mean(axis=1)

        # Normalize
        signal = signal - signal.mean()
        signal = signal / signal.std()

        # Compute autocorrelation
        autocorr = np.correlate(signal, signal, mode='full')
        autocorr = autocorr[len(autocorr)//2:]  # Keep positive lags
        autocorr = autocorr / autocorr[0]  # Normalize

        return autocorr[:max_lag]

    def get_training_history(self) -> dict:
        """Get training history if metrics are available."""
        if self.metrics is None:
            return None

        history = {}
        for key, values in self.metrics.items():
            if values:
                history[key] = {
                    'values': values,
                    'final': values[-1] if values else None,
                    'best': min(values) if 'loss' in key.lower() else max(values),
                    'epochs': len(values)
                }

        return history

    def get_metric(self, metric_name: str, version: str = 'avg') -> Optional[list]:
        """
        Get a specific metric from the loaded metrics.

        Args:
            metric_name: Base metric name (e.g., 'train_losses', 'val_vamp1_scores')
            version: 'last', 'avg', or 'both' (default: 'avg')

        Returns:
            List of metric values, or dict with 'last' and 'avg' keys if version='both'
        """
        if self.metrics is None:
            return None

        # Try new format (with _last and _avg suffixes)
        metric_last = self.metrics.get(f'{metric_name}_last')
        metric_avg = self.metrics.get(f'{metric_name}_avg')

        # If new format exists
        if metric_last is not None or metric_avg is not None:
            if version == 'last':
                return metric_last
            elif version == 'avg':
                return metric_avg
            elif version == 'both':
                return {'last': metric_last, 'avg': metric_avg}

        # Fall back to old format (without suffix)
        return self.metrics.get(metric_name)

    def export_summary(self, output_path: str):
        """
        Export summary statistics to JSON.

        Args:
            output_path: Path to save summary JSON
        """
        summary = {
            'predictions_stats': self.get_summary_stats(),
            'training_history': self.get_training_history(),
        }

        with open(output_path, 'w') as f:
            json.dump(summary, f, indent=2)

        print(f"Summary exported to: {output_path}")

    def plot_training_metrics(self, save_path: Optional[str] = None, show_both: bool = True):
        """
        Plot training metrics with support for both last and avg values.

        Args:
            save_path: Optional path to save the plot
            show_both: If True, plot both last and avg values (default: True)

        Returns:
            matplotlib figure
        """
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            print("matplotlib not installed. Install with: pip install matplotlib")
            return None

        if self.metrics is None:
            print("No metrics available to plot")
            return None

        fig, axes = plt.subplots(2, 2, figsize=(15, 10))

        # Training and validation loss
        ax = axes[0, 0]
        train_loss = self.get_metric('train_losses', 'both' if show_both else 'avg')
        val_loss = self.get_metric('val_losses', 'both' if show_both else 'avg')

        if show_both and isinstance(train_loss, dict):
            if train_loss.get('avg'):
                ax.plot(train_loss['avg'], label='Train Loss (Avg)', alpha=0.7, linewidth=2)
            if train_loss.get('last'):
                ax.plot(train_loss['last'], label='Train Loss (Last)', alpha=0.4, linewidth=1, linestyle='--')
            if val_loss.get('avg'):
                ax.plot(val_loss['avg'], label='Val Loss (Avg)', alpha=0.7, linewidth=2)
            if val_loss.get('last'):
                ax.plot(val_loss['last'], label='Val Loss (Last)', alpha=0.4, linewidth=1, linestyle='--')
        else:
            if train_loss:
                ax.plot(train_loss, label='Train Loss', alpha=0.7)
            if val_loss:
                ax.plot(val_loss, label='Val Loss', alpha=0.7)

        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss')
        ax.set_title('Training and Validation Loss')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # VAMP1 scores
        ax = axes[0, 1]
        train_vamp1 = self.get_metric('train_vamp1_scores', 'both' if show_both else 'avg')
        val_vamp1 = self.get_metric('val_vamp1_scores', 'both' if show_both else 'avg')

        if show_both and isinstance(train_vamp1, dict):
            if train_vamp1.get('avg'):
                ax.plot(train_vamp1['avg'], label='Train VAMP1 (Avg)', alpha=0.7, linewidth=2)
            if train_vamp1.get('last'):
                ax.plot(train_vamp1['last'], label='Train VAMP1 (Last)', alpha=0.4, linewidth=1, linestyle='--')
            if val_vamp1.get('avg'):
                ax.plot(val_vamp1['avg'], label='Val VAMP1 (Avg)', alpha=0.7, linewidth=2)
            if val_vamp1.get('last'):
                ax.plot(val_vamp1['last'], label='Val VAMP1 (Last)', alpha=0.4, linewidth=1, linestyle='--')
        else:
            if train_vamp1:
                ax.plot(train_vamp1, label='Train VAMP1', alpha=0.7)
            if val_vamp1:
                ax.plot(val_vamp1, label='Val VAMP1', alpha=0.7)

        ax.set_xlabel('Epoch')
        ax.set_ylabel('VAMP1 Score')
        ax.set_title('VAMP1 Metric')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # VAMP2 scores
        ax = axes[1, 0]
        train_vamp2 = self.get_metric('train_vamp2_scores', 'both' if show_both else 'avg')
        val_vamp2 = self.get_metric('val_vamp2_scores', 'both' if show_both else 'avg')

        if show_both and isinstance(train_vamp2, dict):
            if train_vamp2.get('avg'):
                ax.plot(train_vamp2['avg'], label='Train VAMP2 (Avg)', alpha=0.7, linewidth=2)
            if train_vamp2.get('last'):
                ax.plot(train_vamp2['last'], label='Train VAMP2 (Last)', alpha=0.4, linewidth=1, linestyle='--')
            if val_vamp2.get('avg'):
                ax.plot(val_vamp2['avg'], label='Val VAMP2 (Avg)', alpha=0.7, linewidth=2)
            if val_vamp2.get('last'):
                ax.plot(val_vamp2['last'], label='Val VAMP2 (Last)', alpha=0.4, linewidth=1, linestyle='--')
        else:
            if train_vamp2:
                ax.plot(train_vamp2, label='Train VAMP2', alpha=0.7)
            if val_vamp2:
                ax.plot(val_vamp2, label='Val VAMP2', alpha=0.7)

        ax.set_xlabel('Epoch')
        ax.set_ylabel('VAMP2 Score')
        ax.set_title('VAMP2 Metric')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Loss comparison (generalization gap)
        ax = axes[1, 1]
        if show_both and isinstance(train_loss, dict) and isinstance(val_loss, dict):
            if train_loss.get('avg') and val_loss.get('avg'):
                train_arr = np.array(train_loss['avg'])
                val_arr = np.array(val_loss['avg'])
                gap = val_arr - train_arr
                ax.plot(gap, label='Val - Train Gap (Avg)', color='red', alpha=0.7, linewidth=2)
            if train_loss.get('last') and val_loss.get('last'):
                train_arr = np.array(train_loss['last'])
                val_arr = np.array(val_loss['last'])
                gap = val_arr - train_arr
                ax.plot(gap, label='Val - Train Gap (Last)', color='orange', alpha=0.4, linewidth=1, linestyle='--')
        elif train_loss and val_loss:
            train_arr = np.array(train_loss)
            val_arr = np.array(val_loss)
            gap = val_arr - train_arr
            ax.plot(gap, label='Val - Train Gap', color='red', alpha=0.7)

        ax.axhline(y=0, color='black', linestyle='--', alpha=0.3)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss Gap')
        ax.set_title('Generalization Gap')
        ax.legend()
        ax.grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Plot saved to: {save_path}")

        return fig


def analyze_predictions(predictions_path: str,
                       metrics_path: Optional[str] = None,
                       output_dir: Optional[str] = None) -> GDYNetAnalyzer:
    """
    Quick analysis of GDYNet predictions.

    Args:
        predictions_path: Path to predictions file
        metrics_path: Optional path to metrics.json
        output_dir: Optional directory to save outputs

    Returns:
        GDYNetAnalyzer instance
    """
    analyzer = GDYNetAnalyzer(predictions_path, metrics_path)

    print("\n" + "="*60)
    print("GDYNet Prediction Analysis")
    print("="*60)

    # Summary stats
    stats = analyzer.get_summary_stats()
    print(f"\nPrediction Shape: {stats['shape']}")
    print(f"  Frames: {stats['n_frames']}")
    print(f"  Atoms: {stats['n_atoms']}")
    print(f"  States: {stats['n_states']}")

    print(f"\nOverall Statistics:")
    print(f"  Mean: {stats['mean']:.4f}")
    print(f"  Std:  {stats['std']:.4f}")
    print(f"  Min:  {stats['min']:.4f}")
    print(f"  Max:  {stats['max']:.4f}")

    print(f"\nPer-State Statistics:")
    for i in range(stats['n_states']):
        print(f"  State {i}: mean={stats[f'state_{i}_mean']:.4f}, std={stats[f'state_{i}_std']:.4f}")

    # Training history
    if analyzer.metrics:
        print(f"\nTraining History:")
        history = analyzer.get_training_history()
        for key, info in history.items():
            if info:
                print(f"  {key}:")
                print(f"    Final: {info['final']:.4f}")
                print(f"    Best:  {info['best']:.4f}")
                print(f"    Epochs: {info['epochs']}")

    # Export if output directory provided
    if output_dir:
        output_path = Path(output_dir) / 'analysis_summary.json'
        analyzer.export_summary(str(output_path))

    return analyzer


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Analyze GDYNet predictions')
    parser.add_argument('predictions', type=str, help='Path to predictions file (.npy or .pt)')
    parser.add_argument('--metrics', type=str, default=None, help='Path to metrics.json')
    parser.add_argument('--output', type=str, default=None, help='Output directory for analysis')

    args = parser.parse_args()

    analyze_predictions(args.predictions, args.metrics, args.output)
