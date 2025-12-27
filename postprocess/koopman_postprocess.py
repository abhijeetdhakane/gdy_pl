"""
Koopman Postprocessing Module

A self-contained module for analyzing VAMPnet predictions using Koopman operator theory.
Includes implied timescales, Chapman-Kolmogorov tests, and eigenvalue analysis.
"""

import numpy as np
from scipy import optimize
import matplotlib.pyplot as plt


class KoopmanAnalysis:
    """
    Class for Koopman operator analysis of trajectory predictions.
    
    Parameters
    ----------
    epsilon : float, optional, default=1e-10
        Threshold for eigenvalues to be considered different from zero.
    """
    
    def __init__(self, epsilon=1e-10):
        self.epsilon = epsilon
    
    def estimate_koopman_op(self, traj, tau):
        """
        Estimate the Koopman operator for a given trajectory at the specified lag time.
        
        Formula: K = C00^-1 @ C01
        
        Parameters
        ----------
        traj : np.ndarray, shape (n_timesteps, n_traj, n_states) or (n_timesteps, n_states)
            Trajectory data (probability of each state over time)
        tau : int
            Lag time for estimation
            
        Returns
        -------
        koopman_op : np.ndarray, shape (n_states, n_states)
            Koopman operator estimated at lag time tau
        """
        # Handle different input shapes and convert to float64 for numerical stability
        if traj.ndim == 3:
            n_classes = traj.shape[-1]
            prev = traj[:-tau].reshape(-1, n_classes).astype(np.float64)
            post = traj[tau:].reshape(-1, n_classes).astype(np.float64)
        else:
            prev = traj[:-tau].astype(np.float64)
            post = traj[tau:].astype(np.float64)
        
        # Compute covariance matrices
        c_0 = prev.T @ prev
        c_tau = prev.T @ post
        
        # Regularized inverse of c_0 using eigendecomposition
        eigv, eigvec = np.linalg.eigh(c_0)  # Use eigh for symmetric matrices
        
        include = eigv > self.epsilon
        eigv = eigv[include]
        eigvec = eigvec[:, include]
        c0_inv = eigvec @ np.diag(1.0 / eigv) @ eigvec.T
        
        koopman_op = c0_inv @ c_tau
        return koopman_op
    
    def get_its(self, traj, lags):
        """
        Compute implied timescales from trajectory at a series of lag times.
        
        Matches the original VAMPnet implementation exactly.
        
        Parameters
        ----------
        traj : np.ndarray, shape (n_timesteps, n_traj, n_states) or (n_timesteps, n_states)
            Trajectory data
        lags : array-like
            Lag times at which to estimate implied timescales
            
        Returns
        -------
        its : np.ndarray, shape (n_states - 1, n_lags)
            Implied timescales (excluding the stationary process)
        """
        n_states = traj.shape[-1]
        its = np.zeros((n_states - 1, len(lags)))
        
        for t, tau_lag in enumerate(lags):
            koopman_op = self.estimate_koopman_op(traj, tau_lag)
            k_eigvals, _ = np.linalg.eig(np.real(koopman_op))
            # Sort ascending, remove largest (stationary eigenvalue ~1)
            k_eigvals = np.sort(np.absolute(k_eigvals))
            k_eigvals = k_eigvals[:-1]
            # ITS formula: -tau / log(lambda)
            its[:, t] = -tau_lag / np.log(k_eigvals)
        
        return its
    
    def get_ck_test(self, traj, steps, tau):
        """
        Chapman-Kolmogorov test for the Koopman operator.
        
        Compares predicted (K^n) vs estimated (K(n*tau)) transition probabilities.
        
        Parameters
        ----------
        traj : np.ndarray, shape (n_timesteps, n_traj, n_states) or (n_timesteps, n_states)
            Trajectory data
        steps : int
            Number of lag time multiples to evaluate
        tau : int
            Base lag time
            
        Returns
        -------
        predicted : np.ndarray, shape (n_states, n_states, steps)
            Predicted transition probabilities from K^n
        estimated : np.ndarray, shape (n_states, n_states, steps)
            Estimated transition probabilities from K(n*tau)
        """
        if traj.ndim == 3:
            n_states = traj.shape[-1]
        else:
            n_states = traj.shape[-1]
        
        predicted = np.zeros((n_states, n_states, steps))
        estimated = np.zeros((n_states, n_states, steps))
        
        # Initial condition (identity at step 0)
        predicted[:, :, 0] = np.eye(n_states)
        estimated[:, :, 0] = np.eye(n_states)
        
        # Base Koopman operator
        koop = self.estimate_koopman_op(traj, tau)
        
        for i, vector in enumerate(np.eye(n_states)):
            for n in range(1, steps):
                # Predicted: K^n
                koop_pred = np.linalg.matrix_power(koop, n)
                # Estimated: K(n*tau)
                koop_est = self.estimate_koopman_op(traj, tau * n)
                
                predicted[i, :, n] = vector @ koop_pred
                estimated[i, :, n] = vector @ koop_est
        
        return predicted, estimated
    
    def get_stationary_distribution(self, traj, tau):
        """
        Compute the stationary distribution from the Koopman operator.
        
        Parameters
        ----------
        traj : np.ndarray
            Trajectory data
        tau : int
            Lag time
            
        Returns
        -------
        stationary : np.ndarray, shape (n_states,)
            Stationary distribution
        koopman_op : np.ndarray
            The Koopman operator
        eigvals : np.ndarray
            Eigenvalues of K^T
        eigvecs : np.ndarray
            Eigenvectors of K^T
        """
        koopman_op = self.estimate_koopman_op(traj, tau)
        eigvals, eigvecs = np.linalg.eig(koopman_op.T)
        
        # Find the eigenvector corresponding to eigenvalue ~1
        idx = np.argmax(np.real(eigvals))
        stationary = np.real(eigvecs[:, idx])
        stationary = stationary / np.sum(stationary)  # Normalize
        
        return stationary, koopman_op, eigvals, eigvecs


def plot_timescales(predictions, lags, n_splits=1, split_axis=0, 
                    time_unit_in_ns=1.0, epsilon=1e-10, figsize=(5, 4),
                    save_path=None):
    """
    Plot implied timescales from trajectory predictions.
    
    Parameters
    ----------
    predictions : np.ndarray, shape (n_timesteps, n_traj, n_states)
        Probability of each state over time
    lags : array-like
        Lag times (in timesteps) for ITS calculation
    n_splits : int, optional
        Number of splits for uncertainty estimation
    split_axis : int, optional
        Axis to split (0=time, 1=trajectory)
    time_unit_in_ns : float, optional
        Time unit conversion (e.g., 1e-4 means 0.1 ps = 1e-4 ns per timestep)
    epsilon : float, optional
        Regularization threshold
    figsize : tuple, optional
        Figure size
    save_path : str, optional
        Path to save the figure
        
    Returns
    -------
    fig : matplotlib.figure.Figure
    ax : matplotlib.axes.Axes
    its_mean : np.ndarray
        Mean implied timescales
    """
    if split_axis not in [0, 1]:
        raise ValueError("split_axis must be 0 (time) or 1 (trajectory)")
    
    analyzer = KoopmanAnalysis(epsilon=epsilon)
    
    # Split predictions for uncertainty estimation
    if n_splits > 1:
        splited_preds = np.array_split(predictions, n_splits, axis=split_axis)
        splited_its = np.stack([analyzer.get_its(p, lags) for p in splited_preds])
    else:
        splited_its = analyzer.get_its(predictions, lags)[np.newaxis, ...]
    
    # Convert to physical time units
    lags_ns = np.array(lags) * time_unit_in_ns
    splited_its_ns = splited_its * time_unit_in_ns
    
    # Compute statistics in log space for better error estimation
    its_log_mean = np.mean(np.log(splited_its_ns + 1e-20), axis=0)
    its_log_std = np.std(np.log(splited_its_ns + 1e-20), axis=0)
    
    its_mean = np.exp(its_log_mean)
    its_upper = np.exp(its_log_mean + its_log_std * 1.96 / np.sqrt(max(n_splits, 1)))
    its_lower = np.exp(its_log_mean - its_log_std * 1.96 / np.sqrt(max(n_splits, 1)))
    
    # Create plot
    fig, ax = plt.subplots(figsize=figsize)
    
    n_timescales = its_mean.shape[0]
    colors = plt.cm.tab10(np.linspace(0, 1, max(n_timescales, 1)))
    
    for i in range(n_timescales):
        ax.semilogy(lags_ns, its_mean[i], color=colors[i], 
                    label=f'Timescale {i+1}', linewidth=2)
        if n_splits > 1:
            ax.fill_between(lags_ns, its_lower[i], its_upper[i], 
                           alpha=0.2, color=colors[i])
    
    # Plot the diagonal (lag time = timescale boundary)
    ax.semilogy(lags_ns, lags_ns, 'k-', linewidth=1.5, label='Lag time')
    
    # Fill the "forbidden" region below the diagonal
    y_min = ax.get_ylim()[0]
    ax.fill_between(lags_ns, y_min, lags_ns, alpha=0.2, color='gray')
    
    ax.set_xlabel('Lag time (ns)', fontsize=12)
    ax.set_ylabel('Implied timescales (ns)', fontsize=12)
    ax.legend(loc='best', fontsize=10)
    ax.set_xlim(lags_ns[0], lags_ns[-1])
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved figure to {save_path}")
    
    return fig, ax, its_mean


def plot_ck_tests(predictions, tau_msm, steps, n_splits=1, split_axis=0,
                  time_unit_in_ns=1.0, epsilon=1e-10, figsize=None,
                  save_path=None):
    """
    Plot Chapman-Kolmogorov test results.
    
    Parameters
    ----------
    predictions : np.ndarray, shape (n_timesteps, n_traj, n_states)
        Probability of each state over time
    tau_msm : int
        Lag time for Koopman model (in timesteps)
    steps : int
        Number of lag time multiples to validate
    n_splits : int, optional
        Number of splits for uncertainty estimation
    split_axis : int, optional
        Axis to split (0=time, 1=trajectory)
    time_unit_in_ns : float, optional
        Time unit conversion
    epsilon : float, optional
        Regularization threshold
    figsize : tuple, optional
        Figure size (auto-determined if None)
    save_path : str, optional
        Path to save the figure
        
    Returns
    -------
    fig : matplotlib.figure.Figure
    axes : np.ndarray of matplotlib.axes.Axes
    """
    if split_axis not in [0, 1]:
        raise ValueError("split_axis must be 0 (time) or 1 (trajectory)")
    
    n_states = predictions.shape[-1]
    analyzer = KoopmanAnalysis(epsilon=epsilon)
    
    # Split predictions for uncertainty estimation
    if n_splits > 1:
        splited_preds = np.array_split(predictions, n_splits, axis=split_axis)
    else:
        splited_preds = [predictions]
    
    splited_predicted, splited_estimated = [], []
    for p in splited_preds:
        predicted, estimated = analyzer.get_ck_test(p, steps, tau_msm)
        splited_predicted.append(predicted)
        splited_estimated.append(estimated)
    
    splited_predicted = np.stack(splited_predicted)
    splited_estimated = np.stack(splited_estimated)
    
    # Convert lag time to physical units
    tau_ns = tau_msm * time_unit_in_ns
    time_axis = np.arange(0, steps * tau_ns, tau_ns)
    
    # Determine figure size
    if figsize is None:
        figsize = (3 * n_states, 2.5 * n_states)
    
    fig, axes = plt.subplots(n_states, n_states, sharex=True, sharey=True,
                              figsize=figsize)
    
    # Handle single state case
    if n_states == 1:
        axes = np.array([[axes]])
    
    for i in range(n_states):
        for j in range(n_states):
            ax = axes[i, j]
            
            # Predicted (from K^n)
            pred_mean = splited_predicted[:, i, j].mean(axis=0)
            pred_std = splited_predicted[:, i, j].std(axis=0)
            
            ax.plot(time_axis, pred_mean, 'b-', linewidth=2, label='Predicted')
            if n_splits > 1:
                ax.fill_between(
                    time_axis,
                    pred_mean - pred_std * 1.96 / np.sqrt(n_splits),
                    pred_mean + pred_std * 1.96 / np.sqrt(n_splits),
                    alpha=0.2, color='b'
                )
            
            # Estimated (from K(n*tau))
            est_mean = splited_estimated[:, i, j].mean(axis=0)
            est_std = splited_estimated[:, i, j].std(axis=0)
            
            ax.errorbar(
                time_axis, est_mean,
                yerr=est_std * 1.96 / np.sqrt(max(n_splits, 1)) if n_splits > 1 else None,
                color='r', linestyle='--', linewidth=2,
                capsize=3, label='Estimated'
            )
            
            ax.set_title(f'{i} → {j}', fontsize=10)
            
            if i == n_states - 1:
                ax.set_xlabel('Time (ns)', fontsize=10)
            if j == 0:
                ax.set_ylabel('Probability', fontsize=10)
    
    axes[0, 0].set_ylim(-0.1, 1.1)
    axes[0, 0].set_xlim(0, (steps - 1) * tau_ns)
    axes[0, -1].legend(loc='upper right', fontsize=8)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved figure to {save_path}")
    
    return fig, axes


def analyze_koopman(predictions, tau_msm, time_unit_in_ns=1.0, epsilon=1e-10):
    """
    Perform full Koopman operator analysis.
    
    Parameters
    ----------
    predictions : np.ndarray, shape (n_timesteps, n_traj, n_states)
        Probability of each state over time
    tau_msm : int
        Lag time (in timesteps)
    time_unit_in_ns : float, optional
        Time unit conversion
    epsilon : float, optional
        Regularization threshold
        
    Returns
    -------
    results : dict
        Dictionary containing:
        - 'koopman_op': Koopman operator matrix
        - 'eigvals': Eigenvalues of K^T
        - 'eigvecs': Eigenvectors of K^T
        - 'stationary': Stationary distribution
        - 'timescales': Implied timescales from eigenvalues
    """
    analyzer = KoopmanAnalysis(epsilon=epsilon)
    
    # Estimate Koopman operator
    koopman_op = analyzer.estimate_koopman_op(predictions, tau_msm)
    
    # Eigendecomposition of K^T
    eigvals, eigvecs = np.linalg.eig(koopman_op.T)
    
    # Sort by eigenvalue magnitude (descending)
    idx = np.argsort(np.abs(eigvals))[::-1]
    eigvals = eigvals[idx]
    eigvecs = eigvecs[:, idx]
    
    # Stationary distribution (eigenvector of eigenvalue ~1)
    stationary = np.real(eigvecs[:, 0])
    stationary = stationary / np.sum(stationary)
    
    # Implied timescales from eigenvalues (excluding stationary)
    eigvals_process = np.abs(eigvals[1:])
    eigvals_process = np.clip(eigvals_process, epsilon, 1.0 - epsilon)
    timescales = -tau_msm * time_unit_in_ns / np.log(eigvals_process)
    
    results = {
        'koopman_op': koopman_op,
        'eigvals': eigvals,
        'eigvecs': eigvecs,
        'stationary': stationary,
        'timescales': timescales,
        'tau_msm': tau_msm,
        'tau_ns': tau_msm * time_unit_in_ns
    }
    
    return results


def plot_eigenvector(eigvec, title=None, figsize=(4, 4), save_path=None):
    """
    Plot a single eigenvector as a bar chart.
    
    Parameters
    ----------
    eigvec : np.ndarray
        Eigenvector to plot
    title : str, optional
        Title for the plot
    figsize : tuple, optional
        Figure size
    save_path : str, optional
        Path to save the figure
    """
    plt.figure(figsize=figsize)
    plt.bar(range(len(eigvec)), np.real(eigvec), width=0.4)
    plt.xticks(range(len(eigvec)))
    plt.xlabel('States')
    plt.ylabel('Eigvectors')
    if title:
        plt.title(title)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')


def plot_eigenanalysis(koopman_op, tau_msm, time_unit_in_ns=1.0, figsize=(4, 4),
                       save_prefix=None, show=True, normalize_stationary=True):
    """
    Plot eigenvectors and print eigenvalue analysis.
    
    Parameters
    ----------
    koopman_op : np.ndarray
        Koopman operator matrix
    tau_msm : int
        Lag time used for estimation
    time_unit_in_ns : float
        Time unit conversion
    figsize : tuple
        Figure size for each eigenvector plot
    save_prefix : str, optional
        Prefix for saving figures (e.g., 'eigvec' -> 'eigvec_0.png', 'eigvec_1.png')
    show : bool
        Whether to call plt.show()
    normalize_stationary : bool
        Whether to normalize the stationary eigenvector as a probability distribution
        
    Returns
    -------
    eigvals : np.ndarray
        Eigenvalues sorted by magnitude (descending)
    eigvecs : np.ndarray
        Corresponding eigenvectors
    """
    # Eigendecomposition of K^T
    eigvals, eigvecs = np.linalg.eig(koopman_op.T)
    
    # Sort by eigenvalue magnitude (descending)
    sorted_indices = sorted(enumerate(eigvals), key=lambda x: np.abs(x[1]), reverse=True)
    
    for rank, (i, eigval) in enumerate(sorted_indices):
        print(f'Eig {i}')
        print(f'Value: {eigval}')
        
        vec = np.real(eigvecs[:, i].copy())
        
        # Compute timescale (skip if eigenvalue is ~1)
        if np.abs(eigval) < 1 - 1e-6:
            timescale = -tau_msm / np.log(np.abs(eigval)) * time_unit_in_ns
            print(f'Timescale: {timescale} ns')
        else:
            print('Timescale: inf (stationary)')
            # Normalize stationary distribution to be positive and sum to 1
            if normalize_stationary:
                if vec.sum() < 0:
                    vec = -vec
                vec = vec / vec.sum()
                print(f'Stationary distribution: {vec}')
        
        print(f'Vector: {eigvecs[:, i]}')
        print()
        
        # Plot eigenvector
        save_path = f'{save_prefix}_{i}.png' if save_prefix else None
        plot_eigenvector(vec, title=f'Eigenvector {i}', 
                        figsize=figsize, save_path=save_path)
    
    if show:
        plt.show()
    
    return eigvals, eigvecs


def print_analysis_summary(results):
    """
    Print a summary of Koopman analysis results.
    
    Parameters
    ----------
    results : dict
        Output from analyze_koopman()
    """
    print("=" * 60)
    print("KOOPMAN OPERATOR ANALYSIS SUMMARY")
    print("=" * 60)
    
    print(f"\nLag time: {results['tau_msm']} steps ({results['tau_ns']:.4e} ns)")
    
    print("\n--- Koopman Operator ---")
    print(np.array2string(results['koopman_op'], precision=4, suppress_small=True))
    
    print("\n--- Eigenvalues of K^T ---")
    for i, ev in enumerate(results['eigvals']):
        print(f"  λ_{i+1} = {ev:.6f} (|λ| = {np.abs(ev):.6f})")
    
    print("\n--- Stationary Distribution ---")
    for i, p in enumerate(results['stationary']):
        print(f"  π_{i+1} = {p:.4f} ({p*100:.1f}%)")
    
    print("\n--- Implied Timescales ---")
    for i, ts in enumerate(results['timescales']):
        print(f"  τ_{i+1} = {ts:.4e} ns")
    
    print("=" * 60)


# =============================================================================
# Example usage
# =============================================================================

if __name__ == "__main__":
    # Example with synthetic data
    print("Running example with synthetic 2-state system...")
    
    # Create synthetic predictions (random walk between 2 states)
    np.random.seed(42)
    n_timesteps = 100000
    n_states = 2
    
    # Generate a simple Markov chain
    transition_prob = 0.01  # Probability of switching states
    states = np.zeros(n_timesteps, dtype=int)
    for t in range(1, n_timesteps):
        if np.random.random() < transition_prob:
            states[t] = 1 - states[t-1]
        else:
            states[t] = states[t-1]
    
    # Convert to "soft" predictions
    preds = np.zeros((n_timesteps, 1, n_states))
    preds[np.arange(n_timesteps), 0, states] = 0.9
    preds[np.arange(n_timesteps), 0, 1-states] = 0.1
    
    print(f"Predictions shape: {preds.shape}")
    
    # Analysis parameters
    time_unit_in_ns = 1e-4  # 0.1 ps per timestep
    tau_msm = 100
    max_tau = 500
    lags = np.arange(10, max_tau, 20)
    
    # Plot implied timescales
    print("\nPlotting implied timescales...")
    fig1, ax1, its = plot_timescales(
        preds, lags, 
        n_splits=1,
        time_unit_in_ns=time_unit_in_ns,
        save_path='its_plot.png'
    )
    
    # Plot CK test
    print("\nPlotting Chapman-Kolmogorov test...")
    fig2, ax2 = plot_ck_tests(
        preds, tau_msm, steps=10,
        n_splits=1,
        time_unit_in_ns=time_unit_in_ns,
        save_path='ck_test.png'
    )
    
    # Full Koopman analysis
    print("\nPerforming Koopman analysis...")
    results = analyze_koopman(preds, tau_msm, time_unit_in_ns=time_unit_in_ns)
    print_analysis_summary(results)
    
    plt.show()