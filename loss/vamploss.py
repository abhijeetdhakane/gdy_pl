"""
VAMP losses and gradients used for VAMPNet / GDyNet-style training.

Adapted from the Deeptime VAMPNet implementation (_vampnet.py, commit 8dc85b3):
https://github.com/deeptime-ml/deeptime

Core method introduced in VAMPNet:
Mardt, A., Pasquali, L., Wu, H., & Noé, F. (2018).
"VAMPnets for deep learning of molecular kinetics."
Nature Communications, 9(1), 5.

Related usage in graph-based dynamical learning:
Xie, T., France-Lanord, A., Wang, Y., Shao-Horn, Y., & Grossman, J. C. (2019).
"Graph dynamical networks for unsupervised learning of atomic scale dynamics in materials."
Nature Communications, 10, 2667.

Related work (GDyNet / ferroelectrics):
Dhakane, A., et al. (2023).
"A Graph Dynamical neural network approach for decoding dynamical states in ferroelectrics."
Carbon Trends, 11, 100264.

Modifications:
- PyTorch Geometric compatibility
- Added experimental `VAMP_sym` loss (symmetric mixed covariances)

Maintainer and primary contributor: Abhijeet Dhakane
"""


from typing import Optional, Union, Callable, Tuple

import numpy as np
import torch
import torch.nn as nn

# wrappers for older pytorch versions that lack linalg module
eigh = torch.linalg.eigh if hasattr(torch, 'linalg') else lambda x: torch.symeig(x, eigenvectors=True)
multi_dot = torch.linalg.multi_dot if hasattr(torch, 'linalg') and hasattr(torch.linalg, 'multi_dot') else \
    lambda args: torch.chain_matmul(*args)

def symeig_reg(mat, epsilon: float = 1e-6, mode='regularize', eigenvectors=True) \
        -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    r""" Solves a eigenvector/eigenvalue decomposition for a hermetian matrix also if it is rank deficient.
    Parameters
    ----------
    mat : torch.Tensor
        the hermetian matrix
    epsilon : float, default=1e-6
        Cutoff for eigenvalues.
    mode : str, default='regularize'
        Whether to truncate eigenvalues if they are too small or to regularize them by taking the absolute value
        and adding a small positive constant. :code:`trunc` leads to truncation, :code:`regularize` leads to epsilon
        being added to the eigenvalues after taking the absolute value
    eigenvectors : bool, default=True
        Whether to compute eigenvectors.
    Returns
    -------
    (eigval, eigvec) : Tuple[torch.Tensor, Optional[torch.Tensor]]
        Eigenvalues and -vectors.
    """
    assert mode in sym_inverse.valid_modes, f"Invalid mode {mode}, supported are {sym_inverse.valid_modes}"

    if mode == 'regularize':
        identity = torch.eye(mat.shape[0], dtype=mat.dtype, device=mat.device)
        mat = mat + epsilon * identity

    # Calculate eigvalues and potentially eigvectors
    eigval, eigvec = eigh(mat)

    if eigenvectors:
        eigvec = eigvec.transpose(0, 1)

    if mode == 'trunc':
        # Filter out Eigenvalues below threshold and corresponding Eigenvectors
        mask = eigval > epsilon
        eigval = eigval[mask]
        if eigenvectors:
            eigvec = eigvec[mask]
    elif mode == 'regularize':
        # Calculate eigvalues and eigvectors
        eigval = torch.abs(eigval)
    elif mode == 'clamp':
        eigval = torch.clamp_min(eigval, min=epsilon)

    return eigval, eigvec


def sym_inverse(mat, epsilon: float = 1e-6, return_sqrt=False, mode='regularize'):
    """ Utility function that returns the inverse of a matrix, with the
    option to return the square root of the inverse matrix.
    Parameters
    ----------
    mat: numpy array with shape [m,m]
        Matrix to be inverted.
    epsilon : float
        Cutoff for eigenvalues.
    return_sqrt: bool, optional, default = False
        if True, the square root of the inverse matrix is returned instead
    mode: str, default='trunc'
        Whether to truncate eigenvalues if they are too small or to regularize them by taking the absolute value
        and adding a small positive constant. :code:`trunc` leads to truncation, :code:`regularize` leads to epsilon
        being added to the eigenvalues after taking the absolute value
    Returns
    -------
    x_inv: numpy array with shape [m,m]
        inverse of the original matrix
    """
    eigval, eigvec = symeig_reg(mat, epsilon, mode)

    # Build the diagonal matrix with the filtered eigenvalues or square
    # root of the filtered eigenvalues according to the parameter
    if return_sqrt:
        diag = torch.diag(torch.sqrt(1. / eigval))
    else:
        diag = torch.diag(1. / eigval)

    return multi_dot([eigvec.t(), diag, eigvec])


sym_inverse.valid_modes = ('trunc', 'regularize', 'clamp')


def koopman_matrix(x: torch.Tensor, y: torch.Tensor, epsilon: float = 1e-6, mode: str = 'trunc',
                   c_xx: Optional[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = None) -> torch.Tensor:
    r""" Computes the Koopman matrix
    .. math:: K = C_{00}^{-1/2}C_{0t}C_{tt}^{-1/2}
    based on data over which the covariance matrices :math:`C_{\cdot\cdot}` are computed.
    Parameters
    ----------
    x : torch.Tensor
        Instantaneous data.
    y : torch.Tensor
        Time-lagged data.
    epsilon : float, default=1e-6
        Cutoff parameter for small eigenvalues.
    mode : str, default='trunc'
        Regularization mode for Hermetian inverse. See :meth:`sym_inverse`.
    c_xx : tuple of torch.Tensor, optional, default=None
        Tuple containing c00, c0t, ctt if already computed.
    Returns
    -------
    K : torch.Tensor
        The Koopman matrix.
    """
    if c_xx is not None:
        c00, c0t, ctt = c_xx
    else:
        _, _, _, c00, c0t, ctt, _ = covariances(x, y, remove_mean=True)

    c00_sqrt_inv = sym_inverse(c00, return_sqrt=True, epsilon=epsilon, mode=mode)
    ctt_sqrt_inv = sym_inverse(ctt, return_sqrt=True, epsilon=epsilon, mode=mode)

    return multi_dot([c00_sqrt_inv, c0t, ctt_sqrt_inv]).t()

def covariances(x: torch.Tensor, y: torch.Tensor, remove_mean: bool = True) -> torch.Tensor:
    """Computes instantaneous and time-lagged covariances matrices.
    Parameters
    ----------
    x : (T, n) torch.Tensor
        Instantaneous data.
    y : (T, n) torch.Tensor
        Time-lagged data.
    remove_mean: bool, default=True
        Whether to remove the mean of x and y.
    Returns
    -------
    cov_00 : (n, n) torch.Tensor
        Auto-covariance matrix of x.
    cov_0t : (n, n) torch.Tensor
        Cross-covariance matrix of x and y.
    cov_tt : (n, n) torch.Tensor
        Auto-covariance matrix of y.
    See Also
    --------
    deeptime.covariance.Covariance : Estimator yielding these kind of covariance matrices based on raw numpy arrays
                                     using an online estimation procedure.
    """

    assert x.shape == y.shape, "x and y must be of same shape"
    batch_size = x.shape[0]

    if remove_mean:
        x = x - x.mean(dim=0, keepdim=True)
        y = y - y.mean(dim=0, keepdim=True)

    # Calculate the cross-covariance
    y_t = y.transpose(0, 1)
    x_t = x.transpose(0, 1)
    cov_01 = 1 / (batch_size - 1) * torch.matmul(x_t, y)
    cov_10 = 1 / (batch_size - 1) * torch.matmul(y_t, x)

    # Calculate the auto-correlations
    cov_00 = 1 / (batch_size - 1) * torch.matmul(x_t, x)
    cov_11 = 1 / (batch_size - 1) * torch.matmul(y_t, y)

    return x_t, y_t, batch_size, cov_00, cov_01, cov_11, cov_10


valid_score_methods = ('VAMP1', 'VAMP2','VAMP_sym')

def vamp_score(data: torch.Tensor, data_lagged: torch.Tensor, method='VAMP2', epsilon: float = 1e-6, mode='trunc'):
    r"""Computes the VAMP score based on data and corresponding time-shifted data.
    Parameters
    ----------
    data : torch.Tensor
        (N, d)-dimensional torch tensor
    data_lagged : torch.Tensor
        (N, k)-dimensional torch tensor
    method : str, default='VAMP2'
        The scoring method. See :meth:`score <deeptime.decomposition.CovarianceKoopmanModel.score>` for details.
    epsilon : float, default=1e-6
        Cutoff parameter for small eigenvalues, alternatively regularization parameter.
    mode : str, default='trunc'
        Regularization mode for Hermetian inverse. See :meth:`sym_inverse`.
    Returns
    -------
    score : torch.Tensor
        The score. It contains a contribution of :math:`+1` for the constant singular function since the
        internally estimated Koopman operator is defined on a decorrelated basis set.
    """
    assert method in valid_score_methods, f"Invalid method '{method}', supported are {valid_score_methods}"
    assert data.shape == data_lagged.shape, f"Data and data_lagged must be of same shape but were {data.shape} " \
                                            f"and {data_lagged.shape}."
    out = None
    if method == 'VAMP1':
        koopman = koopman_matrix(data, data_lagged, epsilon=epsilon, mode=mode)
        out = torch.norm(koopman, p='nuc')

        return out

    elif method == 'VAMP2':
        koopman = koopman_matrix(data, data_lagged, epsilon=epsilon, mode=mode)
        out = torch.pow(torch.norm(koopman, p='fro'), 2)

        return out

    elif method == 'VAMP_sym':
        """
        # -------------------------------------------------------------------------
        # Experimental symmetric mixed-covariance gradient-style loss (TF-inspired)
        # -------------------------------------------------------------------------
        # IMPORTANT: this branch intentionally detaches inputs 
        """

        x, y, batch_size, c00, c0t, ctt, ct0 = covariances(data.clone().detach(), data_lagged.clone().detach(), remove_mean=True)

        cross_cov = 0.5 * (c0t + ct0)
        auto_cov = 0.5 * (c00 + ctt)

        auto_cov_sqrt_inv = sym_inverse(auto_cov, epsilon=epsilon, return_sqrt=True, mode=mode)

        koopman = torch.matmul(auto_cov_sqrt_inv, torch.matmul(cross_cov, auto_cov_sqrt_inv))

        U,D,V = torch.linalg.svd(koopman, full_matrices=True)
        diag = torch.diag(D)

        # Base-changed covariance matrices
        x_base = torch.matmul(auto_cov_sqrt_inv, U)
        y_base = torch.matmul(V, auto_cov_sqrt_inv)

        # Derivative for the output of both networks.
        nabla_01 = torch.matmul(x_base, y_base)
        nabla_00 = -0.5 * torch.linalg.multi_dot([x_base,diag,x_base.t()])

        # Derivative for the output of both networks.
        x_der = 2/(batch_size - 1) * (torch.matmul(nabla_00, x) + torch.matmul(nabla_01, y))
        y_der = 2/(batch_size - 1) * (torch.matmul(nabla_00, y) + torch.matmul(nabla_01, x))

        x_1d = torch.t(x_der)
        y_1d = torch.t(y_der)

        # Concatenate it again
        concat_derivatives = torch.cat([x_1d,y_1d], dim=-1)

        concat_preds = torch.cat([x.t(),y.t()], dim=-1)

        out = - concat_derivatives * concat_preds

        return out.mean() ,x.t(), -x_1d, y.t(), -y_1d

def vamp_metric_calc(data: torch.Tensor, data_lagged: torch.Tensor, method='VAMP2', epsilon: float = 1e-6, mode='trunc'):
    r"""Computes the VAMP score based on data and corresponding time-shifted data.
    Parameters
    ----------
    data : torch.Tensor
        (N, d)-dimensional torch tensor
    data_lagged : torch.Tensor
        (N, k)-dimensional torch tensor
    method : str, default='VAMP2'
        The scoring method. See :meth:`score <deeptime.decomposition.CovarianceKoopmanModel.score>` for details.
    epsilon : float, default=1e-6
        Cutoff parameter for small eigenvalues, alternatively regularization parameter.
    mode : str, default='trunc'
        Regularization mode for Hermetian inverse. See :meth:`sym_inverse`.
    Returns
    -------
    score : torch.Tensor
        The score. It contains a contribution of :math:`+1` for the constant singular function since the
        internally estimated Koopman operator is defined on a decorrelated basis set.
    """
    assert method in valid_score_methods, f"Invalid method '{method}', supported are {valid_score_methods}"
    assert data.shape == data_lagged.shape, f"Data and data_lagged must be of same shape but were {data.shape} " \
                                            f"and {data_lagged.shape}."
    out = None
    if method == 'VAMP1':
        koopman = koopman_matrix(data, data_lagged, epsilon=epsilon, mode=mode)
        # Select the K highest singular values of the VAMP matrix
        #u, s, v = torch.svd(vamp_matrix)
        #diag = s
        diag = torch.linalg.svd(koopman, full_matrices=False).S
        # Option to select top k singular values (currently k=0 means use all)
        k = 0
        cond = k > 0
        top_k_val = torch.topk(diag, k=k)[0] if k > 0 else diag

        # Sum the singular values
        eig_sum = torch.sum(top_k_val) if cond else torch.sum(diag)

        return eig_sum

    elif method == 'VAMP2':
        koopman = koopman_matrix(data, data_lagged, epsilon=epsilon, mode=mode)
        # Select the K highest singular values of the VAMP matrix
        diag = torch.linalg.svd(koopman, full_matrices=False).S
        # Option to select top k singular values (currently k=0 means use all)
        k = 0
        cond = k > 0
        top_k_val = torch.topk(diag, k=k)[0] if k > 0 else diag

        # Square the singular values and sum them
        pow2_topk = torch.sum(top_k_val ** 2)
        pow2_diag = torch.sum(diag ** 2)
        eig_sum_sq = pow2_topk if cond else pow2_diag

        return eig_sum_sq

def vampnet_loss(data: torch.Tensor, data_lagged: torch.Tensor, method='VAMP2', epsilon: float = 1e-6, mode: str = 'trunc'):
    r"""Loss function that can be used to train VAMPNets. It evaluates as :math:`-\mathrm{score}`. The score
    is implemented in :meth:`score`."""
    return -1. * vamp_score(data, data_lagged, method=method, epsilon=epsilon, mode=mode)

def vamp_metric(data: torch.Tensor, data_lagged: torch.Tensor, epsilon: float = 1e-6, mode: str = 'trunc'):
    r"""Metric that can be used to train VAMPNets. It evaluates as :math:`-\mathrm{score}`. The score
    is implemented in :meth:`score`."""
    vamp1 = vamp_metric_calc(data, data_lagged, method='VAMP1', epsilon=epsilon, mode=mode)
    vamp2 = vamp_metric_calc(data, data_lagged, method='VAMP2', epsilon=epsilon, mode=mode)
    return vamp1, vamp2
