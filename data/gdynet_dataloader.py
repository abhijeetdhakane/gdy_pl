from typing import List
import os
import warnings

import numpy as np
import torch
from torch.utils.data import Dataset
from torch_geometric.data import Data
from torch_geometric.nn.models.schnet import GaussianSmearing


class MyData(Data):
    def __init__(self, **kwargs):
        super(MyData, self).__init__(**kwargs)

    def __inc__(self, key, value, *args, **kwargs):
        if key == 'target':
            return self.num_nodes
        return super().__inc__(key, value, *args, **kwargs)


class PyGMDStackGen_vanilla(Dataset):
    """
    Vanilla dataloader without atom direction features.
    For use with gdynet_vanilla model.

    Loads from four .npy files:
      atom_types:   (N,)
      target_index: (n,)
      nbr_lists:    (F, N, n_nbrs)
      nbr_dists:    (F, N, n_nbrs)
    """

    def __init__(self, fnames: List[str], tau: int, cutoff: float, num_gaussians: int):
        super().__init__()

        self.fnames = fnames
        self.tau = int(tau)
        self.cutoff = float(cutoff)
        self.num_gaussians = int(num_gaussians)

        self.distance_expansion = GaussianSmearing(0., self.cutoff, self.num_gaussians)
        self.n_traj = 1

        # Validate required files exist before loading
        self._validate_required_files()

        # ------------------------------------------------------------
        # Small constant arrays: load normally (NOT memmap) and cache as torch tensors
        # This avoids the "non-writable" warning since regular np.load returns writable arrays
        # ------------------------------------------------------------
        atom_types_np = np.load(self.fnames[0])  # writable ndarray
        target_index_np = np.load(self.fnames[1])  # writable ndarray

        # Convert once in __init__ and reuse - no warning since arrays are writable
        self.atom_types_t = torch.from_numpy(atom_types_np).long().squeeze()
        self.target_index_t = torch.from_numpy(target_index_np).long().squeeze()

        # Store numpy versions for validation
        self._atom_types_np = atom_types_np
        self._target_index_np = target_index_np

        # ------------------------------------------------------------
        # Big arrays: keep memmapped read-only for memory efficiency
        # ------------------------------------------------------------
        self.nbr_lists = np.load(self.fnames[2], mmap_mode='r')
        self.nbr_dists = np.load(self.fnames[3], mmap_mode='r')

        # Shapes / indices
        self.n_frames = int(self.nbr_lists.shape[0])
        self.num_atoms = int(self.nbr_lists.shape[1])
        self.num_nbrs = int(self.nbr_lists.shape[2])

        if self.n_frames <= self.tau:
            raise ValueError(f"n_frames ({self.n_frames}) must be > tau ({self.tau}).")

        self.indices = np.arange(self.n_frames - self.tau, dtype=np.int64)

        # Precompute source indices (same for every sample)
        self._src = torch.arange(self.num_atoms, dtype=torch.long).view(-1, 1).repeat(1, self.num_nbrs).reshape(-1)

        # Validate data after loading
        self._validate_target_index()
        self._validate_shapes()

        print(f'atom_types      : {atom_types_np.shape}  (cached torch: {tuple(self.atom_types_t.shape)})')
        print(f'target_index    : {target_index_np.shape} (cached torch: {tuple(self.target_index_t.shape)})')
        print(f'nbr_lists       : {self.nbr_lists.shape} (memmap)')
        print(f'nbr_dists       : {self.nbr_dists.shape} (memmap)')
        print(f'num_samples     : {len(self.indices)}')
        print("VANILLA DATA LOADING!! (Validation passed)")

    def __len__(self):
        return int(self.indices.shape[0])

    @staticmethod
    def _to_writable_tensor(arr, dtype):
        """
        Convert a potentially read-only numpy array to a writable torch tensor.
        Uses np.array() to create a writable copy before conversion.
        """
        # np.array() always returns a writable copy
        writable_arr = np.array(arr, dtype=dtype, order='C')
        return torch.from_numpy(writable_arr)

    def __getitem__(self, index: int):
        idx = int(self.indices[index])

        # --- Neighbor lists (memmap slices) -> writable copy -> torch ---
        nbr1 = self._to_writable_tensor(self.nbr_lists[idx].ravel(), dtype=np.int64)
        nbr2 = self._to_writable_tensor(self.nbr_lists[idx + self.tau].ravel(), dtype=np.int64)

        # edge_index shape: (2, N*n_nbrs)
        edge_list1 = torch.stack((self._src, nbr1), dim=0)
        edge_list2 = torch.stack((self._src, nbr2), dim=0)

        # --- Neighbor distances (memmap slices) -> writable copy -> torch -> expand ---
        d1 = self._to_writable_tensor(self.nbr_dists[idx].ravel(), dtype=np.float32)
        d2 = self._to_writable_tensor(self.nbr_dists[idx + self.tau].ravel(), dtype=np.float32)

        nbr_dist_1 = self.distance_expansion(d1)
        nbr_dist_2 = self.distance_expansion(d2)

        # Use cached constant tensors from __init__
        data_t = MyData(
            x=self.atom_types_t,
            edge_index=edge_list1,
            edge_attr=nbr_dist_1,
            target=self.target_index_t,
            indc=torch.tensor([idx], dtype=torch.long)
        )
        data_lt = MyData(
            x=self.atom_types_t,
            edge_index=edge_list2,
            edge_attr=nbr_dist_2,
            target=self.target_index_t,
            indc=torch.tensor([idx], dtype=torch.long)
        )

        return data_t, data_lt

    def _validate_required_files(self):
        """Validate that all required data files exist."""
        required_files = ['atom_types.npy', 'target_index.npy', 'nbr_lists.npy', 'nbr_dists.npy']

        if len(self.fnames) < 4:
            raise ValueError(
                f"Expected 4 file paths for vanilla dataset, got {len(self.fnames)}.\n"
                f"Required files: {required_files}"
            )

        missing_files = [fpath for fpath in self.fnames[:4] if not os.path.exists(fpath)]

        if missing_files:
            raise FileNotFoundError(
                f"Missing required data files for PyGMDStackGen_vanilla:\n"
                f"Missing files: {missing_files}\n"
                f"Required files: {required_files}"
            )

    def _validate_target_index(self):
        """Validate target_index values are within valid atom index range."""
        max_target_idx = self._target_index_np.max()
        num_atoms = self._atom_types_np.shape[0]

        if max_target_idx >= num_atoms:
            raise ValueError(
                f"Invalid target_index detected!\n"
                f"  Max target_index: {max_target_idx}\n"
                f"  Number of atoms: {num_atoms}\n"
                f"  All target indices must be < {num_atoms} (valid range: 0-{num_atoms-1})"
            )

        if self._target_index_np.min() < 0:
            raise ValueError(
                f"target_index contains negative values!\n"
                f"  Min target_index: {self._target_index_np.min()}\n"
                f"  All indices must be >= 0"
            )

        unique_targets = np.unique(self._target_index_np)
        if len(unique_targets) < len(self._target_index_np):
            warnings.warn(
                f"target_index contains {len(self._target_index_np) - len(unique_targets)} duplicate entries. "
                f"This may be intentional, but please verify your data.",
                UserWarning
            )

    def _validate_shapes(self):
        """Validate data shape consistency across arrays."""
        if self.nbr_lists.shape != self.nbr_dists.shape:
            raise ValueError(
                f"Shape mismatch between nbr_lists and nbr_dists!\n"
                f"  nbr_lists shape: {self.nbr_lists.shape}\n"
                f"  nbr_dists shape: {self.nbr_dists.shape}\n"
                f"  These must be identical: (F, N, n_nbrs)"
            )

        if self.nbr_lists.shape[1] != self.num_atoms:
            raise ValueError(
                f"Inconsistent number of atoms!\n"
                f"  nbr_lists reports: {self.nbr_lists.shape[1]} atoms\n"
                f"  Expected: {self.num_atoms} atoms"
            )


class PyGMDStackGen_ferro(Dataset):
    """
    Enhanced dataloader with atom direction features.
    For use with gdynet_ferro model.

    Loads from five .npy files:
      atom_types:      (N,)
      target_index:    (n,)
      nbr_lists:       (F, N, n_nbrs)
      nbr_dists:       (F, N, n_nbrs)
      atom_directions: (F, N, 3) -- LOCAL polarization of each unit cell centered at Ti (BaTiO3)
    """

    def __init__(self, fnames: List[str], tau: int, cutoff: float, num_gaussians: int):
        super().__init__()

        self.fnames = fnames
        self.tau = int(tau)
        self.cutoff = float(cutoff)
        self.num_gaussians = int(num_gaussians)

        self.distance_expansion = GaussianSmearing(0., self.cutoff, self.num_gaussians)
        self.n_traj = 1

        # Validate required files exist before loading
        self._validate_required_files()

        # ------------------------------------------------------------
        # Small constant arrays: load normally (NOT memmap) and cache as torch tensors
        # ------------------------------------------------------------
        atom_types_np = np.load(self.fnames[0])
        target_index_np = np.load(self.fnames[1])

        self.atom_types_t = torch.from_numpy(atom_types_np).long().squeeze()
        self.target_index_t = torch.from_numpy(target_index_np).long().squeeze()

        self._atom_types_np = atom_types_np
        self._target_index_np = target_index_np

        # ------------------------------------------------------------
        # Big arrays: keep memmapped read-only for memory efficiency
        # ------------------------------------------------------------
        self.nbr_lists = np.load(self.fnames[2], mmap_mode='r')
        self.nbr_dists = np.load(self.fnames[3], mmap_mode='r')
        self.atom_directions = np.load(self.fnames[4], mmap_mode='r')

        # Shapes / indices
        self.n_frames = int(self.nbr_lists.shape[0])
        self.num_atoms = int(self.nbr_lists.shape[1])
        self.num_nbrs = int(self.nbr_lists.shape[2])

        if self.n_frames <= self.tau:
            raise ValueError(f"n_frames ({self.n_frames}) must be > tau ({self.tau}).")

        self.indices = np.arange(self.n_frames - self.tau, dtype=np.int64)

        # Precompute source indices
        self._src = torch.arange(self.num_atoms, dtype=torch.long).view(-1, 1).repeat(1, self.num_nbrs).reshape(-1)

        # Validate data after loading
        self._validate_target_index()
        self._validate_shapes()

        print(f'atom_types      : {atom_types_np.shape}  (cached torch: {tuple(self.atom_types_t.shape)})')
        print(f'target_index    : {target_index_np.shape} (cached torch: {tuple(self.target_index_t.shape)})')
        print(f'nbr_lists       : {self.nbr_lists.shape} (memmap)')
        print(f'nbr_dists       : {self.nbr_dists.shape} (memmap)')
        print(f'atom_directions : {self.atom_directions.shape} (memmap)')
        print(f'num_samples     : {len(self.indices)}')
        print("FERRO DATA LOADING!! (Validation passed)")

    def __len__(self):
        return int(self.indices.shape[0])

    @staticmethod
    def _to_writable_tensor(arr, dtype):
        """
        Convert a potentially read-only numpy array to a writable torch tensor.
        Uses np.array() to create a writable copy before conversion.
        """
        writable_arr = np.array(arr, dtype=dtype, order='C')
        return torch.from_numpy(writable_arr)

    def __getitem__(self, index: int):
        idx = int(self.indices[index])

        # --- Neighbor lists ---
        nbr1 = self._to_writable_tensor(self.nbr_lists[idx].ravel(), dtype=np.int64)
        nbr2 = self._to_writable_tensor(self.nbr_lists[idx + self.tau].ravel(), dtype=np.int64)

        edge_list1 = torch.stack((self._src, nbr1), dim=0)
        edge_list2 = torch.stack((self._src, nbr2), dim=0)

        # --- Neighbor distances ---
        d1 = self._to_writable_tensor(self.nbr_dists[idx].ravel(), dtype=np.float32)
        d2 = self._to_writable_tensor(self.nbr_dists[idx + self.tau].ravel(), dtype=np.float32)

        nbr_dist_1 = self.distance_expansion(d1)
        nbr_dist_2 = self.distance_expansion(d2)

        # --- Atom directions ---
        atom_dir_1 = self._to_writable_tensor(self.atom_directions[idx], dtype=np.float32).view(-1, 3)
        atom_dir_2 = self._to_writable_tensor(self.atom_directions[idx + self.tau], dtype=np.float32).view(-1, 3)

        # Use cached constant tensors
        data_t = MyData(
            x=self.atom_types_t,
            edge_index=edge_list1,
            edge_attr=nbr_dist_1,
            atom_direction=atom_dir_1,
            target=self.target_index_t,
            indc=torch.tensor([idx], dtype=torch.long)
        )
        data_lt = MyData(
            x=self.atom_types_t,
            edge_index=edge_list2,
            edge_attr=nbr_dist_2,
            atom_direction=atom_dir_2,
            target=self.target_index_t,
            indc=torch.tensor([idx], dtype=torch.long)
        )

        return data_t, data_lt

    def _validate_required_files(self):
        """Validate that all required data files exist."""
        required_files = ['atom_types.npy', 'target_index.npy', 'nbr_lists.npy', 'nbr_dists.npy', 'atom_directions.npy']

        if len(self.fnames) < 5:
            raise ValueError(
                f"Expected 5 file paths for ferro dataset, got {len(self.fnames)}.\n"
                f"Required files: {required_files}"
            )

        missing_files = [fpath for fpath in self.fnames[:5] if not os.path.exists(fpath)]

        if missing_files:
            raise FileNotFoundError(
                f"Missing required data files for PyGMDStackGen_ferro:\n"
                f"Missing files: {missing_files}\n"
                f"Required files: {required_files}"
            )

    def _validate_target_index(self):
        """Validate target_index values are within valid atom index range."""
        max_target_idx = self._target_index_np.max()
        num_atoms = self._atom_types_np.shape[0]

        if max_target_idx >= num_atoms:
            raise ValueError(
                f"Invalid target_index detected!\n"
                f"  Max target_index: {max_target_idx}\n"
                f"  Number of atoms: {num_atoms}\n"
                f"  All target indices must be < {num_atoms} (valid range: 0-{num_atoms-1})"
            )

        if self._target_index_np.min() < 0:
            raise ValueError(
                f"target_index contains negative values!\n"
                f"  Min target_index: {self._target_index_np.min()}\n"
                f"  All indices must be >= 0"
            )

        unique_targets = np.unique(self._target_index_np)
        if len(unique_targets) < len(self._target_index_np):
            warnings.warn(
                f"target_index contains {len(self._target_index_np) - len(unique_targets)} duplicate entries. "
                f"This may be intentional, but please verify your data.",
                UserWarning
            )

    def _validate_shapes(self):
        """Validate data shape consistency across arrays."""
        if self.nbr_lists.shape != self.nbr_dists.shape:
            raise ValueError(
                f"Shape mismatch between nbr_lists and nbr_dists!\n"
                f"  nbr_lists shape: {self.nbr_lists.shape}\n"
                f"  nbr_dists shape: {self.nbr_dists.shape}\n"
                f"  These must be identical: (F, N, n_nbrs)"
            )

        if self.nbr_lists.shape[1] != self.num_atoms:
            raise ValueError(
                f"Inconsistent number of atoms!\n"
                f"  nbr_lists reports: {self.nbr_lists.shape[1]} atoms\n"
                f"  Expected: {self.num_atoms} atoms"
            )

        expected_directions_shape = (self.n_frames, self.num_atoms, 3)
        if self.atom_directions.shape != expected_directions_shape:
            raise ValueError(
                f"Invalid atom_directions shape!\n"
                f"  Current shape: {self.atom_directions.shape}\n"
                f"  Expected shape: {expected_directions_shape} (F, N, 3)\n"
                f"  where F={self.n_frames}, N={self.num_atoms}"
            )
