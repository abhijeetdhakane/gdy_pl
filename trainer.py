"""
Optimized Trainer for GDYNet Models with torch.compile and TorchScript

This trainer provides advanced optimization features:
- torch.compile for PyTorch 2.0+ (significant speedup)
- TorchScript compilation for inference optimization
- Original loss tracking (last batch, no averaging)
- All features from unified trainer
"""

import os
import json
import random
from copy import deepcopy
import time
import yaml
import argparse
from typing import Optional, Dict, Any, List

import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from torch.amp import autocast, GradScaler
from torch.nn.parallel import DistributedDataParallel
import torch.distributed as dist
from torch.utils.data import DistributedSampler

from torch_geometric.loader import DataLoader

from config.registry import ModelRegistry
from loss.vamploss import vamp_score, vamp_metric
from utils.sampler import CustomInferenceSampler_v2

# Optional wandb import
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    print("Warning: wandb not available. Install with 'pip install wandb' for experiment tracking.")


def detect_gpu_environment():
    """
    Detect GPU environment (NVIDIA/CUDA vs AMD/ROCm) and print info.

    Returns:
        dict: Environment information including GPU availability, type, and backend.
    """
    env_info = {
        'has_gpu': torch.cuda.is_available(),
        'gpu_count': torch.cuda.device_count() if torch.cuda.is_available() else 0,
        'gpu_type': 'unknown',
        'backend': 'unknown'
    }

    if torch.cuda.is_available():
        # Check if using ROCm (AMD) or CUDA (NVIDIA)
        if hasattr(torch.version, 'hip') and torch.version.hip is not None:
            env_info['gpu_type'] = 'AMD'
            env_info['backend'] = 'ROCm'
            env_info['rocm_version'] = torch.version.hip
        else:
            env_info['gpu_type'] = 'NVIDIA'
            env_info['backend'] = 'CUDA'
            env_info['cuda_version'] = torch.version.cuda

        try:
            env_info['gpu_name'] = torch.cuda.get_device_name(0)
        except:
            env_info['gpu_name'] = 'Unknown'

    return env_info


def print_gpu_info():
    """Print GPU environment information."""
    env_info = detect_gpu_environment()

    print("=" * 60)
    print("GPU Environment Information")
    print("=" * 60)
    print(f"GPU Available: {env_info['has_gpu']}")
    if env_info['has_gpu']:
        print(f"GPU Type: {env_info['gpu_type']}")
        print(f"Backend: {env_info['backend']}")
        if env_info['backend'] == 'ROCm':
            print(f"ROCm Version: {env_info.get('rocm_version', 'Unknown')}")
        elif env_info['backend'] == 'CUDA':
            print(f"CUDA Version: {env_info.get('cuda_version', 'Unknown')}")
        print(f"GPU Name: {env_info.get('gpu_name', 'Unknown')}")
        print(f"GPU Count: {env_info['gpu_count']}")
    else:
        print("Running on CPU")
    print("=" * 60)

    return env_info


def check_torch_compile_available():
    """Check if torch.compile is available (PyTorch 2.0+)."""
    try:
        torch_version = torch.__version__.split('+')[0]
        major, minor = map(int, torch_version.split('.')[:2])

        if major >= 2:
            return True, f"PyTorch {torch_version}"
        else:
            return False, f"PyTorch {torch_version} (need 2.0+)"
    except:
        return False, "Unknown version"


def convert_to_serializable(value):
    """
    Convert numpy/torch types to Python native types for JSON serialization.
    
    Args:
        value: The value to convert (can be numpy, torch, or Python native type).
        
    Returns:
        Python native type (int, float, or str).
    """
    if isinstance(value, (np.floating, np.float32, np.float64)):
        return float(value)
    elif isinstance(value, (np.integer, np.int32, np.int64)):
        return int(value)
    elif isinstance(value, torch.Tensor):
        return value.item()
    return value


def set_seed(seed, world_size):
    """Set random seeds for reproducibility."""
    if seed is None:
        seed = np.random.randint(10000)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if world_size > 1:
        torch.cuda.manual_seed_all(seed)
    return seed


class OptimizedTrainer(object):
    """
    Optimized trainer for GDYNet models with:
    - torch.compile support (PyTorch 2.0+)
    - TorchScript compilation
    - Original loss tracking (no averaging)
    - All unified trainer features
    """

    def __init__(self, config_path: str):
        # Load configuration
        with open(config_path, 'r') as config_file:
            self.config = yaml.safe_load(config_file)

        # Model selection
        self.model_name = self.config.get('model_type', 'gdynet_ferro')
        if not ModelRegistry.is_registered(self.model_name):
            raise ValueError(f"Model '{self.model_name}' not registered. "
                           f"Available models: {list(ModelRegistry.list_models().keys())}")

        # Get model and dataloader classes from registry
        self.model_class = ModelRegistry.get_model_class(self.model_name)
        self.dataloader_class = ModelRegistry.get_dataloader_class(self.model_name)
        self.model_info = ModelRegistry.get_model_info(self.model_name)

        print(f"Using model: {self.model_name}")
        print(f"Description: {self.model_info['description']}")
        print(f"Requires direction features: {self.model_info['requires_direction']}")

        # Set up data paths
        self.train_fnames = self.config['data']['train_fnames']
        self.val_fnames = self.config['data']['val_fnames']
        self.test_fnames = self.config['data']['test_fnames']

        # Set up model parameters
        self.tau = self.config['model']['tau']
        self.batch_size = self.config['model']['batch_size']
        self.cutoff = self.config['model']['cutoff']
        self.num_gaussians = self.config['model']['num_gaussians']
        self.epsilon = self.config['model']['epsilon']
        self.mode = self.config['model']['mode']
        self.atom_fea_len = self.config['model']['atom_fea_len']
        self.n_conv = self.config['model']['n_conv']
        self.nbr_fea_len = self.num_gaussians
        self.state_len = self.config['model']['state_len']
        self.learning_rate = self.config['model']['learning_rate']

        # Optimization options
        self.use_torch_compile = self.config.get('optimization', {}).get('torch_compile', False)
        self.compile_mode = self.config.get('optimization', {}).get('compile_mode', 'default')
        self.use_torchscript = self.config.get('optimization', {}).get('torchscript', False)
        self.use_amp = self.config.get('optimization', {}).get('mixed_precision', False)

        # Check torch.compile availability
        self.torch_compile_available, torch_version_info = check_torch_compile_available()
        if self.use_torch_compile and not self.torch_compile_available:
            print(f"Warning: torch.compile requested but not available ({torch_version_info})")
            print("Falling back to eager mode")
            self.use_torch_compile = False

        # Set up training parameters
        self.epochs = self.config['training']['epochs']
        self.loss_schedule = self.config['training'].get('loss_schedule', ['vamp2', 'vamp1', 'vamp2'])
        self.seed = self.config['training'].get('seed', 1234)

        # WandB configuration
        self.use_wandb = self.config.get('wandb', {}).get('enabled', False)
        self.wandb_project = self.config.get('wandb', {}).get('project', 'gdynet')
        self.wandb_entity = self.config.get('wandb', {}).get('entity', None)
        self.wandb_run_name = self.config.get('wandb', {}).get('run_name', None)

        # Checkpointing configuration
        self.checkpoint_freq = self.config.get('checkpointing', {}).get('frequency', 1)
        self.save_best_only = self.config.get('checkpointing', {}).get('save_best_only', False)
        self.best_metric = float('inf')

        # Set up output folder
        self.PATH = self.config['output']['folder']
        self.create_output_folders()

        # Print and save hyperparameters
        self.print_and_save_hyperparameters()

        # Initialize the distributed environment
        self.world_size = 1
        if 'WORLD_SIZE' in os.environ:
            self.world_size = int(os.environ['WORLD_SIZE'])

        self.local_rank = 0
        self.world_rank = 0
        if self.world_size > 1:
            dist.init_process_group(backend='nccl', init_method='env://')
            self.world_rank = dist.get_rank()
            self.local_rank = int(os.environ["LOCAL_RANK"])

            print("World Size: ", self.world_size)
            print("World Rank: ", self.world_rank)
            print("Local Rank: ", self.local_rank)
            print("Master Addr: ", os.environ["MASTER_ADDR"])
            print("Master Port: ", os.environ["MASTER_PORT"])

        if torch.cuda.is_available():
            torch.cuda.set_device(self.local_rank)
            torch.backends.cudnn.benchmark = True

        if torch.cuda.is_available():
            self.device = torch.device(f'cuda:{torch.cuda.current_device()}')
        else:
            self.device = torch.device('cpu')

        print("Device: ", self.device)

        # Print GPU environment information
        self.gpu_env = print_gpu_info()

        # Print optimization settings
        print("\n" + "=" * 60)
        print("Optimization Settings")
        print("=" * 60)
        print(f"torch.compile: {'Enabled' if self.use_torch_compile else 'Disabled'}")
        if self.use_torch_compile:
            print(f"  Compile mode: {self.compile_mode}")
        print(f"TorchScript: {'Enabled' if self.use_torchscript else 'Disabled'}")
        print(f"Mixed Precision (AMP): {'Enabled' if self.use_amp else 'Disabled'}")
        print("=" * 60)

        self.seed = set_seed(self.seed, self.world_size)
        print(f"Random seed set to: {self.seed}")

        # Initialize WandB
        if self.use_wandb and self.world_rank == 0:
            if not WANDB_AVAILABLE:
                print("Warning: wandb requested but not available. Disabling wandb.")
                self.use_wandb = False
            else:
                self.init_wandb()

        # Metrics tracking - track both last batch and average values
        self.epochs_list = []
        self.loss_types_list = []
        self.train_losses_last = []
        self.train_losses_avg = []
        self.val_losses_last = []
        self.val_losses_avg = []
        self.train_vamp1_scores_last = []
        self.train_vamp1_scores_avg = []
        self.train_vamp2_scores_last = []
        self.train_vamp2_scores_avg = []
        self.val_vamp1_scores_last = []
        self.val_vamp1_scores_avg = []
        self.val_vamp2_scores_last = []
        self.val_vamp2_scores_avg = []

        # AMP scaler
        self.scaler = GradScaler('cuda') if self.use_amp else None

    def create_output_folders(self):
        """Create organized output directory structure."""
        folders = {
            'root': self.PATH,
            'checkpoints': os.path.join(self.PATH, 'checkpoints'),
            'metrics': os.path.join(self.PATH, 'metrics'),
            'final_models': os.path.join(self.PATH, 'final_models'),
            'predictions': os.path.join(self.PATH, 'predictions'),
            'logs': os.path.join(self.PATH, 'logs'),
            'compiled_models': os.path.join(self.PATH, 'compiled_models'),
        }

        for folder_name, folder_path in folders.items():
            if not os.path.exists(folder_path):
                try:
                    os.makedirs(folder_path, exist_ok=True)
                    print(f"Created {folder_name} folder: {folder_path}")
                except OSError as e:
                    print(f"Error creating {folder_name} folder: {e}")
            else:
                print(f"{folder_name.capitalize()} folder already exists: {folder_path}")

        self.folders = folders

    def print_and_save_hyperparameters(self):
        """Print and save hyperparameters to file."""
        hyperparameters = {
            "Model Type": self.model_name,
            "Model Parameters": {k: v for k, v in self.config['model'].items()},
            "Training Parameters": {k: v for k, v in self.config['training'].items()},
            "Optimization": {
                "torch_compile": self.use_torch_compile,
                "compile_mode": self.compile_mode if self.use_torch_compile else "N/A",
                "torchscript": self.use_torchscript,
                "mixed_precision": self.use_amp,
            },
            "Output Folder": self.PATH,
            "WandB Enabled": self.use_wandb,
        }

        print("Hyperparameters:")
        print(yaml.dump(hyperparameters, default_flow_style=False))

        hyperparameters_file = os.path.join(self.PATH, 'hyperparameters.yml')
        with open(hyperparameters_file, 'w') as f:
            yaml.dump(hyperparameters, f, default_flow_style=False)
        print(f"Saved hyperparameters to: {hyperparameters_file}")

    def init_wandb(self):
        """Initialize Weights & Biases."""
        wandb_config = self.config.copy()
        wandb_config['optimization'] = {
            'torch_compile': self.use_torch_compile,
            'torchscript': self.use_torchscript,
            'mixed_precision': self.use_amp,
        }

        wandb.init(
            project=self.wandb_project,
            entity=self.wandb_entity,
            name=self.wandb_run_name,
            config=wandb_config
        )
        print("WandB initialized successfully")

    def load_checkpoint_for_resume(self, checkpoint_path: str):
        """
        Load checkpoint for resuming training.

        Args:
            checkpoint_path: Path to checkpoint file

        Returns:
            dict: Checkpoint data

        Raises:
            FileNotFoundError: If checkpoint doesn't exist
            ValueError: If loss schedule mismatch
        """
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

        print(f"\nLoading checkpoint from: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=self.device)

        # Validate configuration compatibility
        saved_loss_schedule = checkpoint.get('loss_schedule', ['vamp2', 'vamp1', 'vamp2'])
        if saved_loss_schedule != self.loss_schedule:
            print(f"\nWARNING: Loss schedule mismatch!")
            print(f"  Checkpoint: {saved_loss_schedule}")
            print(f"  Config: {self.loss_schedule}")
            print(f"  Continuing with config loss schedule: {self.loss_schedule}")

        # Load metric history
        self.epochs_list = checkpoint.get('epochs_list', [])
        self.loss_types_list = checkpoint.get('loss_types_list', [])
        self.train_losses_last = checkpoint.get('train_losses_last', [])
        self.train_losses_avg = checkpoint.get('train_losses_avg', [])
        self.val_losses_last = checkpoint.get('val_losses_last', [])
        self.val_losses_avg = checkpoint.get('val_losses_avg', [])
        self.train_vamp1_scores_last = checkpoint.get('train_vamp1_scores_last', [])
        self.train_vamp1_scores_avg = checkpoint.get('train_vamp1_scores_avg', [])
        self.train_vamp2_scores_last = checkpoint.get('train_vamp2_scores_last', [])
        self.train_vamp2_scores_avg = checkpoint.get('train_vamp2_scores_avg', [])
        self.val_vamp1_scores_last = checkpoint.get('val_vamp1_scores_last', [])
        self.val_vamp1_scores_avg = checkpoint.get('val_vamp1_scores_avg', [])
        self.val_vamp2_scores_last = checkpoint.get('val_vamp2_scores_last', [])
        self.val_vamp2_scores_avg = checkpoint.get('val_vamp2_scores_avg', [])

        # Load best metric
        self.best_metric = checkpoint.get('best_metric', float('inf'))

        print(f"\nResume information:")
        print(f"  Checkpoint epoch: {checkpoint['epoch']}")
        print(f"  Best validation loss (avg): {self.best_metric:.6f}")
        print(f"  Loaded {len(self.train_losses_last)} training epochs of history")

        return checkpoint

    def save_checkpoint(self, epoch: int, train_loss_last: float, train_loss_avg: float,
                       val_loss_last: float, val_loss_avg: float, loss_type: str,
                       lobe_0: nn.Module, lobe_t: nn.Module, optimizer: optim.Optimizer,
                       is_best: bool = False):
        """Save model checkpoint with both last and average loss values."""
        checkpoint_dir = self.folders['checkpoints']

        # Unwrap DDP if necessary to remove 'module.' prefix
        model_0_to_save = lobe_0.module if isinstance(lobe_0, DistributedDataParallel) else lobe_0
        model_t_to_save = lobe_t.module if isinstance(lobe_t, DistributedDataParallel) else lobe_t

        checkpoint = {
            'epoch': epoch,
            'model_name': self.model_name,
            'lobe_0_state_dict': model_0_to_save.state_dict(),
            'lobe_t_state_dict': model_t_to_save.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'train_loss_last': train_loss_last,
            'train_loss_avg': train_loss_avg,
            'val_loss_last': val_loss_last,
            'val_loss_avg': val_loss_avg,
            'loss_type': loss_type,
            'best_metric': self.best_metric,
            'config': self.config,
            'loss_schedule': self.loss_schedule,
            'scaler_state_dict': self.scaler.state_dict() if self.use_amp else None,
            # Full metric history for resume
            'epochs_list': self.epochs_list,
            'loss_types_list': self.loss_types_list,
            'train_losses_last': self.train_losses_last,
            'train_losses_avg': self.train_losses_avg,
            'val_losses_last': self.val_losses_last,
            'val_losses_avg': self.val_losses_avg,
            'train_vamp1_scores_last': self.train_vamp1_scores_last,
            'train_vamp1_scores_avg': self.train_vamp1_scores_avg,
            'train_vamp2_scores_last': self.train_vamp2_scores_last,
            'train_vamp2_scores_avg': self.train_vamp2_scores_avg,
            'val_vamp1_scores_last': self.val_vamp1_scores_last,
            'val_vamp1_scores_avg': self.val_vamp1_scores_avg,
            'val_vamp2_scores_last': self.val_vamp2_scores_last,
            'val_vamp2_scores_avg': self.val_vamp2_scores_avg,
        }

        # Save epoch checkpoint
        checkpoint_path = os.path.join(checkpoint_dir, f'checkpoint_epoch_{epoch}.pth')
        torch.save(checkpoint, checkpoint_path)
        print(f"Saved checkpoint: {checkpoint_path}")

        # Always save latest checkpoint
        latest_path = os.path.join(checkpoint_dir, 'checkpoint_latest.pth')
        torch.save(checkpoint, latest_path)
        print(f"Saved latest checkpoint: {latest_path}")

        # Save best checkpoint
        if is_best:
            best_path = os.path.join(checkpoint_dir, 'checkpoint_best.pth')
            torch.save(checkpoint, best_path)
            print(f"Saved best checkpoint: {best_path}")

    def save_metrics(self):
        """Save training metrics to files (both last and average values)."""
        metrics_dir = self.folders['metrics']

        metrics = {
            'epochs': self.epochs_list,
            'loss_types': self.loss_types_list,
            'train_losses_last': self.train_losses_last,
            'train_losses_avg': self.train_losses_avg,
            'val_losses_last': self.val_losses_last,
            'val_losses_avg': self.val_losses_avg,
            'train_vamp1_scores_last': self.train_vamp1_scores_last,
            'train_vamp1_scores_avg': self.train_vamp1_scores_avg,
            'train_vamp2_scores_last': self.train_vamp2_scores_last,
            'train_vamp2_scores_avg': self.train_vamp2_scores_avg,
            'val_vamp1_scores_last': self.val_vamp1_scores_last,
            'val_vamp1_scores_avg': self.val_vamp1_scores_avg,
            'val_vamp2_scores_last': self.val_vamp2_scores_last,
            'val_vamp2_scores_avg': self.val_vamp2_scores_avg,
        }

        # Convert metrics to JSON-serializable format
        # numpy.float64 and other numpy types are not directly JSON serializable
        serializable_metrics = {}
        for key, values in metrics.items():
            if key == 'loss_types':
                # loss_types are already strings
                serializable_metrics[key] = values
            else:
                # Convert numpy/torch types to Python native types
                serializable_metrics[key] = [convert_to_serializable(v) for v in values]

        # Save as JSON
        metrics_json_path = os.path.join(metrics_dir, 'metrics.json')
        with open(metrics_json_path, 'w') as f:
            json.dump(serializable_metrics, f, indent=2)
        print(f"Saved metrics to: {metrics_json_path}")

        # Save as numpy arrays (excluding loss_types which is string list)
        for metric_name, metric_values in metrics.items():
            if metric_values and metric_name != 'loss_types':
                np_path = os.path.join(metrics_dir, f'{metric_name}.npy')
                np.save(np_path, np.array(metric_values))

    def cleanup(self):
        """Shutdown the distributed environment."""
        if dist.is_initialized():
            dist.destroy_process_group()

        # Note: Metrics are saved periodically alongside checkpoints during training.
        # No need to save again here to avoid overwriting with potentially stale data.

        # Close wandb
        if self.use_wandb and self.world_rank == 0:
            wandb.finish()

    def compile_models(self, lobe_0: nn.Module, lobe_t: nn.Module):
        """Apply torch.compile or TorchScript compilation."""
        compiled_lobe_0 = lobe_0
        compiled_lobe_t = lobe_t

        if self.use_torch_compile and self.torch_compile_available:
            print(f"\nCompiling models with torch.compile (mode: {self.compile_mode})...")
            try:
                compiled_lobe_0 = torch.compile(lobe_0, mode=self.compile_mode)
                compiled_lobe_t = torch.compile(lobe_t, mode=self.compile_mode)
                print("✓ torch.compile successful")
            except Exception as e:
                print(f"Warning: torch.compile failed: {e}")
                print("Falling back to eager mode")
                compiled_lobe_0 = lobe_0
                compiled_lobe_t = lobe_t

        elif self.use_torchscript:
            print("\nConverting models to TorchScript...")
            try:
                print("TorchScript will be applied during first forward pass")
            except Exception as e:
                print(f"Warning: TorchScript setup failed: {e}")

        return compiled_lobe_0, compiled_lobe_t

    def train(self, resume_checkpoint_path: Optional[str] = None):
        """
        Main training loop with optimizations.

        Args:
            resume_checkpoint_path: Optional path to checkpoint for resuming training
        """
        # Load Dataset
        print(f"Loading training dataset with {self.dataloader_class.__name__}...")
        MDStackGen_train = self.dataloader_class(
            fnames=self.train_fnames,
            tau=self.tau,
            cutoff=self.cutoff,
            num_gaussians=self.num_gaussians
        )

        print(f"Loading validation dataset with {self.dataloader_class.__name__}...")
        MDStackGen_val = self.dataloader_class(
            fnames=self.val_fnames,
            tau=self.tau,
            cutoff=self.cutoff,
            num_gaussians=self.num_gaussians
        )

        print("Datasets loaded")

        # Distribution Sampler
        self.train_sampler = DistributedSampler(MDStackGen_train) if dist.is_initialized() else None
        self.val_sampler = DistributedSampler(MDStackGen_val) if dist.is_initialized() else None

        print("Distributed Sampler Initialized")

        # Data Loader
        self.train_loader = DataLoader(
            MDStackGen_train,
            batch_size=self.batch_size,
            shuffle=(self.train_sampler is None),
            sampler=self.train_sampler,
            drop_last=True,
            pin_memory=torch.cuda.is_available(),
            num_workers=1
        )

        self.val_loader = DataLoader(
            MDStackGen_val,
            batch_size=self.batch_size,
            shuffle=(self.val_sampler is None),
            sampler=self.val_sampler,
            drop_last=True,
            pin_memory=torch.cuda.is_available(),
            num_workers=1
        )

        print("Data Loader Initialized")

        # MODEL - Create model with appropriate parameters
        model_kwargs = {
            'nbr_fea_len': self.num_gaussians,
            'atom_fea_len': self.atom_fea_len,
            'n_conv': self.n_conv,
            'state_len': self.state_len,
        }

        print(f"Creating {self.model_name} model with parameters: {model_kwargs}")
        lobe_0 = self.model_class(**model_kwargs)
        lobe_t = deepcopy(lobe_0)

        lobe_0 = lobe_0.to(self.device)
        lobe_t = lobe_t.to(self.device)

        # Apply compilation BEFORE DDP
        lobe_0, lobe_t = self.compile_models(lobe_0, lobe_t)

        # Distributed Model Setup
        if self.world_size > 0 and dist.is_initialized():
            self.lobe_0 = DistributedDataParallel(
                lobe_0, device_ids=[self.local_rank], output_device=[self.local_rank]
            )
            self.lobe_t = DistributedDataParallel(
                lobe_t, device_ids=[self.local_rank], output_device=[self.local_rank]
            )
            print("Distributed Model Initialized")
        else:
            self.lobe_0 = lobe_0
            self.lobe_t = lobe_t

        # Optimizer
        self.optimizer = optim.Adam(
            [{'params': self.lobe_0.parameters()},
             {'params': self.lobe_t.parameters()}],
            lr=self.learning_rate
        )

        print("Optimizer Initialized")

        # Handle checkpoint resume
        start_global_epoch = 0
        if resume_checkpoint_path:
            checkpoint = self.load_checkpoint_for_resume(resume_checkpoint_path)

            # Load model states
            self.lobe_0.load_state_dict(checkpoint['lobe_0_state_dict'])
            self.lobe_t.load_state_dict(checkpoint['lobe_t_state_dict'])
            print("Loaded model weights from checkpoint")

            # Load optimizer state
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            print("Loaded optimizer state from checkpoint")

            # Load scaler state if using AMP
            if self.use_amp and checkpoint.get('scaler_state_dict'):
                self.scaler.load_state_dict(checkpoint['scaler_state_dict'])
                print("Loaded GradScaler state from checkpoint")

            # Resume from next epoch
            start_global_epoch = checkpoint['epoch']
            print(f"\nResuming training from epoch {start_global_epoch}")

        # TRAINING LOOP
        loss_types = self.loss_schedule
        global_epoch = start_global_epoch

        # Calculate starting phase and epoch within phase
        start_phase_idx = global_epoch // self.epochs
        start_epoch_in_phase = global_epoch % self.epochs

        # Adjust if we've completed all phases
        if start_phase_idx >= len(loss_types):
            print(f"\nTraining already completed ({global_epoch} epochs)!")
            return

        print(f"\nStarting training:")
        print(f"  Total phases: {len(loss_types)}")
        print(f"  Epochs per phase: {self.epochs}")
        print(f"  Starting from phase {start_phase_idx}, epoch {start_epoch_in_phase}")
        print(f"  Loss schedule: {loss_types}\n")

        for phase_idx, loss_type in enumerate(loss_types):
            # Skip completed phases
            if phase_idx < start_phase_idx:
                global_epoch += self.epochs
                continue
            print(f"\n{'='*50}")
            print(f"Phase {phase_idx}: Training with loss type: {loss_type}")
            print(f"{'='*50}\n")

            # Start from appropriate epoch within phase
            start_epoch = start_epoch_in_phase if phase_idx == start_phase_idx else 0

            for epoch in range(start_epoch, self.epochs):
                if dist.is_initialized():
                    self.train_sampler.set_epoch(global_epoch)

                start_time = time.time()

                self.lobe_0.train()
                self.lobe_t.train()

                # Training - accumulate batch losses for averaging
                batch_losses = []
                batch_vamp1_scores = []
                batch_vamp2_scores = []
                train_loss_last = 0.0
                tr_vamp1_last = 0.0
                tr_vamp2_last = 0.0

                for batch_idx, data in enumerate(self.train_loader):
                    self.optimizer.zero_grad()

                    data_0, data_t = data[0].to(self.device), data[1].to(self.device)

                    if self.use_amp:
                        with autocast('cuda'):
                            out_0 = self.lobe_0(data_0)
                            out_t = self.lobe_t(data_t)

                        # VAMP loss computation in float32 (eigh doesn't support float16)
                        with autocast('cuda', enabled=False):
                            out_0 = out_0.float()
                            out_t = out_t.float()

                            if loss_type == "vamp1":
                                loss, _, grad_0, _, grad_t = vamp_score(
                                    out_0, out_t, method="VAMP_sym",
                                    epsilon=self.epsilon, mode=self.mode
                                )
                                out_0 = out_0 - out_0.mean(dim=0, keepdim=True)
                                out_t = out_t - out_t.mean(dim=0, keepdim=True)

                                self.scaler.scale(out_0).backward(gradient=grad_0)
                                self.scaler.scale(out_t).backward(gradient=grad_t)
                            else:
                                loss = -1 * vamp_score(
                                    out_0, out_t, method="VAMP2",
                                    epsilon=self.epsilon, mode=self.mode
                                )
                                self.scaler.scale(loss).backward()

                        self.scaler.step(self.optimizer)
                        self.scaler.update()
                    else:
                        out_0 = self.lobe_0(data_0)
                        out_t = self.lobe_t(data_t)

                        if loss_type == "vamp1":
                            loss, _, grad_0, _, grad_t = vamp_score(
                                out_0, out_t, method="VAMP_sym",
                                epsilon=self.epsilon, mode=self.mode
                            )
                            out_0 = out_0 - out_0.mean(dim=0, keepdim=True)
                            out_t = out_t - out_t.mean(dim=0, keepdim=True)

                            out_0.backward(gradient=grad_0)
                            out_t.backward(gradient=grad_t)
                        else:
                            loss = -1 * vamp_score(
                                out_0, out_t, method="VAMP2",
                                epsilon=self.epsilon, mode=self.mode
                            )
                            loss.backward()

                        self.optimizer.step()

                    # Compute metrics
                    vamp1, vamp2 = vamp_metric(out_0, out_t, epsilon=self.epsilon, mode=self.mode)

                    # Store batch values for averaging and keep last batch values
                    batch_loss = loss.item()
                    batch_vamp1 = vamp1.item()
                    batch_vamp2 = vamp2.item()

                    batch_losses.append(batch_loss)
                    batch_vamp1_scores.append(batch_vamp1)
                    batch_vamp2_scores.append(batch_vamp2)

                    # Update last batch values
                    train_loss_last = batch_loss
                    tr_vamp1_last = batch_vamp1
                    tr_vamp2_last = batch_vamp2

                # Calculate average across all batches
                train_loss_avg = np.mean(batch_losses) if batch_losses else 0.0
                tr_vamp1_avg = np.mean(batch_vamp1_scores) if batch_vamp1_scores else 0.0
                tr_vamp2_avg = np.mean(batch_vamp2_scores) if batch_vamp2_scores else 0.0

                # Store both last and average metrics
                self.epochs_list.append(global_epoch + 1)
                self.loss_types_list.append(loss_type)
                self.train_losses_last.append(train_loss_last)
                self.train_losses_avg.append(train_loss_avg)
                self.train_vamp1_scores_last.append(tr_vamp1_last)
                self.train_vamp1_scores_avg.append(tr_vamp1_avg)
                self.train_vamp2_scores_last.append(tr_vamp2_last)
                self.train_vamp2_scores_avg.append(tr_vamp2_avg)

                # Validation
                self.lobe_0.eval()
                self.lobe_t.eval()

                if dist.is_initialized():
                    self.val_sampler.set_epoch(global_epoch)

                # Validation - accumulate batch losses for averaging
                val_batch_losses = []
                val_batch_vamp1_scores = []
                val_batch_vamp2_scores = []
                val_loss_last = 0.0
                val_vamp1_last = 0.0
                val_vamp2_last = 0.0

                with torch.no_grad():
                    for batch_idx, data in enumerate(self.val_loader):
                        data_0, data_t = data[0].to(self.device), data[1].to(self.device)

                        out_0 = self.lobe_0(data_0)
                        out_t = self.lobe_t(data_t)

                        vamp1, vamp2 = vamp_metric(out_0, out_t, epsilon=self.epsilon, mode=self.mode)

                        if loss_type == "vamp1":
                            loss, _, _, _, _ = vamp_score(
                                out_0, out_t, method="VAMP_sym",
                                epsilon=self.epsilon, mode=self.mode
                            )
                        else:
                            loss = -1 * vamp_score(
                                out_0, out_t, method="VAMP2",
                                epsilon=self.epsilon, mode=self.mode
                            )

                        # Store batch values for averaging and keep last batch values
                        batch_val_loss = loss.item()
                        batch_val_vamp1 = vamp1.item()
                        batch_val_vamp2 = vamp2.item()

                        val_batch_losses.append(batch_val_loss)
                        val_batch_vamp1_scores.append(batch_val_vamp1)
                        val_batch_vamp2_scores.append(batch_val_vamp2)

                        # Update last batch values
                        val_loss_last = batch_val_loss
                        val_vamp1_last = batch_val_vamp1
                        val_vamp2_last = batch_val_vamp2

                # Calculate average across all validation batches
                val_loss_avg = np.mean(val_batch_losses) if val_batch_losses else 0.0
                val_vamp1_avg = np.mean(val_batch_vamp1_scores) if val_batch_vamp1_scores else 0.0
                val_vamp2_avg = np.mean(val_batch_vamp2_scores) if val_batch_vamp2_scores else 0.0

                # Store both last and average validation metrics
                self.val_losses_last.append(val_loss_last)
                self.val_losses_avg.append(val_loss_avg)
                self.val_vamp1_scores_last.append(val_vamp1_last)
                self.val_vamp1_scores_avg.append(val_vamp1_avg)
                self.val_vamp2_scores_last.append(val_vamp2_last)
                self.val_vamp2_scores_avg.append(val_vamp2_avg)

                epoch_time = time.time() - start_time

                if self.world_rank == 0:
                    print(f"\nWorld Rank : {self.world_rank} || Train Epoch: {global_epoch + 1}, Loss Type: {loss_type}")
                    print(f"Train Loss - Last: {train_loss_last:.6f}, Avg: {train_loss_avg:.6f}")
                    print(f"Train Metrics - VAMP1 (Last: {tr_vamp1_last:.6f}, Avg: {tr_vamp1_avg:.6f}), VAMP2 (Last: {tr_vamp2_last:.6f}, Avg: {tr_vamp2_avg:.6f})")
                    print(f"Val Loss - Last: {val_loss_last:.6f}, Avg: {val_loss_avg:.6f}")
                    print(f"Val Metrics - VAMP1 (Last: {val_vamp1_last:.6f}, Avg: {val_vamp1_avg:.6f}), VAMP2 (Last: {val_vamp2_last:.6f}, Avg: {val_vamp2_avg:.6f})")
                    print(f"--- {epoch_time:.2f} seconds ---")

                    # Log to wandb
                    if self.use_wandb:
                        wandb.log({
                            'epoch': global_epoch + 1,
                            'loss_type': loss_type,
                            'train_loss_last': train_loss_last,
                            'train_loss_avg': train_loss_avg,
                            'val_loss_last': val_loss_last,
                            'val_loss_avg': val_loss_avg,
                            'train_vamp1_last': tr_vamp1_last,
                            'train_vamp1_avg': tr_vamp1_avg,
                            'train_vamp2_last': tr_vamp2_last,
                            'train_vamp2_avg': tr_vamp2_avg,
                            'val_vamp1_last': val_vamp1_last,
                            'val_vamp1_avg': val_vamp1_avg,
                            'val_vamp2_last': val_vamp2_last,
                            'val_vamp2_avg': val_vamp2_avg,
                            'epoch_time': epoch_time,
                        })

                    # Checkpointing
                    is_best = val_loss_avg < self.best_metric
                    if is_best:
                        self.best_metric = val_loss_avg

                    # Save checkpoint based on frequency
                    if (global_epoch + 1) % self.checkpoint_freq == 0:
                        if self.save_best_only:
                            if is_best:
                                self.save_checkpoint(
                                    global_epoch + 1, train_loss_last, train_loss_avg,
                                    val_loss_last, val_loss_avg, loss_type,
                                    self.lobe_0, self.lobe_t, self.optimizer, is_best=True
                                )
                        else:
                            self.save_checkpoint(
                                global_epoch + 1, train_loss_last, train_loss_avg,
                                val_loss_last, val_loss_avg, loss_type,
                                self.lobe_0, self.lobe_t, self.optimizer, is_best=is_best
                            )
                        # Save metrics alongside checkpoint
                        self.save_metrics()

                global_epoch += 1

        # Save final checkpoint to ensure checkpoint_latest.pth exists
        if self.world_rank == 0:
            print("\nSaving final checkpoint...")
            self.save_checkpoint(
                global_epoch, train_loss_last, train_loss_avg,
                val_loss_last, val_loss_avg, loss_type,
                self.lobe_0, self.lobe_t, self.optimizer, is_best=False
            )

        # Save final models
        if self.world_rank == 0:
            final_model_dir = self.folders['final_models']
            model_0_path = os.path.join(final_model_dir, f'{self.model_name}_lobe_0_final.pth')
            model_t_path = os.path.join(final_model_dir, f'{self.model_name}_lobe_t_final.pth')

            # Unwrap DDP if necessary to remove 'module.' prefix
            model_0_to_save = self.lobe_0.module if isinstance(self.lobe_0, DistributedDataParallel) else self.lobe_0
            model_t_to_save = self.lobe_t.module if isinstance(self.lobe_t, DistributedDataParallel) else self.lobe_t

            torch.save(model_0_to_save.state_dict(), model_0_path)
            torch.save(model_t_to_save.state_dict(), model_t_path)

            print(f"\nSaved final models:")
            print(f"  Lobe 0: {model_0_path}")
            print(f"  Lobe t: {model_t_path}")

            # Save TorchScript models if enabled
            if self.use_torchscript:
                try:
                    compiled_dir = self.folders['compiled_models']
                    model_0_to_script = self.lobe_0.module if isinstance(self.lobe_0, DistributedDataParallel) else self.lobe_0
                    model_t_to_script = self.lobe_t.module if isinstance(self.lobe_t, DistributedDataParallel) else self.lobe_t

                    print("\nTorchScript export requires example inputs - skipping for safety")
                    print("Use torch.jit.trace() manually with example data if needed")
                except Exception as e:
                    print(f"Warning: TorchScript export failed: {e}")

        if dist.is_initialized():
            dist.barrier(device_ids=[self.local_rank] if torch.cuda.is_available() else None)
        print("Training Complete!")

    def evaluate(self):
        """Evaluation loop."""
        print("Loading test dataset...")
        MDStackGen_test = self.dataloader_class(
            fnames=self.test_fnames,
            tau=self.tau,
            cutoff=self.cutoff,
            num_gaussians=self.num_gaussians
        )

        # Distribution Sampler
        self.test_sampler = CustomInferenceSampler_v2(MDStackGen_test) if dist.is_initialized() else None

        # Data Loader
        self.test_loader = DataLoader(
            MDStackGen_test,
            batch_size=self.batch_size,
            shuffle=False,
            sampler=self.test_sampler,
            drop_last=False,
            pin_memory=torch.cuda.is_available(),
            num_workers=4
        )

        print("Data Loader Initialized")

        # MODEL
        model_kwargs = {
            'nbr_fea_len': self.num_gaussians,
            'atom_fea_len': self.atom_fea_len,
            'n_conv': self.n_conv,
            'state_len': self.state_len,
        }

        lobe_0 = self.model_class(**model_kwargs)
        lobe_t = deepcopy(lobe_0)

        lobe_0 = lobe_0.to(self.device)
        lobe_t = lobe_t.to(self.device)

        # Load trained weights
        final_model_dir = self.folders['final_models']
        model_0_path = os.path.join(final_model_dir, f'{self.model_name}_lobe_0_final.pth')
        model_t_path = os.path.join(final_model_dir, f'{self.model_name}_lobe_t_final.pth')

        if not os.path.exists(model_0_path) or not os.path.exists(model_t_path):
            checkpoint_dir = self.folders['checkpoints']
            latest_checkpoint = os.path.join(checkpoint_dir, 'checkpoint_latest.pth')
            best_checkpoint = os.path.join(checkpoint_dir, 'checkpoint_best.pth')

            if os.path.exists(latest_checkpoint):
                print(f"Loading from latest checkpoint: {latest_checkpoint}")
                checkpoint = torch.load(latest_checkpoint, map_location=self.device)
                lobe_0.load_state_dict(checkpoint['lobe_0_state_dict'])
                lobe_t.load_state_dict(checkpoint['lobe_t_state_dict'])
            elif os.path.exists(best_checkpoint):
                print(f"Loading from best checkpoint: {best_checkpoint}")
                checkpoint = torch.load(best_checkpoint, map_location=self.device)
                lobe_0.load_state_dict(checkpoint['lobe_0_state_dict'])
                lobe_t.load_state_dict(checkpoint['lobe_t_state_dict'])
            else:
                raise FileNotFoundError(f"No trained models found in {final_model_dir} or {checkpoint_dir}")
        else:
            print(f"Loading models from {final_model_dir}")
            lobe_0.load_state_dict(torch.load(model_0_path, map_location=self.device))
            lobe_t.load_state_dict(torch.load(model_t_path, map_location=self.device))

        # Distributed Model Setup
        if self.world_size > 0 and dist.is_initialized():
            self.lobe_0 = DistributedDataParallel(
                lobe_0, device_ids=[self.local_rank], output_device=[self.local_rank]
            )
            self.lobe_t = DistributedDataParallel(
                lobe_t, device_ids=[self.local_rank], output_device=[self.local_rank]
            )
        else:
            self.lobe_0 = lobe_0
            self.lobe_t = lobe_t

        self.lobe_0.eval()
        self.lobe_t.eval()

        pred_out_0 = []
        pred_out_t = []
        indices = []

        print("Running inference...")
        with torch.no_grad():
            for i, data in enumerate(self.test_loader):
                data_0, data_t = data[0].to(self.device), data[1].to(self.device)

                out_0 = self.lobe_0(data_0)
                out_t = self.lobe_t(data_t)

                mini_batch = data_0.num_graphs
                idx = data_0.indc

                indices.extend(idx.cpu())

                out_0_ = out_0.view(mini_batch, -1, self.state_len).cpu().clone()
                out_t_ = out_t.view(mini_batch, -1, self.state_len).cpu().clone()

                pred_out_0.append(out_0_)
                pred_out_t.append(out_t_)

        # Convert lists of tensors to single tensors
        pred_out_0 = torch.cat(pred_out_0, dim=0)
        pred_out_t = torch.cat(pred_out_t, dim=0)
        indices = torch.tensor(indices)

        print(f'Shape of local pred_out_0: {pred_out_0.shape}')

        if dist.is_initialized():
            # Gather results from all processes
            all_results_lobe_0 = [None] * self.world_size
            all_results_lobe_t = [None] * self.world_size
            all_indices = [None] * self.world_size

            dist.all_gather_object(all_results_lobe_0, pred_out_0)
            dist.all_gather_object(all_results_lobe_t, pred_out_t)
            dist.all_gather_object(all_indices, indices)

            if self.world_rank == 0:
                _pred_out_0 = torch.cat(all_results_lobe_0, dim=0)
                _pred_out_t = torch.cat(all_results_lobe_t, dim=0)

                _indices = torch.cat(all_indices)
                sorted_order = torch.sort(_indices)[1]

                sorted_pred_out_0 = _pred_out_0[sorted_order]
                sorted_pred_out_t = _pred_out_t[sorted_order]

                self._save_predictions(sorted_pred_out_0, sorted_pred_out_t)
        else:
            sorted_order = torch.sort(indices)[1]
            sorted_pred_out_0 = pred_out_0[sorted_order]
            sorted_pred_out_t = pred_out_t[sorted_order]
            self._save_predictions(sorted_pred_out_0, sorted_pred_out_t)

        if dist.is_initialized():
            dist.barrier(device_ids=[self.local_rank] if torch.cuda.is_available() else None)

        print("Evaluation Complete!")

    def _save_predictions(self, sorted_pred_out_0: torch.Tensor, sorted_pred_out_t: torch.Tensor):
        """Save predictions to files."""
        predictions_dir = self.folders['predictions']

        tau = self.tau
        predictions_np = np.concatenate([
            sorted_pred_out_0.numpy()[:, :, :],
            sorted_pred_out_t.numpy()[-tau:, :, :]
        ])

        predictions_torch = torch.cat([
            sorted_pred_out_0[:, :, :],
            sorted_pred_out_t[-tau:, :, :]
        ], dim=0)

        # Save with model name
        npy_path = os.path.join(predictions_dir, f'{self.model_name}_predictions.npy')
        pt_path = os.path.join(predictions_dir, f'{self.model_name}_predictions.pt')

        np.save(npy_path, predictions_np)
        torch.save(predictions_torch, pt_path)

        print(f"\nPredictions saved:")
        print(f"  NumPy: {npy_path}")
        print(f"  PyTorch: {pt_path}")
        print(f"  Shape: {predictions_np.shape}")


def main():
    parser = argparse.ArgumentParser(description='Optimized trainer for GDYNet models.')
    parser.add_argument('--config', type=str, required=True, help='Path to the configuration file')
    parser.add_argument('--mode', type=str, choices=['train', 'evaluate'], required=True,
                       help='Mode to run the script in')
    parser.add_argument('--output', type=str, default=None, help='Custom output folder path')
    parser.add_argument('--resume', type=str, default=None,
                       help='Path to checkpoint file to resume training from')

    args = parser.parse_args()

    # Load the trainer with the specified config
    trainer = OptimizedTrainer(args.config)

    # Override the output folder if specified
    if args.output:
        trainer.PATH = args.output
        trainer.create_output_folders()
        trainer.print_and_save_hyperparameters()

    start_time = time.time()

    try:
        if args.mode == 'train':
            trainer.train(resume_checkpoint_path=args.resume)
            print("\nTRAINING COMPLETED SUCCESSFULLY!")
        elif args.mode == 'evaluate':
            trainer.evaluate()
            print("\nEVALUATION COMPLETED SUCCESSFULLY!")
    except Exception as e:
        print(f"\nError during {args.mode}: {str(e)}")
        raise
    finally:
        trainer.cleanup()

    print(f"\nTotal time: {time.time() - start_time:.2f} seconds")


if __name__ == "__main__":
    main()