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

set -euo pipefail
mkdir -p logs

echo "=== Job start: $(date) ==="
echo "SLURM_JOB_ID=${SLURM_JOB_ID}"
echo "SLURM_JOB_NODELIST=${SLURM_JOB_NODELIST}"
echo "Nodes=${SLURM_JOB_NUM_NODES}  Tasks=${SLURM_NTASKS}  Tasks/node=${SLURM_NTASKS_PER_NODE:-8}"

# ----------------------------
# Modules / env
# ----------------------------
module load PrgEnv-gnu/8.6.0
module load rocm/7.0.2
module load craype-accel-amd-gfx90a
module load miniforge3/23.11.0-0

# Activate your conda environment (edit to match your setup)
# Option A: conda activate by name
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate CONDA_ENV_NAME

# Option B: activate by path (uncomment and edit)
# source activate /path/to/conda_env

# Keep CRAY libs at runtime
export LD_LIBRARY_PATH="${CRAY_LD_LIBRARY_PATH}:${LD_LIBRARY_PATH}"

# ----------------------------
# Frontier comms / RCCL knobs
# ----------------------------
export HDF5_USE_FILE_LOCKING=FALSE
export FI_MR_CACHE_MONITOR=kdreg2
export FI_CXI_DEFAULT_CQ_SIZE=131072
export FI_CXI_DEFAULT_TX_SIZE=2048
export FI_CXI_RX_MATCH_MODE=hybrid

# Pin sockets to Slingshot NIC
export NCCL_SOCKET_IFNAME=hsn0
export GLOO_SOCKET_IFNAME=hsn0

# Optional: debugging / hang-to-error
export NCCL_ASYNC_ERROR_HANDLING=1
export TORCH_DISTRIBUTED_DEBUG=DETAIL
export NCCL_DEBUG=INFO

# MIOpen cache (per-job, local)
export MIOPEN_USER_DB_PATH="/tmp/miopen-cache-${SLURM_JOB_ID}"
export MIOPEN_CUSTOM_CACHE_DIR="${MIOPEN_USER_DB_PATH}"
rm -rf "${MIOPEN_USER_DB_PATH}"
mkdir -p "${MIOPEN_USER_DB_PATH}"

# ----------------------------
# DDP rendezvous (env://)
# ----------------------------
export MASTER_PORT=29500
MASTER_HOST=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_ADDR=$(getent hosts "$MASTER_HOST" | awk '{print $1}' | head -n 1)
export WORLD_SIZE="${SLURM_NTASKS}"

echo "MASTER_ADDR=${MASTER_ADDR}"
echo "MASTER_PORT=${MASTER_PORT}"
echo "WORLD_SIZE=${WORLD_SIZE}"

# ----------------------------
# Repo location
# ----------------------------
# Default: assume repo lives in ./gdy_pl relative to submit directory
export ROOT="${ROOT:-$PWD/gdy_pl}"
cd "$ROOT"
echo "ROOT=$ROOT"
echo "PWD=$(pwd)"

# ----------------------------
# Commands
# ----------------------------
train_cmd="python -W ignore -u trainer.py --config configs/gdynet_ferro_example.yaml --mode train"
eval_cmd="python -W ignore -u trainer.py --config configs/gdynet_ferro_example.yaml --mode evaluate"

# ----------------------------
# SRUN 1: TRAIN
# ----------------------------
echo
echo "=== Step 1: TRAIN ==="
echo "Start TRAIN at $(date)"

srun -n "${SLURM_NTASKS}" --ntasks-per-node=8 \
  --gpu-bind=none \
  -l bash -c '
    set -euo pipefail

    # Ensure ranks can see all GPUs 0..7 (so torch.cuda.set_device(local_rank) works)
    unset ROCR_VISIBLE_DEVICES || true
    unset HIP_VISIBLE_DEVICES  || true
    unset CUDA_VISIBLE_DEVICES || true

    export RANK=$SLURM_PROCID
    export LOCAL_RANK=$SLURM_LOCALID
    export WORLD_SIZE=$SLURM_NTASKS

    echo "TRAIN rank=$RANK local_rank=$LOCAL_RANK host=$(hostname) ROCR_VISIBLE_DEVICES=${ROCR_VISIBLE_DEVICES:-UNSET}"

    '"${train_cmd}"'
  '

train_status=$?

echo "TRAIN finished status=${train_status} at $(date)"
if [ "${train_status}" -ne 0 ]; then
  echo "TRAIN FAILED (status=${train_status}) — skipping EVALUATE."
  echo "=== Job end: $(date) ==="
  exit "${train_status}"
fi

# ----------------------------
# SRUN 2: EVALUATE
# ----------------------------
echo
echo "=== Step 2: EVALUATE ==="
echo "Start EVAL at $(date)"

srun -n "${SLURM_NTASKS}" --ntasks-per-node=8 \
  --gpu-bind=none \
  -l bash -c '
    set -euo pipefail

    # Ensure ranks can see all GPUs 0..7 (so torch.cuda.set_device(local_rank) works)
    unset ROCR_VISIBLE_DEVICES || true
    unset HIP_VISIBLE_DEVICES  || true
    unset CUDA_VISIBLE_DEVICES || true

    export RANK=$SLURM_PROCID
    export LOCAL_RANK=$SLURM_LOCALID
    export WORLD_SIZE=$SLURM_NTASKS

    echo "EVAL  rank=$RANK local_rank=$LOCAL_RANK host=$(hostname) ROCR_VISIBLE_DEVICES=${ROCR_VISIBLE_DEVICES:-UNSET}"

    '"${eval_cmd}"'
  '

echo "=== Job end: $(date) ==="
