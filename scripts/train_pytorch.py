#!/usr/bin/env python3
"""
Training script for PyTorch Pi0 models with distributed training support.

This script provides both single-GPU and multi-GPU training support using PyTorch's
DistributedDataParallel (DDP). It handles data loading, checkpointing, and logging
in a distributed environment.

Usage:
    Single GPU:
        python scripts/train_pytorch.py \
            --model_name pi0_aloha_sim \
            --checkpoint_dir checkpoints/aloha_sim \
            --log_dir logs/aloha_sim \
            --num_epochs 100

    Multiple GPUs (e.g., 4 GPUs):
        python scripts/train_pytorch.py \
            --model_name pi0_aloha_sim \
            --checkpoint_dir checkpoints/aloha_sim \
            --log_dir logs/aloha_sim \
            --num_epochs 100 \
            --num_gpus 4

    Slurm with Multiple GPUs:
        srun --nodes=1 --ntasks-per-node=1 --gpus-per-task=4 \
            python scripts/train_pytorch.py \
            --model_name pi0_aloha_sim \
            --checkpoint_dir checkpoints/aloha_sim \
            --log_dir logs/aloha_sim \
            --num_epochs 100 \
            --num_gpus 4

Arguments:
    --model_name: Name of the model to train
        Choices: pi0_aloha_sim, pi0_aloha_towel, pi0_base
        Required: Yes
    
    --checkpoint_dir: Directory for saving/loading checkpoints
        Default: None (no checkpointing)
        Required: No
    
    --log_dir: Directory for saving training logs
        Default: "logs"
        Required: No
    
    --num_epochs: Number of epochs to train
        Default: 100
        Required: No
    
    --num_gpus: Number of GPUs to use for training
        Default: All available GPUs (torch.cuda.device_count())
        Required: No

Features:
    - Distributed training with DDP
    - Automatic batch size scaling across GPUs
    - Synchronized logging and checkpointing
    - Proper error handling and cleanup
    - Progress tracking and loss monitoring
    - Best model checkpointing

Requirements:
    - PyTorch >= 1.8.0
    - CUDA-capable GPU(s)
    - NCCL backend for distributed training
    - Enough GPU memory to fit model and batch

Notes:
    - The effective batch size is (--batch_size * num_gpus)
    - Only rank 0 process handles logging and checkpointing
    - Each GPU process uses its own seed (42 + rank) for reproducibility
    - Uses NCCL backend for best GPU-GPU communication performance
    - Automatically handles process group initialization and cleanup
"""

import os
import sys
import time
import argparse
import logging
from typing import Dict, Any

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
import numpy as np

from openpi.policies import policy_config as _policy_config
from openpi.training import config as _config
from openpi.data.dataset import create_dataset
from openpi.shared import download
from openpi.utils import setup_logger

logger = logging.getLogger(__name__)

def setup_distributed(rank: int, world_size: int):
    """Initialize distributed training environment."""
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    
    # Initialize process group
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    
    # Set device for this process
    torch.cuda.set_device(rank)
    
    # Set random seeds
    torch.manual_seed(42 + rank)
    np.random.seed(42 + rank)

def cleanup_distributed():
    """Clean up distributed training environment."""
    dist.destroy_process_group()

def create_data_loader(config: Dict[str, Any], rank: int, world_size: int) -> DataLoader:
    """Create distributed data loader."""
    dataset = create_dataset(config)
    sampler = DistributedSampler(
        dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=True,
        seed=42
    )
    
    loader = DataLoader(
        dataset,
        batch_size=config.training.batch_size // world_size,  # Split batch size across GPUs
        sampler=sampler,
        num_workers=4,
        pin_memory=True
    )
    
    return loader, sampler

def train_epoch(epoch: int, 
                model: DDP,
                loader: DataLoader,
                sampler: DistributedSampler,
                optimizer: torch.optim.Optimizer,
                device: torch.device,
                rank: int):
    """Train for one epoch."""
    model.train()
    sampler.set_epoch(epoch)  # Ensure different ordering each epoch
    
    total_loss = 0
    num_batches = 0
    
    for batch_idx, batch in enumerate(loader):
        # Move data to GPU
        batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v 
                for k, v in batch.items()}
        
        # Generate noise
        noise = torch.randn(
            (batch["actions"].shape[0], model.module.action_horizon, model.module.action_dim),
            device=device
        )
        
        # Forward pass
        optimizer.zero_grad()
        loss = model.forward(batch, noise=noise)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        # Update metrics
        total_loss += loss.item()
        num_batches += 1
        
        # Log progress (only on rank 0)
        if rank == 0 and batch_idx % 10 == 0:
            logger.info(f"Epoch {epoch} [{batch_idx}/{len(loader)}]: "
                       f"Loss = {loss.item():.4f}")
    
    # Calculate average loss
    avg_loss = total_loss / num_batches
    
    # Sync loss across GPUs
    if world_size > 1:
        loss_tensor = torch.tensor([avg_loss], device=device)
        dist.all_reduce(loss_tensor, op=dist.ReduceOp.SUM)
        avg_loss = loss_tensor.item() / world_size
    
    return avg_loss

def save_checkpoint(model: DDP,
                   optimizer: torch.optim.Optimizer,
                   epoch: int,
                   loss: float,
                   checkpoint_dir: str,
                   rank: int):
    """Save training checkpoint."""
    if rank == 0:  # Only save on rank 0
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.module.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss,
        }
        
        checkpoint_path = os.path.join(checkpoint_dir, f'checkpoint_epoch_{epoch}.pt')
        torch.save(checkpoint, checkpoint_path)
        logger.info(f"Saved checkpoint to {checkpoint_path}")

def train(rank: int, world_size: int, args: argparse.Namespace):
    """Main training function for each process."""
    try:
        # Initialize distributed
        setup_distributed(rank, world_size)
        
        # Set up logging
        if rank == 0:
            setup_logger(args.log_dir)
        
        # Load config
        config = _config.get_config(args.model_name)
        
        # Create model
        policy = _policy_config.create_trained_policy(
            config,
            args.checkpoint_dir if args.checkpoint_dir else None,
            is_pytorch=True
        )
        
        # Move model to GPU and wrap with DDP
        policy.model.to(rank)
        model = DDP(policy.model, device_ids=[rank])
        
        # Create optimizer
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=config.training.learning_rate
        )
        
        # Create data loader
        train_loader, sampler = create_data_loader(config, rank, world_size)
        
        # Training loop
        best_loss = float('inf')
        start_epoch = 0
        
        for epoch in range(start_epoch, args.num_epochs):
            epoch_start = time.time()
            
            # Train one epoch
            train_loss = train_epoch(
                epoch, model, train_loader, sampler,
                optimizer, rank, rank
            )
            
            epoch_time = time.time() - epoch_start
            
            # Log results (rank 0 only)
            if rank == 0:
                logger.info(f"Epoch {epoch} completed in {epoch_time:.2f}s")
                logger.info(f"Average loss: {train_loss:.4f}")
            
            # Save checkpoint if best so far
            if train_loss < best_loss:
                best_loss = train_loss
                save_checkpoint(
                    model, optimizer, epoch, train_loss,
                    args.checkpoint_dir, rank
                )
            
            # Make sure all processes sync up
            if world_size > 1:
                dist.barrier()
        
    except Exception as e:
        logger.error(f"Error on rank {rank}: {str(e)}")
        raise e
    
    finally:
        cleanup_distributed()

def main():
    parser = argparse.ArgumentParser(description="Train PyTorch Pi0 model with DDP support")
    parser.add_argument("--model_name", type=str, required=True,
                       choices=["pi0_aloha_sim", "pi0_aloha_towel", "pi0_base"],
                       help="Model name to train")
    parser.add_argument("--checkpoint_dir", type=str, default=None,
                       help="Directory to load checkpoint from and save new checkpoints")
    parser.add_argument("--log_dir", type=str, default="logs",
                       help="Directory to save training logs")
    parser.add_argument("--num_epochs", type=int, default=100,
                       help="Number of epochs to train")
    parser.add_argument("--num_gpus", type=int, default=torch.cuda.device_count(),
                       help="Number of GPUs to use for training")
    
    args = parser.parse_args()
    
    # Create directories
    os.makedirs(args.log_dir, exist_ok=True)
    if args.checkpoint_dir:
        os.makedirs(args.checkpoint_dir, exist_ok=True)
    
    # Launch training processes
    if args.num_gpus > 1:
        mp.spawn(
            train,
            args=(args.num_gpus, args),
            nprocs=args.num_gpus,
            join=True
        )
    else:
        # Single GPU training
        train(0, 1, args)

if __name__ == "__main__":
    main() 