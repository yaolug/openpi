#!/usr/bin/env python3
"""
Example script comparing JAX and PyTorch Pi0 model training for one step.

This demonstrates training behavior consistency between implementations.
"""

import argparse
import numpy as np
import jax
import jax.numpy as jnp
import torch
from openpi.policies import policy_config as _policy_config
from openpi.training import config as _config
from openpi.shared import download
import time

def create_example_batch(model_name: str = "pi0_aloha_sim", batch_size: int = 4) -> dict:
    """Create example training batch."""
    
    if model_name == "pi0_aloha_sim":
        # Create example batch for ALOHA sim environment
        batch = {
            "images": {
                "cam_high": np.random.randint(0, 256, size=(batch_size, 3, 224, 224), dtype=np.uint8),
                "cam_low": np.random.randint(0, 256, size=(batch_size, 3, 224, 224), dtype=np.uint8),
            },
            "state": np.random.randn(batch_size, 14).astype(np.float32),  # 14 motors for ALOHA sim
            "prompt": ["Pick up the cube and place it in the bin"] * batch_size,
            "actions": np.random.randn(batch_size, 10, 14).astype(np.float32),  # 10 timesteps, 14 motors
        }
    elif model_name == "pi0_aloha_towel":
        batch = {
            "images": {
                "cam_high": np.random.randint(0, 256, size=(batch_size, 3, 224, 224), dtype=np.uint8),
            },
            "state": np.random.randn(batch_size, 14).astype(np.float32),
            "prompt": ["Fold the towel neatly on the table"] * batch_size,
            "actions": np.random.randn(batch_size, 10, 14).astype(np.float32),
        }
    elif model_name == "pi0_base":
        batch = {
            "images": {
                "cam_high": np.random.randint(0, 256, size=(batch_size, 3, 224, 224), dtype=np.uint8),
                "cam_low": np.random.randint(0, 256, size=(batch_size, 3, 224, 224), dtype=np.uint8),
            },
            "state": np.random.randn(batch_size, 8).astype(np.float32),
            "prompt": ["Pick up the object and move it to the target location"] * batch_size,
            "actions": np.random.randn(batch_size, 10, 8).astype(np.float32),
        }
    else:
        raise ValueError(f"Unknown model name: {model_name}")
    
    return batch

def run_jax_training_step(batch, model_name):
    """Run one training step with JAX Pi0 model."""
    print("=== JAX Pi0 Training Example ===")

    try:
        from openpi.models.pi0_config import Pi0Config
        from openpi.policies.policy import Policy

        config = _config.get_config(model_name)
        checkpoint_dir = download.maybe_download(f"s3://openpi-assets/checkpoints/{model_name}")
        
        # Create and initialize model
        policy = _policy_config.create_trained_policy(config, checkpoint_dir)
        
        # Set random seed for noise
        rng_key = jax.random.key(42)
        noise_shape = (batch["actions"].shape[0], config.model.action_horizon, config.model.action_dim)
        jax_noise = jax.random.normal(rng_key, noise_shape, dtype=jnp.float32)
        noise_np = np.array(jax_noise)
        policy._rng = rng_key

        # Get initial weights
        initial_weights = policy.get_weights()

        # Run one training step
        print("Running JAX training step...")
        start_time = time.time()
        loss, gradients = policy.compute_loss_and_gradients(batch, noise=noise_np)
        policy.apply_gradients(gradients)
        train_time = (time.time() - start_time) * 1000  # Convert to ms

        # Get final weights
        final_weights = policy.get_weights()

        print("JAX training completed!")
        print(f"  - Training time: {train_time:.2f}ms")
        print(f"  - Loss: {loss:.6f}")

        return {
            "loss": loss,
            "initial_weights": initial_weights,
            "final_weights": final_weights,
            "noise": noise_np,
            "training_time": train_time
        }

    except ImportError as e:
        print(f"Failed to run JAX training: {e}")
        return None

def run_pytorch_training_step(batch, model_name, noise):
    """Run one training step with PyTorch Pi0 model."""
    print("\n=== PyTorch Pi0 Training Example ===")

    try:
        from openpi.models.pi0_config import Pi0Config
        from openpi.models_pytorch.pi0_pytorch import PI0Pytorch
        from openpi.policies.policy import Policy

        config = _config.get_config(model_name)
        checkpoint_dir = f"/home/jasonlu/.cache/openpi/openpi-assets/checkpoints/{model_name}_pytorch2"
        
        # Create and initialize model
        policy = _policy_config.create_trained_policy(config, checkpoint_dir, is_pytorch=True)
        
        # Get initial weights
        initial_weights = {name: param.detach().cpu().numpy() 
                         for name, param in policy.model.named_parameters()}

        # Run one training step
        print("Running PyTorch training step...")
        start_time = time.time()
        loss = policy.compute_loss(batch, noise=torch.from_numpy(noise))
        loss.backward()
        policy.optimizer.step()
        policy.optimizer.zero_grad()
        train_time = (time.time() - start_time) * 1000  # Convert to ms

        # Get final weights
        final_weights = {name: param.detach().cpu().numpy() 
                        for name, param in policy.model.named_parameters()}

        print("PyTorch training completed!")
        print(f"  - Training time: {train_time:.2f}ms")
        print(f"  - Loss: {loss.item():.6f}")

        return {
            "loss": loss.item(),
            "initial_weights": initial_weights,
            "final_weights": final_weights,
            "training_time": train_time
        }

    except ImportError as e:
        print(f"Failed to run PyTorch training: {e}")
        return None

def compare_results(jax_result, pytorch_result):
    """Compare training results from both implementations."""
    if jax_result is None or pytorch_result is None:
        print("Cannot compare results - one implementation failed")
        return

    print("\n=== Comparing Results ===")

    # Compare losses
    loss_diff = abs(jax_result["loss"] - pytorch_result["loss"])
    print("Loss comparison:")
    print(f"  - JAX loss: {jax_result['loss']:.6f}")
    print(f"  - PyTorch loss: {pytorch_result['loss']:.6f}")
    print(f"  - Absolute difference: {loss_diff:.6f}")

    # Compare weight changes
    print("\nWeight changes comparison:")
    for jax_layer, pt_layer in zip(jax_result["final_weights"].items(), pytorch_result["final_weights"].items()):
        jax_name, jax_weights = jax_layer
        pt_name, pt_weights = pt_layer
        
        weight_diff = np.abs(jax_weights - pt_weights)
        print(f"\nLayer: {jax_name}")
        print(f"  - Max absolute difference: {np.max(weight_diff):.6f}")
        print(f"  - Mean absolute difference: {np.mean(weight_diff):.6f}")

    # Compare timing
    print("\nTiming comparison:")
    print(f"  - JAX: {jax_result['training_time']:.2f}ms")
    print(f"  - PyTorch: {pytorch_result['training_time']:.2f}ms")
    print(f"  - Speedup: {pytorch_result['training_time'] / jax_result['training_time']:.2f}x (PyTorch time / JAX time)")

def main():
    parser = argparse.ArgumentParser(description="Compare JAX and PyTorch Pi0 model training")
    parser.add_argument("--model_name", type=str, default="pi0_aloha_sim",
                       choices=["pi0_aloha_sim", "pi0_aloha_towel", "pi0_base"],
                       help="Model name to use")
    parser.add_argument("--batch_size", type=int, default=4,
                       help="Batch size for training")
    args = parser.parse_args()

    print("Pi0 Model Training Comparison")
    print("=" * 50)

    # Set random seed for reproducibility
    np.random.seed(42)
    torch.manual_seed(42)

    # Create training batch
    batch = create_example_batch(args.model_name, args.batch_size)

    # Run JAX training step
    jax_result = run_jax_training_step(batch, args.model_name)

    # Reset random seed for fair comparison
    np.random.seed(42)
    torch.manual_seed(42)

    # Run PyTorch training step with same noise
    pytorch_result = run_pytorch_training_step(batch, args.model_name, jax_result["noise"])

    # Compare results
    compare_results(jax_result, pytorch_result)

    print("\n" + "=" * 50)
    print("Example completed!")

if __name__ == "__main__":
    main() 