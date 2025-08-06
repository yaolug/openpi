#!/usr/bin/env python3
"""
Example script showing how to run inference with both JAX and PyTorch Pi0 models.

This demonstrates the basic usage patterns for both implementations.
"""

import argparse
import numpy as np
import jax
import jax.numpy as jnp
from openpi.policies import policy_config as _policy_config
from openpi.training import config as _config
from openpi.shared import download

def create_example_data(model_name: str = "pi0_aloha_sim") -> dict:
    """Create example input data matching the expected format."""
    
    if model_name == "pi0_aloha_sim":
        # Create example data for ALOHA sim environment
        example = {
            "images": {
                "cam_high": np.random.randint(0, 256, size=(3, 224, 224), dtype=np.uint8),
                "cam_low": np.random.randint(0, 256, size=(3, 224, 224), dtype=np.uint8),
            },
            "state": np.random.randn(14).astype(np.float32),  # 14 motors for ALOHA sim
            "prompt": "Pick up the cube and place it in the bin",
        }
    elif model_name == "pi0_aloha_towel":
        # Create example data for ALOHA towel task
        example = {
            "images": {
                "cam_high": np.random.randint(0, 256, size=(3, 224, 224), dtype=np.uint8),
                # Note: towel task typically only uses one camera
            },
            "state": np.random.randn(14).astype(np.float32),  # 14 motors for ALOHA
            "prompt": "Fold the towel neatly on the table",
        }
    elif model_name == "pi0_base":
        # Create example data for base droid policy
        example = {
            "images": {
                "cam_high": np.random.randint(0, 256, size=(3, 224, 224), dtype=np.uint8),
                "cam_low": np.random.randint(0, 256, size=(3, 224, 224), dtype=np.uint8),
            },
            "state": np.random.randn(8).astype(np.float32),  # Joint + gripper positions
            "prompt": "Pick up the object and move it to the target location",
        }
    else:
        raise ValueError(f"Unknown model name: {model_name}")
    
    return example



def run_jax_inference_example(observation, model_name):
    """Example of running inference with JAX Pi0 model."""
    print("=== JAX Pi0 Inference Example ===")

    try:
        import jax

        from openpi.models.pi0_config import Pi0Config
        from openpi.policies.policy import Policy

        config = _config.get_config(model_name)
        checkpoint_dir = download.maybe_download(f"gs://openpi-assets/checkpoints/{model_name}")
        
        # Create trained policy
        policy = _policy_config.create_trained_policy(config, checkpoint_dir)

        rng_key = jax.random.key(42)
        noise_shape = (config.model.action_horizon, config.model.action_dim)  # Use model's expected dimension
        jax_noise = jax.random.normal(rng_key, noise_shape, dtype=jnp.float32)
        noise_np = np.array(jax_noise)
        policy._rng = rng_key

        # Run inference
        print("Running JAX inference...")
        result = policy.infer(observation, noise=noise_np)

        # Print results
        print("JAX inference completed!")
        print(f"  - Inference time: {result['policy_timing']['infer_ms']:.2f}ms")
        print(f"  - Actions shape: {result['actions'].shape}")
        print(f"  - Actions range: [{result['actions'].min():.3f}, {result['actions'].max():.3f}]")

        return result, noise_np

    except ImportError as e:
        print(f"Failed to run JAX inference: {e}")
        return None

def run_pytorch_inference_example(observation, model_name, noise):
    """Example of running inference with PyTorch Pi0 model."""
    print("\n=== PyTorch Pi0 Inference Example ===")

    try:
        from openpi.models.pi0_config import Pi0Config
        from openpi.models_pytorch.pi0_pytorch import PI0Pytorch
        from openpi.policies.policy import Policy

        config = _config.get_config(model_name)
        checkpoint_dir = f"/home/jasonlu/.cache/openpi/openpi-assets/checkpoints/{model_name}_pytorch2"
        
        # Create trained policy
        policy = _policy_config.create_trained_policy(config, checkpoint_dir, is_pytorch=True)

        # Run inference
        print("Running PyTorch inference...")
        result = policy.infer(observation, noise=noise)

        # Print results
        print("PyTorch inference completed!")
        print(f"  - Inference time: {result['policy_timing']['infer_ms']:.2f}ms")
        print(f"  - Actions shape: {result['actions'].shape}")
        print(f"  - Actions range: [{result['actions'].min():.3f}, {result['actions'].max():.3f}]")

        return result

    except ImportError as e:
        print(f"Failed to run PyTorch inference: {e}")
        return None


def compare_results(jax_result, pytorch_result):
    """Compare results from both implementations."""
    if jax_result is None or pytorch_result is None:
        print("Cannot compare results - one implementation failed")
        return

    print("\n=== Comparing Results ===")

    # Compare actions
    actions_diff = np.abs(jax_result["actions"] - pytorch_result["actions"])
    max_diff = np.max(actions_diff)
    mean_diff = np.mean(actions_diff)

    print(f"JAX actions: {jax_result['actions'][0:2, :]}")
    print(f"PyTorch actions: {pytorch_result['actions'][0:2, :]}")

    print("Actions comparison:")
    print(f"  - Max absolute difference: {max_diff:.6f}")
    print(f"  - Mean absolute difference: {mean_diff:.6f}")

    # Calculate relative differences
    relative_diff = np.abs((jax_result["actions"] - pytorch_result["actions"]) / pytorch_result["actions"])
    max_rel_diff = np.max(relative_diff)
    mean_rel_diff = np.mean(relative_diff)

    print(f"  - Max relative difference: {max_rel_diff:.6f}")
    print(f"  - Mean relative difference: {mean_rel_diff:.6f}")
    
    # Additional diagnostic info
    print(f"  - JAX actions stats: min={jax_result['actions'].min():.6f}, max={jax_result['actions'].max():.6f}, mean={jax_result['actions'].mean():.6f}")
    print(f"  - PyTorch actions stats: min={pytorch_result['actions'].min():.6f}, max={pytorch_result['actions'].max():.6f}, mean={pytorch_result['actions'].mean():.6f}")

    # Check if results are close with different tolerances
    if np.allclose(jax_result["actions"], pytorch_result["actions"], rtol=1e-5, atol=1e-6):
        print("✅ Results match within strict tolerance!")
    elif np.allclose(jax_result["actions"], pytorch_result["actions"], rtol=1e-4, atol=1e-5):
        print("⚠️  Results match within moderate tolerance (rtol=1e-4, atol=1e-5)")
    elif np.allclose(jax_result["actions"], pytorch_result["actions"], rtol=1e-2, atol=1e-3):
        print("⚠️  Results match within loose tolerance (rtol=1e-3, atol=1e-4)")
    else:
        print("❌ Results differ significantly even with loose tolerance!")

    # Compare timing
    jax_time = jax_result["policy_timing"]["infer_ms"]
    pytorch_time = pytorch_result["policy_timing"]["infer_ms"]

    print("Timing comparison:")
    print(f"  - JAX: {jax_time:.2f}ms")
    print(f"  - PyTorch: {pytorch_time:.2f}ms")
    print(f"  - Speedup: {jax_time / pytorch_time:.2f}x (JAX time / PyTorch time)")


def main():
    parser = argparse.ArgumentParser(description="Run inference with both JAX and PyTorch Pi0 models")
    parser.add_argument("--model_name", type=str, default="pi0_aloha_sim", 
                       choices=["pi0_aloha_sim", "pi0_aloha_towel", "pi0_base"],
                       help="Model name to use")
    args = parser.parse_args()

    """Run both inference examples and compare results."""
    print("Pi0 Model Inference Comparison")
    print("=" * 50)

    # Set random seed for reproducibility
    np.random.seed(42)

    observation = create_example_data(args.model_name)

    # Run JAX inference
    jax_result, noise = run_jax_inference_example(observation, args.model_name)

    # Reset random seed for fair comparison
    np.random.seed(42)

    # Run PyTorch inference
    pytorch_result = run_pytorch_inference_example(observation, args.model_name, noise)

    # Compare results
    compare_results(jax_result, pytorch_result)

    print("\n" + "=" * 50)
    print("Example completed!")


if __name__ == "__main__":
    main()
