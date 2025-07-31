#!/usr/bin/env python3
"""
Example script showing how to run inference with both JAX and PyTorch Pi0 models.

This demonstrates the basic usage patterns for both implementations.
"""

import logging

import numpy as np

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def create_example_observation():
    """Create an example observation for testing."""
    # Create test images (RGB, shape: height x width x channels)
    image_shape = (224, 224, 3)

    observation = {
        "images": {
            "base_0_rgb": np.random.rand(*image_shape).astype(np.float32),
            "left_wrist_0_rgb": np.random.rand(*image_shape).astype(np.float32),
            "right_wrist_0_rgb": np.random.rand(*image_shape).astype(np.float32),
        },
        "image_masks": {
            "base_0_rgb": True,
            "left_wrist_0_rgb": True,
            "right_wrist_0_rgb": True,
        },
        "state": np.random.rand(32).astype(np.float32),  # Robot state (32-dim)
        "tokenized_prompt": np.random.randint(0, 1000, (48,), dtype=np.int32),  # Language tokens
        "tokenized_prompt_mask": np.ones((48,), dtype=bool),  # Token mask
    }

    return observation


def run_jax_inference_example():
    """Example of running inference with JAX Pi0 model."""
    logger.info("=== JAX Pi0 Inference Example ===")

    try:
        import jax

        from openpi.models.pi0_config import Pi0Config
        from openpi.policies.policy import Policy

        # Create model configuration
        config = Pi0Config(
            action_dim=32,
            action_horizon=50,
            max_token_len=48,
        )

        # Create the model
        rng = jax.random.key(42)
        model = config.create(rng)

        # Create policy wrapper
        policy = Policy(
            model=model,
            sample_kwargs={"num_steps": 10},  # Number of diffusion denoising steps
        )

        # Create test observation
        observation = create_example_observation()

        # Optional: provide custom noise for deterministic results
        noise = np.random.randn(config.action_horizon, config.action_dim).astype(np.float32)

        # Run inference
        logger.info("Running JAX inference...")
        result = policy.infer(observation, noise=noise)

        # Print results
        logger.info("JAX inference completed!")
        logger.info(f"  - Inference time: {result['policy_timing']['infer_ms']:.2f}ms")
        logger.info(f"  - Actions shape: {result['actions'].shape}")
        logger.info(f"  - Actions range: [{result['actions'].min():.3f}, {result['actions'].max():.3f}]")

        return result

    except ImportError as e:
        logger.error(f"Failed to run JAX inference: {e}")
        return None


def run_pytorch_inference_example():
    """Example of running inference with PyTorch Pi0 model."""
    logger.info("\n=== PyTorch Pi0 Inference Example ===")

    try:
        from openpi.models.pi0_config import Pi0Config
        from openpi.models_pytorch.pi0_pytorch import PI0Pytorch
        from openpi.policies.policy import Policy

        # Create model configuration (same as JAX)
        config = Pi0Config(
            action_dim=32,
            action_horizon=50,
            max_token_len=48,
        )

        # Create PyTorch model
        model = PI0Pytorch(config)

        # Create policy with PyTorch backend
        policy = Policy(
            model=model,
            device="cpu",  # Use "cuda" if GPU is available
            sample_kwargs={"num_steps": 10},
            is_pytorch=True,  # Explicitly specify PyTorch backend
        )

        # Create test observation
        observation = create_example_observation()

        # Optional: provide custom noise for deterministic results
        noise = np.random.randn(config.action_horizon, config.action_dim).astype(np.float32)

        # Run inference
        logger.info("Running PyTorch inference...")
        result = policy.infer(observation, noise=noise)

        # Print results
        logger.info("PyTorch inference completed!")
        logger.info(f"  - Inference time: {result['policy_timing']['infer_ms']:.2f}ms")
        logger.info(f"  - Actions shape: {result['actions'].shape}")
        logger.info(f"  - Actions range: [{result['actions'].min():.3f}, {result['actions'].max():.3f}]")

        return result

    except ImportError as e:
        logger.error(f"Failed to run PyTorch inference: {e}")
        return None


def compare_results(jax_result, pytorch_result):
    """Compare results from both implementations."""
    if jax_result is None or pytorch_result is None:
        logger.warning("Cannot compare results - one implementation failed")
        return

    logger.info("\n=== Comparing Results ===")

    # Compare actions
    actions_diff = np.abs(jax_result["actions"] - pytorch_result["actions"])
    max_diff = np.max(actions_diff)
    mean_diff = np.mean(actions_diff)

    logger.info("Actions comparison:")
    logger.info(f"  - Max absolute difference: {max_diff:.6f}")
    logger.info(f"  - Mean absolute difference: {mean_diff:.6f}")

    # Check if results are close
    if np.allclose(jax_result["actions"], pytorch_result["actions"], rtol=1e-5, atol=1e-6):
        logger.info("✅ Results match within tolerance!")
    else:
        logger.warning("❌ Results differ significantly!")

    # Compare timing
    jax_time = jax_result["policy_timing"]["infer_ms"]
    pytorch_time = pytorch_result["policy_timing"]["infer_ms"]

    logger.info("Timing comparison:")
    logger.info(f"  - JAX: {jax_time:.2f}ms")
    logger.info(f"  - PyTorch: {pytorch_time:.2f}ms")
    logger.info(f"  - Speedup: {jax_time / pytorch_time:.2f}x (JAX time / PyTorch time)")


def main():
    """Run both inference examples and compare results."""
    logger.info("Pi0 Model Inference Comparison")
    logger.info("=" * 50)

    # Set random seed for reproducibility
    np.random.seed(42)

    # Run JAX inference
    jax_result = run_jax_inference_example()

    # Reset random seed for fair comparison
    np.random.seed(42)

    # Run PyTorch inference
    pytorch_result = run_pytorch_inference_example()

    # Compare results
    compare_results(jax_result, pytorch_result)

    logger.info("\n" + "=" * 50)
    logger.info("Example completed!")


if __name__ == "__main__":
    main()
