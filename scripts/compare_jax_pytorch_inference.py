#!/usr/bin/env python3
"""
Compare inference results between JAX and PyTorch Pi0 models.

This script creates test observations and runs inference on both JAX and PyTorch
implementations of the Pi0 model to ensure they produce identical results.
"""

import argparse
import logging
import sys
from typing import Any

import numpy as np

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def create_test_observation(config, include_language: bool = True) -> dict[str, Any]:
    """Create a test observation for inference."""
    batch_size = 1

    # Create test images (3 cameras)
    image_shape = (224, 224, 3)  # Standard image resolution
    images = {
        "base_0_rgb": np.random.rand(*image_shape).astype(np.float32),
        "left_wrist_0_rgb": np.random.rand(*image_shape).astype(np.float32),
        "right_wrist_0_rgb": np.random.rand(*image_shape).astype(np.float32),
    }

    # Create image masks (all valid)
    image_masks = {
        "base_0_rgb": True,
        "left_wrist_0_rgb": True,
        "right_wrist_0_rgb": True,
    }

    # Create test state
    state = np.random.rand(config.action_dim).astype(np.float32)

    observation = {
        "images": images,
        "image_masks": image_masks,
        "state": state,
    }

    if include_language:
        # Create test language tokens
        tokenized_prompt = np.random.randint(0, 1000, (config.max_token_len,), dtype=np.int32)
        tokenized_prompt_mask = np.ones((config.max_token_len,), dtype=bool)

        observation.update(
            {
                "tokenized_prompt": tokenized_prompt,
                "tokenized_prompt_mask": tokenized_prompt_mask,
            }
        )

    return observation


def create_test_noise(config) -> np.ndarray:
    """Create deterministic test noise for reproducible results."""
    np.random.seed(42)  # Fixed seed for reproducibility
    return np.random.randn(config.action_horizon, config.action_dim).astype(np.float32)


def compare_arrays(arr1: np.ndarray, arr2: np.ndarray, name: str, rtol: float = 1e-5, atol: float = 1e-6) -> bool:
    """Compare two numpy arrays and report differences."""
    try:
        np.testing.assert_allclose(arr1, arr2, rtol=rtol, atol=atol)
        logger.info(f"✅ {name}: Arrays match within tolerance (rtol={rtol}, atol={atol})")
        return True
    except AssertionError:
        logger.error(f"❌ {name}: Arrays do not match")
        logger.error(f"   JAX shape: {arr1.shape}, PyTorch shape: {arr2.shape}")
        logger.error(f"   Max absolute difference: {np.max(np.abs(arr1 - arr2))}")
        logger.error(f"   Mean absolute difference: {np.mean(np.abs(arr1 - arr2))}")
        logger.error(f"   Relative difference: {np.max(np.abs((arr1 - arr2) / (np.abs(arr1) + 1e-8)))}")
        return False


def run_jax_inference(config, observation: dict[str, Any], noise: np.ndarray) -> dict[str, Any]:
    """Run inference using JAX Pi0 model."""
    try:
        import jax

        from openpi.policies import policy_config
        from openpi.policies.policy import Policy

        logger.info("Running JAX inference...")

        # Create JAX policy
        rng = jax.random.key(42)
        model = config.create(rng)

        jax_policy = Policy(
            model=model,
            sample_kwargs={"num_steps": 10},
        )

        # Run inference
        result = jax_policy.infer(observation, noise=noise)
        logger.info(f"JAX inference completed in {result['policy_timing']['infer_ms']:.2f}ms")

        return result

    except ImportError as e:
        logger.error(f"Failed to import JAX dependencies: {e}")
        return None


def run_pytorch_inference(
    config, observation: dict[str, Any], noise: np.ndarray, device: str = "cpu"
) -> dict[str, Any]:
    """Run inference using PyTorch Pi0 model."""
    try:
        from openpi.models_pytorch.pi0_pytorch import PI0Pytorch
        from openpi.policies.policy import Policy

        logger.info("Running PyTorch inference...")

        # Create PyTorch model
        model = PI0Pytorch(config)

        # Create policy with PyTorch backend
        pytorch_policy = Policy(
            model=model,
            device=device,
            sample_kwargs={"num_steps": 10},
            is_pytorch=True,  # Explicitly specify PyTorch backend
        )

        # Run inference
        result = pytorch_policy.infer(observation, noise=noise)
        logger.info(f"PyTorch inference completed in {result['policy_timing']['infer_ms']:.2f}ms")

        return result

    except ImportError as e:
        logger.error(f"Failed to import PyTorch dependencies: {e}")
        return None


def main():
    parser = argparse.ArgumentParser(description="Compare JAX and PyTorch Pi0 inference")
    parser.add_argument("--device", default="cpu", help="Device for PyTorch (cpu, cuda)")
    parser.add_argument("--num-tests", type=int, default=5, help="Number of test cases to run")
    parser.add_argument("--tolerance", type=float, default=1e-5, help="Numerical tolerance for comparison")
    parser.add_argument("--no-language", action="store_true", help="Test without language tokens")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose logging")

    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # Import Pi0Config
    try:
        from openpi.models.pi0_config import Pi0Config
    except ImportError as e:
        logger.error(f"Failed to import Pi0Config: {e}")
        sys.exit(1)

    # Create test configuration
    config = Pi0Config(
        action_dim=32,
        action_horizon=50,
        max_token_len=48,
    )

    logger.info(f"Running {args.num_tests} test cases...")
    logger.info(f"Model config: action_dim={config.action_dim}, action_horizon={config.action_horizon}")

    all_tests_passed = True

    for test_idx in range(args.num_tests):
        logger.info(f"\n{'=' * 60}")
        logger.info(f"Test Case {test_idx + 1}/{args.num_tests}")
        logger.info(f"{'=' * 60}")

        # Create test data
        observation = create_test_observation(config, include_language=not args.no_language)
        noise = create_test_noise(config)

        # Run JAX inference
        jax_result = run_jax_inference(config, observation, noise)
        if jax_result is None:
            logger.error("JAX inference failed, skipping comparison")
            all_tests_passed = False
            continue

        # Run PyTorch inference
        pytorch_result = run_pytorch_inference(config, observation, noise, args.device)
        if pytorch_result is None:
            logger.error("PyTorch inference failed, skipping comparison")
            all_tests_passed = False
            continue

        # Compare results
        logger.info("\nComparing results...")

        # Compare actions (main output)
        actions_match = compare_arrays(
            jax_result["actions"], pytorch_result["actions"], "Actions", rtol=args.tolerance, atol=args.tolerance
        )

        # Compare states (should be identical as they're just passed through)
        state_match = compare_arrays(jax_result["state"], pytorch_result["state"], "State", rtol=1e-10, atol=1e-10)

        test_passed = actions_match and state_match
        all_tests_passed = all_tests_passed and test_passed

        if test_passed:
            logger.info("✅ Test case passed!")
        else:
            logger.error("❌ Test case failed!")

    # Final summary
    logger.info(f"\n{'=' * 60}")
    logger.info("FINAL RESULTS")
    logger.info(f"{'=' * 60}")

    if all_tests_passed:
        logger.info("🎉 All tests passed! JAX and PyTorch models produce identical results.")
        sys.exit(0)
    else:
        logger.error("💥 Some tests failed! JAX and PyTorch models produce different results.")
        sys.exit(1)


if __name__ == "__main__":
    main()
