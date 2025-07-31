#!/usr/bin/env python3
"""
Example showing the unified Policy interface for both JAX and PyTorch Pi0 models.

This demonstrates how to use the same Policy class for both backends by simply
setting the is_pytorch parameter.
"""

import logging

import numpy as np

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def create_test_observation():
    """Create a test observation."""
    return {
        "images": {
            "base_0_rgb": np.random.rand(224, 224, 3).astype(np.float32),
            "left_wrist_0_rgb": np.random.rand(224, 224, 3).astype(np.float32),
            "right_wrist_0_rgb": np.random.rand(224, 224, 3).astype(np.float32),
        },
        "image_masks": {
            "base_0_rgb": True,
            "left_wrist_0_rgb": True,
            "right_wrist_0_rgb": True,
        },
        "state": np.random.rand(32).astype(np.float32),
        "tokenized_prompt": np.random.randint(0, 1000, (48,), dtype=np.int32),
        "tokenized_prompt_mask": np.ones((48,), dtype=bool),
    }


def main():
    """Demonstrate unified policy interface."""
    logger.info("🤖 Unified Policy Interface Demo")
    logger.info("=" * 50)

    # Create test data
    observation = create_test_observation()
    noise = np.random.randn(50, 32).astype(np.float32)  # Fixed noise for comparison

    # === JAX Version ===
    logger.info("\n🔵 JAX Pi0 Model")
    try:
        import jax

        from openpi.models.pi0_config import Pi0Config
        from openpi.policies.policy import Policy

        # Create JAX model and policy
        config = Pi0Config(action_dim=32, action_horizon=50, max_token_len=48)
        jax_model = config.create(jax.random.key(42))

        jax_policy = Policy(
            model=jax_model,
            sample_kwargs={"num_steps": 10},
            # is_pytorch=False (default)
        )

        # Run inference
        jax_result = jax_policy.infer(observation, noise=noise)
        logger.info(f"✅ JAX inference: {jax_result['policy_timing']['infer_ms']:.1f}ms")
        logger.info(f"   Actions shape: {jax_result['actions'].shape}")

    except ImportError as e:
        logger.warning(f"⚠️  JAX not available: {e}")
        jax_result = None

    # === PyTorch Version ===
    logger.info("\n🟠 PyTorch Pi0 Model")
    try:
        from openpi.models.pi0_config import Pi0Config
        from openpi.models_pytorch.pi0_pytorch import PI0Pytorch
        from openpi.policies.policy import Policy

        # Create PyTorch model and policy
        config = Pi0Config(action_dim=32, action_horizon=50, max_token_len=48)
        pytorch_model = PI0Pytorch(config)

        pytorch_policy = Policy(
            model=pytorch_model,
            device="cpu",
            sample_kwargs={"num_steps": 10},
            is_pytorch=True,  # 🔑 Key difference: explicit PyTorch flag
        )

        # Run inference (same API!)
        pytorch_result = pytorch_policy.infer(observation, noise=noise)
        logger.info(f"✅ PyTorch inference: {pytorch_result['policy_timing']['infer_ms']:.1f}ms")
        logger.info(f"   Actions shape: {pytorch_result['actions'].shape}")

    except ImportError as e:
        logger.warning(f"⚠️  PyTorch not available: {e}")
        pytorch_result = None

    # === Compare Results ===
    if jax_result is not None and pytorch_result is not None:
        logger.info("\n🔍 Comparing Results")
        diff = np.abs(jax_result["actions"] - pytorch_result["actions"])
        max_diff = np.max(diff)
        logger.info(f"   Max difference: {max_diff:.6f}")

        if max_diff < 1e-5:
            logger.info("🎉 Results match! Both backends produce identical outputs.")
        else:
            logger.warning("⚠️  Results differ - check implementation consistency.")

    logger.info("\n" + "=" * 50)
    logger.info("🎯 Key Benefits:")
    logger.info("   • Same Policy class for both JAX and PyTorch")
    logger.info("   • Identical API - just set is_pytorch=True")
    logger.info("   • No separate wrapper classes needed")
    logger.info("   • Easy to switch between backends")


if __name__ == "__main__":
    main()
