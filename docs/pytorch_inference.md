# PyTorch Pi0 Inference Implementation

This document describes the PyTorch implementation of Pi0 inference and how to use both JAX and PyTorch versions of the model.

## Overview

We provide two implementations of the Pi0 model for inference:

1. **JAX Implementation** (`src/openpi/models/pi0.py`) - Original implementation using JAX/Flax
2. **PyTorch Implementation** (`src/openpi/models_pytorch/pi0_pytorch.py`) - New PyTorch port for broader compatibility

Both implementations use the **same unified Policy interface** - you simply pass `is_pytorch=True` to use the PyTorch backend. This provides identical APIs and should produce identical results.

## Quick Start

### JAX Inference

```python
import jax
import numpy as np
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

# Create observation
observation = {
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

# Run inference
result = policy.infer(observation)
actions = result["actions"]  # Shape: (action_horizon, action_dim)
```

### PyTorch Inference

```python
import numpy as np
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

# Create observation (same format as JAX)
observation = {
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

# Run inference
result = policy.infer(observation)
actions = result["actions"]  # Shape: (action_horizon, action_dim)
```

## API Reference

### Pi0Config

The `Pi0Config` class defines the model configuration:

```python
@dataclasses.dataclass(frozen=True)
class Pi0Config:
    dtype: str = "bfloat16"
    paligemma_variant: str = "gemma_2b"
    action_expert_variant: str = "gemma_300m"
    action_dim: int = 32
    action_horizon: int = 50
    max_token_len: int = 48
```

### Observation Format

Observations must include:

- `images`: Dictionary with RGB images for each camera
  - `"base_0_rgb"`: Base camera image (H, W, 3)
  - `"left_wrist_0_rgb"`: Left wrist camera image (H, W, 3)
  - `"right_wrist_0_rgb"`: Right wrist camera image (H, W, 3)
- `image_masks`: Dictionary with boolean masks for each image
- `state`: Robot state vector (action_dim,)
- `tokenized_prompt`: Language tokens (max_token_len,)
- `tokenized_prompt_mask`: Boolean mask for tokens (max_token_len,)

### Unified Policy Interface

Both JAX and PyTorch models use the same `Policy` class. Simply set `is_pytorch=True` for PyTorch models:

```python
# JAX model
policy = Policy(model=jax_model, sample_kwargs={"num_steps": 10})

# PyTorch model  
policy = Policy(model=pytorch_model, device="cuda", is_pytorch=True, sample_kwargs={"num_steps": 10})
```

The `infer` method is identical for both:

```python
def infer(self, obs: dict, *, noise: np.ndarray | None = None) -> dict:
    """Run inference on observation.
    
    Args:
        obs: Observation dictionary
        noise: Optional noise for deterministic results
        
    Returns:
        Dictionary with:
        - "actions": Predicted actions (action_horizon, action_dim)
        - "state": Input state (passed through)
        - "policy_timing": Timing information
    """
```

## Examples

### Basic Usage

Run the basic example:

```bash
python examples/inference_example.py
```

This will run both JAX and PyTorch inference and compare the results.

### Unified Interface Demo

See the unified interface in action:

```bash
python examples/unified_policy_example.py
```

This demonstrates how both backends use the same Policy class - just pass `is_pytorch=True` for PyTorch models.

### Comparison Script

Run detailed comparison tests:

```bash
python scripts/compare_jax_pytorch_inference.py --num-tests 10 --device cpu
```

Options:
- `--device`: PyTorch device ("cpu" or "cuda")
- `--num-tests`: Number of test cases to run
- `--tolerance`: Numerical tolerance for comparison
- `--no-language`: Test without language tokens
- `--verbose`: Enable verbose logging

## Implementation Details

### PyTorch Model Architecture

The PyTorch implementation (`PI0Pytorch`) mirrors the JAX version:

1. **Vision Encoder**: PaliGemma vision model for image embedding
2. **Language Model**: Gemma transformer for sequence modeling
3. **Action Expert**: Separate Gemma expert for action prediction
4. **Diffusion Process**: Flow-matching based action generation

### Key Components

- `embed_prefix()`: Encodes images and language tokens
- `embed_suffix()`: Encodes robot state and noisy actions with timestep
- `sample_actions()`: Runs the full diffusion sampling process
- `denoise_step()`: Single denoising step in the diffusion process

### Numerical Considerations

- Both implementations use the same numerical precision settings
- The PyTorch version includes deterministic seeding options
- Small numerical differences (< 1e-5) are expected due to library differences

## Troubleshooting

### Common Issues

1. **Import Errors**: Ensure both JAX and PyTorch are installed
2. **Memory Issues**: Reduce batch size or use CPU for testing
3. **Shape Mismatches**: Check observation tensor shapes match expected format
4. **Device Errors**: Ensure all tensors are on the same device (PyTorch)

### Performance Tips

1. **GPU Usage**: Use CUDA device for PyTorch for better performance
2. **Batch Processing**: Process multiple observations together when possible
3. **Caching**: The vision embeddings can be cached for repeated inference

## Key Advantages

### Unified Interface
- **Same Policy class** for both JAX and PyTorch models
- **Single parameter** (`is_pytorch=True`) to switch backends  
- **No duplicate code** - one policy implementation handles both
- **Easy migration** between backends without changing user code

### Simple Usage
```python
# JAX (default)
policy = Policy(model=jax_model)

# PyTorch (explicit)  
policy = Policy(model=pytorch_model, device="cuda", is_pytorch=True)

# Same API for inference
result = policy.infer(observation)
```

## Contributing

When modifying the inference implementations:

1. Ensure both JAX and PyTorch versions produce identical results
2. Run the comparison script to validate changes
3. Update this documentation if the API changes
4. Add tests for new functionality
5. Remember that both backends use the same Policy class

## Dependencies

### JAX Implementation
- `jax`
- `flax`
- `einops`

### PyTorch Implementation  
- `torch`
- `transformers`
- `numpy`

Both implementations also require the OpenPI core dependencies. 