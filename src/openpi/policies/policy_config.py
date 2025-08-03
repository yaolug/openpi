from collections.abc import Sequence
import dataclasses
import logging
import pathlib
from typing import Any

import jax.numpy as jnp

import openpi.models.model as _model
import openpi.policies.policy as _policy
import openpi.shared.download as download
from openpi.training import checkpoints as _checkpoints
from openpi.training import config as _config
import openpi.transforms as transforms


@dataclasses.dataclass
class PolicyConfig:
    model: _model.BaseModel
    norm_stats: dict[str, transforms.NormStats]

    input_layers: Sequence[transforms.DataTransformFn]
    output_layers: Sequence[transforms.DataTransformFn]

    model_type: _model.ModelType = _model.ModelType.PI0
    default_prompt: str | None = None
    sample_kwargs: dict[str, Any] | None = None

    
def get_policy_class(name: str):
    """Get the policy's class and config class given a name (matching the policy class' `name` attribute)."""
    if "pi0" in name:
        from openpi.models_pytorch.pi0_pytorch import PI0Pytorch
        return PI0Pytorch
    else:
        raise ValueError(f"Unknown policy name: {name}")


def create_trained_policy(
    train_config: _config.TrainConfig,
    checkpoint_dir: pathlib.Path | str,
    *,
    repack_transforms: transforms.Group | None = None,
    sample_kwargs: dict[str, Any] | None = None,
    default_prompt: str | None = None,
    norm_stats: dict[str, transforms.NormStats] | None = None,
    is_pytorch: bool = False,
) -> _policy.Policy:
    """Create a policy from a trained checkpoint.

    Args:
        train_config: The training config to use to create the model.
        checkpoint_dir: The directory to load the model from.
        repack_transforms: Optional transforms that will be applied before any other transforms.
        sample_kwargs: The kwargs to pass to the `sample_actions` method. If not provided, the default
            kwargs will be used.
        default_prompt: The default prompt to use for the policy. Will inject the prompt into the input
            data if it doesn't already exist.
        norm_stats: The norm stats to use for the policy. If not provided, the norm stats will be loaded
            from the checkpoint directory.
    """
    repack_transforms = repack_transforms or transforms.Group()
    checkpoint_dir = download.maybe_download(str(checkpoint_dir))

    logging.info("Loading model...")
    if is_pytorch:
        print(f"train_config: {train_config}")
        
        # Create a config object compatible with the PyTorch model
        
        # Create the model with the config
        model_class = get_policy_class(train_config.name)
        model = model_class(config=train_config.model)
        
        # Load weights if checkpoint exists
        try:
            import os
            import torch
            from safetensors.torch import load_file
            
            # Try SafeTensors format first
            weight_path = os.path.join(checkpoint_dir, "model.safetensors")
            if os.path.exists(weight_path):
                state_dict = load_file(weight_path)
                model.load_state_dict(state_dict)
                logging.info(f"Loaded PyTorch weights from {weight_path} (removed 'model.' prefix from {len([k for k in state_dict.keys() if k.startswith('model.')])} keys)")
            else:
                logging.warning(f"No PyTorch weights found at {weight_path}, using random initialization")
        except Exception as e:
            logging.warning(f"Failed to load PyTorch weights: {e}, using random initialization")
    else:
        model = train_config.model.load(_model.restore_params(checkpoint_dir / "params", dtype=jnp.bfloat16))

    data_config = train_config.data.create(train_config.assets_dirs, train_config.model)
    if norm_stats is None:
        # We are loading the norm stats from the checkpoint instead of the config assets dir to make sure
        # that the policy is using the same normalization stats as the original training process.
        if data_config.asset_id is None:
            raise ValueError("Asset id is required to load norm stats.")
        norm_stats = _checkpoints.load_norm_stats(checkpoint_dir / "assets", data_config.asset_id)

    return _policy.Policy(
        model,
        transforms=[
            *repack_transforms.inputs,
            transforms.InjectDefaultPrompt(default_prompt),
            *data_config.data_transforms.inputs,
            transforms.Normalize(norm_stats, use_quantiles=data_config.use_quantile_norm),
            *data_config.model_transforms.inputs,
        ],
        output_transforms=[
            *data_config.model_transforms.outputs,
            transforms.Unnormalize(norm_stats, use_quantiles=data_config.use_quantile_norm),
            *data_config.data_transforms.outputs,
            *repack_transforms.outputs,
        ],
        sample_kwargs=sample_kwargs,
        metadata=train_config.policy_metadata,
        is_pytorch=is_pytorch,
        device="cuda" if is_pytorch else None
    )
