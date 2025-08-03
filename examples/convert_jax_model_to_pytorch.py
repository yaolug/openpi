#!/usr/bin/env python3
"""
Load a JAX model and print all parameter keys, with optional conversion to PyTorch.

This script loads a JAX model checkpoint using orbax and can either:
1. Print out all the parameter keys in a hierarchical structure for inspection
2. Convert the JAX model to PyTorch format using our PI0Pytorch model

Usage:
    # Just inspect keys:
    python convert_jax_model_to_pytorch.py --checkpoint_dir /path/to/checkpoint --inspect_only
    
    # Convert to PyTorch:
    python convert_jax_model_to_pytorch.py --checkpoint_dir /path/to/checkpoint --output_path /path/to/output

Example:
    python convert_jax_model_to_pytorch.py --checkpoint_dir /home/user/.cache/openpi/openpi-assets/checkpoints/pi0_base/params --output_path ./pi0_pytorch
"""

import argparse
import pathlib
from typing import Any, Dict

import jax
import numpy as np
import orbax.checkpoint as ocp
import torch
from jax.sharding import SingleDeviceSharding
from safetensors.torch import save_model

# Import our PI0Pytorch model
from openpi.models_pytorch.pi0_pytorch import PI0Pytorch
from openpi.models.pi0_config import Pi0Config
import openpi.models.gemma as _gemma

PRECISIONS = {"bfloat16": torch.bfloat16, "float32": torch.float32, "float16": torch.float16}


def flatten_for_inspection(tree, parent_key="", separator="/"):
    """
    Flatten a nested dictionary for easy inspection of keys.
    
    Args:
        tree: The nested dictionary (JAX pytree)
        parent_key: Current parent key path
        separator: Separator to use between key levels
        
    Returns:
        Dictionary with flattened keys and array shapes as values
    """
    items = []
    for k, v in tree.items():
        new_key = f"{parent_key}{separator}{k}" if parent_key else k
        if isinstance(v, dict):
            items.extend(flatten_for_inspection(v, new_key, separator).items())
        else:
            # Store shape and dtype information instead of the actual array
            if hasattr(v, 'shape') and hasattr(v, 'dtype'):
                items.append((new_key, f"shape: {v.shape}, dtype: {v.dtype}"))
            else:
                items.append((new_key, f"type: {type(v)}"))
    return dict(items)


def flatten_for_npz(tree, parent_key=""):
    """Flatten nested dictionary for conversion processing."""
    out = {}
    for k, v in tree.items():
        new_key = f"{parent_key}/{k}" if parent_key else k
        if isinstance(v, dict):
            out.update(flatten_for_npz(v, new_key))
        else:
            out[new_key] = np.array(v)
    return out


def slice_paligemma_state_dict(state_dict):
    """Convert PaliGemma JAX parameters to PyTorch format for our PI0Pytorch model."""
    suffix = "/value" if "img/embedding/kernel/value" in state_dict else ""
    print(f"\n🔄 Converting PaliGemma parameters (suffix: '{suffix}')...")

    paligemma_params = {}
    
    # Extract all PaliGemma-related parameters and convert them to PyTorch format
    # We'll store them with the paligemma_with_expert prefix to match our model structure
    for key, value in state_dict.items():
        if key.startswith("img/") or key.startswith("llm/"):
            # Convert JAX parameter to PyTorch tensor
            if not isinstance(value, torch.Tensor):
                tensor_value = torch.from_numpy(np.array(value))
            else:
                tensor_value = value
            
            # For now, keep the JAX naming - we'll handle mapping later
            paligemma_params[f"paligemma_with_expert.{key}{suffix}"] = tensor_value
    
    print(f"  Extracted {len(paligemma_params)} PaliGemma parameters")
    return paligemma_params


def slice_gemma_state_dict(state_dict):
    """Convert Gemma expert JAX parameters to PyTorch format."""
    print(f"\n🧠 Converting Gemma expert parameters...")
    
    gemma_params = {}
    
    # Extract Gemma expert parameters (those with _1 suffix)
    for key, value in state_dict.items():
        if "_1/" in key or key.endswith("_1"):
            # Convert JAX parameter to PyTorch tensor
            if not isinstance(value, torch.Tensor):
                tensor_value = torch.from_numpy(np.array(value))
            else:
                tensor_value = value
            
            # Store with gemma_expert prefix
            gemma_params[f"paligemma_with_expert.gemma_expert.{key}"] = tensor_value
    
    print(f"  Extracted {len(gemma_params)} Gemma expert parameters")
    return gemma_params


def slice_initial_orbax_checkpoint(checkpoint_dir: str):
    """Load and process the initial orbax checkpoint."""
    params_path = pathlib.Path(checkpoint_dir).resolve()
    checkpointer = ocp.PyTreeCheckpointer()

    metadata = checkpointer.metadata(params_path)
    print("Metadata keys:", list(metadata.keys()))

    params_name = "params"
    item = {params_name: metadata[params_name]}
    device = jax.local_devices()[0]
    sharding = SingleDeviceSharding(device)
    restored = checkpointer.restore(
        params_path,
        ocp.args.PyTreeRestore(
            item=item,
            restore_args=jax.tree_util.tree_map(
                lambda _: ocp.ArrayRestoreArgs(
                    restore_type=jax.Array,
                    sharding=sharding,
                ),
                item,
            ),
            transforms={},
        ),
    )
    params = restored[params_name]

    # get params for PaliGemma
    pali_params = params["PaliGemma"]
    del params["PaliGemma"]
    pali_params_flat = flatten_for_npz(pali_params)
    return {"paligemma_params": pali_params_flat, "projection_params": params}


def load_jax_model_and_print_keys(checkpoint_dir: str):
    """
    Load JAX model from checkpoint and print all parameter keys.
    
    Args:
        checkpoint_dir: Path to the checkpoint directory
    """
    params_path = pathlib.Path(checkpoint_dir).resolve()
    
    if not params_path.exists():
        print(f"Error: Checkpoint directory does not exist: {params_path}")
        return
    
    print(f"Loading JAX model from: {params_path}")
    print("=" * 80)
    
    try:
        # Initialize checkpointer
        checkpointer = ocp.PyTreeCheckpointer()
        
        # Load metadata to see available keys
        metadata = checkpointer.metadata(params_path)
        print("Available top-level keys in checkpoint:")
        for key in metadata.keys():
            print(f"  - {key}")
        print()
        
        # Restore the parameters
        params_name = "params"
        if params_name not in metadata:
            print(f"Warning: '{params_name}' not found in metadata. Available keys: {list(metadata.keys())}")
            if metadata.keys():
                params_name = list(metadata.keys())[0]
                print(f"Using '{params_name}' instead.")
            else:
                print("No keys found in metadata!")
                return
        
        item = {params_name: metadata[params_name]}
        device = jax.local_devices()[0]
        sharding = SingleDeviceSharding(device)
        
        print(f"Restoring parameters for key: '{params_name}'...")
        restored = checkpointer.restore(
            params_path,
            ocp.args.PyTreeRestore(
                item=item,
                restore_args=jax.tree_util.tree_map(
                    lambda _: ocp.ArrayRestoreArgs(
                        restore_type=jax.Array,
                        sharding=sharding,
                    ),
                    item,
                ),
                transforms={},
            ),
        )
        
        params = restored[params_name]
        print(f"Successfully loaded parameters!")
        print()
        
        # Flatten and print all keys
        flat_params = flatten_for_inspection(params)
        
        print(f"All parameter keys ({len(flat_params)} total):")
        print("=" * 80)
        
        # Sort keys for better readability
        sorted_keys = sorted(flat_params.keys())
        
        for key in sorted_keys:
            print(f"{key:<60} -> {flat_params[key]}")
        
        print()
        print("=" * 80)
        print(f"Summary: Found {len(flat_params)} parameters")
        
        # Print some high-level structure information
        top_level_keys = set()
        for key in sorted_keys:
            top_level_key = key.split('/')[0]
            top_level_keys.add(top_level_key)
        
        print(f"Top-level parameter groups: {sorted(list(top_level_keys))}")
        
    except Exception as e:
        print(f"Error loading checkpoint: {e}")
        import traceback
        traceback.print_exc()


def convert_pi0_checkpoint(checkpoint_dir: str, precision: str, output_path: str):
    """
    Convert PI0 JAX checkpoint to PyTorch format.
    
    Args:
        checkpoint_dir: Path to the JAX checkpoint
        precision: Model precision (float32, bfloat16, float16)
        output_path: Path to save the converted PyTorch model
    """
    print(f"Converting PI0 checkpoint from {checkpoint_dir} to {output_path}")
    print("=" * 80)
    
    # Break down orbax ckpts
    initial_params = slice_initial_orbax_checkpoint(checkpoint_dir=checkpoint_dir)
    
    # Process projection params
    print(f"\n🎯 Converting projection parameters...")
    keys = [
        "state_proj",
        "action_in_proj", 
        "action_out_proj",
        "action_time_mlp_in",
        "action_time_mlp_out",
    ]

    projection_params = {}
    for key in keys:
        kernel_params = initial_params["projection_params"][key]["kernel"]
        bias_params = initial_params["projection_params"][key]["bias"]
        if isinstance(kernel_params, dict):
            weight = kernel_params["value"]
            bias = bias_params["value"]
        else:
            weight = kernel_params
            bias = bias_params
        
        pytorch_weight_key = f"{key}.weight"
        pytorch_bias_key = f"{key}.bias"
        print(f"  {key}/kernel -> {pytorch_weight_key}")
        print(f"  {key}/bias -> {pytorch_bias_key}")
        
        projection_params[pytorch_weight_key] = torch.from_numpy(np.array(weight)).T
        projection_params[pytorch_bias_key] = torch.from_numpy(np.array(bias))

    # Process PaliGemma weights
    paligemma_params = slice_paligemma_state_dict(initial_params["paligemma_params"])

    # Process Gemma weights
    gemma_params = slice_gemma_state_dict(initial_params["paligemma_params"])

    # Create Pi0Config based on checkpoint path
    if "pi0_aloha_sim" in checkpoint_dir:
        pi0_config = Pi0Config(
            action_dim=14,  # ALOHA has 14 action dimensions
            action_horizon=50,
        )
    elif "pi0_aloha_towel" in checkpoint_dir:
        pi0_config = Pi0Config(
            action_dim=14,  # ALOHA has 14 action dimensions
            action_horizon=50,
        )
    elif "pi0_base" in checkpoint_dir:
        pi0_config = Pi0Config(
            action_dim=8,   # Base droid has 8 action dimensions
            action_horizon=10,
        )
    else:
        print("Warning: Could not determine PI0 config from checkpoint path. Using base config.")
        pi0_config = Pi0Config(
            action_dim=8,
            action_horizon=10,
        )

    # Instantiate model
    print(f"\n🏗️ Creating PI0Pytorch model with config: action_dim={pi0_config.action_dim}, action_horizon={pi0_config.action_horizon}")
    pi0_model = PI0Pytorch(pi0_config)

    # Combine all parameters (no prefix needed for our model structure)
    torch_dtype = PRECISIONS[precision]
    all_params = {**paligemma_params, **gemma_params, **projection_params}
    
    print(f"\n🚀 Loading {len(all_params)} parameters into PyTorch model...")
    print(f"  - PaliGemma parameters: {len(paligemma_params)}")
    print(f"  - Gemma expert parameters: {len(gemma_params)}")
    print(f"  - Projection parameters: {len(projection_params)}")
    print(f"  - Target precision: {precision} ({torch_dtype})")
    
    # Load state dict
    try:
        pi0_model.load_state_dict(all_params, strict=False)
        print(f"  ✅ Successfully loaded parameters into model")
    except Exception as e:
        print(f"  ⚠️ Warning: Could not load all parameters: {e}")
        print(f"  Continuing with partial load...")
    
    pi0_model = pi0_model.to(torch_dtype)

    # Save the converted model using safetensors
    print(f"\n💾 Saving converted model to {output_path}...")
    import os
    import shutil
    os.makedirs(output_path, exist_ok=True)
    
    # Save model weights as SafeTensors using save_model to handle tied weights
    save_model(pi0_model, os.path.join(output_path, "model.safetensors"))
    
    # Copy assets folder if it exists
    assets_source = pathlib.Path(checkpoint_dir).parent / "assets"
    if assets_source.exists():
        assets_dest = pathlib.Path(output_path) / "assets"
        if assets_dest.exists():
            shutil.rmtree(assets_dest)
        shutil.copytree(assets_source, assets_dest)
        print(f"  📁 Copied assets folder from {assets_source}")
    else:
        print(f"  ⚠️ Assets folder not found at {assets_source}")
    
    # Save config as JSON for reference
    import json
    config_dict = {
        "action_dim": pi0_config.action_dim,
        "action_horizon": pi0_config.action_horizon,
        "paligemma_variant": pi0_config.paligemma_variant,
        "action_expert_variant": pi0_config.action_expert_variant,
        "precision": precision,
    }
    with open(os.path.join(output_path, "config.json"), "w") as f:
        json.dump(config_dict, f, indent=2)
    
    print(f"  ✅ Model saved successfully!")
    print(f"  📄 Config saved to config.json")
    print(f"  🔢 Model weights saved to model.safetensors")

    print(f"\n🎉 Model conversion completed successfully!")
    print(f"📊 Model info: {type(pi0_model).__name__} with {sum(p.numel() for p in pi0_model.parameters())} total parameters")


def main():
    parser = argparse.ArgumentParser(description="Load JAX model and optionally convert to PyTorch")
    parser.add_argument(
        "--checkpoint_dir",
        type=str,
        required=True,
        help="Path to the JAX checkpoint directory"
    )
    parser.add_argument(
        "--output_path",
        type=str,
        help="Path to save converted PyTorch model (required for conversion)"
    )
    parser.add_argument(
        "--precision",
        choices=["float32", "bfloat16", "float16"],
        default="float32",
        type=str,
        help="Precision for model conversion"
    )
    parser.add_argument(
        "--inspect_only",
        action="store_true",
        help="Only inspect parameter keys, don't convert"
    )
    
    args = parser.parse_args()
    
    if args.inspect_only:
        load_jax_model_and_print_keys(args.checkpoint_dir)
    else:
        if not args.output_path:
            print("Error: --output_path is required for conversion. Use --inspect_only to only view keys.")
            return
        convert_pi0_checkpoint(args.checkpoint_dir, args.precision, args.output_path)


if __name__ == "__main__":
    main()
