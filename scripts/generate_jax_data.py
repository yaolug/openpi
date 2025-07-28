#!/usr/bin/env python3
"""
Generate JAX model inference data for comparison with PyTorch models.

This script creates example.pkl, outputs.pkl, and noise.pkl files that are used
by compare_with_jax.py to validate JAX<->PyTorch model conversion.
"""

import pickle
import numpy as np
from pathlib import Path
import argparse

import jax
import jax.numpy as jnp

from openpi.models import model as _model
from openpi.policies import droid_policy
from openpi.policies import policy_config as _policy_config
from openpi.shared import download
from openpi.training import config as _config
from openpi.training import data_loader as _data_loader


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


def run_jax_inference(config_name: str, model_name: str, example: dict, save_dir: Path):
    """Run JAX model inference and save results."""
    
    print(f"🤖 Loading JAX model for {model_name}...")
    
    # Create config and load policy
    config = _config.get_config(config_name)
    checkpoint_dir = download.maybe_download(f"s3://openpi-assets/checkpoints/{model_name}")
    
    # Create trained policy
    policy = _policy_config.create_trained_policy(config, checkpoint_dir)
    
    print(f"✅ Successfully loaded policy from {checkpoint_dir}")
    
    print(f"📸 Processing example data...")
    print(f"   Images: {list(example['images'].keys())}")
    print(f"   State shape: {example['state'].shape}")
    print(f"   Prompt: {example['prompt']}")
    
    # Get action dimensions from a quick inference to determine noise shape
    print(f"🔍 Determining model dimensions...")
    temp_result = policy.infer(example)
    temp_actions = temp_result['actions']
    
    if len(temp_actions.shape) == 2:
        # JAX returns (action_horizon, action_dim)
        action_horizon, output_action_dim = temp_actions.shape
    else:
        # Fallback
        action_horizon, output_action_dim = temp_actions.shape[-2], temp_actions.shape[-1]
    
    batch_size = 1  # Our example has batch_size = 1
    
    # Get the model's expected action dimension (may differ from output dimension)
    print(f"🔍 Detecting model's internal action dimension...")
    try:
        # Try to infer with test noise using output dimensions
        test_noise = np.random.normal(0, 1, (action_horizon, output_action_dim)).astype(np.float32)
        policy.infer(example, noise=test_noise)
        model_action_dim = output_action_dim  # If no error, dimensions match
        print(f"   Model dimensions match output dimensions: {model_action_dim}")
    except ValueError as e:
        if "does not match model's action dimension" in str(e):
            # Extract the expected dimension from the error message
            error_msg = str(e)
            model_action_dim = int(error_msg.split("model's action dimension ")[1])
            print(f"   Model expects {model_action_dim} action dimensions, output has {output_action_dim}")
        else:
            raise e
    
    print(f"📐 Model dimensions:")
    print(f"   batch_size: {batch_size}")
    print(f"   action_horizon: {action_horizon}")
    print(f"   output_action_dim: {output_action_dim}")
    print(f"   model_action_dim: {model_action_dim}")
    
    # 1. FIRST GENERATE NOISE
    print(f"🎲 Step 1: Generating noise...")
    rng_key = jax.random.key(42)
    noise_shape = (action_horizon, model_action_dim)  # Use model's expected dimension
    jax_noise = jax.random.normal(rng_key, noise_shape, dtype=jnp.float32)
    
    print(f"   Generated noise shape: {jax_noise.shape}")
    print(f"   Noise mean: {jnp.mean(jax_noise):.6f}, Std: {jnp.std(jax_noise):.6f}")
    
    # Convert noise to numpy for passing to the policy
    noise_np = np.array(jax_noise)
    
    # 2. PASS THE NOISE TO INFERENCE AND CAPTURE INTERMEDIATE DATA
    print(f"🎯 Step 2: Running inference with provided noise...")
    policy._rng = rng_key  # Set RNG for reproducibility
    
    # Capture the processed inputs (after normalization & tokenization) before they go to the model
    print(f"📝 Capturing processed inputs (post-normalization/tokenization)...")
    
    # Print all transforms that will be applied
    def print_transform_details(transform_obj, name):
        print(f"🔧 {name}:")
        print(f"   Transform type: {type(transform_obj).__name__}")
        
        # Try different ways to access the individual transforms
        transforms_list = None
        if hasattr(transform_obj, '_transforms'):
            transforms_list = transform_obj._transforms
        elif hasattr(transform_obj, 'transforms'):
            transforms_list = transform_obj.transforms
        elif hasattr(transform_obj, '_fns'):
            transforms_list = transform_obj._fns
        
        if transforms_list:
            print(f"   Number of transforms: {len(transforms_list)}")
            for i, transform in enumerate(transforms_list):
                transform_str = str(transform)
                # Truncate long transform strings
                if len(transform_str) > 100:
                    transform_str = transform_str[:97] + "..."
                print(f"   {i+1}. {type(transform).__name__}: {transform_str}")
        else:
            # If it's a single transform or we can't access the list
            transform_str = str(transform_obj)
            if len(transform_str) > 100:
                transform_str = transform_str[:97] + "..."
            print(f"   Single transform: {transform_str}")
        
        # Print attributes if available
        if hasattr(transform_obj, '__dict__'):
            relevant_attrs = {k: v for k, v in transform_obj.__dict__.items() 
                            if not k.startswith('_') and not callable(v)}
            if relevant_attrs:
                print(f"   Attributes: {relevant_attrs}")
    
    print_transform_details(policy._input_transform, "Input transform pipeline")
    print_transform_details(policy._output_transform, "Output transform pipeline")
    
    # Replicate the same processing that policy.infer() does
    inputs = jax.tree.map(lambda x: x, example)  # Make a copy
    inputs = policy._input_transform(inputs)  # Apply transforms (normalize, tokenize, etc.)
    inputs_batched = jax.tree.map(lambda x: jnp.asarray(x)[np.newaxis, ...], inputs)  # Batch and convert to JAX
    
    # Convert the processed inputs to numpy for saving
    processed_inputs = jax.tree.map(lambda x: np.array(x), inputs_batched)
    
    print(f"   Processed inputs keys: {list(processed_inputs.keys())}")
    
    # Show some details about the processed data
    if 'state' in processed_inputs:
        print(f"   Processed state shape: {processed_inputs['state'].shape}")
        print(f"   Processed state mean: {np.mean(processed_inputs['state']):.6f}, std: {np.std(processed_inputs['state']):.6f}")
    
    if 'tokenized_prompt' in processed_inputs:
        print(f"   Tokenized prompt shape: {processed_inputs['tokenized_prompt'].shape}")
        print(f"   Tokenized prompt (first 10 tokens): {processed_inputs['tokenized_prompt'][0][:10]}")
    
    if 'image' in processed_inputs:
        print(f"   Processed images:")
        for img_name, img_data in processed_inputs['image'].items():
            print(f"     {img_name}: {img_data.shape} (dtype: {img_data.dtype})")
    
    # Now run the actual inference
    result = policy.infer(example, noise=noise_np)
    actions = result['actions']
    
    print(f"   Generated actions shape: {actions.shape}")
    print(f"   Actions mean: {np.mean(actions):.6f}, std: {np.std(actions):.6f}")
    
    # Prepare outputs
    outputs = {
        "actions": np.array(actions)
    }
    
    print(f"💾 Step 3: Preparing data for saving...")
    print(f"   Noise shape to save: {noise_np.shape}")
    print(f"   This is the exact noise used by JAX model's flow matching process")
    print(f"   PyTorch model should use this same noise for comparison")
    print(f"   Processed inputs contain normalized/tokenized data ready for model input")
    
    return example, outputs, noise_np, processed_inputs


def main():
    parser = argparse.ArgumentParser(description="Generate JAX model data for comparison")
    parser.add_argument("--model_name", type=str, default="pi0_aloha_sim", 
                       choices=["pi0_aloha_sim", "pi0_aloha_towel", "pi0_base"],
                       help="Model name to use")
    parser.add_argument("--config_name", type=str, default="pi0_aloha_sim",
                       help="Config name to use")
    parser.add_argument("--output_dir", type=str, default="data",
                       help="Output directory for pickle files")
    
    args = parser.parse_args()
    
    # Create output directory structure
    output_base = Path(args.output_dir) 
    save_dir = output_base / args.model_name / "save"
    save_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"🚀 Starting JAX data generation for {args.model_name}")
    print(f"📁 Output directory: {save_dir}")
    
    try:
        # Create example data
        example = create_example_data(args.model_name)
        
        # Run JAX inference
        example, outputs, noise, processed_inputs = run_jax_inference(args.config_name, args.model_name, example, save_dir)
        
        # Save pickle files
        print(f"💾 Saving pickle files...")
        
        with open(save_dir / "example.pkl", "wb") as f:
            pickle.dump(example, f)
        print(f"   ✅ Saved example.pkl")
        
        with open(save_dir / "outputs.pkl", "wb") as f:
            pickle.dump(outputs, f)
        print(f"   ✅ Saved outputs.pkl")
        
        with open(save_dir / "noise.pkl", "wb") as f:
            pickle.dump(noise, f)
        print(f"   ✅ Saved noise.pkl")
        
        with open(save_dir / "processed_inputs.pkl", "wb") as f:
            pickle.dump(processed_inputs, f)
        print(f"   ✅ Saved processed_inputs.pkl")
        
        print(f"\n🎉 Successfully generated data for {args.model_name}!")
        print(f"📂 Files saved in: {save_dir}")
        print(f"\nGenerated files:")
        print(f"  - example.pkl: Raw input data with images, state, and prompt")
        print(f"  - processed_inputs.pkl: Normalized/tokenized data ready for model input")
        print(f"  - outputs.pkl: JAX model action predictions")
        print(f"  - noise.pkl: Random noise used for sampling (passed to JAX model)")
        
        # Print summary
        print(f"\n📊 Data Summary:")
        print(f"   Raw example images: {list(example['images'].keys())}")
        for key, img in example['images'].items():
            print(f"     {key}: {img.shape} ({img.dtype})")
        print(f"   Raw state: {example['state'].shape} ({example['state'].dtype})")
        print(f"   Processed inputs keys: {list(processed_inputs.keys())}")
        if 'state' in processed_inputs:
            print(f"   Processed state: {processed_inputs['state'].shape} ({processed_inputs['state'].dtype})")
        if 'image' in processed_inputs:
            print(f"   Processed images: {list(processed_inputs['image'].keys())}")
        print(f"   Actions: {outputs['actions'].shape} ({outputs['actions'].dtype})")
        print(f"   Noise: {noise.shape} ({noise.dtype})")
        
        print(f"\n🔧 Usage:")
        print(f"   These files can now be used with compare_with_jax.py to validate")
        print(f"   JAX ↔ PyTorch model conversion for {args.model_name}")
        print(f"   The processed_inputs.pkl file contains the exact normalized/tokenized")
        print(f"   data that the model receives, useful for debugging transform differences")
        
    except Exception as e:
        print(f"❌ Error generating data: {e}")
        raise


if __name__ == "__main__":
    main() 