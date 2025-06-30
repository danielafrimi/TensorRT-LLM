#!/usr/bin/env python3
"""
Script to modify quantized parameters directly in the full model and run inference.

This approach is more efficient than extracting layers because:
1. No need to recreate the model architecture
2. Can test with real inference pipeline
3. Easier to compare with original model
4. Can modify any parameter in the model
"""

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
from safetensors import safe_open
from safetensors.torch import save_file


class QuantizedModelModifier:

    def __init__(self, model_path: str):
        self.model_path = Path(model_path)
        self.config = None
        self.weights = {}
        self.model = None

    def load_config(self) -> Dict[str, Any]:
        """Load model configuration."""
        config_path = self.model_path / "config.json"
        if not config_path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")

        with open(config_path, 'r') as f:
            self.config = json.load(f)

        print(f"Loaded config: {self.config}")
        return self.config

    def load_weights(self) -> Dict[str, torch.Tensor]:
        """Load model weights from safetensors."""
        # Find safetensors files
        safetensors_files = list(self.model_path.glob("*.safetensors"))
        if not safetensors_files:
            raise FileNotFoundError(
                f"No safetensors files found in {self.model_path}")

        print(f"Found safetensors files: {[f.name for f in safetensors_files]}")

        # Load all weights
        for file_path in safetensors_files:
            with safe_open(file_path, framework="pt", device="cpu") as f:
                for key in f.keys():
                    self.weights[key] = f.get_tensor(key)

        print(f"Loaded {len(self.weights)} weight tensors")
        return self.weights

    def list_available_parameters(
            self,
            layer_idx: Optional[int] = None,
            parameter_type: Optional[str] = None) -> List[str]:
        """List available parameters that can be modified."""
        available_params = []

        for key in self.weights.keys():
            # Filter by layer if specified
            if layer_idx is not None:
                if f"model.layers.{layer_idx}." not in key:
                    continue

            # Filter by parameter type if specified
            if parameter_type is not None:
                if parameter_type not in key:
                    continue

            available_params.append(key)

        return sorted(available_params)

    def modify_parameter(self,
                         param_key: str,
                         modification_type: str,
                         value: Optional[float] = None,
                         pattern: Optional[str] = None,
                         scale: Optional[float] = None):

        original_tensor = self.weights[param_key]
        original_dtype = original_tensor.dtype
        original_device = original_tensor.device

        print(f"\nModifying parameter: {param_key}")
        print(f"Original shape: {original_tensor.shape}")
        print(f"Original dtype: {original_dtype}")

        # Convert to float for modification
        if original_dtype == torch.uint8:
            tensor_float = original_tensor.float()
            print("Converting uint8 tensor to float for modification")
        else:
            tensor_float = original_tensor.clone()

        if modification_type == "pattern":
            if pattern == "identity":
                if len(tensor_float.shape) == 2:
                    # For int4 packed tensors, we need to create identity in the packed format
                    if original_dtype == torch.uint8:
                        # Create identity matrix for int4 packed format
                        # Each uint8 contains two int4 values: [low_nibble, high_nibble]
                        # The unpacking maps packed[i,j] to unpacked[i, 2*j] and unpacked[i, 2*j+1]
                        # To get identity matrix, we need 1s at unpacked[i,i] positions
                        # This means:
                        # - For even i: set packed[i, i//2] to 1 (unpacks to [1, 0])
                        # - For odd i: set packed[i, i//2] to 16 (unpacks to [0, 1])
                        modified_tensor = torch.zeros_like(tensor_float)
                        modified_tensor = modified_tensor.T

                        # For identity matrix, we want 1s at diagonal positions [i, i]
                        # In packed format, this means:
                        # - For even i: packed[i, i//2] should unpack to [1, 0] at [i, i] and [i, i+1]
                        # - For odd i: packed[i, i//2] should unpack to [0, 1] at [i, i-1] and [i, i]

                        # But we need to be careful about the transposition that happens during loading
                        # Let's create a pattern that works with the actual loading process

                        min_dim = min(tensor_float.shape[0],
                                      tensor_float.shape[1])
                        for i in range(min_dim):
                            if i % 2 == 0:  # Even position
                                # Set packed[i, i//2] = 1 (unpacks to [1, 0])
                                modified_tensor[i, i // 2] = 1.0
                            else:  # Odd position
                                # Set packed[i, i//2] = 16 (unpacks to [0, 1])
                                modified_tensor[i, i // 2] = 16.0
                    else:
                        # Regular identity matrix for non-packed tensors
                        modified_tensor = torch.zeros_like(tensor_float)
                        min_dim = min(tensor_float.shape[0],
                                      tensor_float.shape[1])
                        for i in range(min_dim):
                            modified_tensor[i, i] = 1.0
                        print(f"Set diagonal elements to 1 (min_dim={min_dim})")
                    print(
                        f"Original shape: {original_tensor.shape}, Modified shape: {modified_tensor.shape}"
                    )
                    modified_tensor = modified_tensor.T
                else:
                    print("Identity pattern only works for 2D tensors")
                    return
            elif pattern == "ones":
                modified_tensor = torch.ones_like(tensor_float)
            elif pattern == "zeros":
                modified_tensor = torch.zeros_like(tensor_float)
            else:
                print(f"Unknown pattern: {pattern}")
                return
        else:
            print(f"Unknown modification type: {modification_type}")
            return

        # Convert back to original dtype
        if original_dtype == torch.uint8:
            # For uint8, we need to ensure values are in valid range [0, 255]

            modified_tensor = torch.clamp(modified_tensor, 0, 255)
            modified_tensor = modified_tensor.to(torch.uint8)
            # TODO this is a check
            unpacker = torch.ops.trtllm.unpack_int4_packed_tensor_to_int8
            unpack_modified_tensor = unpacker(
                modified_tensor.to(torch.int8).cpu()).contiguous().cuda()
            print("Converted back to uint8")
        else:
            modified_tensor = modified_tensor.to(original_dtype)

        # Ensure device is correct
        modified_tensor = modified_tensor.to(original_device)

        # Update the weights
        self.weights[param_key] = modified_tensor

        print(f"Parameter {param_key} modified successfully")
        print(f"Final dtype: {modified_tensor.dtype}")
        print(f"Final shape: {modified_tensor.shape}")

    def save_modified_model(self, output_dir: str):
        """Save the modified model."""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # Copy all files from original model directory
        original_path = Path(self.model_path)
        for file_path in original_path.iterdir():
            if file_path.is_file():
                # Skip the original safetensors file since we'll create a new one
                if file_path.name.endswith('.safetensors'):
                    continue
                # Copy other files
                import shutil
                shutil.copy2(file_path, output_path / file_path.name)
                print(f"Copied: {file_path.name}")

        # Save weights using safetensors
        weights_path = output_path / "model.safetensors"

        # Create metadata for safetensors
        metadata = {}
        for key in self.weights.keys():
            metadata[key] = str(self.weights[key].dtype)

        # Save using safetensors
        save_file(self.weights, weights_path, metadata=metadata)

        print(f"Modified model saved to: {output_path}")
        print(f"  - All original files copied")
        print(f"  - Weights: {weights_path}")

        return output_path


def main():
    parser = argparse.ArgumentParser(
        description="Modify quantized parameters in full model")
    parser.add_argument("--model_path",
                        type=str,
                        required=True,
                        help="Path to the quantized model directory")
    parser.add_argument("--list_params",
                        action="store_true",
                        help="List available parameters")
    parser.add_argument("--layer_idx",
                        type=int,
                        default=None,
                        help="Filter parameters by layer index")
    parser.add_argument(
        "--param_type",
        type=str,
        default=None,
        help="Filter parameters by type (e.g., 'q_proj', 'weight', 'scale')")
    parser.add_argument("--param_key",
                        type=str,
                        default=None,
                        help="Specific parameter key to modify")

    parser.add_argument("--modify",
                        type=str,
                        choices=['set_value', 'scale', 'pattern'],
                        help="Type of modification to apply")
    parser.add_argument("--value",
                        type=float,
                        default=None,
                        help="Value for set_value or scale modifications")
    parser.add_argument("--pattern",
                        type=str,
                        choices=['ones', 'zeros', 'random', 'identity'],
                        help="Pattern for pattern modification")
    parser.add_argument("--output_dir",
                        type=str,
                        default="./modified_model",
                        help="Output directory for modified model")
    parser.add_argument("--compare",
                        action="store_true",
                        help="Compare modified parameter with original")
    parser.add_argument(
        "--hardcoded_q_proj_layer0",
        action="store_true",
        help=
        "Apply hardcoded modifications to all q_proj scale tensors in layer 0 (set to ones) and weight (set to diagonal)"
    )

    args = parser.parse_args()

    # Initialize modifier
    modifier = QuantizedModelModifier(args.model_path)

    def apply_hardcoded_mlp_modifications_layer_0(modifier, output_dir):
        mlp_params = [
            "model.layers.0.mlp.down_proj.input_scale",
            "model.layers.0.mlp.down_proj.pre_quant_scale",
            "model.layers.0.mlp.down_proj.weight_scale",
            "model.layers.0.mlp.down_proj.weight_scale_2",
        ]

        weight_params = ["model.layers.0.mlp.down_proj.weight"]
        for param in mlp_params:
            if param in modifier.weights:
                modifier.modify_parameter(param, "pattern", pattern="ones")

        for param in weight_params:
            if param in modifier.weights:
                print(
                    f"[Hardcoded] Weight tensor shape before modification: {modifier.weights[param].shape}"
                )
                print(
                    f"[Hardcoded] Weight tensor dtype before modification: {modifier.weights[param].dtype}"
                )
                modifier.modify_parameter(param, "pattern", pattern="identity")

            modifier.save_modified_model(output_dir)

    def apply_hardcoded_q_proj_modifications(modifier, output_dir):
        # List of q_proj scale parameters in layer 0
        q_proj_params = [
            "model.layers.0.self_attn.q_proj.pre_quant_scale",
            "model.layers.0.self_attn.q_proj.weight_scale",
            "model.layers.0.self_attn.q_proj.weight_scale_2",
            "model.layers.0.self_attn.q_proj.input_scale",
        ]

        # List of k_proj scale parameters in layer 0
        k_proj_params = [
            "model.layers.0.self_attn.k_proj.pre_quant_scale",
            "model.layers.0.self_attn.k_proj.weight_scale",
            "model.layers.0.self_attn.k_proj.weight_scale_2",
            "model.layers.0.self_attn.k_proj.input_scale",
        ]

        # List of v_proj scale parameters in layer 0
        v_proj_params = [
            "model.layers.0.self_attn.v_proj.pre_quant_scale",
            "model.layers.0.self_attn.v_proj.weight_scale",
            "model.layers.0.self_attn.v_proj.weight_scale_2",
            "model.layers.0.self_attn.v_proj.input_scale",
        ]

        weight_param = [
            "model.layers.0.self_attn.q_proj.weight",
            "model.layers.0.self_attn.k_proj.weight",
            "model.layers.0.self_attn.v_proj.weight"
        ]

        for param in q_proj_params:
            if param in modifier.weights:
                modifier.modify_parameter(param, "pattern", pattern="ones")
            else:
                print(f"[Hardcoded] Warning: {param} not found!")

        for param in k_proj_params:
            if param in modifier.weights:
                modifier.modify_parameter(param, "pattern", pattern="ones")
            else:
                print(f"[Hardcoded] Warning: {param} not found!")

        for param in v_proj_params:
            if param in modifier.weights:
                modifier.modify_parameter(param, "pattern", pattern="ones")
            else:
                print(f"[Hardcoded] Warning: {param} not found!")

        for param in weight_param:
            if param in modifier.weights:
                print(
                    f"[Hardcoded] Weight tensor shape before modification: {modifier.weights[param].shape}"
                )
                print(
                    f"[Hardcoded] Weight tensor dtype before modification: {modifier.weights[param].dtype}"
                )
                modifier.modify_parameter(param, "pattern", pattern="identity")

            modifier.save_modified_model(output_dir)

        # Load model

    print("Loading model configuration...")
    modifier.load_config()

    print("Loading model weights...")
    modifier.load_weights()

    # apply_hardcoded_q_proj_modifications(modifier, args.output_dir)
    apply_hardcoded_mlp_modifications_layer_0(modifier, args.output_dir)


if __name__ == "__main__":
    main()
