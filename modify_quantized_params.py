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
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
from safetensors import safe_open


class QuantizedModelModifier:
    """Class to modify quantized parameters in a full model and run inference."""

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

    def print_parameter_info(self, param_key: str):
        """Print detailed information about a parameter."""
        if param_key not in self.weights:
            print(f"Parameter {param_key} not found!")
            return

        tensor = self.weights[param_key]
        print(f"\nParameter: {param_key}")
        print(f"  Shape: {tensor.shape}")
        print(f"  Dtype: {tensor.dtype}")
        print(f"  Device: {tensor.device}")

        # Convert to float for statistics if needed
        if tensor.dtype == torch.uint8:
            tensor_float = tensor.float()
            print(f"  (Converted to float for statistics)")
        else:
            tensor_float = tensor

        print(f"  Mean: {tensor_float.mean():.6f}")
        print(f"  Std: {tensor_float.std():.6f}")
        print(f"  Min: {tensor_float.min():.6f}")
        print(f"  Max: {tensor_float.max():.6f}")

        # Print first few elements
        flat_tensor = tensor.flatten()
        if len(flat_tensor) <= 10:
            print(f"  Values: {flat_tensor.tolist()}")
        else:
            print(f"  First 10 values: {flat_tensor[:10].tolist()}")

    def modify_parameter(self,
                         param_key: str,
                         modification_type: str,
                         value: Optional[float] = None,
                         pattern: Optional[str] = None) -> torch.Tensor:
        """
        Modify a parameter in the model.

        Args:
            param_key: Key of the parameter to modify
            modification_type: Type of modification ('set_value', 'scale', 'pattern')
            value: Value to set or scale factor
            pattern: Pattern for initialization ('ones', 'zeros', 'random', 'identity')

        Returns:
            Modified tensor
        """
        if param_key not in self.weights:
            raise ValueError(f"Parameter {param_key} not found!")

        original_tensor = self.weights[param_key]
        is_uint8 = original_tensor.dtype == torch.uint8

        # Convert to float for modification if needed
        if is_uint8:
            modified_tensor = original_tensor.float()
            print(f"Converting uint8 tensor to float for modification")
        else:
            modified_tensor = original_tensor.clone()

        print(f"\nModifying parameter: {param_key}")
        print(f"Original tensor shape: {original_tensor.shape}")
        print(f"Modification type: {modification_type}")

        if modification_type == 'set_value':
            if value is None:
                raise ValueError(
                    "Value must be provided for 'set_value' modification")
            modified_tensor.fill_(value)
            print(f"Set all values to: {value}")

        elif modification_type == 'scale':
            if value is None:
                raise ValueError(
                    "Scale factor must be provided for 'scale' modification")
            modified_tensor = modified_tensor * value
            print(f"Scaled by factor: {value}")

        elif modification_type == 'pattern':
            if pattern is None:
                raise ValueError(
                    "Pattern must be provided for 'pattern' modification")

            if pattern == 'ones':
                modified_tensor.fill_(1.0)
                print("Set to ones")
            elif pattern == 'zeros':
                modified_tensor.fill_(0.0)
                print("Set to zeros")
            elif pattern == 'random':
                torch.randn_like(modified_tensor, out=modified_tensor)
                print("Set to random values")
            elif pattern == 'identity':
                if len(modified_tensor.shape) == 2:
                    # For square matrices, set to identity
                    if modified_tensor.shape[0] == modified_tensor.shape[1]:
                        modified_tensor.fill_(0.0)
                        torch.diagonal(modified_tensor).fill_(1.0)
                        print("Set to identity matrix")
                    else:
                        # For non-square, set diagonal elements to 1
                        min_dim = min(modified_tensor.shape)
                        for i in range(min_dim):
                            modified_tensor[i, i] = 1.0
                        print(f"Set diagonal elements to 1 (min_dim={min_dim})")
                else:
                    print("Identity pattern only works for 2D tensors")
            else:
                raise ValueError(f"Unknown pattern: {pattern}")

        else:
            raise ValueError(f"Unknown modification type: {modification_type}")

        # Convert back to uint8 if original was uint8
        if is_uint8:
            # Clamp to valid uint8 range and convert
            modified_tensor = torch.clamp(modified_tensor, 0, 255)
            modified_tensor = modified_tensor.to(torch.uint8)
            print(f"Converted back to uint8")

        # Update the weights dictionary
        self.weights[param_key] = modified_tensor

        # Print statistics
        print(f"Modified tensor stats:")
        if modified_tensor.dtype == torch.uint8:
            tensor_float = modified_tensor.float()
        else:
            tensor_float = modified_tensor
        print(f"  Mean: {tensor_float.mean():.6f}")
        print(f"  Std: {tensor_float.std():.6f}")
        print(f"  Min: {tensor_float.min():.6f}")
        print(f"  Max: {tensor_float.max():.6f}")

        return modified_tensor

    def save_modified_model(self, output_dir: str):
        """Save the modified model."""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # Save config
        config_path = output_path / "config.json"
        with open(config_path, 'w') as f:
            json.dump(self.config, f, indent=2)

        # Save weights
        weights_path = output_path / "model.safetensors"
        with safe_open(weights_path, framework="pt", device="cpu") as f:
            for key, tensor in self.weights.items():
                f.set_tensor(key, tensor)

        print(f"Modified model saved to: {output_path}")
        print(f"  - Config: {config_path}")
        print(f"  - Weights: {weights_path}")

        return output_path
        """Create a script to run inference with the modified model."""
        inference_script = f"""#!/usr/bin/env python3
\"\"\"
Inference script for modified quantized model.

Modified parameters:
{chr(10).join(f"- {{param}}" for param in modified_params)}
\"\"\"

import torch
import json
from pathlib import Path

def run_inference(model_path: str, input_text: str = "Hello, world!"):
    \"\"\"Run inference with the modified model.\"\"\"
    print(f"Loading modified model from: {{model_path}}")

    # Load config
    with open(f"{{model_path}}/config.json", 'r') as f:
        config = json.load(f)

    print(f"Model loaded successfully!")
    print(f"Modified parameters:")
{chr(10).join(f"    - {{param}}" for param in modified_params)}

    # Run inference
    print(f"\\nRunning inference with input: '{{input_text}}'")

    # TODO: Implement actual inference
    # This would involve tokenization and model forward pass
    # For now, just print the model structure

    print(f"Model architecture:")
    print(f"  - Hidden size: {{config.get('hidden_size', 'N/A')}}")
    print(f"  - Num layers: {{config.get('num_hidden_layers', 'N/A')}}")
    print(f"  - Num attention heads: {{config.get('num_attention_heads', 'N/A')}}")
    print(f"  - Quantization: {{config.get('quantization', 'N/A')}}")

    return config

if __name__ == "__main__":
    model_path = "{output_dir}"
    config = run_inference(model_path)
    print("\\n✅ Inference script completed!")
"""

        script_path = Path(output_dir) / "run_inference.py"
        with open(script_path, 'w') as f:
            f.write(inference_script)

        # Make executable
        os.chmod(script_path, 0o755)
        print(f"Created inference script: {script_path}")

    def compare_with_original(self, original_model_path: str, param_key: str):
        """Compare modified parameter with original."""
        original_weights = {}

        # Load original weights
        safetensors_files = list(
            Path(original_model_path).glob("*.safetensors"))
        for file_path in safetensors_files:
            with safe_open(file_path, framework="pt", device="cpu") as f:
                for key in f.keys():
                    if key == param_key:
                        original_weights[key] = f.get_tensor(key)
                        break

        if param_key not in original_weights:
            print(f"Parameter {param_key} not found in original model!")
            return

        original_tensor = original_weights[param_key]
        modified_tensor = self.weights[param_key]

        print(f"\nComparison for parameter: {param_key}")
        print(f"Original tensor:")
        print(f"  Shape: {original_tensor.shape}")
        print(f"  Mean: {original_tensor.mean():.6f}")
        print(f"  Std: {original_tensor.std():.6f}")
        print(f"  Min: {original_tensor.min():.6f}")
        print(f"  Max: {original_tensor.max():.6f}")

        print(f"Modified tensor:")
        print(f"  Shape: {modified_tensor.shape}")
        print(f"  Mean: {modified_tensor.mean():.6f}")
        print(f"  Std: {modified_tensor.std():.6f}")
        print(f"  Min: {modified_tensor.min():.6f}")
        print(f"  Max: {modified_tensor.max():.6f}")

        # Calculate difference
        diff = torch.abs(original_tensor - modified_tensor)
        print(f"Difference:")
        print(f"  Mean absolute difference: {diff.mean():.6f}")
        print(f"  Max absolute difference: {diff.max():.6f}")
        print(f"  Total elements changed: {(diff > 0).sum().item()}")


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

    args = parser.parse_args()

    # Initialize modifier
    modifier = QuantizedModelModifier(args.model_path)

    try:
        # Load model
        print("Loading model configuration...")
        modifier.load_config()

        print("Loading model weights...")
        modifier.load_weights()

        # List parameters if requested
        if args.list_params:
            print(f"\nAvailable parameters:")
            if args.layer_idx is not None:
                print(f"Filtered by layer {args.layer_idx}:")
            if args.param_type is not None:
                print(f"Filtered by type '{args.param_type}':")

            available_params = modifier.list_available_parameters(
                layer_idx=args.layer_idx, parameter_type=args.param_type)

            for param in available_params:
                print(f"  {param}")

            modified_tensor = modifier.modify_parameter(
                param_key=args.param_key,
                modification_type=args.modify,
                value=args.value,
                pattern=args.pattern)

            # Compare with original if requested
            if args.compare:
                modifier.compare_with_original(args.model_path, args.param_key)

            # Save modified model
            print("\nSaving modified model...")
            modifier.save_modified_model(args.output_dir)

        elif not args.list_params and not args.info:
            print("No action specified. Use --list_params, --info, or --modify")
            print("Use --help for more information")

    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
