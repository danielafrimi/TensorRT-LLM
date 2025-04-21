from transformers import AutoConfig
from typing import Optional, Literal
from pydantic import AliasPath, BaseModel, Field, AliasChoices, model_validator
import huggingface_hub
from huggingface_hub.constants import (
    SAFETENSORS_INDEX_FILE,
    SAFETENSORS_MAX_HEADER_LENGTH,
    SAFETENSORS_SINGLE_FILE,
)
from huggingface_hub.utils import SafetensorsRepoMetadata, SafetensorsFileMetadata, TensorInfo
from huggingface_hub.utils import tqdm as hf_tqdm
from tqdm.contrib.concurrent import thread_map
import os
import json
import struct


def parse_safetensors_file_metadata(model_path, filename):

    with open(os.path.join(model_path, filename), "rb") as f:
        metadata_size = f.read(8)
        metadata_size = struct.unpack("<Q", metadata_size)[0]

        if metadata_size > SAFETENSORS_MAX_HEADER_LENGTH:
            raise RuntimeError(
                f"Failed to parse safetensors header for '{filename}' (model_path '{model_path}'): "
                f"safetensors header is too big. Maximum supported size is "
                f"{SAFETENSORS_MAX_HEADER_LENGTH} bytes (got {metadata_size}).")

        metadata_as_bytes = f.read(metadata_size)

    try:
        metadata_as_dict = json.loads(metadata_as_bytes.decode(errors="ignore"))
    except json.JSONDecodeError as e:
        raise RuntimeError(
            f"Failed to parse safetensors header for '{filename}' (model_path '{model_path}'): "
            "header format not recognized. Please make sure this is a correctly formatted safetensors file."
        ) from e

    try:
        return SafetensorsFileMetadata(
            metadata=metadata_as_dict.get("__metadata__", {}),
            tensors={
                key:
                TensorInfo(
                    dtype=tensor["dtype"],
                    shape=tensor["shape"],
                    data_offsets=tuple(tensor["data_offsets"]),  # type: ignore
                )
                for key, tensor in metadata_as_dict.items()
                if key != "__metadata__"
            },
        )
    except (KeyError, IndexError) as e:
        raise RuntimeError(
            f"Failed to parse safetensors header for '{filename}' (model_path '{model_path}'): "
            "header format not recognized. Please make sure this is a correctly formatted safetensors file."
        ) from e


def get_safetensors_metadata(model_name_or_path):
    """ Read the safetensors metadata from HF model. """
    if os.path.isdir(model_name_or_path):
        # Try direct approach first - check for safetensors files in the root directory
        if os.path.exists(
                os.path.join(model_name_or_path, SAFETENSORS_SINGLE_FILE)):
            file_metadata = parse_safetensors_file_metadata(
                model_path=model_name_or_path, filename=SAFETENSORS_SINGLE_FILE)
            return SafetensorsRepoMetadata(
                metadata=None,
                sharded=False,
                weight_map={
                    tensor_name: SAFETENSORS_SINGLE_FILE
                    for tensor_name in file_metadata.tensors.keys()
                },
                files_metadata={SAFETENSORS_SINGLE_FILE: file_metadata},
            )
        elif os.path.exists(
                os.path.join(model_name_or_path, SAFETENSORS_INDEX_FILE)):
            with open(os.path.join(model_name_or_path,
                                   SAFETENSORS_INDEX_FILE)) as f:
                index = json.load(f)

            weight_map = index.get("weight_map", {})

            # Fetch metadata per shard
            files_metadata = {}

            def _parse(filename: str) -> None:
                files_metadata[filename] = parse_safetensors_file_metadata(
                    model_path=model_name_or_path, filename=filename)

            thread_map(
                _parse,
                set(weight_map.values()),
                desc="Parse safetensors files",
                tqdm_class=hf_tqdm,
            )

            return SafetensorsRepoMetadata(
                metadata=index.get("metadata", None),
                sharded=True,
                weight_map=weight_map,
                files_metadata=files_metadata,
            )
        else:
            # Look for safetensor files in subdirectories
            for subdir in os.listdir(model_name_or_path):
                subdir_path = os.path.join(model_name_or_path, subdir)
                if os.path.isdir(subdir_path):
                    # Try to find safetensors in this subdirectory
                    try:
                        return get_safetensors_metadata(subdir_path)
                    except RuntimeError:
                        # If not found, continue to the next subdirectory
                        continue

            # If we've checked all subdirectories and still haven't found anything, search for any .safetensors files
            safetensors_files = []
            for root, _, files in os.walk(model_name_or_path):
                for file in files:
                    if file.endswith('.safetensors'):
                        safetensors_files.append(os.path.join(root, file))

            if safetensors_files:
                # Found at least one safetensors file, use the first one to estimate the metadata
                parent_dir = os.path.dirname(safetensors_files[0])
                filename = os.path.basename(safetensors_files[0])
                file_metadata = parse_safetensors_file_metadata(
                    model_path=parent_dir, filename=filename)
                return SafetensorsRepoMetadata(
                    metadata=None,
                    sharded=False,
                    weight_map={
                        tensor_name: filename
                        for tensor_name in file_metadata.tensors.keys()
                    },
                    files_metadata={filename: file_metadata},
                )

            # Not a safetensors repo
            raise RuntimeError(
                f"'{model_name_or_path}' is not a safetensors repo. Couldn't find '{SAFETENSORS_INDEX_FILE}' or '{SAFETENSORS_SINGLE_FILE}' files."
            )
    else:
        return huggingface_hub.get_safetensors_metadata(model_name_or_path)


class ModelConfig(BaseModel):
    """ Model specific configurations. The parameters are needed in engine
        setting calculation.
    """
    name: str
    model_type: str
    param_count: int
    num_hidden_layers: int = Field(validation_alias=AliasChoices(
        "num_hidden_layers",
        "n_layer",
        AliasPath("text_config", "num_hidden_layers"),
    ))
    num_attention_heads: int = Field(validation_alias=AliasChoices(
        "num_attention_heads",
        "n_head",
        AliasPath("text_config", "num_attention_heads"),
    ))
    num_key_value_heads: Optional[int] = Field(
        default=None,
        validation_alias=AliasChoices(
            "num_key_value_heads",
            "num_kv_heads",
            AliasPath("text_config", "num_key_value_heads"),
        ),
    )
    hidden_size: int = Field(validation_alias=AliasChoices(
        "hidden_size",
        "n_embd",
        AliasPath("text_config", "hidden_size"),
    ))
    head_size: Optional[int] = Field(default=None,
                                     validation_alias=AliasChoices(
                                         "head_size",
                                         "head_dim",
                                         AliasPath("text_config", "head_dim"),
                                     ))
    max_position_embeddings: Optional[int] = Field(
        default=None,
        validation_alias=AliasChoices(
            "max_position_embeddings",
            "n_positions",
            AliasPath("text_config", "max_position_embeddings"),
        ))
    dtype: Literal["float16", "bfloat16",
                   None] = Field(default="float16",
                                 validation_alias=AliasChoices(
                                     "dtype", "torch_dtype"))

    @model_validator(mode="after")
    def set_values_if_none(self):
        """ Set the values if cannot get values from HF config.json. """
        if not self.dtype:  # for GPT-J
            self.dtype = "float16"
        if self.num_key_value_heads is None:
            self.num_key_value_heads = self.num_attention_heads
        if self.head_size is None:
            self.head_size = self.hidden_size // self.num_attention_heads
        return self

    @classmethod
    def get_param_count(cls, model_hf_name, hf_model_path):
        """ Read the parameter count from HF safetensor metadata. """
        model_name_or_path = hf_model_path or model_hf_name

        # Special handling for NVILA models - need to check all components
        if "NVILA" in model_name_or_path and os.path.isdir(model_name_or_path):
            total_params = 0
            # Check all three standard NVILA components
            component_dirs = ["llm", "vision_tower", "mm_projector"]
            for component in component_dirs:
                component_path = os.path.join(model_name_or_path, component)
                if os.path.exists(component_path):
                    try:
                        metadata = get_safetensors_metadata(component_path)
                        total_params += sum(metadata.parameter_count.values())
                    except Exception as e:
                        print(
                            f"Warning: Could not get parameters for {component}: {e}"
                        )

            # If we found parameters, return them
            if total_params > 0:
                return total_params

            # Fallback parameter estimates if we couldn't get actual counts
            if "8B" in model_name_or_path:
                return 8 * 10**9  # 8B parameters
            elif "3B" in model_name_or_path:
                return 3 * 10**9  # 3B parameters
            else:
                print(
                    f"Warning: Using approximate parameter count for {model_name_or_path}"
                )
                return 7 * 10**9  # Default approximation

        # Regular handling for non-NVILA models
        elif model_hf_name == "EleutherAI/gpt-j-6b":  # GPT-J repo doesn't use safetensor format.
            return 6053381344
        else:
            try:
                metadata = get_safetensors_metadata(model_name_or_path)
                return sum(metadata.parameter_count.values())
            except Exception as e:
                raise RuntimeError(
                    f"Failed to get parameter count for {model_name_or_path}: {e}"
                )

    @classmethod
    def from_hf(cls, model_hf_name, hf_model_path):
        model_name_or_path = hf_model_path or model_hf_name
        # Special handling for NVILA models
        if "NVILA" in model_name_or_path and os.path.isdir(model_name_or_path):
            # Try to load from the LLM component first
            llm_path = os.path.join(model_name_or_path, "llm")
            if os.path.exists(llm_path):
                try:
                    hf_config = AutoConfig.from_pretrained(
                        llm_path, trust_remote_code=True).to_dict()
                    # Add multimodal flag
                    hf_config["is_multimodal"] = True
                    param_count = cls.get_param_count(model_hf_name,
                                                      hf_model_path)
                    return cls(name=model_hf_name,
                               param_count=param_count,
                               **hf_config)
                except Exception as e:
                    print(
                        f"Warning: Failed to load config from llm directory: {e}"
                    )
        # Regular model handling
        try:
            hf_config = AutoConfig.from_pretrained(
                model_name_or_path, trust_remote_code=True).to_dict()
        except Exception as e:
            raise RuntimeError(
                f"Failed to load config for {model_hf_name}: {e}")

        param_count = cls.get_param_count(model_hf_name, hf_model_path)
        return cls(name=model_hf_name, param_count=param_count, **hf_config)
