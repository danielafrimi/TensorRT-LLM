import copy
import enum
import math
from abc import ABC, abstractmethod
from collections import OrderedDict, defaultdict
from typing import Dict, List, Optional, Set, Tuple, Union

import torch

import tensorrt_llm
import tensorrt_llm.bindings
from tensorrt_llm.bindings.BuildInfo import ENABLE_MULTI_DEVICE
from tensorrt_llm.sampling_params import SamplingParams

from ..._utils import binding_dtype_size, nvtx_range
from ...logger import logger
from ...mapping import Mapping
from .llm_request import LlmRequest, LlmRequestState, SamplingConfig
from .scheduler import ScheduledRequests

if ENABLE_MULTI_DEVICE:
    from mpi4py import MPI

    from tensorrt_llm._utils import mpi_comm

BufferManagerCpp = tensorrt_llm.bindings.internal.runtime.BufferManager
KVCacheManagerCpp = tensorrt_llm.bindings.internal.batch_manager.KVCacheManager
KvCacheConfigCpp = tensorrt_llm.bindings.executor.KvCacheConfig
CacheTypeCpp = tensorrt_llm.bindings.internal.batch_manager.CacheType
ModelConfig = tensorrt_llm.bindings.ModelConfig
DataType = tensorrt_llm.bindings.DataType
KVCacheEventManagerCpp = tensorrt_llm.bindings.internal.batch_manager.KVCacheEventManager
RequestList = list[LlmRequest]
PeftCacheManagerCpp = tensorrt_llm.bindings.internal.batch_manager.PeftCacheManager
PeftCacheConfig = tensorrt_llm.bindings.executor.PeftCacheConfig
WorldConfig = tensorrt_llm.bindings.WorldConfig
TempAttentionWindowInputs = tensorrt_llm.bindings.internal.batch_manager.TempAttentionWindowInputs
BlocksPerWindow = Dict[int, Tuple[
    int,
    int]]  # window_size -> (blocks_in_primary_pool, blocks_in_secondary_pool)


class ResourceManagerType(enum.Enum):
    KV_CACHE_MANAGER = "KV_CACHE_MANAGER"
    DRAFT_KV_CACHE_MANAGER = "DRAFT_KV_CACHE_MANAGER"
    PEFT_CACHE_MANAGER = "PEFT_CACHE_MANAGER"
    SEQ_SLOT_MANAGER = "SEQ_SLOT_MANAGER"
    SPEC_RESOURCE_MANAGER = "SPEC_RESOURCE_MANAGER"


def compute_page_count(token_count: int, tokens_per_page: int) -> int:
    return (token_count + tokens_per_page) // tokens_per_page


class BaseResourceManager(ABC):

    @abstractmethod
    def get_max_resource_count(self) -> int:
        raise NotImplementedError

    @abstractmethod
    def get_needed_resource_to_completion(self, request: LlmRequest) -> int:
        raise NotImplementedError

    def add_dummy_requests(self, request_ids: List[int]):
        pass

    def prepare_resources(self, scheduled_batch: ScheduledRequests):
        pass

    def update_resources(self, scheduled_batch: ScheduledRequests):
        pass

    def free_resources(self, request: LlmRequest):
        pass

    def shutdown(self):
        pass


def get_pp_layers(
    num_layers: int,
    mapping: Mapping,
    spec_config: Optional["DecodingBaseConfig"] = None,
    layer_mask: Optional[List[bool]] = None,
) -> Tuple[List[int], int]:
    from ..speculative.utils import get_num_spec_layers

    total_num_layers = num_layers
    if layer_mask is not None:
        assert sum(layer_mask) == num_layers, (
            f"The number of enabled layers in layer_mask ({sum(layer_mask)}) "
            f"must match the number of layers ({num_layers}) "
            f"in KV cache manager, but get layer_mask: {layer_mask}")
        total_num_layers = len(layer_mask)
    pp_layers = mapping.pp_layers(total_num_layers)
    if layer_mask is not None:
        pp_layers = [i for i in pp_layers if layer_mask[i]]
    if spec_config is not None:
        num_spec_layers = get_num_spec_layers(spec_config)
        total_num_layers += num_spec_layers
        if mapping.is_last_pp_rank():
            pp_layers.extend(
                range(total_num_layers - num_spec_layers, total_num_layers))
    if len(pp_layers) == 0:
        # Don't support empty KV cache for now, provide at least 1 layer
        pp_layers.append(0)
    return pp_layers, total_num_layers


class KVCacheManager(BaseResourceManager):

    def __init__(
        self,
        kv_cache_config: KvCacheConfigCpp,
        kv_cache_type: CacheTypeCpp,
        *,
        num_layers: int,
        num_kv_heads: Union[int, List[Optional[int]]],
        head_dim: int,
        tokens_per_block: int,
        # Note that max_seq_len is not necessarily equal to kv_cache_config.num_tokens.
        # It's derived from the model's BuildConfig for consistency with the C++ backend.
        max_seq_len: int,
        max_batch_size: int,
        mapping: Mapping,
        dtype: DataType = DataType.HALF,
        spec_config: Optional["DecodingBaseConfig"] = None,
        layer_mask: Optional[List[bool]] = None,
        max_num_tokens: int = 8192,
        model_config: Optional[ModelConfig] = None,
        max_beam_width: int = 1,
    ) -> None:
        self.mapping = mapping
        self.dtype = dtype
        self.kv_cache_type = kv_cache_type
        self.pp_layers, self.num_layers = get_pp_layers(
            num_layers,
            mapping,
            spec_config=spec_config,
            layer_mask=layer_mask,
        )
        self.num_local_layers = len(self.pp_layers)
        self.layer_offsets = {
            idx: offset
            for offset, idx in enumerate(self.pp_layers)
        }

        tp_size = mapping.tp_size
        if mapping.enable_attention_dp:
            tp_size = 1

        if isinstance(num_kv_heads, int):
            self.num_kv_heads_per_layer = [
                (num_kv_heads + tp_size - 1) // tp_size
                for _ in range(self.num_local_layers)
            ]
            self.total_num_kv_heads_per_layer = [
                (num_kv_heads + tp_size - 1) // tp_size
                for _ in range(self.num_layers)
            ]
        else:
            assert len(num_kv_heads) == self.num_layers

            def append_to_kv_heads_per_layer(num_kv_heads_per_layer: List[int],
                                             kv_head: Optional[int]):
                if kv_head is not None:
                    num_kv_heads_per_layer.append(
                        (kv_head + tp_size - 1) // tp_size)
                else:
                    num_kv_heads_per_layer.append(0)

            self.num_kv_heads_per_layer = []
            if self.num_local_layers > 0:
                for i in self.pp_layers:
                    kv_head = num_kv_heads[i]
                    append_to_kv_heads_per_layer(self.num_kv_heads_per_layer,
                                                 kv_head)

            self.total_num_kv_heads_per_layer = []
            for i in range(self.num_layers):
                kv_head = num_kv_heads[i]
                append_to_kv_heads_per_layer(self.total_num_kv_heads_per_layer,
                                             kv_head)

        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.tokens_per_block = tokens_per_block
        self.max_seq_len = max_seq_len
        self.max_batch_size = max_batch_size
        self.kv_factor = 1 if kv_cache_type == CacheTypeCpp.SELFKONLY else 2
        # Some speculative decoding methods need to use different kv lengths for the
        # draft/target layers. Add extra tokens to handle this issue.
        # Import here to avoid circular imports
        from ..speculative import get_num_extra_kv_tokens
        self.num_extra_kv_tokens = get_num_extra_kv_tokens(spec_config)
        self.event_buffer_max_size = kv_cache_config.event_buffer_max_size
        self.max_num_tokens = max_num_tokens

        # Determine max_attention_window_vec
        if kv_cache_config.max_attention_window is None:
            # Use max_seq_len as default max_attention_window
            self.max_attention_window_vec = [max_seq_len]
        else:
            self.max_attention_window_vec = kv_cache_config.max_attention_window.copy(
            )  # Make a copy to avoid modifying original

        sink_token_length = (kv_cache_config.sink_token_length
                             if kv_cache_config.sink_token_length is not None
                             else 0)

        # Determine if this is VSWA (Variable Sliding Window Attention)
        self.is_vswa = len(self.max_attention_window_vec) > 1

        # Calculate blocks per window using appropriate method
        if self.is_vswa:
            # VSWA case: use C++ implementation for variable window sizes
            # model config check
            if model_config is None:
                raise ValueError(
                    "model_config is required for VSWA (Variable Sliding Window Attention)"
                )
            # kv cache config check
            assert isinstance(
                kv_cache_config, KvCacheConfigCpp
            ), "calculate_max_num_blocks_from_cpp only accepts KvCacheConfigCpp"
            blocks_per_window = self.calculate_max_num_blocks_from_cpp(
                kv_cache_config=kv_cache_config,
                model_config=model_config,
                extra_cost_memory=0,
            )
        else:
            # Standard case: use original Python implementation
            self.blocks_in_primary_pool, self.blocks_in_secondary_pool = self.calculate_max_num_blocks(
                kv_cache_config=kv_cache_config,
                head_dim=head_dim,
                tokens_per_block=tokens_per_block,
                mapping=mapping,
                dtype=dtype,
                kv_factor=self.kv_factor,
            )
            blocks_per_window = {
                self.max_attention_window_vec[0]:
                (self.blocks_in_primary_pool, self.blocks_in_secondary_pool)
            }

        # Validate and adjust attention windows against their upper bounds if needed
        blocks_per_window, self.max_seq_len, self.max_attention_window_vec = self._validate_and_adjust_attention_windows(
            max_attention_window_vec=self.max_attention_window_vec,
            blocks_per_window=blocks_per_window,
            tokens_per_block=tokens_per_block,
            sink_token_length=sink_token_length,
            max_seq_len=self.max_seq_len,
            max_beam_width=max_beam_width,
        )

        if kv_cache_type != CacheTypeCpp.SELF:
            assert len(
                blocks_per_window
            ) == 1, "Only one window size is supported for non-self KV cache"
            # rewrite the attention window size in blocks_per_window
            memory_pools = blocks_per_window[self.max_attention_window_vec[0]]
            blocks_per_window = {self.max_seq_len: memory_pools}
            logger.info(
                f"Adjusted attention window size to {self.max_seq_len} in blocks_per_window"
            )

        # Set up temp_attention_window_inputs
        temp_attention_window_inputs = self._set_temp_attention_window_inputs()

        # Note that this stream is unused for now. Will be used for copying to host
        # when that feature is enabled.
        self._stream = torch.cuda.Stream()
        kwargs = {
            'num_kv_heads_per_layer': self.num_kv_heads_per_layer,
            'size_per_head': head_dim,
            'tokens_per_block': tokens_per_block,
            'blocks_per_window': blocks_per_window,
            'max_num_sequences': max_batch_size,
            'max_beam_width': max_beam_width,
            'max_attention_window_vec': self.max_attention_window_vec,
            'temp_attention_window_inputs': temp_attention_window_inputs,
            'dtype': dtype,
            'sink_token_length': sink_token_length,
            'stream': self._stream.cuda_stream,
            'max_sequence_length': max_seq_len,
            'enable_block_reuse': kv_cache_config.enable_block_reuse,
            'onboard_blocks': kv_cache_config.onboard_blocks,
            'cache_type': kv_cache_type,
            'enable_partial_reuse': kv_cache_config.enable_partial_reuse,
            'copy_on_partial_reuse': kv_cache_config.copy_on_partial_reuse,
        }
        if self.event_buffer_max_size > 0:
            kwargs['event_manager'] = KVCacheEventManagerCpp(
                max_kv_event_entries=self.event_buffer_max_size)

        self.impl = KVCacheManagerCpp(**kwargs)

        self.impl.allocate_pools(False)
        self.kv_cache_pool_pointers = self.impl.get_block_pool_pointers()
        self.kv_cache_pool_mapping = self.impl.get_layer_to_pool_mapping()
        self.num_pools = self.impl.num_pools
        self.max_blocks_per_seq = self.impl.max_blocks_per_seq
        self.enable_block_reuse = kv_cache_config.enable_block_reuse

    def shutdown(self):
        self.impl.release_pools()

    @classmethod
    def from_model_config(cls,
                          model_config: ModelConfig,
                          kv_cache_config: KvCacheConfigCpp,
                          mapping: Mapping,
                          kv_cache_type: CacheTypeCpp = CacheTypeCpp.SELF,
                          dtype: DataType = DataType.HALF) -> "KVCacheManager":
        return cls(
            kv_cache_config,
            kv_cache_type,
            num_layers=model_config.num_attention_layers(mapping.pp_size),
            # NOTE: this preserves existing behavior in KV cache manager.
            # But we should change this to pass a list at some point.
            # We're assuming the KV cache is homogeneous here.
            num_kv_heads=model_config.num_kv_heads(0),
            head_dim=model_config.size_per_head,
            tokens_per_block=model_config.tokens_per_block,
            max_seq_len=model_config.max_seq_len,
            max_batch_size=model_config.max_batch_size,
            mapping=mapping,
            dtype=dtype)

    def get_max_resource_count(self) -> int:
        return self.impl.max_num_blocks

    def get_needed_resource_to_completion(self, request: LlmRequest) -> int:
        # TODO: the C++ implementation of this method can be used, but the
        # Python and C++ schedulers currently do not agree on what "needed
        # resource to completion" means. The C++ one excludes already allocated
        # blocks; the Python one includes them. This should be unified, but
        # the Python scheduler needs to be fixed.
        #
        # return self.impl.get_remaining_blocks_to_completion(request)
        context_token_count = request.orig_prompt_len
        num_context_blocks = context_token_count // self.tokens_per_block
        remaining_tokens = context_token_count + request.max_new_tokens - num_context_blocks * self.tokens_per_block
        need_blocks = num_context_blocks + math.ceil(
            remaining_tokens / self.tokens_per_block)
        return need_blocks

    def prepare_resources(self, scheduled_batch: ScheduledRequests):
        context_batch = scheduled_batch.context_requests
        generation_batch = scheduled_batch.generation_requests
        # allocate KV Cache
        for req in context_batch:
            req_beam_width = req.sampling_config.beam_width
            if 'cp_type' in self.mapping.cp_config and 'star_attention' == self.mapping.cp_config[
                    'cp_type']:
                if req.ctx_iters == 0:
                    seq_len = sum(
                        len(ctx_block) for ctx_block in req.ctx_blocks)
                    self.impl.add_sequence(
                        req.py_request_id,
                        seq_len + (len(req.query_id) if self.mapping.cp_rank
                                   == self.mapping.cp_size - 1 else 0),
                        req_beam_width, req)
            else:
                if req.is_first_context_chunk:
                    self.impl.add_sequence(req.py_request_id, req.prompt_len,
                                           req_beam_width, req)
                    for _ in range(self.num_extra_kv_tokens):
                        self.impl.add_token(req.py_request_id)
                    for _ in range(len(req.py_draft_tokens)):
                        self.impl.add_token(req.py_request_id)

        for req in generation_batch:
            self.impl.add_token(req.py_request_id)
            for _ in range(len(req.py_draft_tokens)):
                self.impl.add_token(req.py_request_id)

    def add_dummy_requests(
        self,
        request_ids: List[int],
        # Note that token_nums should be past_kv_len + input_len (without
        # spec decoding). The draft tokens will be added in this function,
        # so we don't need to take care of it in the caller. When preparing
        # token_nums, we should not take the draft tokens into account, so
        # don't use the kv_cache_manager.max_seq_len, which includes both
        # extra tokens and draft tokens.
        token_nums: Optional[List[int]] = None,
        is_gen: bool = False,
        prepare_resource: bool = True,
        max_num_draft_tokens: int = 0,
        use_mrope: bool = False,
        max_beam_width: int = 1,
    ):
        beam_width = max_beam_width
        requests = []
        for i, req_id in enumerate(request_ids):
            # exact choice of n can be ignored for dummy requests
            sampling_params = SamplingParams(n=beam_width,
                                             best_of=beam_width,
                                             use_beam_search=beam_width > 1)
            # Here 1+max_num_draft_tokens is used to extend the prompt length to
            # a non-zero number to skip illegal memory access issue in MLA kernel
            # during warmup.
            token_num = token_nums[
                i] if token_nums is not None else 1 + max_num_draft_tokens
            encoder_input_tokens = [
                1
            ] * token_num if self.impl.cross_kv else None
            # Using 1 instead of 0 prevents NaN during warmup in e.g. Deepseek
            mrope_position_deltas = torch.zeros(
                1, device="cuda", dtype=torch.int32) if use_mrope else None
            req = LlmRequest(request_id=req_id,
                             max_new_tokens=1,
                             input_tokens=[1] * token_num,
                             sampling_config=SamplingConfig(
                                 sampling_params._get_sampling_config()),
                             is_streaming=False,
                             mrope_position_deltas=mrope_position_deltas,
                             encoder_input_tokens=encoder_input_tokens)
            req.is_dummy_request = True
            req.paged_kv_block_ids = []
            if prepare_resource:
                self.impl.add_sequence(req_id, token_num, beam_width, req)
                for _ in range(self.num_extra_kv_tokens):
                    self.impl.add_token(req_id)
            if is_gen:
                req.state = LlmRequestState.GENERATION_IN_PROGRESS
                req.prompt_len = token_num - 1
                req.py_prompt_len = req.prompt_len
                req.py_draft_tokens = [1] * max_num_draft_tokens
                if prepare_resource:
                    for _ in range(max_num_draft_tokens):
                        self.impl.add_token(req_id)
            requests.append(req)
        return requests

    def update_resources(self, scheduled_batch: ScheduledRequests):
        # rewind kv cache
        for request in scheduled_batch.generation_requests:
            if request.state != LlmRequestState.GENERATION_COMPLETE:
                if request.py_rewind_len > 0:
                    self.rewind_kv_cache(request, request.py_rewind_len)

    def free_resources(self, request: LlmRequest):
        self.impl.remove_sequence(request.py_request_id, request)

    def calculate_max_num_blocks(self,
                                 kv_cache_config: KvCacheConfigCpp,
                                 head_dim: int,
                                 tokens_per_block: int,
                                 mapping: Mapping,
                                 dtype: DataType,
                                 kv_factor: int = 2):
        free_mem_fraction = (kv_cache_config.free_gpu_memory_fraction
                             if kv_cache_config.free_gpu_memory_fraction
                             is not None else 0.9)

        cache_size_per_token = kv_factor * sum(
            self.num_kv_heads_per_layer) * head_dim

        if dtype not in (DataType.FP8, DataType.HALF, DataType.BF16,
                         DataType.FLOAT):
            raise ValueError(f'Cannot support {dtype} KV cache.')
        kv_cache_dtype_bytes = binding_dtype_size(dtype)

        cache_size_bytes_per_token = cache_size_per_token * kv_cache_dtype_bytes
        free_mem, total_mem = torch.cuda.mem_get_info()

        assert free_mem_fraction < 1.0, f"Invalid freeMemFraction, freeMemFraction {free_mem_fraction} must be smaller than 1.0"
        max_tokens = free_mem_fraction * free_mem / cache_size_bytes_per_token

        # If user specified a number of tokens
        if kv_cache_config.max_tokens is not None:
            # If user also specified a free gpu memory fraction, take the min
            if kv_cache_config.free_gpu_memory_fraction is not None:
                max_tokens = min(kv_cache_config.max_tokens, max_tokens)
                logger.warning(
                    f'Both free_gpu_memory_fraction and max_tokens are set (to {free_mem_fraction} and {kv_cache_config.max_tokens}, respectively). The smaller value will be used.'
                )
            else:
                max_tokens = kv_cache_config.max_tokens
                logger.info(
                    f"max_tokens is set by kv_cache_config.max_tokens: {max_tokens}"
                )

        if mapping.world_size > 1:
            # make sure all ranks use same value for maxTokens
            max_tokens = mpi_comm().allreduce(max_tokens, op=MPI.MIN)

        # get number of blocks
        blocks_in_primary_pool = math.ceil(max_tokens / tokens_per_block)
        host_cache_size = kv_cache_config.host_cache_size if kv_cache_config.host_cache_size else 0
        max_tokens_secondary = host_cache_size / cache_size_bytes_per_token
        blocks_in_secondary_pool = max(
            0, int(max_tokens_secondary / tokens_per_block))
        return blocks_in_primary_pool, blocks_in_secondary_pool

    def get_max_atten_window_upper_bound(self, blocks_in_primary_pool,
                                         tokens_per_block, max_beam_width,
                                         sink_token_len,
                                         max_seq_len: Optional[int]):
        token_capacity = blocks_in_primary_pool * tokens_per_block
        max_blocks_per_seq = math.floor(token_capacity /
                                        (max_beam_width * tokens_per_block))
        assert max_blocks_per_seq > 0, "Impossible to fit in any sequence in kvCache"

        max_token_num = max_blocks_per_seq * tokens_per_block
        sink_tokens_in_last_block = sink_token_len % tokens_per_block
        sink_bubble_len = 0 if sink_tokens_in_last_block == 0 else tokens_per_block - sink_tokens_in_last_block
        max_atten_window_upper_bound = max_token_num - sink_bubble_len
        if max_seq_len is not None and max_seq_len > max_atten_window_upper_bound and max_beam_width > 1:
            max_atten_window_upper_bound -= tokens_per_block
        assert max_atten_window_upper_bound > 0, "Impossibe to fit in any sequence in kvCache"
        return max_atten_window_upper_bound

    def get_cache_indices(self,
                          request: LlmRequest,
                          window_size: Optional[int] = None) -> List[int]:
        if window_size is None:
            if len(self.max_attention_window_vec) > 1:
                raise ValueError("window_size must be provided for VSWA")
            window_size = self.max_attention_window_vec[0]

        result = self.impl.get_cache_block_ids(request.py_request_id,
                                               window_size)
        assert len(result) == 1
        return result[0]

    def get_batch_cache_indices(
        self,
        request_ids: List[int],
        window_size: Optional[int] = None,
    ) -> List[List[int]]:
        if window_size is None:
            if len(self.max_attention_window_vec) > 1:
                raise ValueError("window_size must be provided for VSWA")
            window_size = self.max_attention_window_vec[0]

        result = self.impl.get_batch_cache_block_ids(request_ids, window_size)
        for i in range(len(result)):
            assert (len(result[i])) == 1
            result[i] = result[i][0]
        return result

    def get_num_free_blocks(self) -> int:
        if self.is_vswa:
            logger.info(
                f"For VSWA case, we return the minimum of the number of free blocks for each window size: {self.impl.get_kv_cache_stats().num_free_blocks_per_window_size}"
            )
            return min(self.impl.get_kv_cache_stats().
                       num_free_blocks_per_window_size.values())
        else:
            return self.impl.get_kv_cache_stats().free_num_blocks

    def get_num_kv_blocks(self, num_tokens: int) -> int:
        return (num_tokens + self.tokens_per_block - 1) // self.tokens_per_block

    def get_num_available_tokens(self, max_num_draft_tokens: int = 0) -> int:
        return (self.get_num_free_blocks() * self.tokens_per_block -
                self.num_extra_kv_tokens - max_num_draft_tokens)

    def get_buffers(self, layer_idx: int) -> Optional[torch.Tensor]:
        layer_offset = self.layer_offsets[layer_idx]
        result = self.impl.get_primary_pool_data(layer_offset)
        return result.reshape(
            result.shape[0],
            self.kv_factor,
            self.tokens_per_block,
            self.num_kv_heads_per_layer[layer_offset],
            self.head_dim,
        )

    def get_block_ids_per_seq(self, request_ids: List[int]) -> torch.Tensor:
        block_ids_per_seq = self.get_batch_cache_indices(request_ids)
        block_ids_per_seq_tensors = [
            torch.tensor(sublist, dtype=torch.int)
            for sublist in block_ids_per_seq
        ]
        padded_tensor = torch.nn.utils.rnn.pad_sequence(
            block_ids_per_seq_tensors, batch_first=True, padding_value=0)
        return padded_tensor

    def flush_iteration_events(self):
        self.impl.flush_iteration_events()

    def get_latest_events(self, timeout_ms: Optional[float] = 0):
        return self.impl.get_latest_events(timeout_ms)

    def get_kv_cache_stats(self):
        return self.impl.get_kv_cache_stats()

    def rewind_kv_cache(self, request: LlmRequest, rewind_len: int):
        self.impl.rewind_kv_cache(request.py_request_id, rewind_len)

    def _get_window_size_to_layers(self) -> dict[int, list[int]]:
        """
        Get the window size to layers mapping.
        The returned map has window sizes as keys and lists of layer indices as values.

        max_attention_window_vec is treated as a repeating pattern.
        """
        window_size_to_layers_map = defaultdict(list)

        if not self.max_attention_window_vec:
            # This case should ideally be prevented by earlier config validation.
            # If num_local_layers is 0, an empty map is fine.
            if self.num_local_layers > 0:
                raise Exception(
                    "max_attention_window_vec cannot be empty if there are local layers."
                )
            return {
            }  # Return an empty dict if no local layers or if somehow vec is empty and no layers.

        # Treat max_attention_window_vec as a repeating pattern.
        pattern_len = len(
            self.max_attention_window_vec
        )  # `sliding_window_pattern`, in HF config terms, e.g. https://huggingface.co/google/gemma-3-1b-it/blob/main/config.json#L32
        # early return if max_attention_window_vec is a single value(SWA)
        if pattern_len == 1:
            return {
                self.max_attention_window_vec[0]:
                list(range(self.num_local_layers))
            }
        for local_layer_idx in range(self.num_local_layers):
            window_size = self.max_attention_window_vec[local_layer_idx %
                                                        pattern_len]
            window_size_to_layers_map[window_size].append(local_layer_idx)
        return window_size_to_layers_map

    @staticmethod
    def adjust_window_sizes_for_vswa(
        window_size_to_layers: Dict[int, List[int]],
        kv_cache_config: KvCacheConfigCpp,
        model_config: ModelConfig,
        pool_memory_bytes: int,
        kv_factor: int,
        dtype: DataType,
        is_cross_attention: bool = False,
    ) -> Dict[int, List[int]]:

        assert is_cross_attention is False, 'Cross attention is not supported'

        max_tokens_from_config = kv_cache_config.max_tokens

        def calculate_cache_size_per_token(layers: Set[int]) -> int:
            # Same as BaseKVCacheManager::calculateCacheSizePerTokenForSingleWindowSize
            total_kv_heads = sum(model_config.num_kv_heads_per_layer[i]
                                 for i in layers)
            return total_kv_heads * kv_factor * model_config.head_size

        # Calculate the required memory bytes per sequence.
        required_mem_bytes_per_seq = 0
        for window_size in sorted(window_size_to_layers):
            layers = window_size_to_layers[window_size]
            cache_size_per_token = calculate_cache_size_per_token(layers)
            cache_size_bytes_per_token = cache_size_per_token * binding_dtype_size(
                dtype)
            required_mem_bytes_per_seq += window_size * cache_size_bytes_per_token
        logger.debug(
            f'Required memory per sequence: {required_mem_bytes_per_seq} bytes')

        if required_mem_bytes_per_seq < pool_memory_bytes:
            # No need to adjust the window sizes.
            return copy.deepcopy(window_size_to_layers)

        logger.debug(
            f'Adjusting the window sizes {list(window_size_to_layers)} to fit '
            f'the memory {pool_memory_bytes} bytes.')
        adjusted_window_size_to_layers = {}

        remaining_mem_bytes = pool_memory_bytes
        remaining_layers = set(i for layers in window_size_to_layers.values()
                               for i in layers)

        accum_max_tokens = 0
        prev_window_size = 0

        for window_size in sorted(window_size_to_layers):
            layers = window_size_to_layers[window_size]
            if remaining_mem_bytes > 0 and remaining_layers:
                # Calculate cache size per token for remaining layers only
                cache_size_per_token = calculate_cache_size_per_token(
                    remaining_layers)
                cache_size_bytes_per_token = cache_size_per_token * binding_dtype_size(
                    dtype)
                logger.debug(
                    f'Cache size per token for {len(remaining_layers)} layers: '
                    f'{cache_size_bytes_per_token} bytes')
                # Calculate max tokens that can fit in this window with remaining memory.
                max_tokens_in_window = min(
                    remaining_mem_bytes // cache_size_bytes_per_token,
                    window_size - prev_window_size)
                remaining_mem_bytes -= max_tokens_in_window * cache_size_bytes_per_token
                accum_max_tokens += max_tokens_in_window
                logger.debug(f'Remaining memory: {remaining_mem_bytes} bytes')
                logger.debug(
                    f'Max token of window {window_size}: {accum_max_tokens}')

                if accum_max_tokens < window_size:
                    logger.debug(
                        f'Max tokens ({accum_max_tokens}) cannot fill the current window ({window_size}). '
                        f'The larger windows will have the same max tokens.')
                    remaining_mem_bytes = 0

                # Clamp the sequence length if provided explicitly.
                if max_tokens_from_config is not None:
                    accum_max_tokens = min(max_tokens_from_config,
                                           accum_max_tokens)
                    # If max tokens from config is reached, stop allocating
                    # more memory. Since the maximum number of tokens is
                    # already reached, for the remaining windows maxTokens
                    # will be set by the current value of accumMaxTokens.
                    if accum_max_tokens == max_tokens_from_config:
                        remaining_mem_bytes = 0

            if accum_max_tokens not in adjusted_window_size_to_layers:
                adjusted_window_size_to_layers[accum_max_tokens] = layers.copy()
            else:
                adjusted_window_size_to_layers[accum_max_tokens].extend(layers)

            remaining_layers -= set(layers)
            prev_window_size = window_size

        return adjusted_window_size_to_layers

    def calculate_max_num_blocks_from_cpp(
            self,
            kv_cache_config: KvCacheConfigCpp,
            model_config: ModelConfig,
            extra_cost_memory: int = 0) -> dict[int, tuple[int, int]]:
        """
        This function is a wrapper of KVCacheManagerCpp.calculate_max_num_blocks.
        The final goal is to switch to the C++ implementation of calculate_max_num_blocks.
        Currently, this function is added to support *ONLY* VSWA.

        Args:
            kv_cache_config: The KV cache configuration object.
            model_config: The model configuration object.
            extra_cost_memory: Extra memory in bytes to exclude from available memory.

        Returns:
            A dict of (max_attention_window, (blocks_in_primary_pool, blocks_in_secondary_pool)).
        """

        # VSWA on Torch backend has not supported the cross attention.
        is_cross_attention = False
        # check model config
        assert model_config.layer_types is not None, "layer_types have to be set correctly for VSWA"

        # Construct WorldConfig from self.mapping
        world_config_cpp = WorldConfig(
            tensor_parallelism=self.mapping.tp_size,
            pipeline_parallelism=self.mapping.pp_size,
            rank=self.mapping.rank,
            gpus_per_node=self.mapping.gpus_per_node)

        window_size_to_layers = self._get_window_size_to_layers()
        logger.debug(f"window_size_to_layers: {window_size_to_layers}")

        free_mem, total_mem = torch.cuda.mem_get_info()
        primary_pool_memory_bytes = free_mem
        secondary_pool_memory_bytes = 0
        logger.debug(
            f"primary_pool_memory_bytes is set to {primary_pool_memory_bytes/1024**3}GB, \n"
            f"secondary_pool_memory_bytes is set to {secondary_pool_memory_bytes/1024**3}GB"
        )

        # Adjust the window sizes to fit the memory if even a single sequence
        # cannot fit in the memory.
        window_size_to_layers = self.adjust_window_sizes_for_vswa(
            window_size_to_layers=window_size_to_layers,
            model_config=model_config,
            kv_cache_config=kv_cache_config,
            pool_memory_bytes=primary_pool_memory_bytes,
            kv_factor=self.kv_factor,
            dtype=self.dtype,
            is_cross_attention=is_cross_attention,
        )

        blocks_per_window = KVCacheManagerCpp.calculate_max_num_blocks(
            config=kv_cache_config,
            # TODO: support cross attention
            is_cross_attention=is_cross_attention,
            dtype=self.dtype,
            model_config=model_config,
            world_config=world_config_cpp,
            window_size_to_layers=window_size_to_layers,
            allotted_primary_mem_bytes=primary_pool_memory_bytes,
            allotted_secondary_mem_bytes=secondary_pool_memory_bytes,
            extra_cost_memory=extra_cost_memory,
            kv_factor=self.kv_factor,
        )
        return blocks_per_window

    def _validate_and_adjust_attention_windows(
        self,
        max_attention_window_vec: List[int],
        blocks_per_window: BlocksPerWindow,
        tokens_per_block: int,
        sink_token_length: int,
        max_seq_len: int,
        max_beam_width: int,
    ) -> Tuple[BlocksPerWindow, int, List[int]]:
        """
        Validate and adjust attention windows against their upper bounds if needed.
        If there is no adjustment, the returned max_attention_window_vec will be the same as the input.

        Args:
            max_attention_window_vec: List of attention window sizes
            blocks_per_window: Dict mapping window size to (primary_blocks, secondary_blocks)
            tokens_per_block: Number of tokens per block
            sink_token_length: Length of sink tokens
            max_seq_len: Maximum sequence length

        Returns:
            Tuple of (adjusted_blocks_per_window, adjusted_max_seq_len, adjusted_max_attention_window_vec)
        """
        window_adjustments = {}
        # Validate each window size in blocks_per_window against its upper bound
        for window_size, (blocks_in_primary_pool,
                          _) in blocks_per_window.items():
            upper_bound = self.get_max_atten_window_upper_bound(
                blocks_in_primary_pool=blocks_in_primary_pool,
                tokens_per_block=tokens_per_block,
                max_beam_width=max_beam_width,
                sink_token_len=sink_token_length,
                max_seq_len=max_seq_len)
            if window_size > upper_bound:
                logger.warning(
                    f"Attention window size {window_size} exceeds upper bound {upper_bound} "
                    f"for available blocks. Reducing to {upper_bound}.")
                window_adjustments[window_size] = upper_bound
        # Apply adjustments to the window vector if any were needed
        if window_adjustments:
            adjusted_window_vec = [
                window_adjustments.get(window, window)
                for window in max_attention_window_vec
            ]
            logger.warning(
                f"Adjusted max_attention_window_vec to {adjusted_window_vec}")
            # update the window size in blocks_per_window if it is adjusted
            adjusted_blocks_per_window = {}
            for window_size, memory_pools in blocks_per_window.items():
                if window_size in window_adjustments:
                    adjusted_window_size = window_adjustments[window_size]
                    adjusted_blocks_per_window[
                        adjusted_window_size] = memory_pools
                    logger.warning(
                        f"Adjusted window size {window_size} to {adjusted_window_size} in blocks_per_window"
                    )
                else:
                    adjusted_blocks_per_window[window_size] = memory_pools
            # Update max_seq_len to the maximum of adjusted windows
            adjusted_max_seq_len = max(adjusted_window_vec)
            logger.warning(f"Adjusted max_seq_len to {adjusted_max_seq_len}")

            return adjusted_blocks_per_window, adjusted_max_seq_len, adjusted_window_vec
        else:
            return blocks_per_window, max_seq_len, max_attention_window_vec

    def _set_temp_attention_window_inputs(
            self) -> Optional[TempAttentionWindowInputs]:
        """
        Set up temp_attention_window_inputs for sliding window.
        """
        is_sliding_window = min(
            self.max_attention_window_vec) < self.max_seq_len
        if is_sliding_window:
            temp_attention_window_inputs = TempAttentionWindowInputs()
            temp_attention_window_inputs.paged_context_fmha = True
            temp_attention_window_inputs.max_input_len = self.max_seq_len - 1
            temp_attention_window_inputs.max_num_tokens = self.max_num_tokens
            return temp_attention_window_inputs
        else:
            return None


class MambaCacheManager(BaseResourceManager):

    def __init__(
        self,
        d_model: int,
        d_state: int,
        d_conv: int,
        expand: int,
        n_groups: int,
        head_dim: int,
        num_layers: int,
        max_batch_size: int,
        mapping: Mapping,
        dtype: torch.dtype,
        layer_mask: Optional[List[bool]] = None,
    ) -> None:

        # get tp size
        tp_size = mapping.tp_size

        # derive mamba parameters for conv and ssm states
        d_inner = d_model * expand
        conv_dim = d_inner + 2 * n_groups * d_state
        nheads = d_inner // head_dim

        # check that can be partitioned
        assert nheads % tp_size == 0, "nheads must be divisible by tp_size"
        assert conv_dim % tp_size == 0, "conv_dim must be divisible by tp_size"

        # partition conv_dim and nheads
        conv_dim = conv_dim // tp_size
        nheads = nheads // tp_size

        # conv and ssm states device
        device = torch.device("cuda")

        pp_layers, num_layers = get_pp_layers(
            num_layers,
            mapping,
            layer_mask=layer_mask,
        )
        num_local_layers = len(pp_layers)
        self.mamba_layer_offsets = {
            idx: offset
            for offset, idx in enumerate(pp_layers)
        }

        # mamba conv states
        self.conv_states = torch.empty(
            size=[
                num_local_layers,
                max_batch_size,
                conv_dim,
                d_conv - 1,
            ],
            dtype=dtype,
            device=device,
        )

        # mamba ssm states
        self.ssm_states = torch.empty(
            size=[
                num_local_layers,
                max_batch_size,
                nheads,
                head_dim,
                d_state,
            ],
            dtype=dtype,
            device=device,
        )

        # mamba cache available blocks
        self.mamba_cache_free_blocks = [i for i in range(max_batch_size)]

        # mamba cache index, maps request_id -> state indices
        self.mamba_cache_index: Dict[int, int] = {}

        # mamba cache state indices
        self.state_indices: torch.Tensor = torch.arange(max_batch_size,
                                                        device=device,
                                                        dtype=torch.int32)

    def _prepare_mamba_cache_blocks(self, request_ids: List[int]):
        state_indices = []
        for r in request_ids:
            # cache hit
            if r in self.mamba_cache_index:
                state_indices.append(self.mamba_cache_index[r])
            # cache miss
            else:
                if len(self.mamba_cache_free_blocks) == 0:
                    raise Exception("run out of mamba cache blocks")
                block = self.mamba_cache_free_blocks.pop()
                self.mamba_cache_index[r] = block
                state_indices.append(block)
        self.state_indices[:len(state_indices)] = torch.as_tensor(
            state_indices, dtype=torch.int32, device=self.ssm_states.device)

    def prepare_resources(self, scheduled_batch: ScheduledRequests):
        context_ids = [
            i.py_request_id for i in scheduled_batch.context_requests
        ]
        generation_ids = [
            i.py_request_id for i in scheduled_batch.generation_requests
        ]
        request_ids = context_ids + generation_ids
        self._prepare_mamba_cache_blocks(request_ids)

    def free_resources(self, request: LlmRequest):
        request_id = request.py_request_id
        if request_id in self.mamba_cache_index:
            block = self.mamba_cache_index.pop(request_id)
            self.mamba_cache_free_blocks.append(block)

    def get_state_indices(self) -> torch.Tensor:
        return self.state_indices

    def get_conv_states(self, layer_idx: int) -> torch.Tensor:
        layer_offset = self.mamba_layer_offsets[layer_idx]
        return self.conv_states[layer_offset]

    def get_ssm_states(self, layer_idx: int) -> torch.Tensor:
        layer_offset = self.mamba_layer_offsets[layer_idx]
        return self.ssm_states[layer_offset]

    def shutdown(self):
        # release tensor memory, keeping python references as tensors
        self.conv_states = torch.tensor([])
        self.ssm_states = torch.tensor([])
        self.state_indices = torch.tensor([])
        torch.cuda.empty_cache()


class MambaHybridCacheManager(KVCacheManager, MambaCacheManager):

    def __init__(
        self,
        # mamba cache parameters
        mamba_d_model: int,
        mamba_d_state: int,
        mamba_d_conv: int,
        mamba_expand: int,
        mamba_n_groups: int,
        mamba_head_dim: int,
        mamba_num_layers: int,
        mamba_layer_mask: List[bool],
        mamba_cache_dtype: torch.dtype,
        # kv cache parameters
        kv_cache_config: KvCacheConfigCpp,
        kv_cache_type: CacheTypeCpp,
        *,
        num_layers: int,
        layer_mask: List[bool],
        num_kv_heads: Union[int, List[Optional[int]]],
        head_dim: int,
        tokens_per_block: int,
        # Note that max_seq_len is not necessarily equal to kv_cache_config.num_tokens.
        # It's derived from the model's BuildConfig for consistency with the C++ backend.
        max_seq_len: int,
        max_batch_size: int,
        mapping: Mapping,
        dtype: DataType = DataType.HALF,
        spec_config: Optional["DecodingBaseConfig"] = None,
    ) -> None:

        # mamba hybrid cache requires block reuse to be disabled in KV cache config
        assert not kv_cache_config.enable_block_reuse, "mamba hybrid cache requires block reuse to be disabled in KV cache config"

        # initialize mamba cache manager
        MambaCacheManager.__init__(
            self,
            mamba_d_model,
            mamba_d_state,
            mamba_d_conv,
            mamba_expand,
            mamba_n_groups,
            mamba_head_dim,
            mamba_num_layers,
            max_batch_size,
            mapping,
            mamba_cache_dtype,
            mamba_layer_mask,
        )

        # initialize kv cache manager
        KVCacheManager.__init__(
            self,
            kv_cache_config,
            kv_cache_type,
            num_layers=num_layers,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
            tokens_per_block=tokens_per_block,
            max_seq_len=max_seq_len,
            max_batch_size=max_batch_size,
            mapping=mapping,
            dtype=dtype,
            spec_config=spec_config,
            layer_mask=layer_mask,
        )

    def prepare_resources(self, scheduled_batch: ScheduledRequests):
        MambaCacheManager.prepare_resources(self, scheduled_batch)
        KVCacheManager.prepare_resources(self, scheduled_batch)

    def free_resources(self, request: LlmRequest):
        MambaCacheManager.free_resources(self, request)
        KVCacheManager.free_resources(self, request)

    def shutdown(self):
        MambaCacheManager.shutdown(self)
        KVCacheManager.shutdown(self)


class SlotManager:

    def __init__(self, max_num_requests: int):
        self.max_num_requests = max_num_requests
        self.slot_mapping = dict()
        self.free_slots = set(range(max_num_requests))

    def get_slot(self, request_id: int):
        return self.slot_mapping.get(request_id, None)

    def fill_slot_id_tensor(self, requests: List[LlmRequest],
                            slot_id_tensor: torch.Tensor):
        for i, request in enumerate(requests):
            slot_id = self.get_slot(request.request_id)
            if slot_id is not None:
                slot_id_tensor[i] = slot_id
            else:
                raise ValueError(f"Request {request.request_id} has no slot id")

    def add_slot(self, request_id: int):
        if len(self.free_slots) == 0:
            raise ValueError("No free slots")
        slot = self.free_slots.pop()
        self.slot_mapping[request_id] = slot
        return slot

    def remove_slot(self, request_id: int):
        if request_id in self.slot_mapping:
            slot = self.slot_mapping.pop(request_id)
            self.free_slots.add(slot)


class ResourceManager:

    def __init__(self, resource_managers: dict[str, BaseResourceManager]):
        self.resource_managers = OrderedDict(resource_managers)

    def __call__(self, name: str):
        return self.resource_managers[name]

    def register_resource_manager(self, name: str,
                                  resource_manager: BaseResourceManager):
        self.resource_managers[name] = resource_manager

    def get_resource_manager(self, name: str) -> BaseResourceManager:
        return self.resource_managers.get(name)

    @nvtx_range("prepare_resources")
    def prepare_resources(self, scheduled_batch: ScheduledRequests):
        for _, resource_manager in self.resource_managers.items():
            if hasattr(resource_manager, "prepare_resources"):
                resource_manager.prepare_resources(scheduled_batch)

    @nvtx_range("update_resources")
    def update_resources(self, scheduled_batch: ScheduledRequests):
        for _, resource_manager in self.resource_managers.items():
            if hasattr(resource_manager, "update_resources"):
                resource_manager.update_resources(scheduled_batch)

    def free_resources(self, request: LlmRequest):
        for _, resource_manager in reversed(self.resource_managers.items()):
            if hasattr(resource_manager, "free_resources"):
                resource_manager.free_resources(request)

    def reorder_pipeline(self, resource_manager_list: list[str]):
        assert set(resource_manager_list) == set(self.resource_managers.keys())
        for resource_manager in resource_manager_list:
            self.resource_managers.move_to_end(resource_manager)


class PeftCacheManager(BaseResourceManager):

    def __init__(self,
                 peft_cache_config: PeftCacheConfig,
                 model_config: ModelConfig,
                 world_config: WorldConfig | None = None):
        import tensorrt_llm.bindings as _tb

        peft_cache_manager_config = _tb.PeftCacheManagerConfig(
            num_host_module_layer=peft_cache_config.num_host_module_layer,
            num_device_module_layer=peft_cache_config.num_device_module_layer,
            optimal_adapter_size=peft_cache_config.optimal_adapter_size,
            max_adapter_size=peft_cache_config.max_adapter_size,
            num_put_workers=peft_cache_config.num_put_workers,
            num_ensure_workers=peft_cache_config.num_ensure_workers,
            num_copy_streams=peft_cache_config.num_copy_streams,
            max_pages_per_block_host=peft_cache_config.max_pages_per_block_host,
            max_pages_per_block_device=peft_cache_config.
            max_pages_per_block_device,
            device_cache_percent=peft_cache_config.device_cache_percent,
            host_cache_size=peft_cache_config.host_cache_size,
            lora_prefetch_dir=peft_cache_config.lora_prefetch_dir,
        )

        if world_config is None:
            world_config = _tb.WorldConfig()

        BufferManager = tensorrt_llm.bindings.internal.runtime.BufferManager
        buffer_manager = BufferManager(torch.cuda.current_stream().cuda_stream,
                                       True)
        self.impl = PeftCacheManagerCpp(config=peft_cache_manager_config,
                                        model_config=model_config,
                                        world_config=world_config,
                                        buffer_manager=buffer_manager)

    def add_request_peft(self, request: LlmRequest):
        self.impl.add_request_peft(request, True)

    def ensure_batch(self,
                     context_batch: List[LlmRequest],
                     generation_batch: List[LlmRequest],
                     reset_gpu_cache: bool = False) -> List[LlmRequest]:
        return self.impl.ensure_batch(context_batch, generation_batch,
                                      reset_gpu_cache)

    def get_max_resource_count(self) -> int:
        return 0

    def get_needed_resource_to_completion(self, request: LlmRequest) -> int:
        return 0

    def prepare_resources(self, scheduled_batch: ScheduledRequests):
        context_batch = scheduled_batch.context_requests
        generation_batch = scheduled_batch.generation_requests
        for req in context_batch:
            if req.lora_weights is not None and req.lora_config is not None:
                req.lora_weights = req.lora_weights.reshape(
                    [1] + list(req.lora_weights.shape))
                req.lora_config = req.lora_config.reshape(
                    [1] + list(req.lora_config.shape))
            self.impl.add_request_peft(req, True)

        py_lora_task_layer_module_configs = self.impl.ensure_batch(
            context_batch, generation_batch, False)

        for req in context_batch:
            req.py_lora_task_layer_module_configs = py_lora_task_layer_module_configs[
                req.
                py_request_id] if req.py_request_id in py_lora_task_layer_module_configs else None
        for req in generation_batch:
            req.py_lora_task_layer_module_configs = py_lora_task_layer_module_configs[
                req.
                py_request_id] if req.py_request_id in py_lora_task_layer_module_configs else None

    def update_resources(self, scheduled_batch: ScheduledRequests):
        pass

    def free_resources(self, request: LlmRequest):
        self.impl.mark_request_done(request)

    def shutdown(self):
        pass
