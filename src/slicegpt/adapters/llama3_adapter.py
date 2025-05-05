# -*- coding:utf-8 -*-　
# Last modify: Liu Wentao
# Description: Adapter for llama3
# Note:
from abc import ABC

from transformers import PretrainedConfig
from transformers.models.llama.modeling_llama import LlamaConfig, LlamaDecoderLayer, LlamaForCausalLM, LlamaRMSNorm
from slicegpt.model_adapter import LayerAdapter, ModelAdapter
from torch.nn import Linear, Module
import torch

class CompressedLlama3DecoderLayer(LlamaDecoderLayer):
    """
    Custom decoder layer for Llama3 with compression support.
    """

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_value: tuple[torch.Tensor] | None = None,
        output_attentions: bool | None = False,
        use_cache: bool | None = False,
        cache_position: torch.LongTensor = None,
        position_embeddings: tuple[torch.Tensor, torch.Tensor] = None,
        **kwargs,
    ) -> tuple:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)

        # Self Attention
        hidden_states, self_attn_weights, present_key_value = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
            cache_position=cache_position,
            position_embeddings=position_embeddings,
            **kwargs,
        )

        if hasattr(self, "attn_shortcut_Q") and self.attn_shortcut_Q is not None:
            rotated_residual = torch.matmul(residual, self.attn_shortcut_Q)
            hidden_states = rotated_residual + hidden_states
        else:
            hidden_states = residual + hidden_states

        # Fully Connected
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)

        if hasattr(self, "mlp_shortcut_Q") and self.mlp_shortcut_Q is not None:
            rotated_residual = torch.matmul(residual, self.mlp_shortcut_Q)
            hidden_states = rotated_residual + hidden_states
        else:
            hidden_states = residual + hidden_states

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (self_attn_weights,)

        if use_cache:
            outputs += (present_key_value,)

        return outputs


class Llama3LayerAdapter(LayerAdapter):
    def __init__(self, layer: LlamaDecoderLayer) -> None:
        super().__init__()
        self._layer: LlamaDecoderLayer = layer

    @property
    def layer(self) -> Module:
        return self._layer

    @property
    def hidden_states_args_position(self) -> int:
        return 0

    @property
    def hidden_states_output_position(self) -> int:
        return 0

    def get_first_layernorm(self) -> Module:
        return self.layer.input_layernorm

    def get_second_layernorm(self) -> Module:
        return self.layer.post_attention_layernorm

    def get_attention_inputs(self) -> list[Linear]:
        return [self.layer.self_attn.q_proj, self.layer.self_attn.k_proj, self.layer.self_attn.v_proj]

    def get_attention_output(self) -> Linear:
        return self.layer.self_attn.o_proj

    def get_mlp_inputs(self) -> list[Linear]:
        return [self.layer.mlp.gate_proj, self.layer.mlp.up_proj]

    def get_mlp_output(self) -> Linear:
        return self.layer.mlp.down_proj


class Llama3ModelAdapter(ModelAdapter):
    def __init__(self, model: LlamaForCausalLM) -> None:
        super().__init__()
        self._model: LlamaForCausalLM = model

    @property
    def model(self) -> Module:
        return self._model

    @property
    def config(self) -> PretrainedConfig:
        return self._model.config

    @property
    def config_type(self) -> type:
        return LlamaConfig

    @property
    def parallel_blocks(self) -> bool:
        return False

    @property
    def seqlen(self) -> int:
        return self.config.max_position_embeddings

    @property
    def hidden_size(self) -> int:
        return self.config.hidden_size

    @property
    def should_bake_mean_into_linear(self) -> bool:
        return False

    @property
    def original_layer_type(self) -> type:
        return LlamaDecoderLayer

    @property
    def original_layer_norm_type(self) -> type:
        return LlamaRMSNorm

    @property
    def layer_adapter_type(self) -> type:
        return Llama3LayerAdapter

    @property
    def compressed_layer_type(self) -> type:
        return CompressedLlama3DecoderLayer

    @property
    def use_cache(self) -> bool:
        return self.config.use_cache

    @use_cache.setter
    def use_cache(self, value: bool) -> None:
        self.config.use_cache = value

    def compute_output_logits(self, input_ids: torch.Tensor) -> torch.FloatTensor:
        return self.model(input_ids=input_ids).logits

    def get_layers(self) -> list[LayerAdapter]:
        return [Llama3LayerAdapter(layer) for layer in self.model.model.layers]

    def convert_layer_to_compressed(self, layer: Module, layer_idx: int | None) -> Module:
        compressed_layer = CompressedLlama3DecoderLayer(self._model.config, layer_idx).to(self._model.config.torch_dtype)
        compressed_layer.load_state_dict(layer.state_dict(), strict=True)
        return compressed_layer

    def post_init(self, tokenizer) -> None:
        tokenizer.pad_token = tokenizer.eos_token
        self._model.config.pad_token_id = tokenizer.pad_token_id

    def get_raw_layer_at(self, index: int) -> Module:
        return self.model.model.layers[index]

    def set_raw_layer_at(self, index: int, new_layer: Module) -> None:
        self.model.model.layers[index] = new_layer

    def get_embeddings(self) -> list[Module]:
        return [self.model.model.embed_tokens]

    def get_pre_head_layernorm(self) -> Module:
        pre_head_layernorm = self.model.model.norm
        assert isinstance(pre_head_layernorm, self.original_layer_norm_type)
        return pre_head_layernorm

    def get_lm_head(self) -> Linear:
        return self.model.lm_head

    @classmethod
    def _from_pretrained(
            cls,
            model_name: str,
            model_path: str,
            *,
            dtype: torch.dtype = torch.float16,
            local_files_only: bool = False,
            token: str | bool | None = None,
    ) -> ModelAdapter | None:
        if not model_name.startswith("meta-llama/Llama-3"):
            return None

        model = LlamaForCausalLM.from_pretrained(
            model_path, torch_dtype=dtype, token=token, local_files_only=local_files_only
        )
        model.config.torch_dtype = dtype
        return cls(model)

    @classmethod
    def _from_uninitialized(
            cls,
            model_name: str,
            model_path: str,
            *,
            dtype: torch.dtype = torch.float16,
            local_files_only: bool = False,
            token: str | bool | None = None,
    ) -> ModelAdapter | None:
        if not model_name.startswith("meta-llama/Llama-3"):
            return None

        class UninitializedLlamaForCausalLM(LlamaForCausalLM):
            def _init_weights(self, _) -> None:
                # Prevent weight initialization
                pass

        config = LlamaConfig.from_pretrained(
            model_path, torch_dtype=dtype, token=token, local_files_only=local_files_only
        )
        model = UninitializedLlamaForCausalLM(config)
        model = model.to(dtype=dtype)

        return Llama3ModelAdapter(model)