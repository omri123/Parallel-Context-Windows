import math
from abc import ABC
from typing import Optional, Tuple, Dict

import torch
from torch import nn
from transformers import LlamaConfig
from transformers.models.llama.modeling_llama import LlamaAttention, apply_rotary_pos_emb, LlamaRMSNorm, \
    LlamaDecoderLayer, LlamaModel, LlamaForCausalLM
from transformers import DynamicCache, Cache

from pcw_wrapper import generate_pcw_position_ids

"""
The following code is mainly copy+paste from the original modelling_llama.py:
LlamaAttention uses a caching mechanism for the positional rotation vectors (using LlamaRotaryEmbedding). 
This mechanism forces us to override LLaMa attention layer, which in turn forces us to override the decoder, 
and model (so that the correct forward function would be called).
"""


class LlamaForCausalLMPCW(LlamaForCausalLM, ABC):
    _no_split_modules = ["LlamaDecoderLayerPCW"]
    _tied_weights_keys = ["lm_head.weight"]

    def __init__(self, config: LlamaConfig):
        super(LlamaForCausalLM, self).__init__(config)
        # using our Llama model variant:
        self.model = LlamaModelPCW(config)
        self.vocab_size = config.vocab_size

        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # Initialize weights and apply final processing
        self.post_init()

    def prepare_inputs_for_generation(self,
                                      input_ids: torch.LongTensor,
                                      past_key_values: Optional[Tuple[Tuple[torch.Tensor]]] = None,
                                      windows_key_values: Optional[Tuple[Tuple[torch.Tensor]]] = None,
                                      max_window_size: Optional[int] = None,
                                      sum_windows_size: Optional[int] = None,
                                      **kwargs
                                      ) -> Dict:
        """input_ids:
            ids of task_tokens.
         attention_mask:
            concatenation of windows + task tokens attentions masks.

         Note (past_key_values vs windows_key_values):
             In the first token generation, past_key_values is None while windows_key_values contains the combined past
             key values of context windows. During following generations, past_key_values is the concatenation of
             windows_key_values + previous generations. Thus, windows_key_values is practically ignored.
             """

        # only last token for inputs_ids if past_key_values is defined in kwargs
        if past_key_values:
            input_ids = input_ids[:, -1:]
        attention_mask = kwargs.get("attention_mask")
        position_ids = kwargs.get("position_ids", None)
        if attention_mask is not None and position_ids is None:
            # create PCW's position_ids on the fly
            position_ids = generate_pcw_position_ids(attention_mask, max_window_size, past_key_values,
                                                     sum_windows_size, windows_key_values)

        if windows_key_values and not past_key_values:
            past_key_values = windows_key_values

        return {
            "input_ids": input_ids,
            "past_key_values": past_key_values,
            "use_cache": kwargs.get("use_cache"),
            "position_ids": position_ids,
            "attention_mask": attention_mask,
        }


class LlamaModelPCW(LlamaModel, ABC):
    """
    Transformer decoder consisting of *config.num_hidden_layers* layers. Each layer is a [`LlamaDecoderLayer`]

    Args:
        config: LlamaConfig
    """

    def __init__(self, config: LlamaConfig):
        super(LlamaModel, self).__init__(config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        # using the alternative decoder layer:
        self.layers = nn.ModuleList(
            [LlamaDecoderLayerPCW(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )
        self._use_sdpa = config._attn_implementation == "sdpa"
        self._use_flash_attention_2 = config._attn_implementation == "flash_attention_2"
        assert not self._use_sdpa
        assert not self._use_flash_attention_2
        
        self.norm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.gradient_checkpointing = False

        # register a causal mask to separate causal and padding mask creation. Merging happends in the attention class
        causal_mask = torch.full((config.max_position_embeddings, config.max_position_embeddings), fill_value=1)
        self.register_buffer("causal_mask", torch.triu(causal_mask, diagonal=1), persistent=False)
        # Initialize weights and apply final processing
        self.post_init()


class LlamaDecoderLayerPCW(LlamaDecoderLayer):
    def __init__(self, config: LlamaConfig, layer_idx: int):
        super().__init__(config, layer_idx=layer_idx)
        # overriding attention:
        self.self_attn = LlamaAttentionPCW(config=config, layer_idx=layer_idx)


def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    This is the equivalent of torch.repeat_interleave(x, dim=1, repeats=n_rep). The hidden states go from (batch,
    num_key_value_heads, seqlen, head_dim) to (batch, num_attention_heads, seqlen, head_dim)
    """
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)


class LlamaAttentionPCW(LlamaAttention):
    # we have to override the forward attention due to the rotary embeddings caching mechanism
    def forward(
            self,
            hidden_states: torch.Tensor,
            attention_mask: Optional[torch.Tensor] = None,
            position_ids: Optional[torch.LongTensor] = None,
            past_key_value: Optional[Cache] = None,
            output_attentions: bool = False,
            use_cache: bool = False,
            **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        assert (position_ids is not None)
        assert "padding_mask" not in kwargs

        bsz, q_len, _ = hidden_states.size()

        query_states = self.q_proj(hidden_states).view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = self.k_proj(hidden_states).view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = self.v_proj(hidden_states).view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        kv_seq_len = key_states.shape[-2]
        past_seen_tokens = 0
        past_key_value = getattr(self, "past_key_value", past_key_value) # why?!?!  
        if past_key_value is not None:
            past_seen_tokens = past_key_value.get_usable_length(kv_seq_len, self.layer_idx)  # add what was seen
            kv_seq_len += past_seen_tokens

        # *** changes to the original code to accommodate PCW:
        # making sure that the model generates rotary embeddings in the correct length:
        seq_len = kv_seq_len if position_ids is None else int(torch.max(position_ids) + 1)
        cos, sin = self.rotary_emb(value_states, seq_len=seq_len)
        # *** End of changes due to PCW, the rest of the function is copy-paste from the original transformer package.

        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin, position_ids)
        # [bsz, nh, t, hd]

        if past_key_value is not None:
            # sin and cos are specific to RoPE models; position_ids needed for the static cache
            # cache_kwargs = {"sin": sin, "cos": cos, "position_ids": new_cache_positions}
            assert isinstance(past_key_value, DynamicCache)
            key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx) # , cache_kwargs)

        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)

        attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(self.head_dim)

        if attention_mask is not None:
            if attention_mask.size() != (bsz, 1, q_len, kv_seq_len):
                raise ValueError(
                    f"Attention mask should be of size {(bsz, 1, q_len, kv_seq_len)}, but is {attention_mask.size()}"
                )
            attn_weights = attn_weights + attention_mask

        # upcast attention to fp32
        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        attn_weights = nn.functional.dropout(attn_weights, p=self.attention_dropout, training=self.training)
        attn_output = torch.matmul(attn_weights, value_states).to(query_states.dtype)

        if attn_output.size() != (bsz, self.num_heads, q_len, self.head_dim):
            raise ValueError(
                f"`attn_output` should be of size {(bsz, self.num_heads, q_len, self.head_dim)}, but is"
                f" {attn_output.size()}"
            )

        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)
        attn_output = self.o_proj(attn_output)

        if not output_attentions:
            attn_weights = None

        return attn_output, attn_weights, past_key_value
