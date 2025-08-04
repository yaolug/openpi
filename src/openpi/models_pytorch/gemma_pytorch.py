from pytest import Cache
import torch
from torch import nn
import torch.version
# from transformers import Gemma3ForConditionalGeneration
from transformers import GemmaForCausalLM, PaliGemmaForConditionalGeneration
from transformers import PreTrainedModel

import openpi.models.gemma as _gemma
from openpi.models import gemma as gemma_jax

from transformers.models.auto import CONFIG_MAPPING


def apply_rope(x, positions, max_wavelength=10_000):
    """
    Applies RoPE positions [B, L] to x [B, L, H, D].
    """
    d_half = x.shape[-1] // 2
    device = x.device
    dtype = x.dtype
    x = x.to(torch.float32)

    freq_exponents = (2.0 / x.shape[-1]) * torch.arange(d_half, dtype=torch.float32, device=device)
    timescale = max_wavelength**freq_exponents
    radians = positions[..., None].to(torch.float32) / timescale[None, None, :].to(torch.float32)

    radians = radians[..., None, :]

    sin = torch.sin(radians)  # .to(dtype=dtype)
    cos = torch.cos(radians)  # .to(dtype=dtype)

    x1, x2 = x.split(d_half, dim=-1)
    res = torch.empty_like(x)
    res[..., :d_half] = x1 * cos - x2 * sin
    res[..., d_half:] = x2 * cos + x1 * sin

    return res.to(dtype)


class PaliGemmaWithExpertModel(nn.Module):
    def __init__(self, vlm_config, action_expert_config):
        super().__init__()

        # TODO simplify config, to get the config from the name and then override necessary fields
        # maybe only need to override the fields in action expert
        vlm_config = CONFIG_MAPPING["paligemma"](
                transformers_version="4.48.1",
                _vocab_size=257152,
                bos_token_id=2,
                eos_token_id=1,
                hidden_size=2048,
                image_token_index=257152,
                model_type="paligemma",
                pad_token_id=0,
                projection_dim=2048,
                text_config={
                    "hidden_activation": "gelu_pytorch_tanh",
                    "hidden_size": vlm_config.width,
                    "intermediate_size": vlm_config.mlp_dim,
                    "model_type": "gemma",
                    "num_attention_heads": vlm_config.num_heads,
                    "head_dim": vlm_config.head_dim,
                    "num_hidden_layers": vlm_config.depth,
                    "num_image_tokens": 256,
                    "num_key_value_heads": vlm_config.num_kv_heads,
                    "torch_dtype": "float32",
                    "vocab_size": 257152,
                },
                vision_config={
                    "hidden_size": 1152,
                    "intermediate_size": 4304,
                    "model_type": "siglip_vision_model",
                    "num_attention_heads": 16,
                    "num_hidden_layers": 27,
                    "num_image_tokens": 256,
                    "patch_size": 14,
                    "projection_dim": 2048,
                    "projector_hidden_act": "gelu_fast",
                    "torch_dtype": "float32",
                    "vision_use_head": False,
                },
            )
        action_expert_config = CONFIG_MAPPING["gemma"](
                attention_bias=False,
                attention_dropout=0.0,
                bos_token_id=2,
                eos_token_id=1,
                head_dim=action_expert_config.head_dim,
                hidden_act="gelu_pytorch_tanh",
                hidden_activation="gelu_pytorch_tanh",
                hidden_size=action_expert_config.width,
                initializer_range=0.02,
                intermediate_size=action_expert_config.mlp_dim,
                max_position_embeddings=8192,
                model_type="gemma",
                num_attention_heads=action_expert_config.num_heads,
                num_hidden_layers=action_expert_config.depth,
                num_key_value_heads=action_expert_config.num_kv_heads,
                pad_token_id=0,
                rms_norm_eps=1e-06,
                rope_theta=10000.0,
                torch_dtype="float32",
                transformers_version="4.48.1",
                use_cache=True,
                vocab_size=257152,
            )

        # TODO convert to pytorch config

        self.paligemma = PaliGemmaForConditionalGeneration(config=vlm_config)
        self.gemma_expert = GemmaForCausalLM(config=action_expert_config)
        # Remove unused embed_tokens
        self.gemma_expert.model.embed_tokens = None

        self.to_bfloat16_like_physical_intelligence()
        # self.set_requires_grad()

        # TODO: remove the following 3 lines
        self.paligemma.vision_tower.eval()
        self.paligemma.eval()
        self.gemma_expert.eval()

    # def set_requires_grad(self):
    #     if self.config.freeze_vision_encoder:
    #         self.paligemma.vision_tower.eval()
    #         for params in self.paligemma.vision_tower.parameters():
    #             params.requires_grad = False

    #     if self.config.train_expert_only:
    #         self.paligemma.eval()
    #         for params in self.paligemma.parameters():
    #             params.requires_grad = False

    # def train(self, mode: bool = True):
    #     super().train(mode)

    #     if self.config.freeze_vision_encoder:
    #         self.paligemma.vision_tower.eval()

    #     if self.config.train_expert_only:
    #         self.paligemma.eval()

    def to_bfloat16_like_physical_intelligence(self):
        self.paligemma = self.paligemma.to(dtype=torch.bfloat16)

        params_to_change_dtype = [
            "language_model.model.layers",
            "gemma_expert.model.layers",
            "vision_tower",
            "multi_modal",
        ]
        for name, param in self.named_parameters():
            if any(selector in name for selector in params_to_change_dtype):
                param.data = param.data.to(dtype=torch.bfloat16)

    def embed_image(self, image: torch.Tensor):
        # Handle different transformers versions
        if hasattr(self.paligemma, "get_image_features"):
            return self.paligemma.get_image_features(image)
        return self.paligemma.model.get_image_features(image)

    def embed_language_tokens(self, tokens: torch.Tensor):
        return self.paligemma.language_model.embed_tokens(tokens)

    # # TODO: break down this huge forward into modules or functions
    # def forward(
    #     self,
    #     attention_mask: torch.Tensor | None = None,
    #     position_ids: torch.LongTensor | None = None,
    #     past_key_values: list[torch.FloatTensor] | Cache | None = None,
    #     inputs_embeds: list[torch.FloatTensor] = None,
    #     use_cache: bool | None = None,
    #     fill_kv_cache: bool | None = None,
    # ):
    #     # Handle past_key_values: if it's a list of [prefix_kv, suffix_kv], extract them
    #     if past_key_values is not None and isinstance(past_key_values, list) and len(past_key_values) == 2:
    #         prefix_past_kv = past_key_values[0]
    #         suffix_past_kv = past_key_values[1]
    #     else:
    #         # For initial call or when past_key_values is None/Cache object
    #         prefix_past_kv = past_key_values
    #         suffix_past_kv = past_key_values

    #     if inputs_embeds[0] is not None:
    #         prefix_output = self.paligemma.language_model.forward(
    #             inputs_embeds=inputs_embeds[0],
    #             attention_mask=attention_mask,
    #             position_ids=position_ids,
    #             past_key_values=prefix_past_kv,
    #             use_cache=use_cache,
    #         )
    #         prefix_past_key_values = prefix_output.past_key_values
    #         prefix_output = prefix_output.last_hidden_state
    #     else:
    #         prefix_output = None
    #         prefix_past_key_values = None

    #     if inputs_embeds[1] is not None:
    #         suffix_output = self.gemma_expert.model.forward(
    #             inputs_embeds=inputs_embeds[1],
    #             attention_mask=attention_mask,
    #             position_ids=position_ids,
    #             past_key_values=suffix_past_kv,
    #             use_cache=use_cache,
    #         )
    #         suffix_past_key_values = suffix_output.past_key_values
    #         suffix_output = suffix_output.last_hidden_state
    #     else:
    #         suffix_output = None
    #         suffix_past_key_values = None

    #     return [prefix_output, suffix_output], [
    #         prefix_past_key_values,
    #         suffix_past_key_values,
    #     ]



    def forward(
        self,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Union[List[torch.FloatTensor], Cache]] = None,
        inputs_embeds: List[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        fill_kv_cache: Optional[bool] = None,
    ):
        models = [self.paligemma.language_model, self.gemma_expert.model]

        for hidden_states in inputs_embeds:
            # TODO this is very inefficient
            # dtype is always the same, batch size too (if > 1 len)
            # device could be trickier in multi gpu edge cases but that's it
            if hidden_states is None:
                continue
            batch_size = hidden_states.shape[0]

        # RMSNorm
        num_layers = self.paligemma.config.text_config.num_hidden_layers
        head_dim = self.paligemma.config.text_config.head_dim
        for layer_idx in range(num_layers):
            query_states = []
            key_states = []
            value_states = []
            for i, hidden_states in enumerate(inputs_embeds):
                if hidden_states is None:
                    continue
                layer = models[i].layers[layer_idx]
                # normalizer = torch.tensor(models[i].config.hidden_size**0.5, dtype=hidden_states.dtype)
                # hidden_states = hidden_states * normalizer
                hidden_states = layer.input_layernorm(hidden_states)

                input_shape = hidden_states.shape[:-1]
                hidden_shape = (*input_shape, -1, layer.self_attn.head_dim)

                hidden_states = hidden_states.to(dtype=torch.bfloat16)
                query_state = layer.self_attn.q_proj(hidden_states).view(hidden_shape)
                key_state = layer.self_attn.k_proj(hidden_states).view(hidden_shape)
                value_state = layer.self_attn.v_proj(hidden_states).view(hidden_shape)

                query_states.append(query_state)
                key_states.append(key_state)
                value_states.append(value_state)

            # B,L,H,D with L sequence length, H number of heads, D head dim
            # concatenate on the number of embeddings/tokens
            query_states = torch.cat(query_states, dim=1)
            key_states = torch.cat(key_states, dim=1)
            value_states = torch.cat(value_states, dim=1)

            query_states = apply_rope(query_states, position_ids)
            key_states = apply_rope(key_states, position_ids)

            if use_cache and past_key_values is None:
                past_key_values = {}

            if use_cache:
                if fill_kv_cache:
                    past_key_values[layer_idx] = {
                        "key_states": key_states,
                        "value_states": value_states,
                    }
                else:
                    # TODO here, some optimization can be done - similar to a `StaticCache` we can declare the `max_len` before.
                    # so we create an empty cache, with just one cuda malloc, and if (in autoregressive case) we reach
                    # the max len, then we (for instance) double the cache size. This implementation already exists
                    # in `transformers`. (molbap)
                    key_states = torch.cat([past_key_values[layer_idx]["key_states"], key_states], dim=1)
                    value_states = torch.cat(
                        [past_key_values[layer_idx]["value_states"], value_states], dim=1
                    )

            attention_interface = self.get_attention_interface()
            att_output = attention_interface(
                attention_mask, batch_size, head_dim, query_states, key_states, value_states
            )
            att_output = att_output.to(dtype=torch.bfloat16)

            # first part of att_output is prefix (up to sequence length, [:, 0:prefix_seq_len])
            outputs_embeds = []
            start = 0
            for i, hidden_states in enumerate(inputs_embeds):
                layer = models[i].layers[layer_idx]

                if hidden_states is not None:
                    end = start + hidden_states.shape[1]

                    if att_output.dtype != layer.self_attn.o_proj.weight.dtype:
                        att_output = att_output.to(layer.self_attn.o_proj.weight.dtype)
                    out_emb = layer.self_attn.o_proj(att_output[:, start:end])

                    # TODO: first dropout (by default 0.0)

                    # first residual
                    out_emb += hidden_states
                    after_first_residual = out_emb.clone()

                    out_emb = layer.post_attention_layernorm(out_emb)
                    out_emb = layer.mlp(out_emb)

                    # TODO: second dropout (by default 0.0)

                    # second residual
                    out_emb += after_first_residual

                    outputs_embeds.append(out_emb)

                    start = end
                else:
                    outputs_embeds.append(None)

            inputs_embeds = outputs_embeds

        # final norm
        outputs_embeds = []
        for i, hidden_states in enumerate(inputs_embeds):
            if hidden_states is not None:
                out_emb = models[i].norm(hidden_states)
                outputs_embeds.append(out_emb)
            else:
                outputs_embeds.append(None)

        return outputs_embeds, past_key_values

    def get_attention_interface(self):
        attention_interface = self.eager_attention_forward
        return attention_interface

    def flash_attention_forward(
        self, attention_mask, batch_size, head_dim, query_states, key_states, value_states
    ):
        raise NotImplementedError("FA2 is not implemented (yet)")

    def eager_attention_forward(
        self, attention_mask, batch_size, head_dim, query_states, key_states, value_states
    ):
        num_att_heads = 8
        num_key_value_heads = 1
        num_key_value_groups = num_att_heads // num_key_value_heads

        # query_states: batch_size, sequence_length, num_att_head, head_dim
        # key_states: batch_size, sequence_length, num_key_value_head, head_dim
        # value_states: batch_size, sequence_length, num_key_value_head, head_dim
        sequence_length = key_states.shape[1]

        key_states = key_states[:, :, :, None, :].expand(
            batch_size, sequence_length, num_key_value_heads, num_key_value_groups, head_dim
        )
        key_states = key_states.reshape(
            batch_size, sequence_length, num_key_value_heads * num_key_value_groups, head_dim
        )

        value_states = value_states[:, :, :, None, :].expand(
            batch_size, sequence_length, num_key_value_heads, num_key_value_groups, head_dim
        )
        value_states = value_states.reshape(
            batch_size, sequence_length, num_key_value_heads * num_key_value_groups, head_dim
        )

        # Attention here is upcasted to float32 to match the original eager implementation.

        query_states = query_states.to(dtype=torch.float32)
        key_states = key_states.to(dtype=torch.float32)

        query_states = query_states.transpose(1, 2)
        key_states = key_states.transpose(1, 2)

        att_weights = torch.matmul(query_states, key_states.transpose(2, 3))
        att_weights *= head_dim**-0.5
        big_neg = -2.3819763e38  # See gemma/modules.py

        masked_att_weights = torch.where(attention_mask[:, None, :, :], att_weights, big_neg)

        probs = nn.functional.softmax(masked_att_weights, dim=-1)
        probs = probs.to(dtype=value_states.dtype)

        # probs: batch_size, num_key_value_head, num_att_head, sequence_length, sequence_length
        # value_states: batch_size, sequence_length, num_att_heads, head_dim

        att_output = torch.matmul(probs, value_states.permute(0, 2, 1, 3))

        print(f"[PyTorch DEBUG] att_output: {att_output.shape}")
        print(f"[PyTorch DEBUG] att_output stats: min={att_output.min():.6f}, max={att_output.max():.6f}, mean={att_output.mean():.6f}")

        att_output = att_output.permute(0, 2, 1, 3)
        # we use -1 because sequence length can change
        att_output = att_output.reshape(batch_size, -1, num_key_value_heads * num_key_value_groups * head_dim)

        return att_output