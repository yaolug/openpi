from pytest import Cache
import torch
from torch import nn
import torch.version
from transformers import Gemma3ForConditionalGeneration
from transformers import GemmaForCausalLM
from transformers import PreTrainedModel

from openpi.models import gemma as gemma_jax


class PaliGemmaWithExpertModel(PreTrainedModel):
    def __init__(self, config: gemma_jax.Config):
        super().__init__(config=config)
        self.config = config
        # self.paligemma = PaliGemmaForConditionalGeneration(config=config.paligemma_config)
        self.paligemma = Gemma3ForConditionalGeneration.from_pretrained("TODO Yao Lu")
        self.paligemma.vision_tower.position_embedding = nn.Embedding(1024, 1152)
        self.gemma_expert = GemmaForCausalLM(config=config.gemma_expert_config)
        # Remove unused embed_tokens
        self.gemma_expert.model.embed_tokens = None

        self.to_bfloat16_like_physical_intelligence()
        self.set_requires_grad()

    def set_requires_grad(self):
        if self.config.freeze_vision_encoder:
            self.paligemma.vision_tower.eval()
            for params in self.paligemma.vision_tower.parameters():
                params.requires_grad = False

        if self.config.train_expert_only:
            self.paligemma.eval()
            for params in self.paligemma.parameters():
                params.requires_grad = False

    def train(self, mode: bool = True):
        super().train(mode)

        if self.config.freeze_vision_encoder:
            self.paligemma.vision_tower.eval()

        if self.config.train_expert_only:
            self.paligemma.eval()

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

    # TODO: break down this huge forward into modules or functions
    def forward(
        self,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: list[torch.FloatTensor] | Cache | None = None,
        inputs_embeds: list[torch.FloatTensor] = None,
        use_cache: bool | None = None,
        fill_kv_cache: bool | None = None,
    ):
        if inputs_embeds[0] is not None:
            prefix_output = self.paligemma.language_model.forward(
                inputs_embeds=inputs_embeds[0],
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                use_cache=use_cache,
            )
            prefix_past_key_values = prefix_output.past_key_values
        else:
            prefix_past_key_values = None

        if inputs_embeds[1] is not None:
            suffix_output = self.gemma_expert.model.forward(
                inputs_embeds=inputs_embeds[1],
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                use_cache=use_cache,
            )
            suffix_past_key_values = suffix_output.past_key_values
        else:
            suffix_past_key_values = None

        if fill_kv_cache:
            if prefix_past_key_values is None:
                prefix_past_key_values = prefix_output.past_key_values
            if suffix_past_key_values is None:
                suffix_past_key_values = suffix_output.past_key_values

        return [prefix_output.last_hidden_state, suffix_output.last_hidden_state], [
            prefix_past_key_values,
            suffix_past_key_values,
        ]
