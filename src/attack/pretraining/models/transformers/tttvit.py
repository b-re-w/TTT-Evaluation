from lattent import TTTForCausalLM, TTTCausalLMOutput
from transformers import PretrainedConfig
from transformers.utils import ModelOutput

from einops.layers.torch import Rearrange

from ..base import BaseModel

from torch.nn import functional as F
from torch import nn
import torch

from typing import *


class TTTVisionConfig(PretrainedConfig):
    """Vision TTT configuration."""

    model_type = "vision_ttt"

    def __init__(
            self,
            image_size=224,
            patch_size=16,
            num_channels=3,
            num_classes=1000,
            hidden_size=768,
            intermediate_size=3072,
            num_hidden_layers=12,
            num_attention_heads=12,
            hidden_act="gelu",
            initializer_range=0.02,
            rms_norm_eps=1e-6,
            pretraining_tp=1,
            use_cache=True,
            rope_theta=10000.0,
            mini_batch_size=16,
            use_gate=False,
            share_qk=False,
            ttt_layer_type="linear",
            ttt_base_lr=1.0,
            pre_conv=True,
            conv_kernel=4,
            scan_checkpoint_group_size=0,
            **kwargs
    ):
        super().__init__(**kwargs)
        self.image_size = image_size
        self.patch_size = patch_size
        self.num_channels = num_channels

        self.num_classes = num_classes
        self.vocab_size = num_classes

        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads

        self.hidden_act = hidden_act
        self.initializer_range = initializer_range
        self.rms_norm_eps = rms_norm_eps
        self.pretraining_tp = pretraining_tp
        self.use_cache = use_cache
        self.rope_theta = rope_theta

        self.use_gate = use_gate
        self.share_qk = share_qk
        self.ttt_layer_type = ttt_layer_type
        self.ttt_base_lr = ttt_base_lr
        self.mini_batch_size = mini_batch_size

        self.pre_conv = pre_conv
        self.conv_kernel = conv_kernel
        self.scan_checkpoint_group_size = scan_checkpoint_group_size

        # Vision-specific attributes
        self.num_patches = (image_size // patch_size) ** 2


class PatchEmbedding(nn.Module):
    """Converts images into patches and projects them into the model dimension."""

    def __init__(self, config: TTTVisionConfig):
        super().__init__()
        self.config = config
        self.image_size = config.image_size
        self.patch_size = config.patch_size
        self.num_patches = config.num_patches
        self.num_channels = config.num_channels

        # Patch splitting and flattening
        patch_dim = self.num_channels * self.patch_size * self.patch_size

        self.to_patch_embedding = nn.Sequential(
            # Split image into patches and flatten
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=self.patch_size, p2=self.patch_size),
            nn.LayerNorm(patch_dim),
            # Project to hidden dimension
            nn.Linear(patch_dim, config.hidden_size),
            nn.LayerNorm(config.hidden_size),
        )

        # Learnable position embeddings for patches
        self.pos_embedding = nn.Parameter(torch.randn(1, self.num_patches, config.hidden_size))

        # [CLS] token embedding
        self.cls_token = nn.Parameter(torch.randn(1, 1, config.hidden_size))

        # Dropout
        self.dropout = nn.Dropout(0.1)

    def forward(self, pixel_values: torch.FloatTensor) -> torch.FloatTensor:
        B = pixel_values.shape[0]

        # Convert image to patches -> (batch_size, num_patches, hidden_size)
        x = self.to_patch_embedding(pixel_values)

        # Add position embeddings
        x = x + self.pos_embedding

        # Add CLS token
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)

        return self.dropout(x)


class TTTForVisionCausalLM(TTTForCausalLM):
    config_class = TTTVisionConfig

    def __init__(self, config: TTTVisionConfig):
        super().__init__(config)
        self.patch_embed = PatchEmbedding(config)

        # Initialize weights
        self.post_init()

    def forward(
            self,
            input_ids: Optional[torch.LongTensor] = None,  # [batch_size, seq_len]
            attention_mask: Optional[torch.Tensor] = None,  # [batch_size, seq_len]
            position_ids: Optional[torch.LongTensor] = None,
            inputs_embeds: Optional[torch.FloatTensor] = None,  # [batch_size, seq_len, hidden_size]
            pixel_values: Optional[torch.FloatTensor] = None,  # [batch_size, channels, height, width]
            cache_params: Optional[Any] = None,
            labels: Optional[torch.LongTensor] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
            use_cache: Optional[bool] = None,
            **kwargs
    ) -> Union[Tuple, TTTCausalLMOutput]:
        batch_size = None
        if pixel_values is not None:
            batch_size = pixel_values.shape[0]
        elif input_ids is not None:
            batch_size = input_ids.shape[0]
        elif inputs_embeds is not None:
            batch_size = inputs_embeds.shape[0]

        if batch_size is None:
            raise ValueError("No valid input provided to determine batch size")

        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        final_attention_mask = None
        final_embeds = None

        # image processing
        if pixel_values is not None:
            # [batch_size, num_patches + 1, hidden_size]
            image_embeds = self.patch_embed(pixel_values)
            image_attention_mask = torch.ones(
                (batch_size, image_embeds.shape[1]),
                dtype=torch.long,
                device=image_embeds.device
            )
            final_embeds = image_embeds  # [batch_size, num_patches + 1, hidden_size]
            final_attention_mask = image_attention_mask  # [batch_size, num_patches + 1]

        # 텍스트 처리 (선택적)
        if input_ids is not None or inputs_embeds is not None:
            if inputs_embeds is None:
                # [batch_size, seq_len] -> [batch_size, seq_len, hidden_size]
                inputs_embeds = self.get_input_embeddings()(input_ids)

            text_attention_mask = attention_mask if attention_mask is not None else torch.ones(
                (batch_size, inputs_embeds.shape[1]),
                dtype=torch.long,
                device=inputs_embeds.device
            )

            if final_embeds is not None:
                # [batch_size, num_patches + 1 + seq_len, hidden_size]
                final_embeds = torch.cat([final_embeds, inputs_embeds], dim=1)
                # [batch_size, num_patches + 1 + seq_len]
                final_attention_mask = torch.cat([final_attention_mask, text_attention_mask], dim=1)
            else:
                final_embeds = inputs_embeds  # [batch_size, seq_len, hidden_size]
                final_attention_mask = text_attention_mask  # [batch_size, seq_len]

        if final_embeds is None:
            raise ValueError("Either pixel_values or input_ids/inputs_embeds must be provided")

        # run ttt model
        outputs = self.model(
            input_ids=None,
            attention_mask=final_attention_mask,
            position_ids=position_ids,
            inputs_embeds=final_embeds,
            cache_params=cache_params,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            use_cache=use_cache,
        )

        # [batch_size, total_seq_len, hidden_size]
        hidden_states = outputs[0]

        # calculate logits
        if self.config.pretraining_tp > 1:
            lm_head_slices = self.lm_head.weight.split(self.vocab_size // self.config.pretraining_tp, dim=0)
            logits = [F.linear(hidden_states, lm_head_slices[i]) for i in range(self.config.pretraining_tp)]
            logits = torch.cat(logits, dim=-1)  # [batch_size, total_seq_len, vocab_size]
        else:
            # [batch_size, total_seq_len, vocab_size]
            logits = self.lm_head(hidden_states)
        logits = logits.float()

        # calculate loss
        loss = None
        if labels is not None and (input_ids is not None or inputs_embeds is not None):
            text_start = self.config.num_patches + 1 if pixel_values is not None else 0
            # [batch_size, seq_len - 1, vocab_size]
            shift_logits = logits[:, text_start:-1, :].contiguous()
            # [batch_size, seq_len - 1]
            shift_labels = labels[:, 1:].contiguous()

            loss_fct = nn.CrossEntropyLoss()
            shift_logits = shift_logits.view(-1, self.config.vocab_size)  # [batch_size * (seq_len - 1), vocab_size]
            shift_labels = shift_labels.view(-1)  # [batch_size * (seq_len - 1)]
            shift_labels = shift_labels.to(shift_logits.device)
            loss = loss_fct(shift_logits, shift_labels)

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        return TTTCausalLMOutput(
            loss=loss,
            logits=logits,  # [batch_size, total_seq_len, vocab_size]
            cache_params=outputs.cache_params,
            hidden_states=outputs.hidden_states
        )

    def prepare_inputs_for_generation(
            self,
            input_ids: Optional[torch.LongTensor] = None,
            attention_mask: Optional[torch.Tensor] = None,
            cache_params: Optional[Any] = None,
            inputs_embeds: Optional[torch.FloatTensor] = None,
            pixel_values: Optional[torch.FloatTensor] = None,
            **kwargs
    ):
        model_inputs = {}

        if cache_params is not None:
            if input_ids is not None:
                input_ids = input_ids[:, -1].unsqueeze(-1)
            if attention_mask is not None:
                attention_mask = attention_mask[:, -1].unsqueeze(-1)
            if inputs_embeds is not None:
                inputs_embeds = inputs_embeds[:, -1:, :]

        if inputs_embeds is not None and cache_params is None:
            model_inputs["inputs_embeds"] = inputs_embeds
        elif input_ids is not None:
            model_inputs["input_ids"] = input_ids

        if pixel_values is not None and cache_params is None:
            model_inputs["pixel_values"] = pixel_values

        model_inputs.update(
            {
                "cache_params": cache_params,
                "use_cache": kwargs.get("use_cache"),
                "attention_mask": attention_mask,
            }
        )

        return model_inputs

    def _update_model_kwargs_for_generation(
            self,
            outputs: ModelOutput,
            model_kwargs: Dict[str, Any],
            **kwargs
    ) -> Dict[str, Any]:
        model_kwargs["cache_params"] = outputs.get("cache_params", None)

        if "attention_mask" in model_kwargs:
            attention_mask = model_kwargs["attention_mask"]
            if attention_mask is not None:
                model_kwargs["attention_mask"] = torch.cat(
                    [attention_mask, attention_mask.new_ones((attention_mask.shape[0], 1))],
                    dim=-1,
                )

        if "pixel_values" in model_kwargs:
            del model_kwargs["pixel_values"]

        return model_kwargs


class TTTVisionLinear(BaseModel):
    def __init__(self, config: TTTVisionConfig, num_classes: int):
        super().__init__(image_size=config.image_size, num_classes=num_classes)
        self.config = config
        self.model = TTTForVisionCausalLM(config=config)

    def forward(self, x, *args, **kwargs):
        outputs = self.model(pixel_values=x, use_cache=False)
        return outputs.logits[:, -1, :]


class TTTVisionTiny(TTTVisionLinear):
    model_name = "Vision-TTTLinear-Tiny_P4H6E64"

    def __init__(self, image_size: int, num_classes: int):
        config = TTTVisionConfig(
            image_size=image_size,
            patch_size=4,
            encoder_stride=2,
            hidden_size=64,
            num_hidden_layers=6,
            num_attention_heads=8,
            intermediate_size=64,
            scan_checkpoint_group_size=4,
            mini_batch_size=56
        )
        super().__init__(config=config, num_classes=num_classes)


class TTTVisionSmall(TTTVisionLinear):
    model_name = "Vision-TTTLinear-Small_P4H6E192"

    def __init__(self, image_size: int, num_classes: int):
        config = TTTVisionConfig(
            image_size=image_size,
            patch_size=4,           # 1/4 of Vi-T
            encoder_stride=4,       # 1/4 of Vi-T
            hidden_size=192,        # 1/4 of Vi-T
            num_hidden_layers=6,    # 1/2 of Vi-T
            num_attention_heads=8,
            intermediate_size=192,  # 1/4 of Vi-T
            scan_checkpoint_group_size=4,
            mini_batch_size=56
        )
        super().__init__(config=config, num_classes=num_classes)


class TTTVisionBase(TTTVisionLinear):
    model_name = "Vision-TTTLinear-Base_P7H6E384"

    def __init__(self, image_size: int, num_classes: int):
        config = TTTVisionConfig(
            image_size=image_size,
            patch_size=7,
            encoder_stride=2,
            hidden_size=384,
            num_hidden_layers=6,
            num_attention_heads=8,
            intermediate_size=384,
            scan_checkpoint_group_size=4,
            mini_batch_size=56
        )
        super().__init__(config=config, num_classes=num_classes)


class TTTVisionLarge(TTTVisionLinear):
    model_name = "Vision-TTTLinear-Large_P16H12E768"

    def __init__(self, image_size: int, num_classes: int):
        config = TTTVisionConfig(
            image_size=image_size,
            patch_size=16,
            encoder_stride=16,
            hidden_size=768,
            num_hidden_layers=12,
            num_attention_heads=12,
            intermediate_size=768,
            scan_checkpoint_group_size=4,
            mini_batch_size=56
        )
        super().__init__(config=config, num_classes=num_classes)
