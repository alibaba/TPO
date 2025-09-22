from dataclasses import dataclass, field
from typing import Dict, Optional, Sequence, List, Any
import transformers


@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default="facebook/opt-125m")
    version: Optional[str] = field(default="v0")
    freeze_backbone: bool = field(default=False)
    tune_mm_mlp_adapter: bool = field(default=False)
    vision_tower: Optional[str] = field(default=None)
    mm_vision_select_layer: Optional[int] = field(default=-1)  # default to the last layer
    pretrain_mm_mlp_adapter: Optional[str] = field(default=None)
    mm_projector_type: Optional[str] = field(default="linear")
    mm_use_im_start_end: bool = field(default=False)
    mm_use_im_patch_token: bool = field(default=True)
    mm_patch_merge_type: Optional[str] = field(default="flat")
    mm_vision_select_feature: Optional[str] = field(default="patch")
    attn_implementation: Optional[str] = field(default="flash_attention_2")
    perceiver_num_heads: int = field(default=32, metadata={"help": "head num of cross attention in perceiver."})
    perceiver_num_queries: int = field(default=64, metadata={"help": "tokens num of cross attention in perceiver."})
    freeze_perceiver_positions: Optional[str] = field(default=False)


@dataclass
class DataArguments:
    data_path: str = field(default=None, metadata={"help": "Path to the training data."})
    lazy_preprocess: bool = False
    is_multimodal: bool = False
    image_folder: Optional[str] = field(default=None)
    image_aspect_ratio: str = "square"
    conv_template: Optional[Any] = field(default=None)
    default_image_path: Optional[str] = field(default="llava/train/handlers/default_image.jpg")
    default_meta_path: Optional[str] = field(default="llava/train/handlers/default_meta_info.json")


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default="adamw_torch")
    remove_unused_columns: bool = field(default=False)
    freeze_mm_mlp_adapter: bool = field(default=False)
    mpt_attn_impl: Optional[str] = field(default="triton")
    resume: Optional[bool] = field(default=False)
    base_model: Optional[str] = field(
        default="Turing",
        metadata={"choices": ["tbstars", "Turing", "Qwen", "Qwen1.5", "mpt", "llama"]}
    )
    model_max_length: int = field(
        default=512,
        metadata={"help": "Maximum sequence length. Sequences will be right padded (and possibly truncated)."},
    )
    double_quant: bool = field(
        default=True,
        metadata={"help": "Compress the quantization statistics through double quantization."}
    )
    quant_type: str = field(
        default="nf4",
        metadata={"help": "Quantization data type to use. Should be one of `fp4` or `nf4`."}
    )
    bits: int = field(
        default=16,
        metadata={"help": "How many bits to use."}
    )
    lora_enable: bool = False
    lora_r: int = 64
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    lora_weight_path: str = ""
    lora_bias: str = "none"
    mm_projector_lr: Optional[float] = None
    group_by_modality_length: bool = field(default=False)
