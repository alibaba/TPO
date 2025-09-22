# Adopted from https://github.com/lm-sys/FastChat. Below is the original copyright:
# Adopted from tatsu-lab@stanford_alpaca. Below is the original copyright:
#    Copyright 2023 Rohan Taori, Ishaan Gulrajani, Tianyi Zhang, Yann Dubois, Xuechen Li
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.

import os
import copy
from dataclasses import dataclass, field
import json
import logging
from torchvision import transforms
from transformers import logging as hf_logging
from typing import Dict, Optional, Sequence, List
import torch
import transformers
import time
import deepspeed
import re
import pdb
import sys
import pathlib
sys.path.append(str(pathlib.Path(__file__).absolute().parent.parent.parent))

from llava.constants import IGNORE_INDEX, IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from torch.utils.data import Dataset
# from llava.train.llava_trainer import LLaVATrainer
# from llava.train.trainer import LLaVATrainer
from llava.mm_utils import tokenizer_image_token
from PIL import Image
from llava.mm_utils import expand2square
from transformers import AutoConfig, set_seed
from llava.train.trainer import LLaVATrainer
from llava.arguments import ModelArguments, DataArguments
from llava import conversation as conversation_lib
from llava.model import *
from llava.train.scaling_dataset import ScalingLazySupervisedDataset, DataCollatorForScalingSupervisedDataset
from llava.utils import (
    # load_sharded_checkpoint,
    # smart_tokenizer_and_embedding_resize,
    get_latest_mos_uri,
    extract_checkpoint_number,
)
from llava.train.utils import (
    get_mm_adapter_state_maybe_zero_3,
    get_peft_state_non_lora_maybe_zero_3,
    get_peft_state_maybe_zero_3,
    LastStepCallback,
    find_all_linear_names,
    safe_save_model_for_hf_trainer
)
Image.MAX_IMAGE_PIXELS = 900000000

local_rank = None
# 获取 transformers 的日志记录器
logger = hf_logging.get_logger()
logger.setLevel(logging.INFO)
formatter = logging.Formatter(
    "%(asctime)s.%(msecs)03d - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
for safe_handler in logger.handlers:
    safe_handler.setFormatter(formatter)

if os.environ.get("LOCAL_PROCESS_RANK"):
    is_local_first_process = int(os.environ.get("LOCAL_PROCESS_RANK", "0")) == 0
else:
    is_local_first_process = int(os.environ.get("LOCAL_RANK", "0")) == 0
is_global_first_process = int(os.environ.get("RANK", "0")) == 0
rank = int(os.environ.get("RANK", "0"))


def rank0_print(*args):
    if local_rank == 0:
        print(*args)

def pil_to_tensor(image):
    transform = transforms.ToTensor()
    return transform(image)

def tensor_to_pil(tensor):
    transform = transforms.ToPILImage()
    return transform(tensor)

def add_diffusion_noise(image_tensor, noise_step):
    num_steps = 1000  # Number of diffusion steps

    # decide beta in each step
    betas = torch.linspace(-6,6,num_steps)
    betas = torch.sigmoid(betas) * (0.5e-2 - 1e-5) + 1e-5

    # decide alphas in each step
    alphas = 1 - betas
    alphas_prod = torch.cumprod(alphas, dim=0)
    alphas_prod_p = torch.cat([torch.tensor([1]).float(), alphas_prod[:-1]],0) # p for previous
    alphas_bar_sqrt = torch.sqrt(alphas_prod)
    one_minus_alphas_bar_log = torch.log(1 - alphas_prod)
    one_minus_alphas_bar_sqrt = torch.sqrt(1 - alphas_prod)

    def q_x(x_0,t):
        noise = torch.randn_like(x_0)
        alphas_t = alphas_bar_sqrt[t]
        alphas_1_m_t = one_minus_alphas_bar_sqrt[t]
        return (alphas_t*x_0 + alphas_1_m_t*noise)

    noise_delta = int(noise_step) # from 0-999
    noisy_image = image_tensor.clone()
    image_tensor_cd = q_x(noisy_image,noise_step) 

    return image_tensor_cd

@dataclass
class TrainingArguments(transformers.TrainingArguments):
    training_platform: Optional[str] = field(default="nebula")
    # gradient_checkpointing=True
    gradient_checkpointing_kwargs={"use_reentrant":False}
    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default="adamw_torch")
    remove_unused_columns: bool = field(default=False)
    freeze_mm_mlp_adapter: bool = field(default=False)
    mpt_attn_impl: Optional[str] = field(default="triton")
    resume: Optional[bool] = field(default=False)
    tune_base_model: bool = field(default=False)
    base_model: Optional[str] = field(
        default="Turing",
        metadata={"choices": ["tbstars", "Turing", "Qwen", "Qwen1.5", "mpt", "llama", "llava"]}
    )
    training_phase: Optional[str] = field(
        default="finetune",
        metadata={"choices": ["pretrain", "continue-train", "finetune"]}
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
    train_data_path: str = field(default=None, metadata={"help": "Path to the training data."})
    valid_data_path: Optional[str] = field(default=None, metadata={"help": "Path to the validation data."})
    ailake_shuffle: Optional[bool] = field(default=False)
    ailake_conv: Optional[str] = field(default=None)
    tune_vit_from_block: Optional[int] = field(default=None)
    vit_backward_start: Optional[float] = field(default=None)
    skip_loss_spike: bool = field(default=False)
    loss_spike_threshold: float = field(default=1.3)
    loss_buffer_size: int = field(default=20)
    save_bad_cases: bool = field(default=False, metadata={"help": "Whether to save bad cases or not."})
    topk_bad_cases: int = field(default=20)
    logging_dir: str = field(default=None)
    oss_save_dir: str = field(default=None)
    meta_path_template: str = field(default=None)
    model_ignore_mismatched_sizes: Optional[bool] = field(default=False)
    max_patches: int = field(default=784)
    end_lr: Optional[float] = field(default=None)

def preprocess_plain(
    sources: Sequence[str],
    tokenizer: transformers.PreTrainedTokenizer,
) -> Dict:
    # add end signal and concatenate together
    conversations = []
    for source in sources:
        assert len(source) == 2
        assert DEFAULT_IMAGE_TOKEN in source[0]['value']
        source[0]['value'] = DEFAULT_IMAGE_TOKEN
        conversation = source[0]['value'] + source[1]['value'] + conversation_lib.default_conversation.sep
        conversations.append(conversation)
    # tokenize conversations
    input_ids = [tokenizer_image_token(prompt, tokenizer, return_tensors='pt') for prompt in conversations]
    targets = copy.deepcopy(input_ids)
    for target, source in zip(targets, sources):
        tokenized_len = len(tokenizer_image_token(source[0]['value'], tokenizer))
        target[:tokenized_len] = IGNORE_INDEX

    return dict(input_ids=input_ids, labels=targets)


def _tokenize_fn(strings: Sequence[str],
                 tokenizer: transformers.PreTrainedTokenizer) -> Dict:
    """Tokenize a list of strings."""
    tokenized_list = [
        tokenizer(
            text,
            return_tensors="pt",
            padding="longest",
            max_length=tokenizer.model_max_length,
            truncation=True,
        ) for text in strings
    ]
    input_ids = labels = [
        tokenized.input_ids[0] for tokenized in tokenized_list
    ]
    input_ids_lens = labels_lens = [
        tokenized.input_ids.ne(tokenizer.pad_token_id).sum().item()
        for tokenized in tokenized_list
    ]
    return dict(
        input_ids=input_ids,
        labels=labels,
        input_ids_lens=input_ids_lens,
        labels_lens=labels_lens,
    )


def _mask_targets(target, tokenized_lens, speakers):
    # cur_idx = 0
    cur_idx = tokenized_lens[0]
    tokenized_lens = tokenized_lens[1:]
    target[:cur_idx] = IGNORE_INDEX
    for tokenized_len, speaker in zip(tokenized_lens, speakers):
        if speaker == "human":
            target[cur_idx+2:cur_idx + tokenized_len] = IGNORE_INDEX
        cur_idx += tokenized_len


def _add_speaker_and_signal(header, source, get_conversation=True):
    """Add speaker and start/end signal on each round."""
    BEGIN_SIGNAL = "### "
    END_SIGNAL = "\n"
    conversation = header
    for sentence in source:
        from_str = sentence["from"]
        if from_str.lower() == "human":
            from_str = conversation_lib.default_conversation.roles[0]
        elif from_str.lower() == "gpt":
            from_str = conversation_lib.default_conversation.roles[1]
        else:
            from_str = 'unknown'
        sentence["value"] = (BEGIN_SIGNAL + from_str + ": " +
                             sentence["value"] + END_SIGNAL)
        if get_conversation:
            conversation += sentence["value"]
    conversation += BEGIN_SIGNAL
    return conversation


def preprocess_multimodal(
    sources: Sequence[str],
    data_args: DataArguments
) -> Dict:
    is_multimodal = data_args.is_multimodal
    if not is_multimodal:
        return sources

    for source in sources:
        for sentence in source:
            if DEFAULT_IMAGE_TOKEN in sentence['value']:
                sentence['value'] = sentence['value'].replace(DEFAULT_IMAGE_TOKEN, '').strip()
                sentence['value'] = DEFAULT_IMAGE_TOKEN + '\n' + sentence['value']
                sentence['value'] = sentence['value'].strip()
                if "mmtag" in conversation_lib.default_conversation.version:
                    sentence['value'] = sentence['value'].replace(DEFAULT_IMAGE_TOKEN, '<Image>' + DEFAULT_IMAGE_TOKEN + '</Image>')
            replace_token = DEFAULT_IMAGE_TOKEN
            if data_args.mm_use_im_start_end:
                replace_token = DEFAULT_IM_START_TOKEN + replace_token + DEFAULT_IM_END_TOKEN
            sentence["value"] = sentence["value"].replace(DEFAULT_IMAGE_TOKEN, replace_token)

    return sources


def preprocess_llama_2(
    sources,
    tokenizer: transformers.PreTrainedTokenizer,
    has_image: bool = False
) -> Dict:
    conv = conversation_lib.default_conversation.copy()
    roles = {"human": conv.roles[0], "gpt": conv.roles[1]}

    # Apply prompt templates
    conversations = []
    for i, source in enumerate(sources):
        if roles[source[0]["from"]] != conv.roles[0]:
            # Skip the first one if it is not from human
            source = source[1:]

        conv.messages = []
        for j, sentence in enumerate(source):
            role = roles[sentence["from"]]
            assert role == conv.roles[j % 2], f"{i}"
            conv.append_message(role, sentence["value"])
        conversations.append(conv.get_prompt())

    # Tokenize conversations

    if has_image:
        input_ids = torch.stack([tokenizer_image_token(prompt, tokenizer, return_tensors='pt') for prompt in conversations], dim=0)
    else:
        input_ids = tokenizer(
            conversations,
            return_tensors="pt",
            padding="longest",
            max_length=tokenizer.model_max_length,
            truncation=True,
        ).input_ids

    targets = input_ids.clone()

    assert conv.sep_style == conversation_lib.SeparatorStyle.LLAMA_2

    # Mask targets
    sep = "[/INST] "
    for conversation, target in zip(conversations, targets):
        total_len = int(target.ne(tokenizer.pad_token_id).sum())

        rounds = conversation.split(conv.sep2)
        cur_len = 1
        target[:cur_len] = IGNORE_INDEX
        for i, rou in enumerate(rounds):
            if rou == "":
                break

            parts = rou.split(sep)
            if len(parts) != 2:
                break
            parts[0] += sep

            if has_image:
                round_len = len(tokenizer_image_token(rou, tokenizer))
                instruction_len = len(tokenizer_image_token(parts[0], tokenizer)) - 2
            else:
                round_len = len(tokenizer(rou).input_ids)
                instruction_len = len(tokenizer(parts[0]).input_ids) - 2

            target[cur_len : cur_len + instruction_len] = IGNORE_INDEX

            cur_len += round_len
        target[cur_len:] = IGNORE_INDEX

        if cur_len < tokenizer.model_max_length:
            if cur_len != total_len:
                target[:] = IGNORE_INDEX
                print(
                    f"WARNING: tokenization mismatch: {cur_len} vs. {total_len}."
                    f" (ignored)"
                )

    return dict(
        input_ids=input_ids,
        labels=targets,
    )


def preprocess_v1(
    sources,
    tokenizer: transformers.PreTrainedTokenizer,
    has_image: bool = False,
    chosen: int = 1
) -> Dict:
    conv = conversation_lib.default_conversation.copy()
    roles = {"human": conv.roles[0], "gpt": conv.roles[1]}

    # Apply prompt templates
    conversations = []
    prompts = []
    for i, source in enumerate(sources):
        if roles[source[0]["from"]] != conv.roles[0]:
            # Skip the first one if it is not from human
            source = source[1:]

        conv.messages = []
        for j, sentence in enumerate(source):
            role = roles[sentence["from"]]
            assert role == conv.roles[j % 2], f"{i}"
            conv.append_message(role, sentence["value"])
            if j==0:
                prompts.append(conv.get_prompt() + conv.roles[1] + ": ")

        conversations.append(conv.get_prompt())

    # Tokenize conversations

    if has_image:
        input_ids = torch.stack([tokenizer_image_token(prompt, tokenizer, return_tensors='pt') for prompt in conversations], dim=0)
        prompt_input_ids = torch.stack(
            [tokenizer_image_token(prompt, tokenizer, return_tensors='pt') for prompt in prompts], dim=0)

    else:
        input_ids = tokenizer(
            conversations,
            return_tensors="pt",
            padding="longest",
            max_length=tokenizer.model_max_length,
            truncation=True,
        ).input_ids
        prompt_input_ids = tokenizer(
            prompts,
            return_tensors="pt",
            padding="longest",
            max_length=tokenizer.model_max_length,
            truncation=True,
        ).input_ids

    targets = input_ids.clone()

    assert conv.sep_style == conversation_lib.SeparatorStyle.TWO

    # Mask targets
    sep = conv.sep + conv.roles[1] + ": "
    for conversation, target in zip(conversations, targets):
        total_len = int(target.ne(tokenizer.pad_token_id).sum())

        rounds = conversation.split(conv.sep2)
        cur_len = 1
        target[:cur_len] = IGNORE_INDEX
        for i, rou in enumerate(rounds):
            if rou == "":
                break

            parts = rou.split(sep)
            if len(parts) != 2:
                break
            parts[0] += sep

            if has_image:
                round_len = len(tokenizer_image_token(rou, tokenizer))
                instruction_len = len(tokenizer_image_token(parts[0], tokenizer)) - 2
            else:
                round_len = len(tokenizer(rou).input_ids)
                instruction_len = len(tokenizer(parts[0]).input_ids) - 2

            target[cur_len : cur_len + instruction_len] = IGNORE_INDEX

            cur_len += round_len
        target[cur_len:] = IGNORE_INDEX

        if cur_len < tokenizer.model_max_length:
            if cur_len != total_len:
                target[:] = IGNORE_INDEX
                print(
                    f"WARNING: tokenization mismatch: {cur_len} vs. {total_len}."
                    f" (ignored)"
                )


    if chosen == 1:
        return dict(chosen_input_ids=input_ids, chosen_labels=targets, prompt = prompts, prompt_input_ids = prompt_input_ids)
    else:
        return dict(rejected_input_ids=input_ids, rejected_labels=targets)


def preprocess_plain(
    sources: Sequence[str],
    tokenizer: transformers.PreTrainedTokenizer,
) -> Dict:
    # add end signal and concatenate together
    conversations = []
    for source in sources:
        assert len(source) == 2
        assert DEFAULT_IMAGE_TOKEN in source[0]['value']
        source[0]['value'] = DEFAULT_IMAGE_TOKEN
        conversation = source[0]['value'] + source[1]['value'] + conversation_lib.default_conversation.sep
        conversations.append(conversation)
    # tokenize conversations
    input_ids = [tokenizer_image_token(prompt, tokenizer, return_tensors='pt') for prompt in conversations]
    targets = copy.deepcopy(input_ids)
    for target, source in zip(targets, sources):
        tokenized_len = len(tokenizer_image_token(source[0]['value'], tokenizer))
        target[:tokenized_len] = IGNORE_INDEX

    return dict(input_ids=input_ids, labels=targets)


def preprocess(
    sources: Sequence[str],
    tokenizer: transformers.PreTrainedTokenizer,
    has_image: bool = False,
    chosen: int = 1
) -> Dict:
    """
    Given a list of sources, each is a conversation list. This transform:
    1. Add signal '### ' at the beginning each sentence, with end signal '\n';
    2. Concatenate conversations together;
    3. Tokenize the concatenated conversation;
    4. Make a deepcopy as the target. Mask human words with IGNORE_INDEX.
    """
    if conversation_lib.default_conversation.sep_style == conversation_lib.SeparatorStyle.PLAIN:
        return preprocess_plain(sources, tokenizer)
    if conversation_lib.default_conversation.sep_style == conversation_lib.SeparatorStyle.LLAMA_2:
        return preprocess_llama_2(sources, tokenizer, has_image=has_image)
    if conversation_lib.default_conversation.version.startswith("v1"):
        return preprocess_v1(sources, tokenizer, has_image=has_image, chosen = chosen)
    if conversation_lib.default_conversation.version == "mpt":
        return preprocess_mpt(sources, tokenizer)

    # add end signal and concatenate together
    conversations = []
    for source in sources:
        header = f"{conversation_lib.default_conversation.system}\n\n"
        conversation = _add_speaker_and_signal(header, source)
        conversations.append(conversation)

    # tokenize conversations
    def get_tokenize_len(prompts):
        return [len(tokenizer_image_token(prompt, tokenizer)) for prompt in prompts]

    if has_image:
        input_ids = [tokenizer_image_token(prompt, tokenizer, return_tensors='pt') for prompt in conversations]
    else:
        conversations_tokenized = _tokenize_fn(conversations, tokenizer)
        input_ids = conversations_tokenized["input_ids"]

    targets = copy.deepcopy(input_ids)
    for target, source in zip(targets, sources):
        if has_image:
            tokenized_lens = get_tokenize_len([header] + [s["value"] for s in source])
        else:
            tokenized_lens = _tokenize_fn([header] + [s["value"] for s in source], tokenizer)["input_ids_lens"]
        speakers = [sentence["from"] for sentence in source]
        _mask_targets(target, tokenized_lens, speakers)

    if chosen == 1:
        return dict(chosen_input_ids=input_ids, chosen_labels=targets, prompt = conversations[0])
    else:
        return dict(rejected_input_ids=input_ids, rejected_labels=targets)

def _process_image(img, img_aspect_ratio, img_processor, img_crop_processor=None, data_arguments=None):
    if img_aspect_ratio == "NaViT":
        processor_output = img_processor.preprocess(img, max_patches=data_arguments.max_patches, return_tensors="pt")
        return {
            "image": processor_output["pixel_values"][0],
            "patch_attention_mask": processor_output["patch_attention_masks"][0]
        }
    elif img_crop_processor is not None:  # define image list for multi-img-input in future.
        img, _, crop_positions = img_crop_processor(image=[img], text=None)
        return {"image": img, "crop_positions": crop_positions}
    elif img_aspect_ratio == "pad":
        img = expand2square(img, tuple(int(x * 255) for x in img_processor.image_mean))
        processor_output = img_processor.preprocess(img, return_tensors="pt")
        return {"image": processor_output["pixel_values"][0]}
    else:
        processor_output = img_processor.preprocess(img, return_tensors="pt")
        return {"image": processor_output["pixel_values"][0]}

class LazySupervisedDataset(Dataset):
    """Dataset for supervised fine-tuning."""

    def __init__(self, data_path: str,
                 tokenizer: transformers.PreTrainedTokenizer,
                 data_args: DataArguments):
        super(LazySupervisedDataset, self).__init__()
        list_data_dict = json.load(open(data_path, "r"))

        rank0_print("Formatting inputs...Skip in lazy mode")
        self.tokenizer = tokenizer
        self.list_data_dict = list_data_dict
        self.data_args = data_args

    def __len__(self):
        return len(self.list_data_dict)

    @property
    def lengths(self):
        length_list = []
        for sample in self.list_data_dict:
            img_tokens = 128 if 'image' in sample else 0
            length_list.append(sum(len(conv['value'].split()) for conv in sample['conversations']) + img_tokens)
        return length_list

    @property
    def modality_lengths(self):
        length_list = []
        for sample in self.list_data_dict:
            cur_len = sum(len(conv['value'].split()) for conv in sample['conversations'])
            cur_len = cur_len if 'images' in sample else -cur_len
            length_list.append(cur_len)
        return length_list

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        sources = self.list_data_dict[i]
        if isinstance(i, int):
            sources = [sources]
        assert len(sources) == 1, "Don't know why it is wrapped to a list"  # FIXME
        if 'image' in sources[0]:
            image_file = self.list_data_dict[i]['image']
            image_folder = self.data_args.image_folder
            processor = self.data_args.image_processor
            default_image_path = 'llava_data/sft_data/white.jpg'
            crop_processor = self.data_args.crop_processor if "adaptive_crop" in self.data_args.image_aspect_ratio else None
            image_aspect_ratio = self.data_args.image_aspect_ratio
            # image = Image.open(os.path.join(image_folder, image_file)).convert('RGB')
            if self.data_args.image_aspect_ratio == 'pad':
                try:
                    image = Image.open(image_file).convert('RGB')
                except Exception as e:
                    print(f'Image processor exception for image path: {image_file}, exception = {str(e)}', file=sys.stderr)
                    image = Image.open(default_image_path).convert("RGB")
                image_tensor = pil_to_tensor(image)
                image_noisy = add_diffusion_noise(image_tensor, 500)
                image_noisy = tensor_to_pil(image_noisy)
                # image_noisy = Image.open(image_noisy).convert("RGB")
                def expand2square(pil_img, background_color):
                    width, height = pil_img.size
                    if width == height:
                        return pil_img
                    elif width > height:
                        result = Image.new(pil_img.mode, (width, width), background_color)
                        result.paste(pil_img, (0, (width - height) // 2))
                        return result
                    else:
                        result = Image.new(pil_img.mode, (height, height), background_color)
                        result.paste(pil_img, ((height - width) // 2, 0))
                        return result
                image = expand2square(image, tuple(int(x*255) for x in processor.image_mean))
                image = processor.preprocess(image, return_tensors='pt')['pixel_values'][0]
                image_noisy = expand2square(image_noisy, tuple(int(x*255) for x in processor.image_mean))
                image_noisy = processor.preprocess(image_noisy, return_tensors='pt')['pixel_values'][0]

            else:
                # image = processor.preprocess(image, return_tensors='pt')['pixel_values'][0]
                try:
                    image = Image.open(image_file).convert('RGB')
                    image = _process_image(image, image_aspect_ratio, processor, crop_processor, self.data_args)
                except Exception as e:
                    print(f'Image processor exception for image path: {image_file}, exception = {str(e)}', file=sys.stderr)
                    default_img = Image.open(self.data_args.default_image_path).convert("RGB")
                    image = _process_image(default_img, image_aspect_ratio, processor, crop_processor, self.data_args)

            sources = preprocess_multimodal(
                copy.deepcopy([e["conversations"] for e in sources]),
                self.data_args)
        else:
            sources = copy.deepcopy([e["conversations"] for e in sources])
        data_dict = preprocess(
            sources,
            self.tokenizer,
            has_image=('image' in self.list_data_dict[i]),
            chosen=1
        )
        if isinstance(i, int):
            data_dict = dict(chosen_input_ids=data_dict["chosen_input_ids"][0],
                             chosen_labels=data_dict["chosen_labels"][0],
                             prompt_input_ids = data_dict["prompt_input_ids"][0],
                             prompt = data_dict["prompt"][0])


        sources = self.list_data_dict[i]
        if isinstance(i, int):
            sources = [sources]
        assert len(sources) == 1, "Don't know why it is wrapped to a list"  # FIXME
        if 'image' in sources[0]:
            rejected_sources = preprocess_multimodal(
                copy.deepcopy([e["rejected_conversations"] for e in sources]),
                self.data_args)
        else:
            rejected_sources = copy.deepcopy([e["rejected_conversations"] for e in sources])
        rejected_data_dict = preprocess(
            rejected_sources,
            self.tokenizer,
            has_image=('image' in self.list_data_dict[i]),
            chosen=0)
        if isinstance(i, int):
            rejected_data_dict = dict(rejected_input_ids=rejected_data_dict["rejected_input_ids"][0],
                             rejected_labels=rejected_data_dict["rejected_labels"][0],
                             )
        data_dict.update(rejected_data_dict)
        # image exist in the data
        if 'image' in self.list_data_dict[i]:
            data_dict['images'] = image
            data_dict['images_noisy'] = image_noisy
            # data_dict.update(image)
        elif self.data_args.is_multimodal:
            # image does not exist in the data, but the model is multimodal
            crop_size = self.data_args.image_processor.crop_size
            data_dict['images'] = torch.zeros(3, crop_size['height'], crop_size['width'])
            
        return data_dict


@dataclass
class DataCollatorForSupervisedDataset(object):
    """Collate examples for supervised fine-tuning."""

    tokenizer: transformers.PreTrainedTokenizer

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        chosen_input_ids, chosen_labels = tuple([instance[key] for instance in instances]
                                  for key in ("chosen_input_ids", "chosen_labels"))
        prompt_input_ids = [instance["prompt_input_ids"] for instance in instances]
        max_length = max([item.shape[0] for item in prompt_input_ids])
        prompt_input_ids = torch.nn.utils.rnn.pad_sequence(
            prompt_input_ids,
            batch_first=True,
            padding_value=self.tokenizer.pad_token_id)

        prompt_input_ids = prompt_input_ids[:, :max_length]

        chosen_input_ids = torch.nn.utils.rnn.pad_sequence(
            chosen_input_ids,
            batch_first=True,
            padding_value=self.tokenizer.pad_token_id)
        chosen_labels = torch.nn.utils.rnn.pad_sequence(chosen_labels,
                                                 batch_first=True,
                                                 padding_value=IGNORE_INDEX)
        chosen_input_ids = chosen_input_ids[:, :self.tokenizer.model_max_length]
        chosen_labels = chosen_labels[:, :self.tokenizer.model_max_length]

        batch = dict(
            chosen_input_ids = chosen_input_ids,
            chosen_labels= chosen_labels,
            chosen_attention_mask=chosen_input_ids.ne(self.tokenizer.pad_token_id),
            prompt_input_ids = prompt_input_ids,
            prompt_attention_mask = prompt_input_ids.ne(self.tokenizer.pad_token_id),
            prompt = [instance["prompt"] for instance in instances],
        )
        rejected_input_ids, rejected_labels = tuple([instance[key] for instance in instances]
                                  for key in ("rejected_input_ids", "rejected_labels"))
        rejected_input_ids = torch.nn.utils.rnn.pad_sequence(
            rejected_input_ids,
            batch_first=True,
            padding_value=self.tokenizer.pad_token_id)
        rejected_labels = torch.nn.utils.rnn.pad_sequence(rejected_labels,
                                                 batch_first=True,
                                                 padding_value=IGNORE_INDEX)
        rejected_input_ids = rejected_input_ids[:, :self.tokenizer.model_max_length]
        rejected_labels = rejected_labels[:, :self.tokenizer.model_max_length]
        rejected_batch = dict(
            rejected_input_ids = rejected_input_ids,
            rejected_labels= rejected_labels,
            rejected_attention_mask=rejected_input_ids.ne(self.tokenizer.pad_token_id),
        )
        batch.update(rejected_batch)
        # if 'images' in instances[0]:
        #     images = [instance['images'] for instance in instances]
        #     if all(x is not None and x.shape == images[0].shape for x in images):
        #         batch['images'] = torch.stack(images)
        #     else:
        #         batch['images'] = images
        if "images" in instances[0]:
            images = [_["images"] for _ in instances]
            if "crop_positions" in instances[0] and instances[0]["crop_positions"] is not None:
                batch["crop_positions"] = torch.cat([idx for idx in [_["crop_positions"] for _ in instances]], dim=0)
                batch["images"] = torch.cat([img for img in images if img is not None], dim=0)
            elif "patch_attention_mask" in instances[0] and instances[0]["patch_attention_mask"] is not None:
                h_max, w_max = np.max(np.asarray([_.shape for _ in images]), axis=0)[1:]
                images = [F.pad(_, (0, w_max - _.shape[2], 0, h_max - _.shape[1]), "constant", 0) for _ in images]
                batch["images"] = torch.stack(images, dim=0)

                patch_attention_masks = [_["patch_attention_mask"] for _ in instances]
                p_h_max, p_w_max = np.max(np.asarray([_.shape for _ in patch_attention_masks]), axis=0)
                patch_attention_masks = [F.pad(_, (0, p_w_max - _.shape[1], 0, p_h_max - _.shape[0]), "constant", 0) for _ in patch_attention_masks]
                batch["patch_attention_mask"] = torch.stack(patch_attention_masks, dim=0).bool()
            elif all(x is not None and x.shape == images[0].shape for x in images):
                batch["images"] = torch.stack(images)
            else:
                batch["images"] = images
            images_noisy = [instance['images_noisy'] for instance in instances]
            if all(x is not None and x.shape == images_noisy[0].shape for x in images_noisy):
                batch['images_noisy'] = torch.stack(images_noisy)
            else:
                batch['images_noisy'] = images_noisy
        return batch

def make_supervised_data_module(
        tokenizer: transformers.PreTrainedTokenizer,
        data_args: DataArguments,
        training_args: TrainingArguments,
) -> Dict:
    """Make dataset and collator for supervised fine-tuning."""

    logger.info("Formatting inputs...Skip in lazy mode")
    if training_args.data_path.startswith("ailake://"):
        from mdl.distribute_dataset import DistributeDataset
        from llava.train.scaling_dataset import convert_function
        train_dataset = DistributeDataset(
            training_args.data_path,
            num_threads=8,
            capacity=256,
            shuffle=data_args.ailake_shuffle,
            data_convert=convert_function,
            tokenizer=tokenizer,
            training_phase=training_args.training_phase,
            data_args=data_args
        )
        logger.info(train_dataset.state_dict())
        if training_args.resume:
            resume_steps = extract_checkpoint_number(training_args.resume_from_checkpoint)
            # resume_steps = 10000
            resume_data_state_filepath = os.path.join(
                training_args.oss_save_dir, f"train_dataset_state/checkpoint-{resume_steps}-rank-{rank}.json")
            train_dataset.load_state_dict(json.load(open(resume_data_state_filepath, 'r')))
    else:
        # train_dataset = ScalingLazySupervisedDataset(
        #     tokenizer=tokenizer,
        #     data_path=training_args.data_path,
        #     data_args=data_args
        # )
        train_dataset = LazySupervisedDataset(
            tokenizer=tokenizer,
            data_path=data_args.data_path,
            data_args=data_args
        )


    if data_args.valid_data_path is not None:
        assert not data_args.valid_data_path.startswith("ailake://"), NotImplementedError()
        eval_dataset = ScalingLazySupervisedDataset(
            tokenizer=tokenizer,
            data_path=data_args.valid_data_path,
            data_args=data_args
        )
        logger.info("Eval set size: {}".format(len(eval_dataset)))
    else:
        eval_dataset = None

    logger.info("Training set size: {}".format(len(train_dataset)))
    # data_collator = DataCollatorForScalingSupervisedDataset(tokenizer=tokenizer)
    data_collator = DataCollatorForSupervisedDataset(tokenizer=tokenizer)


    return dict(
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator
    )


def postprocess_args(model_arguments, data_arguments, training_arguments):
    training_arguments.logging_dir = os.environ.get("SUMMARY_DIR", training_arguments.logging_dir)
    if training_arguments.resume and training_arguments.resume_from_checkpoint is None:
        logger.info("Auto load latest mos uri...")
        training_arguments.resume_from_checkpoint = get_latest_mos_uri(model_arguments.model_name_or_path)
        training_arguments.resume = False if training_arguments.resume_from_checkpoint is None else True
    training_arguments.resume_from_checkpoint = False if not training_arguments.resume else training_arguments.resume_from_checkpoint
    logger.info(f"Resume from checkpoint path final setting: {training_arguments.resume_from_checkpoint}")
    training_arguments.data_path = training_arguments.train_data_path if training_arguments.train_data_path is not None else data_arguments.data_path
    training_arguments.ignore_data_skip = True if training_arguments.resume and training_arguments.data_path.startswith("ailake://") else False
    training_arguments.save_bad_cases = False if not training_arguments.skip_loss_spike else training_arguments.save_bad_cases
    model_arguments.base_model = training_arguments.base_model
    model_arguments.ignore_mismatched_sizes = training_arguments.model_ignore_mismatched_sizes
    data_arguments.data_path = training_arguments.data_path
    data_arguments.train_data_path = training_arguments.train_data_path
    data_arguments.valid_data_path = training_arguments.valid_data_path
    data_arguments.meta_path_template = training_arguments.meta_path_template
    data_arguments.ailake_shuffle = training_arguments.ailake_shuffle
    data_arguments.ailake_conv = training_arguments.ailake_conv
    data_arguments.max_patches = training_arguments.max_patches

    if training_arguments.oss_save_dir is not None and not os.path.exists(training_arguments.oss_save_dir):
        if is_global_first_process: os.makedirs(training_arguments.oss_save_dir, exist_ok=True)
    if training_arguments.skip_loss_spike:
        oss_save_subdir = os.path.join(training_arguments.oss_save_dir, "loss_buffer_state")
        training_arguments.loss_state_filepath = os.path.join(oss_save_subdir, "checkpoint-{step}-rank-{rank}.pkl")
        if is_global_first_process: os.makedirs(oss_save_subdir, exist_ok=True)
    if training_arguments.data_path.startswith("ailake://"):
        oss_save_subdir = os.path.join(training_arguments.oss_save_dir, "train_dataset_state")
        training_arguments.data_state_filepath = os.path.join(oss_save_subdir, "checkpoint-{step}-rank-{rank}.json")
        if is_global_first_process: os.makedirs(oss_save_subdir, exist_ok=True)
    if training_arguments.save_bad_cases:
        oss_save_subdir = os.path.join(training_arguments.oss_save_dir, "bad_cases")
        training_arguments.bad_cases_filepath = os.path.join(oss_save_subdir, "global_iter_{iter}_rank_{rank}.json")
        if is_global_first_process: os.makedirs(oss_save_subdir, exist_ok=True)

    if "open-clip" in model_arguments.vision_tower.lower() and training_arguments.training_platform == "nebula":
        from llava.utils import load_mos_model
        vision_load_dir = '/data/mdl/open-clip-pretrained-model'
        print('load vision model from {}'.format(model_arguments.vision_tower))
        load_mos_model(vision_load_dir, model_arguments.vision_tower)
        model_arguments.vision_tower = vision_load_dir

    deepspeed.comm.barrier()
    time.sleep(30)
    if training_arguments.oss_save_dir is not None:
        logger.info(f"{training_arguments.oss_save_dir} exists at rank {rank}: {os.path.exists(training_arguments.oss_save_dir)}")

    return model_arguments, data_arguments, training_arguments


def train():
    global local_rank

    parser = transformers.HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = postprocess_args(*parser.parse_args_into_dataclasses())
    training_args.gradient_checkpointing=True
    training_args.gradient_checkpointing_kwargs={'use_reentrant':False}
    logger.info("Model Args:\n{}\n".format(model_args))
    logger.info("Data Args:\n{}\n".format(data_args))
    logger.info("Training Args:\n{}\n".format(training_args))

    set_seed(training_args.seed)
    local_rank = model_args.local_rank = training_args.local_rank
    compute_dtype = (torch.float16 if training_args.fp16 else (torch.bfloat16 if training_args.bf16 else torch.float32))

    bnb_model_from_pretrained_args = {}
    if training_args.bits in [4, 8]:
        from transformers import BitsAndBytesConfig
        bnb_model_from_pretrained_args.update(dict(
            device_map={"": training_args.device},
            load_in_4bit=training_args.bits == 4,
            load_in_8bit=training_args.bits == 8,
            quantization_config=BitsAndBytesConfig(
                load_in_4bit=training_args.bits == 4,
                load_in_8bit=training_args.bits == 8,
                llm_int8_skip_modules=["mm_projector"],
                llm_int8_threshold=6.0,
                llm_int8_has_fp16_weight=False,
                bnb_4bit_compute_dtype=compute_dtype,
                bnb_4bit_use_double_quant=training_args.double_quant,
                bnb_4bit_quant_type=training_args.quant_type  # {"fp4", "nf4"}
            )
        ))

    assert model_args.vision_tower is not None
    logger.info("loading model...")
    config = transformers.AutoConfig.from_pretrained(model_args.model_name_or_path, trust_remote_code=True)
    config.use_cache = False
    config.base_model = model_args.base_model
    config.finetuned_vit = False if training_args.training_phase == "pretrain" else True
    config.vision_tower = model_args.vision_tower
    config.mm_vision_select_layer = model_args.mm_vision_select_layer
    config.mm_vision_select_feature = model_args.mm_vision_select_feature
    config.pretrain_mm_mlp_adapter = model_args.pretrain_mm_mlp_adapter
    config.mm_projector_type = getattr(model_args, "mm_projector_type", "linear")
    config.perceiver_num_heads = model_args.perceiver_num_heads
    config.perceiver_num_queries = model_args.perceiver_num_queries
    config.conv_template = model_args.version
    config.image_aspect_ratio = data_args.image_aspect_ratio
    logger.info(config)

    logger.info(f"Starting create LLM model from {model_args.model_name_or_path}")
    if model_args.base_model == "tbstars":
        model = TBStarsVLForCausalLM.from_pretrained(
            model_args.model_name_or_path,
            config=config,
            cache_dir=training_args.cache_dir,
            ignore_mismatched_sizes=True,
            **bnb_model_from_pretrained_args
        )
    elif model_args.base_model == "Turing":
        model = TuringVLForCausalLM.from_pretrained(
            model_args.model_name_or_path,
            config=config,
            cache_dir=training_args.cache_dir,
            ignore_mismatched_sizes=model_args.ignore_mismatched_sizes,
            **bnb_model_from_pretrained_args
        )
        if hasattr(model, "enable_flash_attention"):
            logger.info("Enable flash attention to speed up training.")
            model.enable_flash_attention()
    elif model_args.base_model == "Qwen1.5":
        config._attn_implementation = "flash_attention_2"
        model = LlavaQwen2ForCausalLM.from_pretrained(
            model_args.model_name_or_path,
            config=config,
            cache_dir=training_args.cache_dir,
            **bnb_model_from_pretrained_args
        )
    elif model_args.base_model == "Qwen":
        config.use_flash_attn = True
        model = LlavaQwenForCausalLM.from_pretrained(
            model_args.model_name_or_path,
            config=config,
            cache_dir=training_args.cache_dir,
            **bnb_model_from_pretrained_args
        )
    elif model_args.base_model == "mpt":
        config.attn_config["attn_impl"] = training_args.mpt_attn_impl
        model = LlavaMptForCausalLM.from_pretrained(
            model_args.model_name_or_path,
            config=config,
            cache_dir=training_args.cache_dir,
            **bnb_model_from_pretrained_args
        )
    elif model_args.base_model == 'llava':
        # config.attention_bias = False
        # config.attention_dropout = 0.0
        # config.rope_theta = 10000.0
        # model = LlavaLlamaForCausalLM.from_pretrained(
        #         model_args.model_name_or_path,
        #         config=config,
        #         cache_dir=training_args.cache_dir,
        #         attn_implementation=model_args.attn_implementation,
        #         torch_dtype=(torch.bfloat16 if training_args.bf16 else None),
        #         **bnb_model_from_pretrained_args
        #     )
        model = LlavaLlamaForCausalLM.from_pretrained(
            model_args.model_name_or_path,
            cache_dir=training_args.cache_dir,
            attn_implementation=model_args.attn_implementation,
            torch_dtype=(torch.bfloat16 if training_args.bf16 else None),
            **bnb_model_from_pretrained_args
        )
        model.config.use_cache = False
    else:
        model = LlavaLlamaForCausalLM.from_pretrained(
            model_args.model_name_or_path,
            cache_dir=training_args.cache_dir,
            attn_implementation=model_args.attn_implementation,
            torch_dtype=(torch.bfloat16 if training_args.bf16 else None),
            **bnb_model_from_pretrained_args
        )

    logger.info(f"{model.get_vision_tower().config}")
    logger.info(f"{model.get_vision_tower()}")

    model.to(dtype=torch.bfloat16 if training_args.bf16 else torch.float16, device=training_args.device)
    logger.info(model.config)

    if model_args.freeze_backbone:
        model.model.requires_grad_(False)

    if training_args.bits in [4, 8]:
        from peft import prepare_model_for_kbit_training
        model.config.torch_dtype = (torch.float32 if training_args.fp16 else
                                    (torch.bfloat16 if training_args.bf16 else torch.float32))
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=training_args.gradient_checkpointing)

    if training_args.gradient_checkpointing:
        # def make_inputs_require_grad(module, input, output):
        #     output.requires_grad_(True)

        # vision_tower = model.get_vision_tower()
        # vision_tower.to(dtype=torch.bfloat16 if training_args.bf16 else torch.float16, device=training_args.device)
        # if hasattr(vision_tower, "enable_input_require_grads"):
        #     vision_tower.enable_input_require_grads()
        # else:
        #     vision_tower.get_input_embeddings().register_forward_hook(make_inputs_require_grad)

        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
        else:
            model.get_input_embeddings().register_forward_hook(make_inputs_require_grad)

    if training_args.lora_enable:
        from peft import LoraConfig, get_peft_model
        # In case it is frozen by LoRA
        for p in model.get_model().mm_projector.parameters():
            p.requires_grad = True
        lora_config = LoraConfig(
            r=training_args.lora_r,
            lora_alpha=training_args.lora_alpha,
            target_modules=find_all_linear_names(model),
            lora_dropout=training_args.lora_dropout,
            bias=training_args.lora_bias,
            task_type="CAUSAL_LM",
        )
        if training_args.bits == 16:
            if training_args.bf16:
                model.to(torch.bfloat16)
            if training_args.fp16:
                model.to(torch.float16)
        logger.info("Adding LoRA adapters...")
        model = get_peft_model(model, lora_config)

    special_tokens_dict = dict(additional_special_tokens=[
        "<s>", "</s>", "<|im_start|>", "<|im_end|>",
        "<img_url>", "</img_url>", "<img_patch>",
        "<ref>", "</ref>", "<box>", "</box>", "<quad>", "</quad>", "<entity>", "</entity>", "<pred>", "</pred>"
    ])
    if model_args.base_model == "tbstars":
        tokenizer = transformers.AutoTokenizer.from_pretrained(
            model_args.model_name_or_path,
            cache_dir=training_args.cache_dir,
            model_max_length=training_args.model_max_length,
            padding_side="right",
            trust_remote_code=True,
            use_fast=True,
        )
    elif model_args.base_model == "Turing":
        from llava.model.language_model.flot import FlotTokenizerFast
        tokenizer = FlotTokenizerFast.from_pretrained(
            model_args.model_name_or_path,
            cache_dir=training_args.cache_dir,
            model_max_length=training_args.model_max_length,
            padding_side="right",
        )
    elif model_args.base_model == "Qwen1.5":
        special_tokens_dict.update(dict(unk_token="<unk>"))
        tokenizer = transformers.AutoTokenizer.from_pretrained(
            model_args.model_name_or_path,
            cache_dir=training_args.cache_dir,
            model_max_length=training_args.model_max_length,
            padding_side="right",
            use_fast=False,
        )
    elif model_args.base_model == "Qwen":
        tokenizer = transformers.AutoTokenizer.from_pretrained(
            pretrained_model_name_or_path="./llava/model/language_model/qwen",
            model_max_length=training_args.model_max_length,
            padding_side="right",
            use_fast=False,
        )
    elif model_args.base_model == 'llava':
        tokenizer = transformers.AutoTokenizer.from_pretrained(
            model_args.model_name_or_path,
            cache_dir=training_args.cache_dir,
            model_max_length=training_args.model_max_length,
            padding_side="right",
            use_fast=False,
        )
        if model_args.version == "v0":
            if tokenizer.pad_token is None:
                smart_tokenizer_and_embedding_resize(
                    special_tokens_dict=dict(pad_token="[PAD]"),
                    tokenizer=tokenizer,
                    model=model,
                )
        elif model_args.version == "v0.5":
            tokenizer.pad_token = tokenizer.unk_token
        else:
            tokenizer.pad_token = tokenizer.unk_token
            if model_args.version in conversation_lib.conv_templates:
                conversation_lib.default_conversation = conversation_lib.conv_templates[model_args.version]
            else:
                conversation_lib.default_conversation = conversation_lib.conv_templates["vicuna_v1"]

    else:
        tokenizer = transformers.AutoTokenizer.from_pretrained(
            model_args.model_name_or_path,
            cache_dir=training_args.cache_dir,
            model_max_length=training_args.model_max_length,
            padding_side="right",
            use_fast=True if model_args.base_model == "mpt" else False,
        )

    # smart_tokenizer_and_embedding_resize(
    #     special_tokens_dict=special_tokens_dict,
    #     tokenizer=tokenizer,
    #     model=model,
    # )
    model.get_model().initialize_vision_modules(
        model_args=model_args,
        fsdp=training_args.fsdp
    )
    vision_tower = model.get_vision_tower()
    vision_tower.to(dtype=torch.bfloat16 if training_args.bf16 else torch.float16, device=training_args.device)

    data_args.image_processor = vision_tower.image_processor
    data_args.is_multimodal = True

    model.config.image_aspect_ratio = data_args.image_aspect_ratio
    model.config.tokenizer_padding_side = tokenizer.padding_side
    model.config.tokenizer_model_max_length = tokenizer.model_max_length
    logger.info(model.config)
    assert model_args.vision_tower is not None
    # model.get_model().initialize_vision_modules(
    #     model_args=model_args,
    #     fsdp=training_args.fsdp
    # )
    
    # data_args.conv_template = conversation_lib.conv_templates[model_args.version]
    # data_args.image_processor = model.get_vision_tower().image_processor
    # if "adaptive_crop" in data_args.image_aspect_ratio:
    #     from llava.mm_utils import DocImgProcessor
    #     data_args.crop_processor = DocImgProcessor(
    #         image_expand=True if data_args.image_aspect_ratio == "adaptive_crop_global_pad" else False,
    #         image_size=config.image_size
    #     )
    # data_args.is_multimodal = True

    model.config.tune_mm_mlp_adapter = training_args.tune_mm_mlp_adapter = model_args.tune_mm_mlp_adapter
    model.config.tune_vit_from_block = training_args.tune_vit_from_block
    if training_args.training_phase == "pretrain":
        assert model_args.tune_mm_mlp_adapter
        model.requires_grad_(False)
        if training_args.tune_base_model:
            model.get_model().requires_grad_(True)
        for p in model.get_model().mm_projector.parameters():
            p.requires_grad = True
    elif training_args.training_phase == "continue-train":
        assert training_args.tune_vit_from_block is not None
        model.requires_grad_(True)
    elif training_args.training_phase == "finetune":
        assert training_args.tune_vit_from_block is None
        model.requires_grad_(True)
    else:
        raise NotImplementedError(f"Unknown training phase: {training_args.training_phase}")

    model.config.freeze_mm_mlp_adapter = training_args.freeze_mm_mlp_adapter
    if training_args.freeze_mm_mlp_adapter:
        for p in model.get_model().mm_projector.parameters():
            p.requires_grad = False

    # model.config.freeze_perceiver_positions = model_args.freeze_perceiver_positions
    # if model_args.freeze_perceiver_positions and isinstance(model.get_model().mm_projector, Resampler):
    #     model.get_model().mm_projector.pos_embed.requires_grad_(False)

    vision_tower.requires_grad_(False)
    if training_args.tune_vit_from_block is not None:
        if training_args.tune_vit_from_block == -1:  # full finetune, include the patch embedding and pre_layrnorm
            model.get_vision_tower().requires_grad_(True)
        else:
            # freeze the patch embedding and pre_layrnorm layer
            if "open-clip" in model_args.vision_tower.lower():
                raise NotImplementedError
            else:
                vision_config = model.get_vision_tower().vision_tower.vision_model.config
                for idx in range(training_args.tune_vit_from_block, vision_config.num_hidden_layers):
                    model.get_vision_tower().vision_tower.vision_model.encoder.layers[idx].requires_grad_(True)
                model.get_vision_tower().vision_tower.vision_model.post_layernorm.requires_grad_(True)


    if training_args.bits in [4, 8]:
        model.get_model().mm_projector.to(dtype=compute_dtype, device=training_args.device)

    model.config.mm_use_im_start_end = data_args.mm_use_im_start_end = model_args.mm_use_im_start_end
    model.config.mm_projector_lr = training_args.mm_projector_lr
    print('*************mm_projector_lr***************:')
    print(training_args.mm_projector_lr)
    for p in model.get_model().mm_projector.parameters():
        if p.requires_grad == False:
            print('*************do not turn mm_projector_lr************')
    training_args.use_im_start_end = model_args.mm_use_im_start_end
    model.config.mm_use_im_patch_token = model_args.mm_use_im_patch_token
    model.initialize_vision_tokenizer(model_args, tokenizer=tokenizer)
    logger.info("Init vision models completed.")

    if training_args.bits in [4, 8]:
        from peft.tuners.lora import LoraLayer
        for name, module in model.named_modules():
            if isinstance(module, LoraLayer):
                if training_args.bf16:
                    module = module.to(torch.bfloat16)
            if "norm" in name:
                module = module.to(torch.float32)
            if "lm_head" in name or "embed_tokens" in name:
                if hasattr(module, "weight"):
                    if training_args.bf16 and module.weight.dtype == torch.float32:
                        module = module.to(torch.bfloat16)

    for param_name, param in model.named_parameters():
        logger.info(f"\tParameter: {local_rank}, {param.device}, {param.dtype}, {param.requires_grad}, {param_name}")

    model_ref = copy.deepcopy(model)
    for param in model_ref.parameters():
        param.requires_grad = False
    model_ref.model.requires_grad_(False)

    data_module = make_supervised_data_module(tokenizer=tokenizer, data_args=data_args, training_args=training_args)
    trainer = LLaVATrainer(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        callbacks=[LastStepCallback],
        **data_module
    )
    trainer.train(resume_from_checkpoint=training_args.resume_from_checkpoint)

    trainer.save_state()
    model.config.use_cache = True
    if training_args.lora_enable:
        state_dict = get_peft_state_maybe_zero_3(model.named_parameters(), training_args.lora_bias)
        non_lora_state_dict = get_peft_state_non_lora_maybe_zero_3(model.named_parameters())
        if training_args.local_rank == 0 or training_args.local_rank == -1:
            model.config.save_pretrained(training_args.output_dir)
            model.save_pretrained(training_args.output_dir, state_dict=state_dict)
            torch.save(non_lora_state_dict, os.path.join(training_args.output_dir, "non_lora_trainables.bin"))
    else:
        safe_save_model_for_hf_trainer(trainer=trainer, output_dir=training_args.output_dir)


if __name__ == "__main__":
    train()
