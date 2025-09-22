import os
import copy
from dataclasses import dataclass, field
import json
from typing import Dict, Optional, Sequence, List, Any, Iterator
import random

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Sampler
import sys
from io import BytesIO
import transformers
from llava.mm_utils import expand2square
from llava.constants import IGNORE_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, \
    DEFAULT_IM_END_TOKEN
from torch.utils.data import Dataset
from llava import conversation as conversation_lib
from llava.preprocess import (
    preprocess_plain,
    preprocess_llama_2,
    preprocess_mpt,
    preprocess_v1,
    preprocess_default,
    preprocess_chatml,
    preprocess_turing,
    preprocess_turing_v1
)
from llava.arguments import DataArguments
from PIL import Image
from transformers.utils import logging

logger = logging.get_logger(__name__)


def preprocess_multimodal(
        sources: Sequence[str],
        data_args: DataArguments
) -> Dict:
    is_multimodal = data_args.is_multimodal
    if not is_multimodal:
        return sources

    for source in sources:
        for sentence in source:
            if DEFAULT_IMAGE_TOKEN in sentence["value"]:
                sentence["value"] = sentence["value"].replace(DEFAULT_IMAGE_TOKEN, "").strip()
                sentence["value"] = DEFAULT_IMAGE_TOKEN + "\n" + sentence["value"]
                sentence["value"] = sentence["value"].strip()
                if "mmtag" in data_args.conv_template.version:
                    sentence["value"] = sentence["value"].replace(
                        DEFAULT_IMAGE_TOKEN, "<Image>" + DEFAULT_IMAGE_TOKEN + "</Image>")
            replace_token = DEFAULT_IMAGE_TOKEN
            if data_args.mm_use_im_start_end:
                replace_token = DEFAULT_IM_START_TOKEN + replace_token + DEFAULT_IM_END_TOKEN
            sentence["value"] = sentence["value"].replace(DEFAULT_IMAGE_TOKEN, replace_token)

    return sources


def preprocess(
        sources: Sequence[str],
        tokenizer: transformers.PreTrainedTokenizer,
        conv_template,
        has_image: bool = False
) -> Dict:
    """
    Given a list of sources, each is a conversation list. This transform:
    1. Add signal "### " at the beginning each sentence, with end signal "\n";
    2. Concatenate conversations together;
    3. Tokenize the concatenated conversation;
    4. Make a deepcopy as the target. Mask human words with IGNORE_INDEX.
    """
    if conv_template.sep_style == conversation_lib.SeparatorStyle.PLAIN:
        return preprocess_plain(sources, tokenizer, conv_template)
    if conv_template.sep_style == conversation_lib.SeparatorStyle.LLAMA_2:
        return preprocess_llama_2(sources, tokenizer, conv_template, has_image=has_image)
    if conv_template.version == "turing":
        return preprocess_turing(sources, tokenizer, conv_template, has_image=has_image)
    if conv_template.version == "turing_v1":
        return preprocess_turing_v1(sources, tokenizer, conv_template, has_image=has_image)
    if conv_template.version.startswith("v1"):
        return preprocess_v1(sources, tokenizer, conv_template, has_image=has_image)
    if conv_template.version == "mpt":
        return preprocess_mpt(sources, tokenizer, conv_template)
    if conv_template.version == "chatml":
        return preprocess_chatml(sources, tokenizer, conv_template, has_image=has_image)

    return preprocess_default(sources, tokenizer, conv_template, has_image)


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
        
def process_images(data_args, image_filepath=None, base64_image=None, base64_id=None):
    if image_filepath is not None:
        image_path = os.path.join(data_args.image_folder, image_filepath)
        try:
            image = Image.open(image_path).convert("RGB")
        except Exception as e:
            print(f'Image.open() exception for image path: {image_path}, exception = {str(e)}', file=sys.stderr)
            image = Image.open(data_args.default_image_path).convert("RGB")
    elif base64_image is not None:
        image_path = base64_id
        try:
            image = Image.open(BytesIO(base64_image)).convert("RGB")
        except Exception as e:
            print(f'Image.open() exception for image path: {image_path}, exception = {str(e)}', file=sys.stderr)
            image = Image.open(data_args.default_image_path).convert("RGB")
    else:
        raise ValueError("You have to specify either image_filepath or image_base64")

    image_processor = data_args.image_processor
    crop_processor = data_args.crop_processor if "adaptive_crop" in data_args.image_aspect_ratio else None
    image_aspect_ratio = data_args.image_aspect_ratio

    try:
        image_processed = _process_image(image, image_aspect_ratio, image_processor, crop_processor, data_args)
    except Exception as e:
        print(f'Image processor exception for image path: {image_path}, exception = {str(e)}', file=sys.stderr)
        image = Image.open(data_args.default_image_path).convert("RGB")
        image_processed = _process_image(image, image_aspect_ratio, image_processor, crop_processor, data_args)

    return image_processed

def process_images_multi(data_args, image_filepaths=None, base64_images=None, base64_ids=None):
    image_processor = data_args.image_processor
    crop_processor = data_args.crop_processor if "adaptive_crop" in data_args.image_aspect_ratio else None
    image_aspect_ratio = data_args.image_aspect_ratio
    
    if isinstance(image_filepaths, str) or image_filepaths is None:
        image_filepaths = [image_filepaths]
    
    images_processed = []
    
    for image_path in image_filepaths:
        if image_path is not None:
            image_full_path = os.path.join(data_args.image_folder, image_path)
            try:
                image = Image.open(image_full_path).convert("RGB")
                image_processed = _process_image(image, image_aspect_ratio, image_processor, crop_processor, data_args)
            except Exception as e:
                print(f'Image processor exception for image path: {image_path}, exception = {str(e)}', file=sys.stderr)
                default_img = Image.open(data_args.default_image_path).convert("RGB")
                image_processed = _process_image(default_img, image_aspect_ratio, image_processor, crop_processor, data_args)
            images_processed.append(image_processed)

    final_output = {}
    for key in images_processed[0].keys():
        final_output[key] = torch.cat([x[key] for x in images_processed], dim=0)
    
    return final_output

class LazySupervisedDataset(Dataset):
    """Dataset for supervised fine-tuning."""

    def __init__(
            self,
            data_path: str,
            tokenizer: transformers.PreTrainedTokenizer,
            data_args: DataArguments
    ):
        super(LazySupervisedDataset, self).__init__()

        self.tokenizer = tokenizer
        self.list_data_dict = [json.loads(_) for _ in list(open(data_path, "r"))] \
            if data_path.endswith(".jsonl") else json.load(open(data_path, "r"))
        self.data_args = data_args

    def __len__(self):
        return len(self.list_data_dict)

    @property
    def lengths(self):
        length_list = []
        for sample in self.list_data_dict:
            img_tokens = 128 if "image" in sample else 0
            length_list.append(sum(len(conv["value"].split()) for conv in sample["conversations"]) + img_tokens)
        return length_list

    @property
    def modality_lengths(self):
        length_list = []
        for sample in self.list_data_dict:
            cur_len = sum(len(conv["value"].split()) for conv in sample["conversations"])
            cur_len = cur_len if "image" in sample else -cur_len
            length_list.append(cur_len)
        return length_list

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        sources = self.list_data_dict[i]
        has_image = True if "image" in sources else False

        if isinstance(i, int):
            sources = [sources]
        assert len(sources) == 1, "Don't know why it is wrapped to a list"  # FIXME
        if has_image:
            image_processed = process_images(self.data_args, image_filepath=sources[0]["image"])
            sources = preprocess_multimodal(copy.deepcopy([e["conversations"] for e in sources]), self.data_args)
        else:
            sources = copy.deepcopy([e["conversations"] for e in sources])

        data_dict = preprocess(
            sources,
            tokenizer=self.tokenizer,
            conv_template=self.data_args.conv_template,
            has_image=has_image
        )
        if isinstance(i, int):
            data_dict = dict(input_ids=data_dict["input_ids"][0], labels=data_dict["labels"][0])

        # image exist in the data
        if has_image:
            data_dict.update(image_processed)
        elif self.data_args.is_multimodal:
            # image does not exist in the data, but the model is multimodal
            input_size = self.data_args.image_processor.crop_size
            image = torch.zeros(3, input_size["height"], input_size["width"])
            if "adaptive_crop" in self.data_args.image_aspect_ratio:
                data_dict["image"] = torch.stack([image, image], dim=0)
                data_dict["crop_positions"] = self.data_args.crop_processor.default_crop_position
            else:
                data_dict["image"] = image
        return data_dict


@dataclass
class DataCollatorForSupervisedDataset(object):
    """Collate examples for supervised fine-tuning."""

    tokenizer: transformers.PreTrainedTokenizer

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        input_ids = [_["input_ids"] for _ in instances]
        labels = [_["labels"] for _ in instances]

        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id
        )
        labels = torch.nn.utils.rnn.pad_sequence(
            labels, batch_first=True, padding_value=IGNORE_INDEX
        )
        input_ids = input_ids[:, :self.tokenizer.model_max_length]
        attention_mask = input_ids.ne(self.tokenizer.pad_token_id)
        labels = labels[:, :self.tokenizer.model_max_length]

        batch = dict(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
        )

        if "image" in instances[0]:
            images = [_["image"] for _ in instances]
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

        return batch


def split_to_even_chunks(indices, lengths, num_chunks):
    """
    Split a list of indices into `chunks` chunks of roughly equal lengths.
    """

    if len(indices) % num_chunks != 0:
        return [indices[i::num_chunks] for i in range(num_chunks)]

    num_indices_per_chunk = len(indices) // num_chunks

    chunks = [[] for _ in range(num_chunks)]
    chunks_lengths = [0 for _ in range(num_chunks)]
    for index in indices:
        shortest_chunk = chunks_lengths.index(min(chunks_lengths))
        chunks[shortest_chunk].append(index)
        chunks_lengths[shortest_chunk] += lengths[index]
        if len(chunks[shortest_chunk]) == num_indices_per_chunk:
            chunks_lengths[shortest_chunk] = float("inf")

    return chunks


def get_modality_length_grouped_indices(lengths, batch_size, world_size, generator=None):
    # We need to use torch for the random part as a distributed sampler will set the random seed for torch.
    assert all(l != 0 for l in lengths), "Should not have zero length."
    if all(l > 0 for l in lengths) or all(l < 0 for l in lengths):
        # all samples are in the same modality
        return get_length_grouped_indices(lengths, batch_size, world_size, generator=generator)
    mm_indices, mm_lengths = zip(*[(i, l) for i, l in enumerate(lengths) if l > 0])
    lang_indices, lang_lengths = zip(*[(i, -l) for i, l in enumerate(lengths) if l < 0])

    mm_shuffle = [mm_indices[i] for i in get_length_grouped_indices(mm_lengths, batch_size, world_size, generator=None)]
    lang_shuffle = [lang_indices[i] for i in get_length_grouped_indices(lang_lengths, batch_size, world_size, generator=None)]
    megabatch_size = world_size * batch_size
    mm_megabatches = [mm_shuffle[i : i + megabatch_size] for i in range(0, len(mm_shuffle), megabatch_size)]
    lang_megabatches = [lang_shuffle[i : i + megabatch_size] for i in range(0, len(lang_shuffle), megabatch_size)]

    last_mm = mm_megabatches[-1]
    last_lang = lang_megabatches[-1]
    additional_batch = last_mm + last_lang
    megabatches = mm_megabatches[:-1] + lang_megabatches[:-1]
    megabatch_indices = torch.randperm(len(megabatches), generator=generator)
    megabatches = [megabatches[i] for i in megabatch_indices]

    if len(additional_batch) > 0:
        megabatches.append(sorted(additional_batch))

    return [i for megabatch in megabatches for i in megabatch]


def get_length_grouped_indices(lengths, batch_size, world_size, generator=None, merge=True):
    # We need to use torch for the random part as a distributed sampler will set the random seed for torch.
    indices = torch.randperm(len(lengths), generator=generator)
    megabatch_size = world_size * batch_size
    megabatches = [indices[i : i + megabatch_size].tolist() for i in range(0, len(lengths), megabatch_size)]
    megabatches = [sorted(megabatch, key=lambda i: lengths[i], reverse=True) for megabatch in megabatches]
    megabatches = [split_to_even_chunks(megabatch, lengths, world_size) for megabatch in megabatches]

    return [i for megabatch in megabatches for batch in megabatch for i in batch]


class LengthGroupedSampler(Sampler):
    r"""
    Sampler that samples indices in a way that groups together features of the dataset of roughly the same length while
    keeping a bit of randomness.
    """

    def __init__(
        self,
        batch_size: int,
        world_size: int,
        lengths: Optional[List[int]] = None,
        generator=None,
        group_by_modality: bool = False,
    ):
        if lengths is None:
            raise ValueError("Lengths must be provided.")

        self.batch_size = batch_size
        self.world_size = world_size
        self.lengths = lengths
        self.generator = generator
        self.group_by_modality = group_by_modality

    def __len__(self):
        return len(self.lengths)

    def __iter__(self):
        if self.group_by_modality:
            indices = get_modality_length_grouped_indices(self.lengths, self.batch_size, self.world_size, generator=self.generator)
        else:
            indices = get_length_grouped_indices(self.lengths, self.batch_size, self.world_size, generator=self.generator)
        return iter(indices)
