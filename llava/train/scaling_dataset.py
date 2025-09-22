import os
import copy
from dataclasses import dataclass, field
import json
from typing import Dict, Optional, Sequence, List, Any, Iterator
import random
import torch
import transformers
from torch.utils.data import Sampler
import sys
from io import BytesIO
from contextlib import contextmanager
from llava.mm_utils import expand2square
from llava.constants import IGNORE_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, \
    DEFAULT_IM_END_TOKEN, IMAGE_TOKEN_INDEX
from torch.utils.data import Dataset
from llava.dataset import preprocess_multimodal, preprocess, process_images, process_images_multi
from llava.arguments import DataArguments
from PIL import Image
from transformers.utils import logging
import numpy as np
import torch.nn.functional as F
import torch.distributed as dist

logger = logging.get_logger(__name__)
is_local_first_process = int(os.environ.get("LOCAL_RANK", "0")) == 0
is_global_first_process = int(os.environ.get("RANK", "0")) == 0

@contextmanager
def local_main_process_first():
    """The main process runs first, followed by other processes."""
    if not is_local_first_process:
        dist.barrier()
    yield
    if is_local_first_process:
        dist.barrier()

def convert_function(sample, **kwargs):  # TODO: Adapt to multi-image scenarios
    tokenizer = kwargs["tokenizer"]
    data_args = kwargs["data_args"]

    if data_args.ailake_conv == "eval":
        idx, conversations, base64_image = sample[0], sample[2], sample[-1]
        sources = {"conversations": json.loads(json.dumps(eval(conversations)))}
    elif data_args.ailake_conv == "plain":
        idx, caption, base64_image = sample[0], sample[4], sample[-1]
        sources = {"conversations": [
            {"from": "human", "value": f"{DEFAULT_IMAGE_TOKEN}\n"},
            {"from": "gpt", "value": caption}
        ]}
    elif data_args.ailake_conv == "1-turn":
        idx, instruction, response, base64_image = sample[0], sample[5], sample[6], sample[-1]
        sources = {"conversations": [
            {"from": "human", "value": instruction},
            {"from": "gpt", "value": response}
        ]}
    else:
        raise NotImplementedError()
    has_image = True if base64_image else False

    sources = [sources]
    assert len(sources) == 1, "Don't know why it is wrapped to a list"  # FIXME

    if has_image:
        image_processed = process_images(data_args, base64_image=base64_image, base64_id=idx)
        sources = preprocess_multimodal(copy.deepcopy([e["conversations"] for e in sources]), data_args)
    else:
        sources = copy.deepcopy([e["conversations"] for e in sources])

    data_dict = preprocess(
        sources,
        tokenizer=tokenizer,
        conv_template=data_args.conv_template,
        has_image=has_image
    )
    data_dict = dict(input_ids=data_dict["input_ids"][0], labels=data_dict["labels"][0])

    data_dict["id"] = torch.tensor(tokenizer(str(idx)).input_ids, dtype=torch.long).view(-1)
    # image exist in the data
    if has_image:
        data_dict.update(image_processed)
    elif data_args.is_multimodal:
        # image does not exist in the data, but the model is multimodal
        input_size = data_args.image_processor.crop_size
        image = torch.zeros(3, input_size["height"], input_size["width"])
        if "adaptive_crop" in data_args.image_aspect_ratio:
            data_dict["image"] = torch.stack([image, image], dim=0)
            data_dict["crop_positions"] = data_args.crop_processor.default_crop_position
        else:
            data_dict["image"] = image

    return data_dict


class ScalingLazySupervisedDataset(Dataset):
    """Dataset for large-scale pretrain."""

    def __init__(
            self,
            data_path: str,
            tokenizer: transformers.PreTrainedTokenizer,
            data_args: DataArguments
    ):
        super(ScalingLazySupervisedDataset, self).__init__()

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
            cur_len = sample["length"] if "length" in sample \
                else sum(len(conv["value"].split()) for conv in sample["conversations"])
            img_tokens = 256 if "image" in sample or sample.get("has_image", False) else 0
            length_list.append(cur_len + img_tokens)
        return length_list

    @property
    def modality_lengths(self):
        length_list = []
        for sample in self.list_data_dict:
            cur_len = sample["length"] if "length" in sample \
                else sum(len(conv["value"].split()) for conv in sample["conversations"])
            cur_len = cur_len if "image" in sample or sample.get("has_image", False) else -cur_len
            length_list.append(cur_len)
        return length_list

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        if "uid" in self.list_data_dict[i] and self.data_args.meta_path_template is not None:
            try:
                meta_file = self.data_args.meta_path_template.format(self.list_data_dict[i]["uid"])
                with open(meta_file, "r") as f:
                    sources = json.load(f)
            except Exception as e:
                print(f'json.load() exception for meta file path: {meta_file}, exception = {str(e)}', file=sys.stderr)
                with open(self.data_args.default_meta_path, "r") as f:
                    sources = json.load(f)
            has_image = self.list_data_dict[i]["has_image"]
            record_id = copy.deepcopy(sources["id"])
        else:
            sources = self.list_data_dict[i]
            has_image = True if "image" in sources else False
            record_id = copy.deepcopy(sources.get("id", "UNKNOWN_RECORD_ID"))

        if has_image and "/dataset_multitask/commonCrawl/PDF/3000-4000/" in sources["image"]:
            sources["image"] = sources["image"].replace("/dataset_multitask/commonCrawl/PDF/3000-4000/", "/dataset_multitask/commonCrawl/PDF/3000-3999/")

        if isinstance(i, int):
            sources = [sources]
        assert len(sources) == 1, "Don't know why it is wrapped to a list"  # FIXME
        if has_image:
            image_filepaths = sources[0]['image']
            # image_processed = process_images(self.data_args, image_filepath=sources[0]["image"])
            image_processed = process_images_multi(self.data_args, image_filepaths=sources[0]["image"])
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
            if self.data_args.image_aspect_ratio == "NaViT":
                data_dict["image"] = torch.zeros(3, 392, 392)
                data_dict["patch_attention_mask"] = torch.ones(size=(28, 28), dtype=torch.long)
            elif "adaptive_crop" in self.data_args.image_aspect_ratio:
                input_size = self.data_args.image_processor.crop_size
                image = torch.zeros(3, input_size["height"], input_size["width"])
                data_dict["image"] = torch.stack([image, image], dim=0)
                data_dict["crop_positions"] = self.data_args.crop_processor.default_crop_position
            else:
                input_size = self.data_args.image_processor.crop_size
                image = torch.zeros(3, input_size["height"], input_size["width"])
                data_dict["image"] = image

        data_dict["id"] = torch.tensor(self.tokenizer(str(record_id)).input_ids, dtype=torch.long).view(-1)
        return data_dict


class ScalingPackingLazySupervisedDataset(Dataset):
    """Dataset for large-scale pretrain with packing."""

    def __init__(
            self,
            data_path: str,
            tokenizer: transformers.PreTrainedTokenizer,
            data_args: DataArguments
    ):
        super(ScalingLazySupervisedDataset, self).__init__()

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
            cur_len = sample["length"] if "length" in sample \
                else sum(len(conv["value"].split()) for conv in sample["conversations"])
            img_tokens = 256 if "image" in sample or sample.get("has_image", False) else 0
            length_list.append(cur_len + img_tokens)
        return length_list

    @property
    def modality_lengths(self):
        length_list = []
        for sample in self.list_data_dict:
            cur_len = sample["length"] if "length" in sample \
                else sum(len(conv["value"].split()) for conv in sample["conversations"])
            cur_len = cur_len if "image" in sample or sample.get("has_image", False) else -cur_len
            length_list.append(cur_len)
        return length_list

    def tokenize_func(self, i) -> Dict[str, torch.Tensor]:
        if "uid" in self.list_data_dict[i] and self.data_args.meta_path_template is not None:
            try:
                meta_file = self.data_args.meta_path_template.format(self.list_data_dict[i]["uid"])
                with open(meta_file, "r") as f:
                    sources = json.load(f)
            except Exception as e:
                print(f'json.load() exception for meta file path: {meta_file}, exception = {str(e)}', file=sys.stderr)
                with open(self.data_args.default_meta_path, "r") as f:
                    sources = json.load(f)
            has_image = self.list_data_dict[i]["has_image"]
            record_id = copy.deepcopy(sources["id"])
        else:
            sources = self.list_data_dict[i]
            has_image = True if "image" in sources else False
            record_id = copy.deepcopy(sources.get("id", "UNKNOWN_RECORD_ID"))

        if has_image and "/dataset_multitask/commonCrawl/PDF/3000-4000/" in sources["image"]:
            sources["image"] = sources["image"].replace("/dataset_multitask/commonCrawl/PDF/3000-4000/", "/dataset_multitask/commonCrawl/PDF/3000-3999/")

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
            if self.data_args.image_aspect_ratio == "NaViT":
                data_dict["image"] = torch.zeros(3, 392, 392)
                data_dict["patch_attention_mask"] = torch.ones(size=(28, 28), dtype=torch.long)
            elif "adaptive_crop" in self.data_args.image_aspect_ratio:
                input_size = self.data_args.image_processor.crop_size
                image = torch.zeros(3, input_size["height"], input_size["width"])
                data_dict["image"] = torch.stack([image, image], dim=0)
                data_dict["crop_positions"] = self.data_args.crop_processor.default_crop_position
            else:
                input_size = self.data_args.image_processor.crop_size
                image = torch.zeros(3, input_size["height"], input_size["width"])
                data_dict["image"] = image

        data_dict["id"] = torch.tensor(self.tokenizer(str(record_id)).input_ids, dtype=torch.long).view(-1)
        return data_dict

    def packing_dataset(self):
        with local_main_process_first():
            self.tokenized_data = self.list_data_dict.map(
                self.tokenize_func,
                batched=True,
                num_proc=1,
                desc="Tokenizing"
            )

            self.tokenized_data = self.tokenized_data.filter(
                lambda d: d["input_ids"] != [] and self.tokenizer.model_max_length >= len(d["input_ids"]) > 3
            )
            self.unpacked_data = copy.deepcopy(self.tokenized_data)
            self.tokenized_data = self.ungrouped_data.map(
                self.group_by_multimodal_length,
                batched=True,
                num_proc=1,
                remove_columns=self.ungrouped_data.column_names,
                desc="Multimodal-Packing",
            )
            
    def group_by_multimodal_length(self, batch):
        """Concatenate sequences with similar length to minimize padding"""
        input_ids = batch["input_ids"]
        labels = batch["labels"]
        
        if self.data_args.is_multimodal:
            images = batch['image']
            if "adaptive_crop" in self.data_args.image_aspect_ratio:
                crop_positions = batch['crop_positions']

        inputids_labels_pair = list(zip(input_ids, labels))
        input_ids, labels = zip(*inputids_labels_pair)
        input_ids, labels = list(input_ids), list(labels)

        input_ids_buckets = []
        labels_buckets = []
        input_id_bucket = []
        label_bucket = []

        for input_id, label in zip(input_ids, labels):
            if len(input_id_bucket) == 0:
                input_id_bucket = copy.deepcopy(input_id)
                label_bucket = copy.deepcopy(label)
            elif len(input_id_bucket) + len(input_id) <= self.tokenizer.model_max_length:
                input_id_bucket += input_id
                label_bucket += label
            else:
                input_ids_buckets.append(input_id_bucket)
                input_id_bucket = copy.deepcopy(input_id)
                labels_buckets.append(label_bucket)
                label_bucket = copy.deepcopy(label)

        input_ids_buckets.append(input_id_bucket)
        labels_buckets.append(label_bucket)

        return {"input_ids": input_ids_buckets, "labels": labels_buckets}
    
    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        input_ids = torch.tensor(self.tokenized_data[i]["input_ids"])
        labels = torch.tensor(self.tokenized_data[i]["labels"])
        return {"input_ids": input_ids, "labels": labels}




@dataclass
class DataCollatorForScalingSupervisedDataset(object):
    """Collate examples for supervised fine-tuning."""

    tokenizer: transformers.PreTrainedTokenizer

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:

        record_ids_padding_length = 30

        record_ids = [_["id"] for _ in instances]
        input_ids = [_["input_ids"] for _ in instances]
        labels = [_["labels"] for _ in instances]

        if getattr(self.tokenizer, "padding_side", "right") == "left":
            record_ids = torch.nn.utils.rnn.pad_sequence(
                [_[::-1] for _ in record_ids], batch_first=True, padding_value=self.tokenizer.pad_token_id
            ).flip(dims=[1])
            input_ids = torch.nn.utils.rnn.pad_sequence(
                [_[::-1] for _ in input_ids], batch_first=True, padding_value=self.tokenizer.pad_token_id
            ).flip(dims=[1])
            labels = torch.nn.utils.rnn.pad_sequence(
                [_[::-1] for _ in labels], batch_first=True, padding_value=IGNORE_INDEX
            ).flip(dims=[1])
        else:
            record_ids = torch.nn.utils.rnn.pad_sequence(
                record_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id
            )
            input_ids = torch.nn.utils.rnn.pad_sequence(
                input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id
            )
            labels = torch.nn.utils.rnn.pad_sequence(
                labels, batch_first=True, padding_value=IGNORE_INDEX
            )

        record_ids = record_ids[:, :record_ids_padding_length]
        input_ids = input_ids[:, :self.tokenizer.model_max_length]
        attention_mask = input_ids.ne(self.tokenizer.pad_token_id)
        labels = labels[:, :self.tokenizer.model_max_length]

        batch = dict(
            record_ids=record_ids,
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


        # sample = dict(
        #     input_ids=batch["input_ids"][:1, :],
        #     attention_mask=batch["attention_mask"][:1, :],
        #     labels=batch["labels"][:1, :],
        #     images=batch["images"][:1, :],
        # )
        # print("input_ids:", sample["input_ids"][:, :256])
        # print("labels:", sample["labels"][:, :256])
        # print("images shape:", sample["images"].shape)
        #
        # sample["input_ids"][sample["input_ids"] == IMAGE_TOKEN_INDEX] = self.tokenizer.unk_token_id
        # print("input_text:", self.tokenizer.batch_decode(sample["input_ids"])[0])
        #
        # sample["labels"][sample["labels"] == IGNORE_INDEX] = self.tokenizer.pad_token_id
        # print("labels_text:", self.tokenizer.batch_decode(sample["labels"])[0])

        return batch


class DistributedLengthGroupedSampler(Sampler):
    r"""
    Distributed Sampler that samples indices in a way that groups together features of the dataset of roughly the same
    length while keeping a bit of randomness.
    """

    # Copied and adapted from PyTorch DistributedSampler.
    def __init__(
            self,
            batch_size: int,
            world_size: int,
            lengths: List[int],
            seed: Optional[int] = 42,
            generator=None
    ):
        self.batch_size = batch_size
        self.world_size = world_size

        if isinstance(lengths, torch.Tensor):
            logger.info(
                "If lengths is a torch.Tensor, LengthGroupedSampler will be slow. Converting lengths to List[int]..."
            )
            lengths = lengths.tolist()

        self.lengths = lengths
        self.generator = generator
        self.seed = seed

        print(f"******************* Init Sampler: WORLD_SIZE {self.world_size}, SEED {self.seed} ")

    def shuffle_chunks(self, indices, num_indices_per_chunk):
        """
        Split a list of indices into `chunks` chunks of roughly equal lengths.
        """
        if len(indices) % num_indices_per_chunk != 0: return indices
        chunks = [indices[i: i + num_indices_per_chunk] for i in range(0, len(indices), num_indices_per_chunk)]
        random.Random(self.seed).shuffle(chunks)
        return [i for chunk in chunks for i in chunk]

    def get_length_grouped_indices(self, lengths, batch_size, mega_batch_mult=None, generator=None):
        """
        Return a list of indices so that each slice of `batch_size` consecutive indices correspond to elements of similar
        lengths. To do this, the indices are:

        - randomly permuted
        - grouped in mega-batches of size `mega_batch_mult * batch_size`
        - sorted by length in each mega-batch

        The result is the concatenation of all mega-batches, with the batch of `batch_size` containing the element of
        maximum length placed first, so that an OOM happens sooner rather than later.
        """
        # Default for mega_batch_mult: 50 or the number to get 4 megabatches, whichever is smaller.
        if mega_batch_mult is None:
            mega_batch_mult = min(len(lengths) // (batch_size * self.world_size * 4), self.world_size * 50)
            # Just in case, for tiny datasets
            if mega_batch_mult == 0:
                mega_batch_mult = 1

        # We need to use torch for the random part as a distributed sampler will set the random seed for torch.
        indices = torch.randperm(len(lengths), generator=generator)
        megabatch_size = batch_size * mega_batch_mult
        megabatches = [indices[i: i + megabatch_size].tolist() for i in range(0, len(lengths), megabatch_size)]
        megabatches = [sorted(megabatch, key=lambda i: lengths[i], reverse=True) for megabatch in megabatches]
        megabatches = [self.shuffle_chunks(megabatch, self.world_size * batch_size) for megabatch in megabatches]
        return [i for megabatch in megabatches for i in megabatch]

    def __len__(self):
        return len(self.lengths)

    def __iter__(self) -> Iterator:
        indices = self.get_length_grouped_indices(self.lengths, self.batch_size, generator=self.generator)
        return iter(indices)
