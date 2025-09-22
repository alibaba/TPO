import argparse
import torch
import os
import ast
import json
import math
import shortuuid
from tqdm import tqdm
from PIL import Image
from typing import Dict, List, Any
from torch.utils.data import Dataset, DataLoader
from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from llava.conversation import conv_templates
from llava.utils import disable_torch_init
from llava.mm_utils import tokenizer_image_token, process_images, get_model_name_from_path
from transformers import PretrainedConfig, PreTrainedTokenizerBase, AutoTokenizer, AutoConfig
from llava import conversation as conversation_lib
from llava.model import *
# from llava.model.multimodal_encoder.openclip.configuration_openclip import OPEN_CLIP_CONFIG
import logging
from transformers import logging as hf_logging
from transformers import StoppingCriteria, StoppingCriteriaList
import torch.distributed as dist
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
from llava.model.builder import load_pretrained_model

os.environ["TOKENIZERS_PARALLELISM"] = "false"
Image.MAX_IMAGE_PIXELS = 1000000000
# 获取 transformers 的日志记录器
logger = hf_logging.get_logger()
logger.setLevel(logging.INFO)
formatter = logging.Formatter(
    "%(asctime)s.%(msecs)03d - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
for safe_handler in logger.handlers:
    safe_handler.setFormatter(formatter)

def setup_distributed(world_size, rank):
    dist.init_process_group(
        backend='nccl',     # or 'gloo' if you are not using CUDA
        init_method='env://',  # typically 'env://' which reads environment variables
        world_size=world_size,
        rank=rank
    )

def gather_answers(files, output_file):
    with open(output_file, 'w', encoding='utf-8') as outfile:
        for filename in files:
            with open(filename, 'r', encoding='utf-8') as infile:  # 指定编码为utf-8
                for line in infile:
                    outfile.write(line)

def split_list(lst, n):
    """Split a list into n (roughly) equal-sized chunks"""
    chunk_size = math.ceil(len(lst) / n)  # integer division
    return [lst[i:i + chunk_size] for i in range(0, len(lst), chunk_size)]

def get_chunk(lst, n, k):
    chunks = split_list(lst, n)
    return chunks[k]

# Custom dataset class
class CustomDataset(Dataset):
    def __init__(self, prompts, image_folder, tokenizer, model_config, conv_template, image_processor, crop_processor=None):
        self.prompts = prompts
        self.image_folder = image_folder
        self.tokenizer = tokenizer
        self.image_processor = image_processor
        self.crop_processor = crop_processor
        self.model_config = model_config
        self.conv_template = conv_template
        self.defaut_image = 'hallusion/images/black.jpg'

    def __getitem__(self, index):
        
        line = self.prompts[index]
        #print('get_item_line:')
        print(line)
        if 'image' in line:
            image_file = line['image']
        else:
            image_file = ''
        qs = line["text"]
    
        if image_file != '':
            if self.model_config.mm_use_im_start_end:
                qs = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + qs
            else:
                qs = DEFAULT_IMAGE_TOKEN + '\n' + qs

        if self.conv_template == "plain":
            prompt = qs.replace(f"{DEFAULT_IMAGE_TOKEN}\n", DEFAULT_IMAGE_TOKEN)
        else:
            conv = conv_templates[self.conv_template].copy()
            conv.append_message(conv.roles[0], qs)
            conv.append_message(conv.roles[1], None)
            prompt = conv.get_prompt()
            

        if image_file != '':
            image = Image.open(os.path.join(self.image_folder, image_file)).convert('RGB')
            input_ids = tokenizer_image_token(prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt')
        else:
            image = Image.open(self.defaut_image).convert('RGB')
            input_ids = self.tokenizer(
                prompt,
                return_tensors="pt",
            ).input_ids.squeeze()

        if self.crop_processor is not None:
            image_tensor, _, crop_positions = self.crop_processor(image=[image], text=None)
            print('using_crop_processer##########################')
            return input_ids, image_tensor, crop_positions, prompt, image_file
        
        if image_file != '':
            image = Image.open(os.path.join(self.image_folder, image_file)).convert('RGB')
        else:
            image = Image.open(self.defaut_image).convert('RGB')
            #print(image_file)
        image_tensor = process_images([image], self.image_processor, self.model_config)
        #print(input_ids,image_tensor[0],prompt,image_file)
        return input_ids, image_tensor[0], prompt, image_file
    
    def __len__(self):
        return len(self.prompts)


# DataLoader
def create_data_loader(
        prompts: List[Dict],
        image_folder: str,
        image_processor: Any,
        crop_processor: Any,
        tokenizer: PreTrainedTokenizerBase,
        model_config: PretrainedConfig,
        conv_template: str,
        batch_size: int = 1,
        num_workers: int = 4
):
    assert batch_size == 1, "batch_size must be 1"
    dataset = CustomDataset(prompts, image_folder, tokenizer, model_config, conv_template, image_processor, crop_processor)

    data_loader = DataLoader(dataset, batch_size=batch_size, num_workers=num_workers, shuffle=False)
    return data_loader

def make_stopping_criteria_list(tokenizer, stop_token):
    class KeywordStoppingCriteria(StoppingCriteria):
        def __init__(self, keyword_ids: list):
            self.keyword_ids = keyword_ids

        def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
            if input_ids[0][-1] in self.keyword_ids:
                return True
            return False

    stop_ids = [tokenizer.encode(stop_token)[0]]
    return StoppingCriteriaList([KeywordStoppingCriteria(stop_ids)])

def eval_model(args):
    # model
    disable_torch_init()
    
    model_name = get_model_name_from_path(args.model_path)
    config = AutoConfig.from_pretrained(args.model_path, trust_remote_code=True)
    config.update(dict(use_cache=True, finetuned_vit=True, vision_tower=args.vision_tower))
    kwargs = {"device_map": "cuda", "torch_dtype": torch.float16}
    if args.base_model == "tbstars":
        model = TBStarsVLForCausalLM.from_pretrained(args.model_path, config=config, **kwargs)
        tokenizer = AutoTokenizer.from_pretrained(
            pretrained_model_name_or_path=args.model_path,
            model_max_length=config.tokenizer_model_max_length,
            padding_side="right",
            trust_remote_code=True,
            use_fast=True,
        )
    elif args.base_model == "Turing":
        from llava.model.language_model.flot import FlotTokenizerFast
        model = TuringVLForCausalLM.from_pretrained(args.model_path, config=config, **kwargs)
        tokenizer = FlotTokenizerFast.from_pretrained(
            args.model_path,
            model_max_length=config.tokenizer_model_max_length,
            padding_side="right"
        )
    elif args.base_model == "Qwen1.5":
        config._attn_implementation = "flash_attention_2"
        model = LlavaQwen2ForCausalLM.from_pretrained(args.model_path, config=config, **kwargs)
        tokenizer = AutoTokenizer.from_pretrained(
            pretrained_model_name_or_path=args.model_path,
            model_max_length=config.tokenizer_model_max_length,
            padding_side="right",
            use_fast=False,
        )
    elif args.base_model == "Qwen":
        config.use_flash_attn = True
        model = LlavaQwenForCausalLM.from_pretrained(args.model_path, config=config, **kwargs)
        tokenizer = AutoTokenizer.from_pretrained(
            pretrained_model_name_or_path="./llava/model/language_model/qwen",
            model_max_length=4096,
            padding_side="right",
            trust_remote_code=True,
            use_fast=False,
        )
    elif args.base_model == 'llava':
        # if "open-clip" in config.vision_tower.lower():
        #     from llava.utils import load_mos_model
        #     vision_load_dir = '/data/mdl/open-clip-pretrained-model'
        #     print('load vision model from {}'.format(config.vision_tower))
        #     load_mos_model(vision_load_dir,config.vision_tower)
        model_name='llava-1.5-7b'
        if 'lora' in args.model_path:
            model_base="ckpt/llava-1.5-7b/"
            model_name="llava_POVID_stage_two_lora"
        else:
            model_base=None
        tokenizer,model,image_processor,_= load_pretrained_model(args.model_path, model_base, model_name, **kwargs)
    else:
        raise NotImplementedError()

    model.to(torch.float16)

    # dataloader
    prompts = [json.loads(q) for q in open(os.path.expanduser(args.question_file), "r")]
    world_size = int(os.environ.get('WORLD_SIZE'))
    # world_size = 1
    rank = int(os.getenv("RANK", "0"))
    if world_size > 1:
        setup_distributed(world_size, rank)
    prompts = get_chunk(prompts, world_size, rank)

    # image_processor, crop_processor = model.get_vision_tower().image_processor, None
    if "adaptive_crop" in getattr(config, "image_aspect_ratio", "square"):
        from llava.mm_utils import DocImgProcessor
        crop_processor = DocImgProcessor(
            image_expand=True if config.image_aspect_ratio == "adaptive_crop_global_pad" else False,
            image_size=config.image_size
        )

    conv_template = conversation_lib.conv_templates[getattr(config, "conv_template", args.conv_template)]
    if conv_template.sep_style == conversation_lib.SeparatorStyle.PLAIN:
        stop_token = conv_template.sep
    elif conv_template.version in ["turing", "chatml"]:
        stop_token = conv_template.sep
    elif conv_template.version == "turing_v1":
        stop_token = conv_template.sep2
    else:
        # raise NotImplementedError(f"Unknown conversation template for Turing-VL inference: {config.conv_template}")
        stop_token = '</s>'
    stopping_criteria = make_stopping_criteria_list(tokenizer, stop_token=stop_token)

    data_loader = create_data_loader(
        prompts=prompts,
        image_folder=args.image_folder,
        image_processor=image_processor,
        crop_processor=None,
        tokenizer=tokenizer,
        model_config=model.config,
        conv_template=args.conv_template
    )
    answers_file_root = args.answers_file.split('.')[0]
    answers_file = f"{answers_file_root}-cuda-{rank}.jsonl"
    print(f"the answer file is: {answers_file}")
    os.makedirs(os.path.dirname(answers_file), exist_ok=True)
    ans_file = open(answers_file, "w")

    with torch.inference_mode():
        
        for (input_ids, images, prompt, image_file), line in tqdm(zip(data_loader, prompts), total=len(prompts)):
            # print(prompt,image_file)
            input_ids = input_ids.to(device=model.device, non_blocking=True)
            # generate_kwargs = {
            #     'input_ids': input_ids,
            #     'do_sample': True if args.temperature > 0 else False,
            #     'temperature': args.temperature,
            #     'top_p': args.top_p,
            #     'num_beams': args.num_beams,
            #     'max_new_tokens': args.max_new_tokens,
            #     'use_cache': True,
            #     'stopping_criteria': stopping_criteria,
            #     'pad_token_id': tokenizer.pad_token_id
            # }
            # print('###############')
            # print(images)
            if image_file:  # Check if image_file is not None
                images = images.to(dtype=torch.float16, device=model.device, non_blocking=True)
                # if crop_positions is not None:
                #     images = images.squeeze(0)
                #     crop_positions = crop_positions.to(device=model.device, non_blocking=True).squeeze(0)
                # generate_kwargs['images'] = images 
            for name, param in model.named_parameters():
                if param.data.sum().item() == 0:
                    print(name)
            with torch.inference_mode():
                # output_ids = model.generate(**generate_kwargs)
                # print(model)
                output_ids = model.generate(
                    input_ids,
                    images=images,
                    do_sample=True if args.temperature > 0 else False,
                    temperature=args.temperature,
                    top_p=args.top_p,
                    num_beams=args.num_beams,
                    max_new_tokens=128,
                    use_cache=True,
                    stopping_criteria=stopping_criteria
                )
            # print(output_ids)
            # print(tokenizer.batch_decode(output_ids[:, :], skip_special_tokens=True)[0])
            input_token_len = input_ids.shape[1]
            # n_diff_input_output = (input_ids != output_ids[:, :input_token_len]).sum().item()
            # if n_diff_input_output > 0:
            #     print(f'[Warning] {n_diff_input_output} output_ids are not the same as the input_ids')
            # outputs = tokenizer.batch_decode(output_ids[:, input_token_len:], skip_special_tokens=True)[0]
            outputs = tokenizer.batch_decode(output_ids[:, :], skip_special_tokens=True)[0]
            outputs = outputs.strip()
            print(outputs)
            # print(prompt, outputs)
            ans_id = shortuuid.uuid()
            prediction = {
                "answer": outputs,
                "prompt": prompt,
                "answer_id": ans_id,
                "model_id": model_name,
                "image": image_file if image_file else '',
                "metadata": {}
            }
            #print(prediction)
            ans_file.write(json.dumps(prediction, ensure_ascii=False) + "\n")
            # ans_file.flush()
        ans_file.close()
        # torch.distributed.barrier()

    # if rank == 0:
    #     all_files = [f"{answers_file_root}-cuda-{rank_idx}.jsonl" for rank_idx in range(world_size)]
    #     gather_answers(all_files, os.path.expanduser(f"{answers_file_root}.jsonl"))
    #     for rank_file in all_files:
    #         os.remove(rank_file)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--base-model", type=str, default="Turing", choices=["tbstars", "Turing", "Qwen", "Qwen1.5", "mpt", "llama","llava"]
    )
    parser.add_argument("--model-path", type=str, default="facebook/opt-350m")
    parser.add_argument("--vision-tower", type=str, default="/tmp/oss_data/clip-vit-large-patch14-336")
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--image-folder", type=str, default="")
    parser.add_argument("--question-file", type=str, default="tables/question.jsonl")
    parser.add_argument("--answers-file", type=str, default="answer.jsonl")
    parser.add_argument("--conv-template", type=str, default="v1")
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--top_p", type=float, default=None)
    parser.add_argument("--num_beams", type=int, default=1)
    parser.add_argument("--max_new_tokens", type=int, default=1000)
    parser.add_argument("--has-timestamp", default=False, type=ast.literal_eval)

    args = parser.parse_args()
    eval_model(args)
