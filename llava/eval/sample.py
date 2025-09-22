from utils import *
from accelerate.utils import gather_object
import torch
from transformers import AutoProcessor, LlavaForConditionalGeneration
from transformers import AutoTokenizer
import json
from accelerate import Accelerator
from PIL import Image
import os
import argparse
import sys
print(sys.path)
sys.path.append('/checkpoint/binary/train_package')
from llava.model.builder import load_pretrained_model
from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from llava.conversation import conv_templates
from llava.mm_utils import tokenizer_image_token, process_images, get_model_name_from_path
from PIL import Image
import math
from torchvision import transforms
import random
import numpy as np
import tqdm
def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
seed_everything(42)
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

def load_hf_llava_model(model_path):
    # Weights are loaded directly with hf-llava version
    model = LlavaForConditionalGeneration.from_pretrained(model_path, device_map='cpu', torch_dtype=torch.float16)
    model_tokenizer = AutoTokenizer.from_pretrained(model_path)
    base_tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False, padding_side='left') 
    return model, model_tokenizer, base_tokenizer


def load_llava_model(model_path, base_hf_model_path, mapping_path):
    # Weights should be specially loaded with other llava versions
    kwargs = {"device_map": "cuda", "torch_dtype": torch.float16}
    tokenizer,model,image_processor,_= load_pretrained_model(model_path, None, 'llava-1.5-7b', **kwargs)
    # model = LlavaForConditionalGeneration.from_pretrained(base_hf_model_path, device_map='cpu', torch_dtype=torch.float16)
    # model_tokenizer = AutoTokenizer.from_pretrained(model_path)
    # base_tokenizer = AutoTokenizer.from_pretrained(base_hf_model_path, use_fast=False, padding_side='left')
    # state_dicts={}
    # for key, value in model2.state_dict().items():
    #     if key not in state_dicts:
    #         state_dicts[key] = value.data
    # # state_dicts = load_and_merge_models(model_path)
    # with open(mapping_path, 'r', encoding='utf-8') as f1:
    #     mapping_keys = json.load(f1)

    # modified_weights = {}
    # for old_key, value in state_dicts.items():
    #     new_key = mapping_keys.get(old_key, old_key)
    #     modified_weights[new_key] = value
    # modified_weights['language_model.model.embed_tokens.weight'] = model.state_dict()['language_model.model.embed_tokens.weight']
    # modified_weights['language_model.lm_head.weight'] = model.state_dict()['language_model.lm_head.weight']
    # # import copy
    # # model3=copy.deepcopy(model)
    # model.load_state_dict(modified_weights, strict=True)
    return model, tokenizer, image_processor


def sentence_level_beam_search_tree(qid, model, accelerator, processor, tokenizer, initial_text, image, sentence_end_id, max_length, max_new_tokens, num_beams, num_beam_group, token_level_beams, diversity_penalty):
    # root = Node(initial_text, 0, 0)
    # active_nodes = [root]
    input_ids = tokenizer_image_token(initial_text, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).to(model.device)
    image_tensor = process_images([image], processor, model.config)[0]
    # inputs = processor(text=initial_text, images=images, return_tensors="pt").to(model.device)
    with torch.inference_mode():
        output_ids = model.generate(
            input_ids,
            images=image_tensor.unsqueeze(0).half().to(model.device),
            image_sizes=[image.size],
            num_beam_groups=5,
            num_beams=5,
            num_return_sequences=5,
            diversity_penalty=5.0,
            max_new_tokens=1024,
            use_cache=True
        )
    
    outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=True)
    score_list=[]
    log_list=[]
    rank1_list=[]
    rank2_list=[]
    image_tensor_ = pil_to_tensor(image)
    image_noisy = add_diffusion_noise(image_tensor_, 500)
    image_noisy = tensor_to_pil(image_noisy)
    image2=image_noisy
    image_tensor2 = process_images([image2], processor, model.config)[0]

    for ans in outputs:
        ans=ans.strip()
        mm=tokenizer.encode(ans)
        mm=mm[1:]
        mm.append(2)
        input_ids2=torch.cat((input_ids, torch.tensor(mm).unsqueeze(0).to(input_ids.device)), dim=1)
        length=len(mm)
        logits_list=[]
        r1_list=[]
        r2_list=[]
        for i in range(len(mm)-1):
            current_input_ids = input_ids2[:, :-len(mm)+i]
            with torch.inference_mode():
                torch.manual_seed(42)
                output_ids = model.forward(
                    current_input_ids,
                    images=image_tensor.unsqueeze(0).half().cuda(),
                    image_sizes=[image.size])
            with torch.inference_mode():
                torch.manual_seed(42)
                output_ids2 = model.forward(
                    current_input_ids,
                    images=image_tensor2.unsqueeze(0).half().cuda(),
                    image_sizes=[image2.size]
                    )
            target_log=mm[i]
            log=output_ids.logits[0,-1,:]
            log_for_target = log[target_log]
            # rank1 = find_rank(log, target_log)

            log2=output_ids2.logits[0,-1,:]
            log_for_target2 = log2[target_log]
            diff = log_for_target - log_for_target2
            # rank2 = find_rank(log2, target_log)
            logits_list.append(diff.tolist())
            # r1_list.append(log_for_target.tolist())
            # r2_list.append(log_for_target2.tolist())
        avg_score=sum(logits_list)/(length-1)
        log_list.append(logits_list)
        score_list.append(avg_score)
    return [{'qid': qid, 'text': initial_text, 'outputs': outputs, 'score_list': score_list, 'log_list': log_list}]

def eval_model(args):
    accelerator = Accelerator()
    model_path = args.model_path
    base_hf_model_path = args.base_hf_model_path
    mapping_path = args.weight_mapping_path
    output_dir = args.output_dir

    # Load Model
    # processor = AutoProcessor.from_pretrained(base_hf_model_path)
    if args.is_hf:
        model, model_tokenizer, base_tokenizer = load_hf_llava_model(model_path)
    else:
        model, model_tokenizer, processor = load_llava_model(model_path, base_hf_model_path, mapping_path)
    model.to(accelerator.device)

    # Load Dataset
    with open(args.dataset_path, 'r', encoding='utf8') as fp:
        my_dataset = json.load(fp)
    llava_loader = get_llava_dataloader(my_dataset, 1)
    llava_loader, processor = accelerator.prepare(llava_loader, processor)
    rank = int(os.getenv("RANK", "0"))
    answers_file_root = 'llava_beam_ans/llava_answers_rlhfv_beam3'
    answers_file = f"{answers_file_root}-cuda-{rank}.jsonl"
    print(f"the answer file is: {answers_file}")
    os.makedirs(os.path.dirname(answers_file), exist_ok=True)
    ans_file = open(answers_file, "w")
    with torch.no_grad():
        for data in tqdm.tqdm(llava_loader, desc="Loading Data"):
            input_questions = data['input']
            input_questions = [q.replace("<image>\n", "").replace("\n<image>", "").replace("<image>", "") for q in input_questions]
            image_paths = data['image']
            qid = data['question_ids']
            images = []

            for image_path in image_paths:
                images.append(Image.open(os.path.join(args.images_dir, image_path)))

            prompts = get_prompts(input_questions)
            sentence_end_id = int(args.period_id)
            max_length = int(args.max_length)
            token_level_beams = int(args.num_token_beams)
            max_new_tokens = int(args.max_new_tokens)
            diversity_penalty = float(args.diversity_penalty)
            num_beams = int(args.num_beams)
            num_beam_group = int(args.num_beam_group)

            # Batched inference is not supported yet
            result = gather_object(sentence_level_beam_search_tree(
                qid[0],
                model,
                accelerator,
                processor,
                model_tokenizer,
                prompts[0],
                images[0],
                sentence_end_id,
                max_length,
                max_new_tokens,
                num_beams,
                num_beam_group,
                token_level_beams,
                diversity_penalty
            ))
            if accelerator.is_main_process:
                print('##')
            for obj in result:
                ans_file.write(json.dumps({"question_id": obj['qid'],
                                            "prompt": obj['text'],
                                            "ans1": obj['outputs'][0].strip(),
                                            "log1":str(obj['log_list'][0])+str(obj['score_list'][0]),# +' rank_o:'+str(rank1_list[0])+' rank_n:'+str(rank2_list[0]),
                                            "ans2": obj['outputs'][1].strip(),
                                            "log2":str(obj['log_list'][1])+str(obj['score_list'][1]),# +' rank_o:'+str(rank1_list[1])+' rank_n:'+str(rank2_list[1]),
                                            "ans3": obj['outputs'][2].strip(),
                                            "log3":str(obj['log_list'][2])+str(obj['score_list'][2]),# +' rank_o:'+str(rank1_list[2])+' rank_n:'+str(rank2_list[2]),
                                            "ans4": obj['outputs'][3].strip(),
                                            "log4":str(obj['log_list'][3])+str(obj['score_list'][3]),# +' rank_o:'+str(rank1_list[3])+' rank_n:'+str(rank2_list[3]),
                                            "ans5": obj['outputs'][4].strip(),
                                            "log5":str(obj['log_list'][4])+str(obj['score_list'][4]),# +' rank_o:'+str(rank1_list[4])+' rank_n:'+str(rank2_list[4]),
                                            "metadata": {}}) + "\n")

            torch.cuda.empty_cache()
            accelerator.wait_for_everyone()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default='llava-hf/llava-1.5-7b-hf', help="Path to your model")
    parser.add_argument("--base_hf_model_path", type=str, default='llava-hf/llava-1.5-7b-hf', help="Path to huggingface base model")
    parser.add_argument("--is_hf", type=int, default=1, help="If it's a hf model")
    parser.add_argument("--dataset_path", type=str, default='./data/CSR-Prompt-Dataset-12k.json', help="Path to the prompt dataset")
    parser.add_argument("--images_dir", type=str, default="./data/images/train2014", help="Directory to images")
    parser.add_argument("--output_dir", type=str, default="./outputs/sample", help="Path to step1's result")
    parser.add_argument("--diversity_penalty", type=float, default=3.0)
    parser.add_argument("--num_beams", type=int, default=5)
    parser.add_argument("--num_beam_group", type=int, default=5)
    parser.add_argument("--num_token_beams", type=int, default=5)
    parser.add_argument("--max_length", type=int, default=1024)
    parser.add_argument("--max_new_tokens", type=int, default=70)
    parser.add_argument("--period_id", type=int, default=29889)
    parser.add_argument("--weight_mapping_path", type=str, default='./model_mapping/key_mapping_hf_7b.json', help="To load non-hf model specially")
    args = parser.parse_args()

    eval_model(args)
