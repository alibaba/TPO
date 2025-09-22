#!/bin/bash
set -x

# ============================ 用户需配置参数 (User-configurable settings) ============================
# 本地训练使用的GPU数量
WORLD_SIZE=8
# 本地GPU类型 (仅用于命名输出文件夹，不影响训练)
GPU_TYPE="A100"
# 训练入口文件
ENTRY_FILE="llava/train/scaling_train_copy.py"
# DeepSpeed 配置文件路径
deepspeed="scripts/general/zero2.json"

# --- 模型和数据路径 (请根据您的本地环境修改) ---
# 预训练模型加载路径
LOAD_MODEL="./checkpoints/llava-v1.5-7b/"
# 训练数据json文件路径
data_path="/tpo/offline_dpo_rlhfv_5K_reform.json"
# 训练图片文件夹路径
image_folder="./data/sft_data/"
# Vision Tower (CLIP) 模型路径
vision_tower="./checkpoints/clip-vit-large/"

# --- 输出和日志路径 ---
# 本次训练的版本名 (用于创建唯一的输出和日志文件夹)
VERSION="local_tpo_${WORLD_SIZE}x${GPU_TYPE}"
# 模型检查点输出目录
output_dir="./output/checkpoints/${VERSION}"
# 日志和其它训练状态保存目录
log_dir="./output/logs/${VERSION}"


# ============================ 模型训练超参数 (Model Training Hyperparameters) ============================
# 训练阶段
training_phase="finetune"
# conversation 版本
conversation_version="v1"
# 基座模型类型
base_model="llava"
# 是否微调语言模型基座
tune_base_model="False"
# "mos" 是星云平台特定类型, 本地加载通常不需要或可设为 "local"
# 请确保您的训练代码能正确处理此参数或忽略它
load_model_type="local" 

mm_projector_type="mlp2x_gelu"
perceiver_num_heads=32
perceiver_num_queries=144
freeze_perceiver_positions="True"
mm_vision_select_layer=-2
mm_use_im_start_end="False"
mm_use_im_patch_token="False"
image_aspect_ratio="pad"
group_by_modality_length="True"
is_multimodal="True"
bf16="True"
num_train_epochs=4
per_device_train_batch_size=2
per_device_eval_batch_size=1
gradient_accumulation_steps=1
evaluation_strategy="no"
save_strategy="steps"
save_steps=200
eval_steps=200
save_total_limit=10
learning_rate=5e-8
end_lr=1e-9
weight_decay=0.0
warmup_steps=20
lr_scheduler_type="cosine"
logging_steps=1
tf32="True"
model_max_length=4096
gradient_checkpointing="True"
dataloader_num_workers=4
lazy_preprocess="True"
report_to="tensorboard"
log_level="info"

# Loss Spike 相关参数
skip_loss_spike="False"
loss_buffer_size=32
loss_spike_threshold=1.5
save_bad_cases="False"
topk_bad_cases=$((per_device_train_batch_size / 4))

# 是否从检查点恢复训练
resume="False"

# ============================ 组合所有参数 (Assemble all arguments) ============================
# 注意: 将原 --oss_save_dir 参数名保留，但路径指向本地的 ${log_dir}
args="
    --training_phase ${training_phase} \
    --seed 42 \
    --deepspeed ${deepspeed} \
    --ddp_timeout=1800000 \
    --data_path ${data_path} \
    --base_model ${base_model} \
    --tune_base_model ${tune_base_model} \
    --model_name_or_path ${LOAD_MODEL} \
    --load_model_type ${load_model_type} \
    --resume ${resume} \
    --eval_steps ${eval_steps} \
    --vision_tower ${vision_tower} \
    --output_dir ${output_dir} \
    --version ${conversation_version} \
    --image_folder ${image_folder} \
    --oss_save_dir ${log_dir} \
    --skip_loss_spike ${skip_loss_spike} \
    --loss_buffer_size ${loss_buffer_size} \
    --loss_spike_threshold ${loss_spike_threshold} \
    --save_bad_cases ${save_bad_cases} \
    --topk_bad_cases ${topk_bad_cases} \
    --mm_projector_type ${mm_projector_type} \
    --perceiver_num_heads ${perceiver_num_heads} \
    --perceiver_num_queries ${perceiver_num_queries} \
    --freeze_perceiver_positions ${freeze_perceiver_positions} \
    --mm_vision_select_layer ${mm_vision_select_layer} \
    --mm_use_im_start_end ${mm_use_im_start_end} \
    --mm_use_im_patch_token ${mm_use_im_patch_token} \
    --image_aspect_ratio ${image_aspect_ratio} \
    --group_by_modality_length ${group_by_modality_length} \
    --is_multimodal ${is_multimodal} \
    --bf16 ${bf16} \
    --num_train_epochs ${num_train_epochs} \
    --per_device_train_batch_size ${per_device_train_batch_size} \
    --per_device_eval_batch_size ${per_device_eval_batch_size} \
    --gradient_accumulation_steps ${gradient_accumulation_steps} \
    --evaluation_strategy ${evaluation_strategy} \
    --save_strategy ${save_strategy} \
    --save_steps ${save_steps} \
    --save_total_limit ${save_total_limit} \
    --learning_rate ${learning_rate} \
    --end_lr ${end_lr} \
    --weight_decay ${weight_decay} \
    --warmup_steps ${warmup_steps} \
    --lr_scheduler_type ${lr_scheduler_type} \
    --logging_steps ${logging_steps} \
    --tf32 ${tf32} \
    --model_max_length ${model_max_length} \
    --gradient_checkpointing ${gradient_checkpointing} \
    --dataloader_num_workers ${dataloader_num_workers} \
    --lazy_preprocess ${lazy_preprocess} \
    --report_to ${report_to} \
    --log_level ${log_level}"

# ============================ 启动训练 (Launch Training) ============================
# 使用 accelerate launch 启动多卡训练
# --multi_gpu 会自动检测可用的GPU
# --num_processes 可以手动指定使用的GPU数量
# 如果您的 accelerate 环境配置好了，可以直接使用 --multi_gpu
# 如果没有，可以使用 --num_processes=${WORLD_SIZE}

echo "Starting training with ${WORLD_SIZE} GPUs..."

accelerate launch --multi_gpu llava/train/scaling_train.py ${args}
