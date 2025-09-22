#!/bin/bash
# source ~/miniconda3/bin/activate your_env
#!/bin/bash
set -x

MODEL_PATH="ckpt/llava-1.5-7b/"
BASE_HF_MODEL_PATH="ckpt/llava-1.5-7b/"
IS_HF="0"
DATASET_PATH="rlhfv2.json"
IMAGES_DIR="llava_data/sft_data/"
OUTPUT_DIR="dpo_data"
DIVERSITY_PENALTY="5.0"
NUM_BEAMS="5"
NUM_BEAM_GROUP="5"
NUM_TOKEN_BEAMS="5"
MAX_LENGTH="1024"
MAX_NEW_TOKENS="70"
PERIOD_ID="29889"
WEIGHT_MAPPING_PATH="rlhfv3.json"

python llava/eval/sample.py
  --model_path $MODEL_PATH \
  --base_hf_model_path $BASE_HF_MODEL_PATH \
  --is_hf $IS_HF \
  --dataset_path $DATASET_PATH \
  --images_dir $IMAGES_DIR \
  --output_dir $OUTPUT_DIR \
  --diversity_penalty $DIVERSITY_PENALTY \
  --num_beams $NUM_BEAMS \
  --num_beam_group $NUM_BEAM_GROUP \
  --num_token_beams $NUM_TOKEN_BEAMS \
  --max_length $MAX_LENGTH \
  --max_new_tokens $MAX_NEW_TOKENS \
  --period_id $PERIOD_ID \
  --weight_mapping_path $WEIGHT_MAPPING_PATH

