#!/usr/bin/env bash
set -x

source "$(conda info --base 2>/dev/null)/etc/profile.d/conda.sh"
conda activate lam

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export HF_HOME="${SCRIPT_DIR}/hf_models"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

export WANDB_PROJECT="star-nemo"
export WANDB_RUN_ID="finetune-lora"
export WANDB_RESUME="allow"
export PYTHONPATH=$PYTHONPATH:$(pwd)/pkg
GPUS=${GPUS:-1}
NNODES=${1:-1}
OUTPUT_DIR=${2:-"datasets/robot-detection/work_dirs/locany_lora"}
NODE_RANK=${NODE_RANK:-0}
PORT=${PORT:-29500}
MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}

PER_DEVICE_BATCH_SIZE=${PER_DEVICE_BATCH_SIZE:-1}
GRADIENT_ACC=1
echo $NODE_RANK

if [ ! -d "$OUTPUT_DIR" ]; then
  mkdir -p "$OUTPUT_DIR"
fi
export NCCL_DEBUG=INFO

script_path=${BASH_SOURCE[0]}
script_name=$(basename "$script_path")
MODEL_PATH=${MODEL_PATH:-"nvidia/LocateAnything-3B"}

LAUNCHER=pytorch torchrun \
    --nnodes=$NNODES \
    --node_rank=$NODE_RANK \
    --master_addr=$MASTER_ADDR \
    --nproc_per_node=$GPUS \
    --master_port=$PORT \
   pkg/eaglevl/train/locany_finetune_magi_stream.py \
  --model_name_or_path ${MODEL_PATH} \
  --output_dir ${OUTPUT_DIR} \
  --meta_path "datasets/robot-detection/locany_recipe/detection_recipe.json" \
  --overwrite_output_dir False \
  --block_size 6 \
  --attn_implementation sdpa \
  --causal_attn False \
  --freeze_llm True \
  --freeze_mlp True \
  --freeze_backbone True \
  --use_llm_lora 16 \
  --use_backbone_lora 8 \
  --vision_select_layer -1 \
  --dataloader_num_workers 2 \
  --bf16 True \
  --num_train_epochs 30 \
  --per_device_train_batch_size ${PER_DEVICE_BATCH_SIZE} \
  --gradient_accumulation_steps ${GRADIENT_ACC} \
  --save_strategy "epoch" \
  --save_total_limit 3 \
  --learning_rate 5e-5 \
  --weight_decay 0.01 \
  --warmup_steps 50 \
  --lr_scheduler_type "cosine" \
  --logging_steps 1 \
  --video_total_pixels 4096 \
  --sample_log_interval 1 \
  --packing_buffer_size 16 \
  --max_seq_length 4096 \
  --max_num_tokens_per_sample 4096 \
  --max_num_tokens 4096 \
  --do_train True \
  --grad_checkpoint True \
  --group_by_length False \
  --report_to "tensorboard"\
  --run_name $script_name \
  --use_onelogger True \
  --mlp_connector_layers 2 \
  2>&1 | tee -a "${OUTPUT_DIR}/training_log.txt"