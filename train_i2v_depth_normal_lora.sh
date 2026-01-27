#!/bin/bash
# HuggingFace offline mode - skip network checks
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1

export TORCH_LOGS="+dynamo,recompiles,graph_breaks"
export TORCHDYNAMO_VERBOSE=1
export WANDB_MODE="offline"
export NCCL_P2P_DISABLE=1
export TORCH_NCCL_ENABLE_MONITORING=0
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=1

# Add NCCL debug and optimization settings
export NCCL_DEBUG=INFO
export NCCL_IB_DISABLE=1
export NCCL_SOCKET_IFNAME=eth0
export NCCL_IB_TIMEOUT=1800
export NCCL_SOCKET_TIMEOUT=1800
export TORCH_NCCL_BLOCKING_WAIT=1

NUM_GPUS=$(nvidia-smi -L | wc -l)
PORT=$(shuf -i 20000-60000 -n 1)
MASTER_ADDR=$HOSTNAME
MASTER_PORT=$SLURM_JOB_ID
NNODES=$SLURM_JOB_NUM_NODES
NODE_RANK=$SLURM_NODEID

# Optional: Use gloo backend if NCCL fails (uncomment if needed)
# export TORCH_DISTRIBUTED_BACKEND=gloo

NUM_GPUS=$(nvidia-smi -L | wc -l)

torchrun --nproc_per_node=$NUM_GPUS --master_port=29500 \
  tesseract/i2v_depth_normal_lora.py \
  --pretrained_model_name_or_path THUDM/CogVideoX-5b-I2V \
  --dataset_file data-cache.jsonl \
  --output_dir ./output/lora \
  --height_buckets 240 256 480 512 720 \
  --width_buckets 320 512 640 854 1280 \
  --frame_buckets 9 17 25 33 49 \
  --dataloader_num_workers 2 \
  --pin_memory \
  --seed 42 \
  --height 480 \
  --width 640 \
  --guidance_scale 7.5 \
  --max_num_frames 49 \
  --train_batch_size 1 \
  --max_train_steps 200000 \
  --checkpointing_steps 200 \
  --checkpoints_total_limit 15 \
  --validation_steps 200 \
  --gradient_accumulation_steps 1 \
  --learning_rate 5e-5 \
  --lr_scheduler constant_with_warmup \
  --lr_warmup_steps 200 \
  --lr_num_cycles 1 \
  --noised_image_dropout 0.05 \
  --rank 512 \
  --lora_alpha 512 \
  --mixed_precision bf16 \
  --gradient_checkpointing \
  --enable_slicing \
  --enable_tiling \
  --optimizer adamw \
  --beta1 0.9 \
  --beta2 0.95 \
  --weight_decay 0.001 \
  --max_grad_norm 1.0 \
  --allow_tf32 \
  --report_to wandb \
  --ignore_learned_positional_embeddings \
  --nccl_timeout 1800 \
  --resume_from_checkpoint latest
