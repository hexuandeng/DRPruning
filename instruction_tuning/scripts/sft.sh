# Multi-nodes are also supported

export NCCL_SOCKET_IFNAME=eth1
export NCCL_IB_GID_INDEX=3
export NCCL_IB_SL=3
export NCCL_NET_GDR_READ=1

export MASTER_ADDR="${CHIEF_IP:=localhost}"
export MASTER_PORT="${MASTER_PORT:=29600}"

cd instruction_tuning
model_name_or_path=$1
output_dir=$2
dataset_name=$3
epoch=$4

torchrun --nnodes $HOST_NUM --node_rank $INDEX --nproc_per_node 8 \
  --master_addr $MASTER_ADDR --master_port $MASTER_PORT  \
  supervised.py \
  --model_name_or_path "${model_name_or_path}" \
  --eval_size 10 \
  --fp16 False \
  --bf16 True \
  --seed 42 \
  --output_dir "${output_dir}" \
  --dataset_path "" \
  --dataset_name "${dataset_name}" \
  --num_train_epochs "${epoch}" \
  --per_device_train_batch_size 2 \
  --per_device_eval_batch_size 4 \
  --gradient_accumulation_steps 8 \
  --eval_steps 100 \
  --save_strategy "epoch" \
  --save_total_limit 1 \
  --learning_rate 5e-5 \
  --weight_decay 0.0 \
  --warmup_ratio 0.03 \
  --lr_scheduler_type "cosine" \
  --evaluation_strategy "steps" \
  --logging_steps 10 \
  --tf32 True \
  --model_max_length 512 \
  --ddp_timeout 1800 \
  --fsdp "full_shard auto_wrap" \
  --fsdp_transformer_layer_cls_to_wrap "LlamaDecoderLayer"
