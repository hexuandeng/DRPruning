# Multi-nodes are also supported

export NCCL_SOCKET_IFNAME=eth1
export NCCL_IB_GID_INDEX=3
export NCCL_IB_SL=3
export NCCL_NET_GDR_READ=1

export MASTER_ADDR="${CHIEF_IP:=localhost}"
export MASTER_PORT="${MASTER_PORT:=29500}"

train_path=drpruning/hf_train.py
model_path=$1
update_type=$2
dataset_path=$3
model_save=$4
domains=$5
proportion=$6
reference_loss=$7

# HOST_NUM will be 1
torchrun --nnodes $HOST_NUM --node_rank $INDEX --nproc_per_node 8 \
    --master_addr $MASTER_ADDR --master_port $MASTER_PORT  \
    ${train_path} \
    --model_name_or_path ${model_path} \
    --dataset_path ${dataset_path} \
    --output_dir ${model_save} \
    --domains ${domains} \
    --proportion ${proportion} \
    --reference_loss ${reference_loss} \
    --update_type ${update_type} \
    --dataloader_pin_memory True \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 32 \
    --max_steps 48000 \
    --learning_rate 1e-4 \
    --weight_decay 0.0 \
    --adam_beta1 0.9 \
    --adam_beta2 0.95 \
    --adam_epsilon 1e-8 \
    --max_grad_norm 1.0 \
    --lr_scheduler_type "cosine" \
    --warmup_ratio 0.03 \
    --logging_steps 100 \
    --block_size 4096 \
    --do_train True \
    --do_eval True \
    --evaluation_strategy steps \
    --eval_steps 400 \
    --bf16 True \
    --bf16_full_eval True \
    --torch_dtype bfloat16 \
    --ddp_timeout 3600 \
    --seed 42 \
    --streaming True \
    --report_to none \
    --log_on_each_node False \
    --save_strategy "steps" \
    --save_steps 10000 \
    --save_total_limit 5 \
    --max_eval_samples 500 \
    --remove_unused_columns False \
    --gradient_checkpointing True \
    --disable_tqdm True
