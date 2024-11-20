decoder_name_or_path=$1
dataset_path=$2
dataset_name=$3
output_path_to_store_samples=$4
num_return_sequences=$5
temperature=0

python inference.py \
    --task "run_inference" \
    --decoder_name_or_path $decoder_name_or_path \
    --num_return_sequences $num_return_sequences \
    --temperature $temperature \
    --per_device_batch_size 4 \
    --mixed_precision "bf16" \
    --tf32 True \
    --output_path $output_path_to_store_samples \
    --max_new_tokens 512 \
    --dataset_path "" \
    --dataset_name $dataset_name \
    --prompt_dict_path "prompts/sft_prompt.json"
