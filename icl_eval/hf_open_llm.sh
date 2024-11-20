#!/bin/bash

bsz="${bsz:-4}"
cmd="python3 main.py --model=hf --model_args="pretrained=$1,dtype=float16" --tasks=$2 --num_fewshot=$3 --batch_size=$bsz --output_path=$4"
echo $cmd
if [[ -n $5 ]]; then cmd="$cmd --limit=$5"; fi

$cmd 
