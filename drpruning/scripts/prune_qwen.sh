# pruning qwen2 7b -> 1.8b
update_type=$1
from_model=7b # source model size
to_model=1.8b # target model size
# Specify $PROJ_DIR in scripts/launch.sh and scripts/srun_launch.sh if using slurm
PROJ_DIR=$2
LAUNCH_SCRIPT=${PROJ_DIR}/drpruning/scripts/launch.sh
DATA_DIR=${PROJ_DIR}/drpruning/data/CulturaX_pt/prune
OUTPUT_DIR=${PROJ_DIR}/drpruning/out/pruning_1.8b_${update_type}
TRAIN_SCRIPT=${PROJ_DIR}/drpruning/train.py

cd ${PROJ_DIR}
for file in /dev/shm/*; do
   if [ -e "$file" ]; then
       echo "Removing $file..."
       rm "$file"
   fi
done

config_file=${PROJ_DIR}/drpruning/configs/qwen2/${from_model}.yaml

# data setup
data_local=${DATA_DIR}

# basic setup
max_seq_len=4096
device_train_microbatch_size=1
global_train_batch_size=32
device_eval_batch_size=2

# learning setup
lr=1e-4 # learning rate for the main parameters
max_duration=3200ba # 0.42B tokens
save_interval=400ba # save in the end
t_warmup=320ba # 10% learning rate warmup 

# dynamic loading setup
dynamic=True
set_names=[en,ru,zh,ja,ar,tr,ko,th] # domain names
proportion=[0.2770426850721432,0.1847173587915918,0.12973923521607217,0.10377832031139692,0.09089790581585629,0.0888547133132728,0.06673991552832112,0.0582298659513457] # initial proportion of RP, make sure that the sum(proportion) = 1
# sheared: update weights with exponential descent
# constant: keep the weights constant
reference_loss=[2.389493,1.795453,2.626018,2.362056,2.453045,2.535048,2.155811,1.835680] # 1.3b predicted loss from scaling law
eval_split_name=eval_merge # eval on all domains
eval_target_model=false # evaluate on the current model, not the target model, otherwise the loss will be inaccurate
eval_interval=50ba # eval every 50 batches and update the loading proportion

# pruning setup
lag_lr=1.0 # learning rate or l0_module
lagr_warmup=640ba # 20% sparsity warmup

# save directroy
run_name=qwen2_${from_model}_pruning_${update_type}_to${to_model}_sl${max_seq_len}
save_dir=${OUTPUT_DIR}/${run_name}

# Run in bash, it will automatically use resources available in the current environment 
nohup python ${PROJ_DIR}/drpruning/callbacks/DRO_server.py \
    --for_prune True \
    --max_duration ${max_duration} \
    > ${save_dir}/DRO_server.log 2>&1 &

composer $PROJ_DIR/drpruning/train.py \
    $config_file \
    run_name=${run_name} \
    data_local=${data_local} \
    eval_loader.dataset.split=${eval_split_name} \
    global_train_batch_size=${global_train_batch_size} \
    device_train_microbatch_size=${device_train_microbatch_size} \
    device_eval_batch_size=${device_eval_batch_size} \
    max_seq_len=${max_seq_len} \
    max_duration=${max_duration} \
    eval_first=true \
    scheduler.t_warmup=${t_warmup} \
    save_folder=${save_dir} \
    eval_interval=${eval_interval} \
    save_interval=${save_interval} \
    optimizer.lr=${lr} \
    optimizer.lag_lr=${lag_lr} \
    model.l0_module.lagrangian_warmup_steps=${lagr_warmup} \
    model.l0_module.pruning_modules='[head,intermediate,layer,hidden]' \
    model.l0_module.eval_target_model=${eval_target_model} \
    callbacks.data_loading.dynamic=${dynamic} \
    callbacks.data_loading.set_names=${set_names} \
    callbacks.data_loading.proportion=${proportion} \
    callbacks.data_loading.update_type=${update_type} \
    callbacks.data_loading.reference_loss=${reference_loss} \
    train_loader.num_workers=0 \
    train_loader.prefetch_factor=null \
    train_loader.persistent_workers=false \
    autoresume=false \
    python_log_level=DEBUG \
    console_log_interval=5ba \
    callbacks.data_loading.rho=0.1 \
    callbacks.data_loading.ema=0.1

python3 -m drpruning.utils.post_pruning_processing prune_and_save_model ${save_dir}/latest-rank0.pt
