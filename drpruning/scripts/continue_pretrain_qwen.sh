# pruning llama2 7b -> 3b or 1.3b
export NCCL_DEBUG=INFO
update_type=$1 

PROJ_DIR=$2
DATA_DIR=${PROJ_DIR}/drpruning/data/CulturaX_pt/train
OUTPUT_DIR=${PROJ_DIR}/drpruning/out/pruning_1.5b_${update_type}
TRAIN_SCRIPT=${PROJ_DIR}/drpruning/train.py

model=1.5b # target model size
config_file=${PROJ_DIR}/drpruning/configs/qwen2/${model}.yaml
prune_run_name=qwen2_7b_pruning_${update_type}_to${model}_sl4096
path=${OUTPUT_DIR}/${prune_run_name}/pruned-latest-rank0.pt # path to the 

cd ${PROJ_DIR}
for file in /dev/shm/*; do
   if [ -e "$file" ]; then
       echo "Removing $file..."
       rm "$file"
   fi
done

# data setup
data_local=${DATA_DIR}

# basic setup
max_seq_len=4096
device_train_microbatch_size=2
global_train_batch_size=256
device_eval_batch_size=4

# learning setup
lr=1e-4 # learning rate for the main parameters
max_duration=40000ba
save_interval=8000ba # save every 3200ba
t_warmup=1440ba # 3% learning rate warmup 

# dynamic loading setup
dynamic=True
set_names=[en,ru,zh,ja,ar,tr,ko,th] # domain names
proportion=[0.2770426850721432,0.1847173587915918,0.12973923521607217,0.10377832031139692,0.09089790581585629,0.0888547133132728,0.06673991552832112,0.0582298659513457] # initial proportion of RP, make sure that the sum(proportion) = 1
reference_loss=[2.389493,1.795453,2.626018,2.362056,2.453045,2.535048,2.155811,1.535680] # 1.3b predicted loss from scaling law
eval_split_name=eval_merge # eval on all domains
eval_interval=400ba # eval every 50 batches and update the loading proportion

# save directroy
run_name=${prune_run_name}_ft${max_duration}
save_dir=${OUTPUT_DIR}/${run_name}

# Run in bash, it will automatically use resources available in the current environment
nohup python ${PROJ_DIR}/drpruning/callbacks/DRO_server.py \
    --for_prune False \
    --max_duration ${max_duration} \
    > ${save_dir}/DRO_server.log 2>&1 &

composer $TRAIN_SCRIPT \
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
    model.l0_module=null \
    model.path=${path} \
    callbacks.data_loading.dynamic=${dynamic} \
    callbacks.data_loading.set_names=${set_names} \
    callbacks.data_loading.proportion=${proportion} \
    callbacks.data_loading.update_type=${update_type} \
    callbacks.data_loading.reference_loss=${reference_loss} \
    train_loader.num_workers=0 \
    train_loader.prefetch_factor=null \
    train_loader.persistent_workers=false \
    python_log_level=DEBUG \
    console_log_interval=20ba \
    autoresume=false \
    callbacks.data_loading.rho=0.1 \
    callbacks.data_loading.ema=0.1
