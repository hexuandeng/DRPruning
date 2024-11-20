# pruning llama2 7b -> 3b or 1.3b
export NCCL_DEBUG=INFO
update_type=$1
from_model=$2 # source model size
model=$3 # target model size

PROJ_DIR=$4
DATA_DIR=${PROJ_DIR}/drpruning/data/SlimPajama_pt/train
OUTPUT_DIR=${PROJ_DIR}/drpruning/out/pruning_${model}_${update_type}
TRAIN_SCRIPT=${PROJ_DIR}/drpruning/train.py

config_file=${PROJ_DIR}/drpruning/configs/llama2/${model}.yaml
prune_run_name=llama2_${from_model}_pruning_${update_type}_to${model}_sl4096
path=${OUTPUT_DIR}/${prune_run_name}/pruned-latest-rank0.pt

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
max_duration=48000ba # 50B tokens
save_interval=8000ba # save every 3200ba
t_warmup=1440ba # 3% learning rate warmup 

# dynamic loading setup
dynamic=True
set_names=[common_crawl,github,book,stackexchange,wikipedia,arxiv,c4] # domain names
if [[ $update_type == sheared ]]; then
    proportion=[0.2192,0.0002,0.0791,0.0064,0.0096,0.001,0.6845] # final proportion of pruning
else
    proportion=[0.67,0.045,0.045,0.02,0.045,0.025,0.15] # initial proportion of RP, make sure that the sum(proportion) = 1
fi
if [[ $model == 1.3b ]]; then
    reference_loss=[1.9643,0.7459,2.1393,1.6117,1.7590,1.4449,2.1251] # 1.3b predicted loss from scaling law
elif [[ $model == 2.7b ]]; then
    reference_loss=[1.8712,0.6883,2.0325,1.5353,1.6297,1.3560,2.0328] # 2.7b predicted loss from scaling law
elif [[ $model == 370m ]]; then
    reference_loss=[2.1401,0.8694,2.3625,1.7791,2.047,1.6637,2.3139] # 410m predicted loss from scaling law
fi
eval_split_name=eval_merge # eval on all domains
eval_interval=400ba # eval every 50 batches and update the loading proportion

# save directroy
run_name=${prune_run_name}_ft${max_duration}
save_dir=${OUTPUT_DIR}/${run_name}

# Run in bash, it will automatically use resources available in the current environment
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
