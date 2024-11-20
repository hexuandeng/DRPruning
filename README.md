# 🚀 DRPruning: Efficient Large Language Model Pruning through Distributionally Robust Optimization

This repository provides the codebase for **DRPruning**, a method that incorporates refined Distributionally Robust Optimization (DRO) into structured pruning to address uneven performance degradation across domains in large language models (LLMs). By dynamically adjusting the data distribution and hyperparameters during training, DRPruning targets underperforming areas, promoting balanced recovery and reducing bias. This approach yields efficient, smaller models with robust, balanced capabilities, outperforming similarly sized models in both monolingual and multilingual settings.

<p align="center">
  <img src="pic/main.png" width="900" alt="DRPruning Overview" />
</p>

## 🔗 Quick Links

- [Brief Introduction](#brief-introduction)
- [Installation Requirements](#installation-requirements)
- [Data Preparation](#data-preparation)
- [Model Preparation](#model-preparation)
- [Sample Scripts for Pruning and Continued Pre-training](#sample-scripts-for-pruning-and-continued-pre-training)
- [Convert Composer Model to Hugging Face Model](#convert-composer-model-to-hugging-face-model)
- [Training Configurations](#training-configurations)
  - [Data Configurations](#data-configurations)
  - [Basic Training Configurations](#basic-training-configurations)
  - [Pruning Configurations](#pruning-configurations)
  - [Dynamic Data Proportion Configurations](#dynamic-data-proportion-configurations)

## Brief Introduction

DRPruning builds upon the [LLM-Shearing](https://github.com/princeton-nlp/LLM-Shearing) framework, specifically optimizing it for LLM pre-training and pruning. By integrating DRO, DRPruning more effectively targets worst-case scenarios during pruning, ensuring robust and balanced model performance across various domains. The codebase is organized as follows:

- `drpruning.data`: Data processing scripts.
- `drpruning.datasets`: Customized datasets for dynamic loading.
- `drpruning.callbacks`: Callbacks for dynamic loading and pruning.
- `drpruning.models`: Model definitions.
- `drpruning.scripts`: Execution scripts.
- `drpruning.utils`: Utility functions (e.g., model conversion).

## Installation Requirements

To set up the environment, please follow the installation instructions from [LLM-Shearing](https://github.com/princeton-nlp/LLM-Shearing). Alternatively, you can use the provided [Dockerfile](Dockerfile) to build the environment. We have updated several package versions to support newer LLMs. After setting up the environment, install the `drpruning` package in editable mode:

```bash
pip install -e .
```

## Data Preparation

We use the pruning data from [LLM-Shearing](https://github.com/princeton-nlp/LLM-Shearing), which is available on Google Drive [here](https://drive.google.com/drive/folders/1A_-88BqcOGa1Pbo-1ZShG2saU0cdRZgK). You can access it by clicking on the link. Alternatively, you have the option to process your own data as follows.

We provide preprocessing code to tokenize, sample, and process RedPajama data into MDS format ([Mosaic's streaming package](https://docs.mosaicml.com/projects/streaming/en/stable/index.html)). For monolingual and multilingual settings, we process [cerebras/SlimPajama-627B](https://huggingface.co/datasets/cerebras/SlimPajama-627B) and [uonlp/CulturaX](https://huggingface.co/datasets/uonlp/CulturaX), respectively. Please run [`data/SlimPajama.py`](drpruning/data/SlimPajama.py) and [`data/CulturaX.py`](drpruning/data/CulturaX.py) to obtain the processed huggingface dataset versions.

After that, run  [`data/SlimPajama_save.py`](drpruning/data/SlimPajama_save.py) and [`data/CulturaX_save.py`](drpruning/data/CulturaX_save.py) to obtain datasets that meet the requirements of the Composer repository. Note that the `eval` folder must include `eval_merge`, which is a single split that contains validation data from all domains. We provide a utility script [`data/merge_data.py`](drpruning/data/merge_data.py) to merge data from multiple splits into one split. An example of running the script is as follows:

```bash
python3 -m drpruning.data.merge_data \
        --input_dir $INPUT_DIR \
        --output_dir $OUTPUT_DIR \
        --output_split eval_merge \
        --split_names domain1 domain2
```

## Model Preparation

To use Hugging Face transformer models with the Composer repository employed by LLM-Shearing, you need to convert the model weights into the key format expected by Composer. Below is an example of converting the weights from the Hugging Face model `llama2` into a compatible format:

```bash
# Define the Hugging Face model name and the output path
HF_MODEL_NAME=meta-llama/Llama-2-7b-hf
OUTPUT_PATH=models/Llama-2-7b-composer.pt

# Create the necessary directory if it doesn't exist
mkdir -p $(dirname $OUTPUT_PATH)

# Convert the Hugging Face model to Composer key format
python3 -m drpruning.utils.hf_to_composer $HF_MODEL_NAME $OUTPUT_PATH
```

Our current implementation supports Pythia, LLaMA, LLaMA2, LLaMA3, and Qwen2 models. It should also be straightforward to adapt it for other models such as Mistral-7B.

## Sample Scripts for Pruning and Continued Pre-training

For pruning, refer to the example script [`drpruning/scripts/prune.sh`](drpruning/scripts/prune.sh). Due to the higher computational cost of pruning compared to continued pre-training, we halt training with the pruning objective after a specific number of steps (typically 3,200 steps in our experiments). We then proceed with further pre-training of the pruned model. After pruning, the saved models consist of the full parameters of the source model accompanied by a set of masks.

We process the masking variables by:

1. Removing substructures where the masking variables are near zero.
2. Incorporating the masking variables into the model parameters through matrix-vector multiplication, resulting in a more compact model.
3. Renaming the weight keys so that they can be seamlessly loaded into the target model architecture, ensuring that the layer names are all consecutive.

This processing can be done using the following command:

```bash
MODEL_PATH=$MODEL_DIR/latest-rank0.pt
python3 -m drpruning.utils.post_pruning_processing prune_and_save_model $MODEL_PATH
```

The pruned model will be saved as `$(dirname $MODEL_PATH)/pruned-latest-rank0.pt`. This step is automatically performed at the end of the pruning script.

After model conversion, continue with the pre-training of the pruned model. The process is similar to pre-training a standard model. Refer to the example script [`drpruning/scripts/continue_pretrain.sh`](drpruning/scripts/continue_pretrain.sh).

<!-- For continued pre-training under the Hugging Face framework, we have provided an implementation that closely follows [ParroT](https://github.com/wxjiao/ParroT). This facilitates simpler implementation and broader applicability for future work. See the example script [`drpruning/scripts/continue_pretrain_hf.sh`](drpruning/scripts/continue_pretrain_hf.sh) for details. -->

## Convert Composer Model to Hugging Face Model

After training, if you would like to use Hugging Face for inference or fine-tuning, you can convert your Composer model into a Hugging Face model using the script [`drpruning/scripts/composer_to_hf.py`](drpruning/scripts/composer_to_hf.py). Here's an example:

```bash
MODEL_PATH=$MODEL_DIR/latest-rank0.pt
OUTPUT_PATH=$MODEL_DIR/hf-latest_rank0
MODEL_CLASS=Llama2
HIDDEN_SIZE=2048
NUM_ATTENTION_HEADS=16
NUM_HIDDEN_LAYERS=24
INTERMEDIATE_SIZE=5504
MODEL_NAME=Pruned-Llama-1.3B

python3 -m drpruning.utils.composer_to_hf $MODEL_PATH $OUTPUT_PATH \
    model_class=${MODEL_CLASS} \
    hidden_size=${HIDDEN_SIZE} \
    num_attention_heads=${NUM_ATTENTION_HEADS} \
    num_hidden_layers=${NUM_HIDDEN_LAYERS} \
    intermediate_size=${INTERMEDIATE_SIZE} \
    num_key_value_heads=${NUM_ATTENTION_HEADS} \
    _name_or_path=${MODEL_NAME}
```

**Note:** The parameter names correspond to the Hugging Face configurations of most LLMs and may differ for other models.

## Training Configurations

This section provides an in-depth guide on configuring parameters within the [training scripts](drpruning/scripts) and the [YAML configuration](drpruning/configs) files for training. The configurations cover data setup, basic training settings, pruning settings, and dynamic data loading configurations.

### Data Configurations

- `data_local`: The local directory containing the data.
- `eval_loader.dataset.split`: For evaluation, provide the name of a combined split that includes data from all domains.
- `train_loader.dataset.split`: When `dynamic=True` (see [Dynamic Data Proportion Configurations](#dynamic-data-proportion-configurations)), there is no need to set this value. If `dynamic=False`, you must specify a training split.

### Basic Training Configurations

These configurations largely follow those of the original `Composer` package. For comprehensive details, refer to [Composer's official documentation](https://docs.mosaicml.com/projects/composer/en/stable/). Key training parameters include:

- `max_duration`: Maximum training duration, specified in steps (e.g., `3200ba`) or epochs (e.g., `1ep`). In our experiments, the pruning duration was `3200ba`, and the continued pre-training duration was `48000ba`.
- `save_interval`: Frequency of saving the model state. Set to `3200ba` for both the pruning and continued pre-training stages.
- `t_warmup`: Duration of learning rate warm-up. For pruning, set to `320ba` (10% of training); for continued pre-training, set to `1440ba` (3% of training).
- `optimizer.lr`: Learning rate for the primary model parameters (default `1e-4`).
- `max_seq_len`: Maximum sequence length, set to `4096` following LLaMA2's training methodology.
- `device_train_microbatch_size`: Batch size per device during training. Set to `4` for pruning and `16` for continued pre-training.
- `global_train_batch_size`: Global batch size across all GPUs. Set to `32` for pruning and `256` for continued pre-training.
- `autoresume`: Enable by setting to `true` when resuming a run. Note that while it works during continued pre-training, compatibility with the pruning stage is not guaranteed.

<!-- Due to computational constraints, an exhaustive hyperparameter search was not conducted; better hyperparameters may exist. For the Hugging Face implementation, we closely follow [ParroT](https://github.com/wxjiao/ParroT)'s settings and principles. Please refer to [`run_clm_llms.py`](https://github.com/wxjiao/ParroT/blob/master/transformers/examples/pytorch/language-modeling/run_clm_llms.py) for more details. -->

### Pruning Configurations

The pruning process allows pruning a source model to a specific target shape. Essential parameters include:

- `from_model`: Specifies the source model size, corresponding to a config file.
- `to_model`: Defines the target model size; the source model will be pruned to match this configuration.
- `optimizer.lag_lr`: Learning rate for learning the masking variables and Lagrangian multipliers during pruning (default `1.0`).

Pruning-specific arguments grouped under `model.l0_module`:

- `model.l0_module.lagrangian_warmup_steps`: In the initial warm-up phase, the pruning rate increases from 0 to the desired target value, determined by the target model's predefined structure. Typically set to approximately 20% of the total training steps.
- `model.l0_module.pruning_modules`: By default, this prunes various aspects of the model, including the head, intermediate dimensions, hidden dimensions, and layers.
- `model.l0_module.eval_target_model`: When `true`, evaluates a submodel matching the target model's structure. If `false`, evaluates the current model, considering the masking values.
- `model.l0_module.target_model.d_model`: Hidden dimension of the target model.
- `model.l0_module.target_model.n_heads`: Number of heads in the target model.
- `model.l0_module.target_model.n_layers`: Number of layers in the target model.
- `model.l0_module.target_model.intermediate_size`: Number of intermediate dimensions in the target model.

These parameters allow precise control over the pruning process.

### Dynamic Data Proportion Configurations

Parameters for configuring dynamic data proportion are defined within the `DynamicLoadingCallback` and `DRPruningCallback`. Configure them in the YAML file under the `callbacks.data_loading` section:

- `callbacks.data_loading.dynamic`: Boolean to enable dynamic data loading. Set to `true` to load data dynamically from various domains or streams.
- `callbacks.data_loading.set_names`: Domain or stream names used for dynamic data loading.
- `callbacks.data_loading.proportion`: Initial data loading proportion for each domain or stream. The sum must equal `1`.
- `callbacks.data_loading.update_type`: Update type for adjusting data loading proportions:
  - **`dro*`**: Strings starting with `dro` indicate the use of the method proposed in this paper. Additional characters after `dro` specify extra configurations, based on the implementation from [DRO-MNMT](https://github.com/violet-zct/fairseq-dro-mnmt):
    - `+Eval`: Use validation loss instead of training loss to determine data proportions.
    - `+Dy`: Use dynamic reference loss (`callbacks.data_loading.reference_loss`) instead of always using the default setting.
    - `+DyAll`: Use both dynamic reference loss (`callbacks.data_loading.reference_loss`) and dynamic reference data ratio (`callbacks.data_loading.proportion`) instead of always using the default settings. It is not recommended to only use dynamic reference data ratio.
  - **`sheared`**: Directly copies code from [Sheared LLaMA](https://arxiv.org/abs/2310.06694) to achieve the same functionality.
  - **`constant`**: Keeps data loading proportions constant throughout training (equivalent to disabling dynamic data loading).
- `callbacks.data_loading.reference_loss`: Target validation loss predetermined before training. Loading proportions adjust based on the difference between the current loss and the target loss.
- `eval_interval`: Frequency of evaluations during training. If `dynamic=True`, data loading proportions adjust after each evaluation.

For DRO, additional configurations are needed:

- `callbacks.data_loading.rho`: Specifies the constraint size of the $f$-divergence ball used in DRO.
- `callbacks.data_loading.ema`: The alpha parameter for the exponential moving average.

<!-- For the Hugging Face version, the variable names only need to retain the part after the last `.`, i.e., `callbacks.data_loading.update_type` becomes `update_type`. The remaining settings remain consistent. -->

**Note:** The code currently supports only local data, functions with a single worker for the dataloader, and does not offer prefetch support. In our testing, these restrictions do not incur additional computational overhead.
