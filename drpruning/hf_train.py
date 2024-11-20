#!/usr/bin/env python
# coding=utf-8
# Adopt from https://github.com/wxjiao/ParroT
import os
import logging
import math
import sys
import random
from dataclasses import dataclass, field
from typing import Optional

import datasets
import torch
from datasets import load_from_disk

import transformers
from transformers import (
    CONFIG_MAPPING,
    MODEL_FOR_CAUSAL_LM_MAPPING,
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    HfArgumentParser,
    TrainingArguments,
    default_data_collator,
    set_seed,
)
from transformers.utils import check_min_version, send_example_telemetry, is_sagemaker_mp_enabled
from transformers.utils.versions import require_version
from drpruning.callbacks.DRO_loading_callback_hf import DRPruningCallback
from drpruning.callbacks.dynamic_loading_callback_hf import ShearedCallback
from drpruning.callbacks.trainer_hf import MyTrainer

logger = logging.getLogger(__name__)

MODEL_CONFIG_CLASSES = list(MODEL_FOR_CAUSAL_LM_MAPPING.keys())
MODEL_TYPES = tuple(conf.model_type for conf in MODEL_CONFIG_CLASSES)


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune, or train from scratch.
    """

    model_name_or_path: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "The model checkpoint for weights initialization.Don't set if you want to train a model from scratch."
            )
        },
    )
    model_type: Optional[str] = field(
        default=None,
        metadata={"help": "If training from scratch, pass a model type from the list: " + ", ".join(MODEL_TYPES)},
    )
    target_model: Optional[str] = field(
        default=None,
        metadata={"help": "If training from scratch, pass a model type from the list: " + ", ".join(MODEL_TYPES)},
    )
    config_overrides: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "Override some existing default config settings when a model is trained from scratch. Example: "
                "n_embd=10,resid_pdrop=0.2,scale_attn_weights=false,summary_type=cls_index"
            )
        },
    )
    config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    tokenizer_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Where do you want to store the pretrained models downloaded from huggingface.co"},
    )
    use_fast_tokenizer: bool = field(
        default=True,
        metadata={"help": "Whether to use one of the fast tokenizer (backed by the tokenizers library) or not."},
    )
    model_revision: str = field(
        default="main",
        metadata={"help": "The specific model version to use (can be a branch name, tag name or commit id)."},
    )
    use_auth_token: bool = field(
        default=False,
        metadata={
            "help": (
                "Will use the token generated when running `huggingface-cli login` (necessary to use this script "
                "with private models)."
            )
        },
    )
    torch_dtype: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "Override the default `torch.dtype` and load the model under this dtype. If `auto` is passed, the "
                "dtype will be automatically derived from the model's weights."
            ),
            "choices": ["auto", "bfloat16", "float16", "float32"],
        },
    )
    low_cpu_mem_usage: bool = field(
        default=False,
        metadata={
            "help": (
                "It is an option to create the model as an empty shell, then only materialize its parameters when the pretrained weights are loaded."
                "set True will benefit LLM loading time and RAM consumption."
            )
        },
    )

    def __post_init__(self):
        if self.config_overrides is not None and (self.config_name is not None or self.model_name_or_path is not None):
            raise ValueError(
                "--config_overrides can't be used in combination with --config_name or --model_name_or_path"
            )


@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """

    dataset_path: Optional[str] = field(
        default=None, metadata={"help": "The name of the dataset to use (via the datasets library)."}
    )
    domains: Optional[str] = field(
        default=None, metadata={"help": "The name of the domains to use (via the datasets library)."}
    )
    proportion: Optional[str] = field(
        default=None, metadata={"help": "Initial data loading proportion for each domain or stream. The sum must equal 1."}
    )
    update_type: Optional[str] = field(
        default=None, metadata={"help": "Update type for adjusting data loading proportions."}
    )
    reference_loss: Optional[str] = field(
        default=None, metadata={"help": "Target validation loss predetermined before training. Loading proportions adjust based on the difference between the current loss and the target loss."}
    )
    rho: Optional[float] = field(
        default=0.1, metadata={"help": "Specifies the constraint size of the $f$-divergence ball used in DRO."}
    )
    ema: Optional[float] = field(
        default=0.1, metadata={"help": "The alpha parameter for the exponential moving average."}
    )
    min_prob: Optional[float] = field(
        default=0.2, metadata={"help": "Specifies the minimum probability for DRO computation."}
    )
    clamp_q_to_min: Optional[bool] = field(
        default=True, metadata={"help": "Indicates whether to clamp probabilities by the minimum value in DRO."}
    )
    task: Optional[str] = field(
        default="train", metadata={"help": "Specifies the current task (e.g., 'train')."}
    )
    max_train_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of training examples to this "
                "value if set."
            )
        },
    )
    max_eval_samples: Optional[int] = field(
        default=500,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of evaluation examples to this "
                "value if set."
            )
        },
    )
    streaming: bool = field(default=True, metadata={"help": "Enable streaming mode"})
    block_size: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "Optional input sequence length after tokenization. "
                "The training dataset will be truncated in block of this size for training. "
                "Default to the model max input length for single sentence inputs (take into account special tokens)."
            )
        },
    )
    overwrite_cache: bool = field(
        default=False, metadata={"help": "Overwrite the cached training and evaluation sets"}
    )
    validation_split_percentage: Optional[int] = field(
        default=5,
        metadata={
            "help": "The percentage of the train set used as validation set in case there's no validation split"
        },
    )
    preprocessing_num_workers: Optional[int] = field(
        default=None,
        metadata={"help": "The number of processes to use for the preprocessing."},
    )
    keep_linebreaks: bool = field(
        default=True, metadata={"help": "Whether to keep line breaks when using TXT files or not."}
    )
    dynamic_update_interval: Optional[int] = field(
        default=400,
        metadata={
            "help": "The percentage of the train set used as validation set in case there's no validation split"
        },
    )

    def __post_init__(self):
        if self.streaming:
            require_version("datasets>=2.0.0", "The streaming feature requires `datasets>=2.0.0`")


class DomainDataset(torch.utils.data.IterableDataset):
    def __init__(self, args: DataTrainingArguments, validation: bool = False):
        super(DomainDataset).__init__()
        self.rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
        world_size = torch.distributed.get_world_size() if torch.distributed.is_initialized() else 1

        self.domains = args.domains.split(',')
        self.domains_to_id = {self.domains[i]: i for i in range(len(self.domains))}
        self.domains_epoch = [0] * len(self.domains)
        self.used_domains = [0] * len(self.domains)
        self.proportion = [float(i) for i in args.proportion.split(',')]
        self.validation = validation
        self.len = 0

        split = args.task
        if validation:
            split = "valid"
        self.datasets = []
        self.iters = []
        for domain in self.domains:
            self.datasets.append(load_from_disk(os.path.join(args.dataset_path, domain, split)))
            if validation:
                self.datasets[-1] = self.datasets[-1].shuffle(seed=42).select(range(args.max_eval_samples))
            l = self.datasets[-1].num_rows
            self.len += l

            if args.streaming:
                self.datasets[-1] = self.datasets[-1].to_iterable_dataset()
            self.datasets[-1] = self.datasets[-1].shuffle(seed=42)

            # start_idx = l * self.rank // world_size
            # self.datasets[-1] = self.datasets[-1].skip(start_idx)
            self.datasets[-1] = self.datasets[-1].shuffle(seed=42)
            self.iters.append(iter(self.datasets[-1]))

    def reinit(self, domain):
        self.iters[domain] = iter(self.datasets[domain])

    def __len__(self):
        return self.len

    def __iter__(self):
        assert self.rank == 0
        if self.validation:
            self.stream_id = 0

        while True:
            if self.validation:
                stream_id = self.stream_id
            else:
                stream_id = random.choices(range(len(self.domains)), weights=self.proportion, k=1)[0]

            try:
                item = next(self.iters[stream_id])
            except StopIteration:
                self.domains_epoch[stream_id] += 1
                self.reinit(stream_id)
                if self.validation:
                    self.stream_id += 1
                    stream_id = self.stream_id
                    if stream_id == len(self.domains):
                        break
                    item = next(self.iters[stream_id])
                else:
                    logger.warning(f"Domain {self.domains[stream_id]} run out of items!")
                    item = next(self.iters[stream_id])

            self.used_domains[stream_id] += 1

            item["input_ids"] = item["data"]
            item["labels"] = item["input_ids"].copy()
            item["domain"] = stream_id
            del item["data"]
            yield item


def main():
    # See all possible arguments in src/transformers/training_args.py
    # or by passing the --help flag to this script.
    # We now keep distinct sets of args, for a cleaner separation of concerns.
    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    # Sending telemetry. Tracking the example usage helps us better allocate resources to maintain them. The
    # information sent is the one passed as arguments along with your Python/PyTorch versions.
    send_example_telemetry("run_clm", model_args, data_args)

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    if training_args.local_rank != -1 and training_args.local_rank != 0:
        transformers.utils.logging.disable_progress_bar()
        datasets.disable_progress_bars()

    if training_args.should_log:
        # The default of training_args.log_level is passive, so we set log level at info here to have that default.
        transformers.utils.logging.set_verbosity_info()

    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    # Log on each process the small summary:
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
        + f"distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
    )
    logger.info(f"Training/evaluation parameters {training_args}")

    # TODO: Breakpoint recovery implementation
    training_args.overwrite_output_dir = True
    training_args.dataloader_num_workers = 0
    training_args.label_names = ['labels']

    # Set seed before initializing model.
    set_seed(training_args.seed)

    # Get the preprocessed datasets:
    train_dataset = DomainDataset(data_args)
    eval_dataset = DomainDataset(data_args, validation=True)

    # See more about loading any type of standard or custom dataset (from files, python dict, pandas DataFrame, etc) at
    # https://huggingface.co/docs/datasets/loading_datasets.html.

    # Load pretrained model and tokenizer
    #
    # Distributed training:
    # The .from_pretrained methods guarantee that only one local process can concurrently
    # download model & vocab.

    config_kwargs = {
        "cache_dir": model_args.cache_dir,
        "revision": model_args.model_revision,
        "use_auth_token": True if model_args.use_auth_token else None,
    }
    if model_args.config_name:
        config = AutoConfig.from_pretrained(model_args.config_name, **config_kwargs)
    elif model_args.model_name_or_path:
        config = AutoConfig.from_pretrained(model_args.model_name_or_path, **config_kwargs)
    else:
        config = CONFIG_MAPPING[model_args.model_type]()
        logger.warning("You are instantiating a new config instance from scratch.")
        if model_args.config_overrides is not None:
            logger.info(f"Overriding config: {model_args.config_overrides}")
            config.update_from_string(model_args.config_overrides)
            logger.info(f"New config: {config}")

    if model_args.target_model is not None:
        config.target_model = model_args.target_model
        config.pruning_modules = ["head", "intermediate", "layer", "hidden"]
        config.start_sparsity = 0.0
        config.lagrangian_warmup_steps = int(0.2 * training_args.max_steps)
        config.eval_target_model = False

    tokenizer_kwargs = {
        "cache_dir": model_args.cache_dir,
        "use_fast": model_args.use_fast_tokenizer,
        "revision": model_args.model_revision,
        "use_auth_token": True if model_args.use_auth_token else None,
    }
    if model_args.tokenizer_name:
        tokenizer = AutoTokenizer.from_pretrained(model_args.tokenizer_name, **tokenizer_kwargs)
    elif model_args.model_name_or_path:
        tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path, **tokenizer_kwargs)
    else:
        raise ValueError(
            "You are instantiating a new tokenizer from scratch. This is not supported by this script."
            "You can do it from another script, save it, and load it from here, using --tokenizer_name."
        )

    if model_args.model_name_or_path:
        torch_dtype = (
            model_args.torch_dtype
            if model_args.torch_dtype in ["auto", None]
            else getattr(torch, model_args.torch_dtype)
        )
        model = AutoModelForCausalLM.from_pretrained(
            model_args.model_name_or_path,
            from_tf=bool(".ckpt" in model_args.model_name_or_path),
            config=config,
            cache_dir=model_args.cache_dir,
            revision=model_args.model_revision,
            use_auth_token=True if model_args.use_auth_token else None,
            torch_dtype=torch_dtype,
            attn_implementation="flash_attention_2"
        )
    else:
        model = AutoModelForCausalLM.from_config(config)
        n_params = sum({p.data_ptr(): p.numel() for p in model.parameters()}.values())
        logger.info(f"Training new model from scratch - Total size={n_params/2**20:.2f}M params")

    # We resize the embeddings only when necessary to avoid index errors. If you are creating a model from scratch
    # on a small vocab and want a smaller embedding size, remove this test.
    embedding_size = model.get_input_embeddings().weight.shape[0]
    if len(tokenizer) > embedding_size:
        model.resize_token_embeddings(len(tokenizer))

    # Initialize our Trainer
    callbacks = []
    if 'dro' in data_args.update_type:
        callbacks.append(DRPruningCallback(
            reference_loss = [float(i) for i in data_args.reference_loss.split(',')],
            save_folder = training_args.output_dir,
            dynamic_update_interval = data_args.dynamic_update_interval,
            dynamic_proportion = "DyAll" in data_args.update_type,
            dynamic_baseline = "Dy" in data_args.update_type,
            use_eval = "Eval" in data_args.update_type
        ))
    else:
        assert data_args.update_type in ["sheared", "bandit", "constant"]
        callbacks.append(ShearedCallback(
            reference_loss = [float(i) for i in data_args.reference_loss.split(',')],
            update_type = data_args.update_type
        ))
    trainer = MyTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset if training_args.do_train else None,
        eval_dataset=eval_dataset if training_args.do_eval else None,
        tokenizer=tokenizer,
        # Data collator will default to DataCollatorWithPadding, so we change it.
        data_collator=default_data_collator,
        callbacks=callbacks
    )

    # Training
    if training_args.do_train:
        train_result = trainer.train()
        trainer.save_model()  # Saves the tokenizer too for easy upload
        metrics = train_result.metrics
        max_train_samples = (
            data_args.max_train_samples if data_args.max_train_samples is not None else len(train_dataset)
        )
        metrics["train_samples"] = min(max_train_samples, len(train_dataset))

        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()

    # Evaluation
    if training_args.do_eval:
        logger.info("*** Evaluate ***")

        metrics = trainer.evaluate()

        max_eval_samples = data_args.max_eval_samples if data_args.max_eval_samples is not None else len(eval_dataset)
        metrics["eval_samples"] = min(max_eval_samples, len(eval_dataset))
        try:
            perplexity = math.exp(metrics["eval_loss"])
        except OverflowError:
            perplexity = float("inf")
        metrics["perplexity"] = perplexity

        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)

    kwargs = {"finetuned_from": model_args.model_name_or_path, "tasks": "text-generation"}
    if data_args.dataset_name is not None:
        kwargs["dataset_tags"] = data_args.dataset_name
        if data_args.dataset_config_name is not None:
            kwargs["dataset_args"] = data_args.dataset_config_name
            kwargs["dataset"] = f"{data_args.dataset_name} {data_args.dataset_config_name}"
        else:
            kwargs["dataset"] = data_args.dataset_name

    if training_args.push_to_hub:
        trainer.push_to_hub(**kwargs)
    else:
        trainer.create_model_card(**kwargs)

if __name__ == "__main__":
    main()
