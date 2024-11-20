import os
import shutil
import numpy as np
import argparse
from transformers import AutoTokenizer
from datasets import load_dataset, DatasetDict, concatenate_datasets
from huggingface_hub import snapshot_download

os.environ["TOKENIZERS_PARALLELISM"] = "false"
script_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(script_dir)

def download():
    snapshot_download(repo_id="MBZUAI-LLM/SlimPajama-627B-DC", repo_type="dataset", local_dir="SlimPajama")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokenizer", type=str, help="Tokenizer Path")
    parser.add_argument("--target_dir", type=str, help="Target directory to save tokenized numpy")
    parser.add_argument("--seq_length", type=int, default=4096, help="Sequence length")
    args = parser.parse_args()

    folders = ["RedPajamaArXiv", "RedPajamaBook", "RedPajamaC4", "RedPajamaCommonCrawl", "RedPajamaGithub", "RedPajamaStackExchange", "RedPajamaWikipedia"]
    saves = ["arxiv", "book", "c4", "common_crawl", "github", "stackexchange", "wikipedia"]
    tok = AutoTokenizer.from_pretrained(args.tokenizer)
    def reshape(batch):
        data = []
        buffer = []
        for line in batch["text"]:
            tokens = buffer + tok.encode((tok.bos_token if tok.bos_token is not None else '') + line + tok.eos_token, add_special_tokens=False)
            buffer = []
            for start_id in range(0, len(tokens), args.seq_length):
                if start_id + args.seq_length < len(tokens):
                    data.append(np.array(tokens[start_id: start_id + args.seq_length], dtype=np.uint16))
                else:
                    buffer = tokens[start_id:]
                    break
        return {"data": data}
    
    for folder, sv in zip(folders, saves):
        dataset = load_dataset(f"drpruning/data/SlimPajama/train/{folder}", 
                               num_proc=32)["train"]
        if sv == "common_crawl":
            dataset = dataset.train_test_split(test_size=0.3)["test"]
        if sv == "c4":
            dataset = dataset.train_test_split(test_size=0.5)["test"]
        print(sv, dataset)

        # tokenize and reshape datasets
        all_datasets = []
        splits = 50
        for i in range(splits):
            if not os.path.exists("drpruning/data/dataset_cache/"):
                os.makedirs("drpruning/data/dataset_cache/")
            all_datasets.append(dataset.shard(splits, index=i).map(
                reshape, 
                batched=True,
                batch_size=128,
                writer_batch_size=128,
                remove_columns=dataset.column_names, 
                num_proc=32,
                keep_in_memory=False,
                cache_file_name="drpruning/data/dataset_cache/.arrow",
                desc=sv
            ))
            shutil.rmtree("drpruning/data/dataset_cache/")

        valid = load_dataset(f"drpruning/data/SlimPajama/validation/{folder}", trust_remote_code=True, num_proc=4)["train"]
        print(sv, valid)
        # tokenize and reshape datasets
        if not os.path.exists("drpruning/data/dataset_cache/"):
            os.makedirs("drpruning/data/dataset_cache/")
        valid = valid.map(
            reshape, 
            batched=True,
            batch_size=128,
            writer_batch_size=128,
            remove_columns=valid.column_names, 
            num_proc=8,
            keep_in_memory=False,
            cache_file_name="drpruning/data/dataset_cache/.arrow",
            desc=sv
        )
        shutil.rmtree("drpruning/data/dataset_cache/")

        train_test_valid_dataset = DatasetDict({
            'train': concatenate_datasets(all_datasets[:-2]),
            'prune': concatenate_datasets(all_datasets[-2:]),
            'valid': valid,
        })
        print(train_test_valid_dataset)
        train_test_valid_dataset.save_to_disk(os.path.join(args.target_dir, sv))
