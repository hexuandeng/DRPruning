import os
import torch
import argparse
from transformers import AutoTokenizer
from datasets import load_dataset, DatasetDict
from huggingface_hub import snapshot_download

script_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(script_dir)

ROOT = "CulturaX"
langs = {
    "en": 2846.970578793,
    "ru": 737.201800363,
    "zh": 227.055380882,
    "ja": 107.873841351,
    "ar": 69.354335076,
    "tr": 64.292787164,
    "ko": 24.765448392,
    "th": 15.717374014,
}

def download(lang, patten):
    if os.path.isfile(f"{ROOT}/{lang}/checksum.sha256"):
        os.remove(f"{ROOT}/{lang}/checksum.sha256")
    snapshot_download(repo_id="uonlp/CulturaX", repo_type="dataset", allow_patterns=f"{lang}/{lang}_part_{patten}.parquet", local_dir=ROOT)
    snapshot_download(repo_id="uonlp/CulturaX", repo_type="dataset", allow_patterns=f"{lang}/checksum.sha256", local_dir=ROOT)
    mem = []
    with open(f"{ROOT}/{lang}/checksum.sha256", 'r', encoding="utf-8") as f:
        for line in f:
            cnt = line.split()[1].strip()
            if os.path.isfile(f"{ROOT}/{lang}/{cnt}"):
                mem.append(line)
    with open(f"{ROOT}/{lang}/checksum.sha256", 'w', encoding="utf-8") as f:
        for i in mem:
            f.write(i)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokenizer", type=str, help="Tokenizer Path")
    parser.add_argument("--target_dir", type=str, help="Target directory to save tokenized numpy")
    parser.add_argument("--seq_length", type=int, default=4096, help="Sequence length")
    args = parser.parse_args()

    # download datasets
    download("en", "*[05]0")
    download("ru", "*[02468]0")
    download("zh", "*0")
    download("ja", "*[05]")
    download("ar", "*[05]")
    download("tr", "*[05]")
    download("ko", "*[02468]")
    download("th", "*")
    snapshot_download(repo_id="uonlp/CulturaX", repo_type="dataset", allow_patterns="CulturaX_loading_script.py", local_dir=ROOT)

    # tokenize and reshape datasets
    tok = AutoTokenizer.from_pretrained(args.tokenizer)
    def reshape(batch):
        data = []
        buffer = []
        for line in batch["text"]:
            tokens = buffer + tok.encode(tok.bos_token if tok.bos_token is not None else '' + line + tok.eos_token, add_special_tokens=False)
            buffer = []
            for start_id in range(0, len(tokens), args.seq_length):
                if start_id + args.seq_length < len(tokens):
                    data.append(torch.tensor(tokens[start_id: start_id + args.seq_length]))
                else:
                    buffer = tokens[start_id:]
                    break
        return {"data": data}

    for k, v in langs.items():
        dataset = load_dataset(f"{ROOT}/CulturaX_loading_script.py", k, trust_remote_code=True, num_proc=16)["train"]
        print(k, dataset)
        print(k, dataset[0])
        dataset = dataset.map(
            reshape, 
            batched=True,
            remove_columns=dataset.column_names, 
            num_proc=32, 
            desc=k
        )
        train_test = dataset.train_test_split(test_size=20000, shuffle=False)
        train = train_test['train'].train_test_split(test_size=0.1)
        train_test_valid_dataset = DatasetDict({
            'train': train['train'],
            'prune': train['test'],
            'valid': train_test['test']
        })
        print(train_test_valid_dataset)
        train_test_valid_dataset.save_to_disk(os.path.join(args.target_dir, k), num_proc=16)
