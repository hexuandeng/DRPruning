import os
import random
import argparse
import numpy as np
from tqdm import tqdm
from streaming import MDSWriter
from datasets import load_from_disk
from multiprocessing import Process

script_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(script_dir)

def make_dir_if_not_ex(path):
    if not os.path.exists(path):
        print("Make target folder:", path)
        os.makedirs(path)

def write_dataset(split, lang):
    dataset = load_from_disk(os.path.join(args.tokenized_dir, lang))[split]
    out = MDSWriter(
        columns={"tokens": "bytes", "set": "str"}, 
        out=os.path.join(args.target_dir, f"{split}", lang), 
        compression=None
    )

    indices = range(len(dataset))
    if split == "valid":
        indices = random.sample(indices, args.eval_seq)
    total = 0
    for idx in tqdm(indices):
        out.write({
            "tokens": np.array(dataset[idx]["data"], dtype=np.uint32).tobytes(),
            "set": lang
        })
        total += 1
    out.finish()
    print(f"Total {lang} for {split}: {total}")

if __name__ == "__main__":
    random.seed(42)
    np.random.seed(42)

    parser = argparse.ArgumentParser()
    parser.add_argument("--tokenized_dir", type=str, help="Target directory to save tokenized numpy")
    parser.add_argument("--target_dir", type=str, help="Target directory to save tokenized numpy")
    parser.add_argument("--eval_seq", type=int, default=500, help="How many sequences to sample for eval for each domain")
    args = parser.parse_args()

    p_apis = []
    make_dir_if_not_ex(os.path.join(args.target_dir))
    for split in ["prune"]:#["train", "valid"]:
        for lang in ["arxiv", "book", "c4", "common_crawl", "github", "stackexchange", "wikipedia"]:
            make_dir_if_not_ex(os.path.join(args.target_dir, f"{split}"))
            p = Process(target=write_dataset, args=[split, lang])
            p.start()
            p_apis.append(p)
    for p in p_apis:
        p.join()

    print("Done.")
