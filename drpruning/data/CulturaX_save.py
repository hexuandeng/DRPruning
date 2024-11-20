import os
import random
import argparse
import numpy as np
from multiprocessing import Process
from drpruning.data.SlimPajama_save import make_dir_if_not_ex, write_dataset

script_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(script_dir)

splits = {
    "valid": 0.8,
    "prune": 1,
    "train": 1,
}
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

if __name__ == "__main__":
    random.seed(42)
    np.random.seed(42)

    parser = argparse.ArgumentParser()
    parser.add_argument("--tokenized_dir", type=str, help="Target directory to save tokenized numpy")
    parser.add_argument("--target_dir", type=str, help="Target directory to save tokenized numpy")
    parser.add_argument("--eval_seq", type=int, default=500, help="How many sequences to sample for eval for each domain")
    args = parser.parse_args()

    data_sum = 0
    for v in langs.values():
        data_sum += v ** 0.3
    for k, v in langs.items():
        langs[k] = v ** 0.3 / data_sum
    print(langs)

    p_apis = []
    make_dir_if_not_ex(os.path.join(args.target_dir))
    for split, ppl in splits.items():
        for lang in langs:
            make_dir_if_not_ex(os.path.join(args.target_dir, split))
            p = Process(target=write_dataset, args=[split, lang])
            p.start()
            p_apis.append(p)
    for p in p_apis:
        p.join()

    print("Done.")
