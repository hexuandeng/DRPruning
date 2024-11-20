import numpy as np
from transformers import AutoTokenizer
from datasets import load_dataset, DatasetDict
import nltk
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from datasets import DatasetDict, Dataset
import math
from collections import defaultdict
from utils import MultiChat, longestCommonSubsequence
import json
import pickle
import random

folders = ["RedPajamaBook", "RedPajamaC4", "RedPajamaCommonCrawl", "RedPajamaGithub", "RedPajamaStackExchange", "RedPajamaWikipedia", "RedPajamaArXiv"]
saves = ["book", "c4", "common_crawl", "github", "stackexchange", "wikipedia", "arxiv"]
random.seed(42)

def split_paragraph(text, max_length=512, github=False):
    if github:
        sentences = text["text"].split("\n\n")
    else:
        sentences = nltk.sent_tokenize(text["text"])
    paragraphs = []
    current_paragraph = []
    current_length = 0

    for sentence in sentences:
        tokens = tokenizer.tokenize(sentence)
        if current_length + len(tokens) + len(current_paragraph) < max_length:
            current_paragraph.append(sentence)
            current_length += len(tokens)
        else:
            cnts = ' '.join(current_paragraph)
            if len(tokenizer.tokenize(cnts)) > max_length:
                while len(tokenizer.tokenize(cnts)) > max_length:
                    split_tokens = tokenizer.tokenize(cnts)[:max_length]
                    split_paragraph = tokenizer.convert_tokens_to_string(split_tokens)
                    paragraphs.append(split_paragraph)
                    cnts = tokenizer.convert_tokens_to_string(tokenizer.tokenize(cnts)[max_length:])
            else:
                paragraphs.append(cnts)
            current_paragraph = [sentence]
            current_length = sum(len(tokenizer.tokenize(s)) for s in current_paragraph)

    if len(current_paragraph):
        cnts = ' '.join(current_paragraph)
        while len(tokenizer.tokenize(cnts)) > max_length:
            split_tokens = tokenizer.tokenize(cnts)[:max_length]
            split_paragraph = tokenizer.convert_tokens_to_string(split_tokens)
            paragraphs.append(split_paragraph)
            cnts = tokenizer.convert_tokens_to_string(tokenizer.tokenize(cnts)[max_length:])

    return paragraphs

def filter_paragraphs_by_ppl(paragraphs, percentile=70):
    ppl_scores = []
    for paragraph in paragraphs:
        paragraph = paragraph["text"]
        with torch.no_grad():
            inputs = tokenizer(paragraph, return_tensors='pt')
            inputs = {k: v.cuda() for k, v in inputs.items()}
            outputs = model(**inputs, labels=inputs["input_ids"]).loss
            ppl_scores.append(outputs.item())

    threshold = np.percentile([x for x in ppl_scores if not math.isnan(x)], percentile)
    print(threshold)
    filtered_paragraphs = [paragraph["text"] for paragraph, ppl in zip(paragraphs, ppl_scores) if not math.isnan(ppl) and ppl <= threshold]
    return filtered_paragraphs

def filter_and_sort_sentences(sentences, use_length=False, max_length=600):
    if use_length:
        if isinstance(sentences[0], list):
            lengths = [(sentence, -len(sentence[0] + sentence[1])) for sentence in sentences]
        else:
            lengths = [(sentence, -len(sentence)) for sentence in sentences]
    else:
        if isinstance(sentences[0], list):
            lengths = [(sentence, len(tokenizer.encode(sentence[0] + sentence[1]))) for sentence in sentences]
        else:
            lengths = [(sentence, len(tokenizer.encode(sentence))) for sentence in sentences]

    sorted_sentences = sorted(lengths, key=lambda x: x[1])

    n = len(sorted_sentences)
    filter_count = int(0.1 * n)

    start_index = min(filter_count, n - max_length - filter_count)
    end_index = max(n - filter_count, max_length)
    if start_index < 0:
        filtered_sorted_sentences = [s[0] for s in sorted_sentences[:end_index]]
    else:
        filtered_sorted_sentences = [s[0] for s in sorted_sentences[start_index:end_index]]

    return filtered_sorted_sentences

if __name__ == "__main__":
    tokenizer = AutoTokenizer.from_pretrained('../LLMs/Llama-2-7b-hf', device_map="auto")
    model = AutoModelForCausalLM.from_pretrained('../LLMs/Llama-2-7b-hf', device_map="auto", torch_dtype=torch.bfloat16).eval()

    out = {}
    for folder, sv in zip(folders, saves):
        dataset = load_dataset(f"drshearing/data/SlimPajama/test/{folder}", trust_remote_code=True, num_proc=4)["train"]
        paragraphs = []
        for it in dataset:
            paragraphs += split_paragraph(it, github=(sv=="github"))
        print(folder, len(paragraphs))

        paragraphs = Dataset.from_dict({"text": paragraphs})
        paragraphs = filter_paragraphs_by_ppl(paragraphs.shuffle(seed=42).select(range(15000)))
        out[sv] = Dataset.from_dict({"text": paragraphs})
        print(sv, out[sv])
        torch.cuda.empty_cache()

    del model
    datasets = DatasetDict(out)

    PROMPT_PARA = 'Please identify two consecutive parts, which could be full sentences or half sentences, from the following input paragraph that have a causal relationship. The two parts must be connected in the original text. Ensure the parts are not too lengthy. If there are no suitable pairs, output "None". Please copy these parts directly from the original text without any modification. Output only these parts; do not include any other information.'
    PROMPT_CODE = 'Please identify a shorter code snippet from the input that implements a complete function. The snippet should be concise yet fully operational. If no suitable code snippets are found, output "None". Only return the code snippet; include no additional information.'

    chat = MultiChat({"gpt-4o_keys": []},
        save_path="icl_eval/continuation/SlimPajama_filter.json",
        model="gpt-4o-mini-2024-07-18",
        temperature=0
    )
    chat.start()

    for k, dataset in datasets.items():
        for cnt, it in enumerate(dataset):
            if k == "github":
                prompt = PROMPT_CODE
            else:
                prompt = PROMPT_PARA
            if cnt > 8000:
                break
            line = {
                "split": k,
                "text": it["text"],
                "prompt": [{"role": "system", "content": prompt}, 
                        {"role": "user", "content": it["text"]}]}
            chat.post(line)

    chat.wait_finish()

    mem = defaultdict(list)
    with open("icl_eval/continuation/SlimPajama_filter.json", "r", encoding="utf-8") as f:
        for line in f:
            line = json.loads(line)
            if "none" in line['response'].lower():
                continue
            subsequence = longestCommonSubsequence(line['text'], line['response'])
            if subsequence < 0.95:
                continue
            mem[line["split"]].append(line['response'].strip())


    chat = MultiChat({"gpt-4o_keys": []},
        save_path="icl_eval/continuation/SlimPajama_causal.json",
        model="gpt-4o-mini-2024-07-18",
        temperature=0
    )
    chat.start()

    PROMPT_PARA = 'Please split the following text into two parts, ensuring that each part is approximately equal in length and that there is a causal relationship between them. Use "|||" as the separator for the two parts, and output the rest of the text without any modifications. Only output "None" if you are very certain that there is no causal relationship between the sentences in the text.'
    PROMPT_CODE = 'Please split the following code into two segments, ensuring that each segment is approximately equal in length and contains one complete function. Use "|||" to separate the two parts of the code, and output the rest of the code without any modifications.'
    for k, dataset in mem.items():
        for cnt, it in enumerate(dataset):
            if k == "github":
                prompt = PROMPT_CODE
            else:
                prompt = PROMPT_PARA
            line = {
                "split": k,
                "text": it,
                "prompt": [{"role": "system", "content": prompt}, 
                        {"role": "user", "content": it}]
            }
            chat.post(line)

    chat.wait_finish()

    asws = {}
    with open('icl_eval/continuation/SlimPajama_causal.json', 'r', encoding='utf-8') as f:
        for line in f:
            line = json.loads(line)
            if "|||" not in line["response"]:
                continue
            subsequence = longestCommonSubsequence(line["response"], line["text"])
            if subsequence < 0.95:
                continue
            if len(line["response"].split("|||")) != 2:
                continue
            asws[line['text']] = line["response"].split("|||")

    for k, v in mem.items():
        new = []
        for it in v:
            if it in asws:
                new.append(asws[it])
        mem[k] = new

    for k, v in mem.items():
        final = []
        for it in v:
            if len(it[-1].strip()) <= 10 or len(it[0].strip()) <= 10:
                continue
            if k != "github":
                if len(it[-1].strip().split()) <= 4 or len(it[0].strip().split()) <= 4:
                    continue
            else:
                if len(it[-1].strip().split("\n")) <= 2 and len(it[0].strip().split("\n")) <= 2:
                    continue
            final.append(it)
        mem[k] = filter_and_sort_sentences(final, use_length=False, max_length=800)
        if len(mem[k]) > 800:
            mem[k] = random.sample(mem[k], 800)

    with open('icl_eval/continuation/SlimPajama_causal.pkl', 'wb') as f:
        pickle.dump(mem, f)
