import json

def get_results(ROOT, end="hf-ba48000_rank0"):
    def get_json(filename):
        with open(f"{ROOT}/{filename}", "r", encoding="utf-8") as f:
            asw = json.load(f)
        return asw

    mmlu = get_json(f"mmlu5shot-{end}")["results"]
    mmlu = [i["acc,none"] for i in mmlu.values()]
    mmlu = sum(mmlu) / len(mmlu)

    results = {
        "arc_c_25shot": get_json(f"arcc25shot-{end}")["results"]["arc_challenge"]["acc_norm,none"] * 100,
        "hellaswag_10shot": get_json(f"hellaswag10shot-{end}")["results"]["hellaswag"]["acc_norm,none"] * 100,
        "mmlu_5shot": mmlu * 100,
        "squad": get_json(f"my0shot-{end}")["results"]["squadv2"]["HasAns_f1,none"],
        "boolq": get_json(f"my0shot-{end}")["results"]["boolq"]["acc,none"] * 100,
        "triviaqa_5shot": get_json(f"my5shot-{end}")["results"]["triviaqa"]["exact_match,remove_whitespace"] * 100,
        "nq_open_5shot": get_json(f"my5shot-{end}")["results"]["nq_open"]["exact_match,remove_whitespace"] * 100,
        "wsc": get_json(f"pythia0shot-{end}")["results"]["wsc"]["acc,none"] * 100,
        "winog": get_json(f"pythia0shot-{end}")["results"]["winogrande"]["acc,none"] * 100,
        "sciq": get_json(f"pythia0shot-{end}")["results"]["sciq"]["acc,none"] * 100,
        "piqa": get_json(f"pythia0shot-{end}")["results"]["piqa"]["acc,none"] * 100,
        "logiqa": get_json(f"pythia0shot-{end}")["results"]["logiqa"]["acc_norm,none"] * 100,
        "lambada_openai": get_json(f"pythia0shot-{end}")["results"]["lambada_openai"]["acc,none"] * 100,
        "arc_easy": get_json(f"pythia0shot-{end}")["results"]["arc_easy"]["acc,none"] * 100,
        "truthfulqa": get_json(f"truthfulqa0shot-{end}")["results"]["truthfulqa"]["acc,none"] * 100,    
        "CC": get_json(f"continuation0shot-{end}")["results"]["common_crawl_continuation"]["acc,none"] * 100,
        "C4": get_json(f"continuation0shot-{end}")["results"]["c4_continuation"]["acc,none"] * 100,
        "GitHub": get_json(f"continuation0shot-{end}")["results"]["github_continuation"]["acc,none"] * 100,
        "Book": get_json(f"continuation0shot-{end}")["results"]["book_continuation"]["acc,none"] * 100,
        "Wiki": get_json(f"continuation0shot-{end}")["results"]["wikipedia_continuation"]["acc,none"] * 100,
        "ArXiv": get_json(f"continuation0shot-{end}")["results"]["arxiv_continuation"]["acc,none"] * 100,
        "StackExchange": get_json(f"continuation0shot-{end}")["results"]["stackexchange_continuation"]["acc,none"] * 100,
    }
    results = dict(sorted(results.items()))
    return results


def dict_to_latex_table(models_results, datasets_per_row=8):
    all_datasets = set().union(*models_results.values())
    all_datasets = sorted(list(all_datasets)) + ["Average"]
    for model, results in models_results.items():
        models_results[model]["Average"] = sum(results.values()) / len(results.values())

    latex_table = "\\begin{table*}[ht]\n\\small\n\\centering\n\\begin{tabular}{l" + "c"*datasets_per_row + "c}\n"
    latex_table += "\\toprule\n"

    for i in range(0, len(all_datasets), datasets_per_row):
        latex_table += " & ".join(["Model"] + [it.replace("_", " ").rstrip("shot") for it in all_datasets[i:i+datasets_per_row]]) + " \\\\\n"
        latex_table += "\\midrule\n"
        for model, results in models_results.items():
            row_results = [results.get(dataset, "-") for dataset in all_datasets[i:i+datasets_per_row]]
            row_data = [model] + [str('%.2f'%val) for val in row_results]
            latex_table += " & ".join(row_data) + " \\\\\n"
        latex_table += "\\midrule\n"

    latex_table += "\\end{tabular}\n\\caption{}\n\\end{table*}"
    return latex_table
