import pickle
import json
from utils import MultiChat
from collections import defaultdict


if __name__ == "__main__":
    with open('icl_eval/continuation/saved_final.pkl', 'rb') as f:
        mem = pickle.load(f)

    chat = MultiChat({"gpt-4o_keys": []},
        save_path="icl_eval/continuation/SlimPajama_causal_filter.json",
        model="gpt-4o-2024-05-13",
        temperature=0,
        max_tokens=2
    )
    chat.start()

    PROMPT_PARA = 'Given sentence A, please determine whether sentence B is correct on a scale from 0 to 10. Please output only a single number between 0 and 10. Do not include any other information.'
    PROMPT_CODE = 'Given code A, please determine whether the completion code B is correct on a scale from 0 to 10. Please output only a single number between 0 and 10. Do not include any other information.'

    for k, v in mem.items():
        for it in v:
            if k == "github":
                line = {
                    "split": k,
                    "text": it,
                    "prompt": [{"role": "system", "content": PROMPT_CODE}, 
                            {"role": "user", "content": f"A. {it[0]}\n\n\nB. {it[1]}"}]
                }
            else:
                line = {
                    "split": k,
                    "text": it,
                    "prompt": [{"role": "system", "content": PROMPT_PARA}, 
                            {"role": "user", "content": f"A. {it[0]}\n\nB. {it[1]}"}]
                }
            chat.post(line)

    chat.wait_finish()

    chat = MultiChat({"gpt-4o_keys": []},
        save_path="icl_eval/continuation/SlimPajama_causal_change.json",
        model="gpt-4o-2024-05-13",
        temperature=0.8
    )
    chat.start()

    PROMPT_SENT = 'Please slightly modify the following sentence to make it completely incorrect or nonsensical. Only provide the modified sentence, without any additional information.'
    PROMPT_CODE = 'Please slightly modify the following code so that it is syntactically correct but completely incorrect in function. Only provide the modified sentence, without any additional information.'

    with open("icl_eval/continuation/SlimPajama_causal_filter.json", "r", encoding="utf-8") as f:
        for line in f:
            line = json.loads(line)
            if int(line["response"]) < 5:
                continue
            if line["split"] == "github":
                prompt = PROMPT_CODE
            else:
                prompt = PROMPT_SENT
            it = line["text"][0]
            for cnt in range(2):
                line = {
                    "text": it,
                    "rand": cnt,
                    "prompt": [{"role": "system", "content": prompt}, 
                            {"role": "user", "content": it}]
                }
                chat.post(line)

    chat.wait_finish()


    map_to_false = defaultdict(list)
    with open("icl_eval/continuation/SlimPajama_causal_change.json", "r", encoding="utf-8") as f:
        for line in f:
            line = json.loads(line)
            map_to_false[line["text"]].append(line["response"])

    chat = MultiChat({"gpt-4o_keys": []},
        save_path="icl_eval/continuation/SlimPajama_causal_wrong.json",
        model="gpt-4o-2024-05-13",
        temperature=0.8
    )
    chat.start()

    with open("icl_eval/continuation/SlimPajama_causal_filter.json", "r", encoding="utf-8") as f:
        for line in f:
            line = json.loads(line)
            if int(line["response"]) < 5:
                continue
            line["score1"] = int(line["response"])
            del line["response"]
            if k == "github":
                for change in map_to_false[line["text"][0]]:
                    lens = len([i for i in line["text"][1].split("\n") if len(i.strip())])
                    line["change"] = change
                    line["prompt"] = [{"role": "system", "content": f'Please finish the following code using {lens} lines.'}, 
                                    {"role": "user", "content": change}]
                    chat.post(line)
            else:
                for change in map_to_false[line["text"][0]]:
                    lens = len([i for i in line["text"][1].split() if len(i.strip())])
                    line["change"] = change
                    line["prompt"] = [{"role": "system", "content": f'Please continue the following sentence using {lens} words.'}, 
                                    {"role": "user", "content": change}]
                    chat.post(line)

    chat.wait_finish()
    
    chat = MultiChat({"gpt-4o_keys": []},
        save_path="icl_eval/continuation/SlimPajama_causal_final.json",
        model="gpt-4o-2024-05-13",
        temperature=0,
        max_tokens=2
    )
    chat.start()

    PROMPT_PARA = 'Given sentence A, please determine whether sentence B is correct on a scale from 0 to 10. Please output only a single number between 0 and 10. Do not include any other information.'
    PROMPT_CODE = 'Given code A, please determine whether the completion code B is correct on a scale from 0 to 10. Please output only a single number between 0 and 10. Do not include any other information.'

    with open("icl_eval/continuation/SlimPajama_causal_wrong.json", "r", encoding="utf-8") as f:
        for line in f:
            line = json.loads(line)
            assert line["part"] == "causal"
            line = {
                "question": line["text"][0],
                "choice1": line["text"][1],
                "choice2": line["response"],
                "split": line["split"],
                "score1": line["score1"],
                "label": 0
            }
            if line["split"] == "github":
                line["prompt"] = [{"role": "system", "content": PROMPT_CODE}, 
                        {"role": "user", "content": f'A. {line["question"]}\n\n\nB. {line["choice2"]}'}]
            else:
                line["prompt"] = [{"role": "system", "content": PROMPT_PARA}, 
                        {"role": "user", "content": f'A. {line["question"]}\n\nB. {line["choice2"]}'}]
            chat.post(line)

    chat.wait_finish()


    final = defaultdict(list)

    with open("icl_eval/continuation/SlimPajama_causal_final.json", "r", encoding="utf-8") as f:
        for line in f:
            line = json.loads(line)
            if int(line["response"]) > 5:
                continue
            final[line["split"]].append((line["score1"] - int(line["response"]), line))
    for k, v in final.items():
        v = sorted(v, key=lambda x: x[0])
        v = [i[1] for i in v[-400: ]]
        with open(f"icl_eval/continuation/{k}_causal.jsonl", "w", encoding="utf-8") as w:
            for line in v:
                del line["score1"]
                del line["response"]
                del line["split"]
                w.write(json.dumps(line) + "\n")
