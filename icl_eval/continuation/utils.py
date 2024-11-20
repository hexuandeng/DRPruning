import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import re
import time
import json
import openai
import random
from loguru import logger
from copy import deepcopy
from queue import Empty
from multiprocessing import Process, Queue, Value


class MultiChat:
    '''
    Generate responses from the ChatGPT API for a set of inputs using multiprocessing.

    Input format: A dictionary containing a key 'prompt' and any additional information you'd like to include.
    Output format: A dictionary containing a key 'response' with the ChatGPT response and any additional information provided in the input.
    
    Arguments:
    config (dict): read from config.json.
    prefix (dict): the prefix for GPT keys. Here turbo for ChatGPT and gpt4 for GPT-4.
    save_path (str): The path where the final responses will be saved, each in a line.
    retry_func (function): A function that takes the input and corresponding ChatGPT response as parameters and returns a tuple (bool, str). 
                           The boolean value indicates whether the model needs to generate another response (True) or not (False), and the string should be prepended to the regenerated response.
    kwargs (dict): Additional parameters for the ChatCompletion function.
    '''
    def __init__(self, config, save_path, retry_func=None, **kwargs):
        if 'claude' in kwargs["model"]:
            self.api_keys = config["claude_keys"]
            if "presence_penalty" in kwargs.keys():
                del kwargs["presence_penalty"]
        elif 'gpt' in kwargs["model"]:
            prefix = '-'.join(kwargs["model"].split('-')[: 2])
            self.api_keys = config[prefix + "_keys"]

        self.retry_func = retry_func
        self.kwargs = kwargs
        self.save_path = save_path
        self.used = []
        if os.path.exists(self.save_path):
            with open(self.save_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = json.loads(line)
                    del line['response']
                    self.used.append(line)
        self.read = Queue(maxsize = 50)
        self.read_retry = Queue()
        self.read_num = 0
        self.write = Queue()
        self.write_num = Value('i', 0)
        self.p_apis = []

    def post(self, prompt):
        # get input and push into waiting queue, and emit it when it is already in the output file
        tmp = deepcopy(prompt)
        del tmp['prompt']
        if tmp not in self.used:
            self.used.append(tmp)
            self.read.put(prompt)
            self.read_num += 1

    def get(self):
        while True:
            try:
                return self.read_retry.get(block=False)
            except Empty:
                try:
                    return self.read.get(timeout=5)
                except Empty:
                    continue

    def _chat_openai(self, prompt, api_key, cnt_key, retry=3):
        for cnt in range(retry):
            try:
                chat = openai.ChatCompletion.create(
                    messages=prompt, 
                    **self.kwargs
                    )
                reply = chat.choices[0].message.content.strip()
                return reply, cnt_key
            except Exception as e:
                if isinstance(api_key, list):
                    cnt_key = (cnt_key + 1) % len(api_key)
                    openai.api_key = api_key[cnt_key]
                if cnt == retry - 1:
                    if isinstance(api_key, list):
                        logger.warning(f'{api_key[cnt_key]} retry {retry} times and failed:\n{e}')
                    else:
                        logger.warning(f'{api_key} retry {retry} times and failed:\n{e}')
                else:
                    time.sleep(random.randint(0, 5))
        if isinstance(api_key, list):
            cnt_key = (cnt_key + 1) % len(api_key)
        return None, api_key
    
    def _chat_claude(self, prompt, claude, api_key, cnt_key, retry=3):
        from anthropic import Anthropic
        for cnt in range(retry):
            try:
                message = claude.messages.create(
                    max_tokens=128,
                    messages=prompt[1: ],
                    system=prompt[0]["content"],
                    **self.kwargs
                )
                reply = message.content[0].text
                return reply, cnt_key
            except Exception as e:
                if isinstance(api_key, list):
                    cnt_key = (cnt_key + 1) % len(api_key)
                    claude = Anthropic(api_key=api_key[cnt_key])
                if cnt == retry - 1:
                    if isinstance(api_key, list):
                        logger.warning(f'{api_key[cnt_key]} retry {retry} times and failed:\n{e}')
                    else:
                        logger.warning(f'{api_key} retry {retry} times and failed:\n{e}')
                else:
                    time.sleep(random.randint(0, 5))
        if isinstance(api_key, list):
            cnt_key = (cnt_key + 1) % len(api_key)
        return None, api_key
    
    def _chat(self, api_key):
        # main function for each subprocess, get input from queue and process
        cnt_key = 0

        if 'claude' in self.kwargs["model"]:
            from anthropic import Anthropic
            if isinstance(api_key, list):
                cnt_key = random.randint(0, len(api_key) - 1)
                claude = Anthropic(api_key=api_key[cnt_key])
            elif isinstance(api_key, str):
                claude = Anthropic(api_key=api_key)
            else:
                raise KeyError(f"Invalid api_key {api_key} given! Need string or list!")
        elif 'gpt' in self.kwargs["model"]:
            if isinstance(api_key, list):
                cnt_key = random.randint(0, len(api_key) - 1)
                openai.api_key = api_key[cnt_key]
            elif isinstance(api_key, str):
                openai.api_key = api_key
            else:
                raise KeyError(f"Invalid api_key {api_key} given! Need string or list!")
        else:
            raise KeyError(f"Invalid model {self.kwargs['model']} given! Only gpt, claude are supported!")
        
        mem = None
        # process the input one by one, and push the result into the writing queue. if failed multiple times, put it back to the reading queue
        while True:
            try:
                if mem is None:
                    prompt = self.get()
                if 'claude' in self.kwargs["model"]:
                    response, cnt_key = self._chat_claude(prompt['prompt'], claude, api_key, cnt_key)
                else:
                    response, cnt_key = self._chat_openai(prompt['prompt'], api_key, cnt_key)
                if response is None:
                    cnt_key = random.randint(0, len(api_key) - 1)
                    # self.read_retry.put(prompt)
                    time.sleep(20)
                else:
                    if self.retry_func is not None:
                        if mem is not None:
                            response = mem + response
                        retry, mem = self.retry_func(prompt, response)
                        if retry:
                            continue
                    mem = None
                    prompt['response'] = response
                    self.write.put(prompt)
            except Empty:
                logger.info(api_key, "finish!")
                break

    def _write_reply(self):
        # write the results into files
        while True:
            line = self.write.get(block=True)
            del line['prompt']
            w = open(self.save_path, 'a', encoding='utf-8')
            w.write(json.dumps(line, sort_keys=True, indent=0, ensure_ascii=False).replace("\n", " ") + "\n")
            w.close()
            self.write_num.value += 1
            if self.write_num.value % 100 == 0:
                logger.info(f"Finish generating {self.write_num.value} items!")

    def start(self):
        # start all the subprocess
        logger.info(f"Starting generating and writing to {self.save_path}!")
        for keys in self.api_keys:
            p = Process(target=self._chat, args=[keys, ])
            p.start()
            self.p_apis.append(p)
        self.p_write = Process(target=self._write_reply, args=[])
        self.p_write.start()

    def try_join(proc):
        # soft join for subprocess
        proc.join(timeout=0)
        if proc.is_alive():
            return False
        return True

    def wait_finish(self):
        # routine until all the input has been processed
        while True:
            if self.write_num.value >= self.read_num:
                for p in self.p_apis:
                    p.kill()
                self.p_write.kill()
                logger.info(f"Finish generating {self.write_num.value} instances in {self.save_path}!")
                return
            else:
                logger.info(f"Finish generating {self.write_num.value} out of {self.read_num} inputs!")
                time.sleep(60)

def clean_string(s):
    return re.sub(r'[^a-zA-Z]', '', s)

def longestCommonSubsequence(str1, str2):
    str1 = clean_string(str1)
    str2 = clean_string(str2)
    
    if len(str2) < 10:
        return 0

    dp = [[0] * (len(str2) + 1) for _ in range(len(str1) + 1)]
    max_len = 0

    for i in range(1, len(str1) + 1):
        for j in range(1, len(str2) + 1):
            if str1[i - 1] == str2[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
                if dp[i][j] > max_len:
                    max_len = dp[i][j]
            else:
                dp[i][j] = 0

    if max_len > 0:
        ratio = max_len / len(str2)
        return ratio
    else:
        return 0
