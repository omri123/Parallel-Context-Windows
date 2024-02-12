import numpy as np
import json
import re
from model_loaders import load_pcw_wrapper
from logits_processor import RestrictiveTokensLogitsProcessor

wrapper = load_pcw_wrapper('TinyLlama/TinyLlama-1.1B-Chat-v1.0', n_windows=2)

def split_windows(txt):
    mark = "talk about the next topic."
    indexes = [match.end() for match in re.finditer(mark, txt)]
    windows = []
    prev = 0
    for index in indexes:
        window = txt[prev:index]
        prev = index
        windows.append(window)
    
    fixed_windows =[]
    topics = []
    for window in windows:
        lines = window.splitlines()
        topic = lines[0].replace('USER: I would like to discuss the topic of the ', '').replace('.', '')
        result = []
        for line in lines:
            line = line.strip()
            line += '</s>'
            result.append(line)
    
        result = '\n'.join(result)
        result = result.replace('USER:', '<|user|>\n')
        result = result.replace('ASSISTANT:', '<|assistant|>\n')
        fixed_windows.append(result)
        topics.append(topic)
    return fixed_windows, topics

def get_samples(): # return a list of windows and a list of topics
    with open('/root/LEval/LEval-data/Closed-ended-tasks/topic_retrieval_longchat.jsonl', 'r') as f:
        for line in f:
            sample = json.loads(line)
            txt = sample['input']
            windows, topics = split_windows(txt)
            yield {'windows': windows, 'topics': topics}

sample = get_samples().__next__()

for topic in sample['topics']:
    print(topic)

windows = sample['windows']
task_text= '<|user|>\nWhat topics did we just discussed?</s>'
# original instruction: '<|user|>\nWhat is the first topic we discussed? Only give me the topic name. Do not summarize yourself.</s>'

output = wrapper.pcw_generate(contexts=windows,
                              task_text=task_text,
                              temperature=1,
                              do_sample=True,
                              max_new_tokens=100)
print('actual result is:')
print(output)

print("next steps: probably \n  1. document choosing using topk for better coherency, and\n  2. dependent windows.")