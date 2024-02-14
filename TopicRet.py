import json
import re
import os
import argparse
from tqdm import tqdm

from model_loaders import load_pcw_wrapper
from logits_processor import RestrictiveTokensLogitsProcessor


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

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--just-one', action='store_true', help='infer the first and print')
    parser.add_argument('--out', type=str, help='path to put results for evaluation')
    parser.add_argument('--temp', type=float, help='sampling temperature, default to zero')
    args = parser.parse_args()
    
    wrapper = load_pcw_wrapper('TinyLlama/TinyLlama-1.1B-Chat-v1.0')
    outputs = []
    for sample in tqdm(get_samples()):
        windows = sample['windows']
        topics = sample['topics']
        task_text= '<|user|>\nWhat topics did we just discussed?</s>'
        do_sample = args.temp == 0
        # original instruction: '<|user|>\nWhat is the first topic we discussed? Only give me the topic name. Do not summarize yourself.</s>'
        output = wrapper.pcw_generate(contexts=windows,
                                    task_text=task_text,
                                    temperature=args.temp,
                                    do_sample=do_sample,
                                    max_new_tokens=100)
        outputs.append(output)
        if args.just_one:
            print(f"topics were:{os.linesep} {os.linesep.join(topics)}\n")
            print(f'model output is {output}')
            return
        
        with open(args.out, 'w+') as f:
            for result in outputs:
                f.write(result)

if __name__ == '__main__':
    main()

print("next steps: probably \n  1. document choosing using topk for better coherency, and\n  2. dependent windows.")
print("Idea: maybe just RAG above contextualized documents?")
