import numpy as np

from model_loaders import load_pcw_wrapper
from logits_processor import RestrictiveTokensLogitsProcessor

wrapper = load_pcw_wrapper('TinyLlama/TinyLlama-1.1B-Chat-v1.0', n_windows=2)

context_a = '<|system|>\nYou are a friendly chatbot named Lucas, and Danny is 35 years old</s>'
context_b = '<|system|>\nYou are a friendly chatbot named Lucas, and Danny is a football player</s>'
task_text= '<|user|>\nHi Lucas, please tell me about Danny.</s>'

# use PCW for generation:
output = wrapper.pcw_generate(contexts=[context_a, context_b],
                              task_text=task_text,
                              temperature=0,
                              do_sample=False,
                              max_new_tokens=100)
print(output)