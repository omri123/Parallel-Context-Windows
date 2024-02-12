from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained('TinyLlama/TinyLlama-1.1B-Chat-v1.0')

messages = [
    {
        "role": "system",
        "content": "You are a friendly chatbot who always responds in the style of a pirate",
    },
    {"role": "user", "content": "How many helicopters can a human eat in one sitting?"},
]

result = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
print(result)