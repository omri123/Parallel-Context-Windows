import torch
from transformers import AutoConfig, LlamaTokenizer, GPT2Tokenizer, PreTrainedTokenizerBase

from modeling_gpt2_with_pcw import GPT2LMHeadPCW
from pcw_wrapper import PCWModelWrapper
from modeling_llama_with_pcw import LlamaForCausalLMPCW
from transformers import AutoTokenizer

GPT2_WINDOW_SIZE = 1024
LLAMA_WINDOW_SIZE = 2048


def validate_model_name(model_name: str) -> None:
    assert 'gpt2' in model_name or model_name == 'TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T' \
            or model_name == 'TinyLlama/TinyLlama-1.1B-Chat-v1.0'


def load_tokenizer(model_name: str) -> PreTrainedTokenizerBase:
    return AutoTokenizer.from_pretrained(model_name)


def load_pcw_wrapper(model_name: str, cache_dir: str = None,
                     right_indentation: bool = False, n_windows: int = 1) -> PCWModelWrapper:
    validate_model_name(model_name)
    config = AutoConfig.from_pretrained(model_name)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    multi_gpus = torch.cuda.device_count() > 1
    model_args = {
        "cache_dir": cache_dir
    }
    if multi_gpus:
        model_args["device_map"] = "auto"
        model_args["low_cpu_mem_usage"] = True
    if hasattr(config, "torch_dtype") and config.torch_dtype is not None:
        model_args["torch_dtype"] = config.torch_dtype

    if 'gpt2' in model_name:
        # we override n_positions to bi pass the model's context window size restriction
        # (for gpt2, n_positions determines the causal attention mask matrix dimension).
        # The correct position embeddings (i.e., gpt2's 1024 trained position embeddings) are re-inserted to the model
        # in GPT2LMHeadWithPCWModel initialization.
        model_args['ignore_mismatched_sizes'] = True
        model_args['n_positions'] = GPT2_WINDOW_SIZE * n_windows
        model_obj = GPT2LMHeadPCW
        context_window_size = GPT2_WINDOW_SIZE
    else:
        #  Note that some LLaMa versions located in HF have an incorrect token mapping, we correct it here:
        # see: https://huggingface.co/decapoda-research/llama-7b-hf/discussions/12
        # also: https://github.com/tloen/alpaca-lora/issues/279
        model_args['bos_token_id'] = 1
        model_args['eos_token_id'] = 2
        model_obj = LlamaForCausalLMPCW
        context_window_size = LLAMA_WINDOW_SIZE

    tokenizer = load_tokenizer(model_name)
    model = model_obj.from_pretrained(model_name, **model_args).eval()
    if not multi_gpus:
        model = model.to(device)

    return PCWModelWrapper(model, tokenizer, device, context_window_size, right_indentation)
