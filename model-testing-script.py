#!/usr/bin/env python3

#%%
import os
import torch
import pandas as pd
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer, StopStringCriteria
from tqdm.auto import tqdm
import json
import time
from datetime import datetime
from pathlib import Path
import dotenv
import importlib
import logging
from collections import OrderedDict

from utils import show_shape, get_hf_token

#%%
# Load environment variables
dotenv.load_dotenv()
my_hf_token = get_hf_token()

#use_model_name = "Qwen/Qwen2.5-7B-Instruct"
use_model_name = "meta-llama/Llama-3.1-8B-Instruct"
use_layers = None #[0, 4, 8, 12, 16, 20, 24]
capture_activations_max = 999  # all

#%%
# Import our dataset generator
#import quotation_test_dataset
#from quotation_test_dataset import generate_quotation_test_dataset
#importlib.reload(quotation_test_dataset)

import simplified_test_dataset
from simplified_test_dataset import generate_simplified_test_dataset
importlib.reload(simplified_test_dataset)

#%%
from typing import Dict, Callable, Optional

# https://github.com/pytorch/pytorch/issues/70455#issuecomment-1002814912
def _remove_all_forward_hooks(
    module: torch.nn.Module, hook_fn_name: Optional[str] = None
) -> None:
    """
    This function removes all forward hooks in the specified module, without requiring
    any hook handles. This lets us clean up & remove any hooks that weren't property
    deleted.

    Warning: Various PyTorch modules and systems make use of hooks, and thus extreme
    caution should be exercised when removing all hooks. Users are recommended to give
    their hook function a unique name that can be used to safely identify and remove
    the target forward hooks.

    Args:

        module (nn.Module): The module instance to remove forward hooks from.
        hook_fn_name (str, optional): Optionally only remove specific forward hooks
            based on their function's __name__ attribute.
            Default: None
    """

    if hook_fn_name is None:
        logging.warn("Removing all active hooks can break some PyTorch modules & systems.")


    def _remove_hooks(m: torch.nn.Module, name: Optional[str] = None) -> None:
        if hasattr(module, "_forward_hooks"):
            if m._forward_hooks != OrderedDict():
                if name is not None:
                    dict_items = list(m._forward_hooks.items())
                    m._forward_hooks = OrderedDict(
                        [(i, fn) for i, fn in dict_items if fn.__name__ != name]
                    )
                else:
                    m._forward_hooks: Dict[int, Callable] = OrderedDict()

    def _remove_child_hooks(
        target_module: torch.nn.Module, hook_name: Optional[str] = None
    ) -> None:
        for name, child in target_module._modules.items():
            if child is not None:
                _remove_hooks(child, hook_name)
                _remove_child_hooks(child, hook_name)

    # Remove hooks from target submodules
    _remove_child_hooks(module, hook_fn_name)

    # Remove hooks from the target module
    _remove_hooks(module, hook_fn_name)

def remove_all_model_hooks(model, hook_fn_name=None):
    # for each layer in the model
    for layer in model.model.layers:
        # remove all forward hooks
        _remove_all_forward_hooks(layer, hook_fn_name)

#%%
def print_model_device_map(model):
    """Print which device each model component is placed on."""
    # For models with explicit device maps
    if hasattr(model, "hf_device_map"):
        print(f"\nHuggingFace Device Map ({model.config._name_or_path}):")
        for key, value in model.hf_device_map.items():
            print(f"{key}: {value}")
    else:
        print(f"\nModel Device Map ({model.config._name_or_path}):")
        for name, param in model.named_parameters():
            print(f"{name}: {param.device}")

#%%
def print_model_structure(model, max_depth=2):
    """Print the structure of the model to understand the layer hierarchy."""
    def _print_structure(module, prefix="", depth=0):
        if depth > max_depth:
            return

        for name, child in module.named_children():
            child_type = child.__class__.__name__
            print(f"{prefix}- {name} ({child_type})")
            _print_structure(child, prefix + "  ", depth + 1)

    model_type = model.__class__.__name__
    print(f"Model: {model_type}")
    _print_structure(model)

#%%

if "MODEL_CACHE" not in globals():
    MODEL_CACHE = {}

#%%
# Let's just try the first one
#models = list(MODEL_CACHE.keys())
#print_model_structure(MODEL_CACHE[models[0]][0])

#%%
def setup_model(model_name=use_model_name, base_model=False, device_map="cuda"):
    """
    Load the specified model and tokenizer.

    Args:
        model_name: Name or path of the model to load
        base_model: If True, load base model instead of instruct version

    Returns:
        tuple: (model, tokenizer)
    """
    # Switch to base model if requested
    if base_model and "Instruct" in model_name:
        model_name = model_name.replace("-Instruct", "")

    # Check if model is already loaded
    if model_name in MODEL_CACHE:
        print(f"Using cached model: {model_name}")
        print_model_device_map(MODEL_CACHE[model_name][0])
        return MODEL_CACHE[model_name]

    print(f"Loading model: {model_name}")

    # Load model with bfloat16 precision to save memory
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map=device_map,
        token=my_hf_token,
    )
    print_model_device_map(model)

    tokenizer = AutoTokenizer.from_pretrained(model_name, token=my_hf_token, padding_side='left')
    if tokenizer.pad_token is None:
        # Set pad_token to eos_token for Llama tokenizers
        tokenizer.pad_token = tokenizer.eos_token
        print(f"Setting pad_token to eos_token: {tokenizer.eos_token}")

    # Cache the model
    MODEL_CACHE[model_name] = (model, tokenizer)
    return model, tokenizer

#%%
default_sys_prompt = "You are a helpful assistant. Follow the user's instructions. Give only single-word answers."

def format_prompt(prompt, instruct=False, sys_prompt=default_sys_prompt):
    if instruct:
        # This is specific to the llama family of models.
        # https://github.com/meta-llama/llama-models/blob/main/models/llama3_2/text_prompt_format.md
        formatted_prompt = f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>

{sys_prompt}<|eot_id|><|start_header_id|>user<|end_header_id|>

{prompt}<|eot_id|><|start_header_id|>assistant<|end_header_id|>

"""
    else:
        formatted_prompt = f"""The following is a transcript of a conversation between a user and an AI assistant.

System:
{sys_prompt}

User:
{prompt}

Assistant:
"""
    return formatted_prompt

#%%
def run_tests_batched(model, tokenizer, test_dataset, output_dir, batch_size=99, gpu_batch_size=99, capture_layers=None, top_n=20):
    """
    Run tests on the model using batched processing for better GPU utilization.

    Args:
        model: The loaded model
        tokenizer: The tokenizer
        test_dataset: DataFrame containing test cases
        output_dir: Directory to save results
        batch_size: Number of examples to process simultaneously
        capture_layers: List of layer indices to capture activations from (None = capture all)

    Returns:
        DataFrame: The test dataset with response columns added
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Add response column to dataset
    test_dataset['response'] = None

    # Store activations separately if requested
    # XXX if capture_layers is not None:
    activation_data = {}

    # Set up stopping criteria for Llama 3 models
    model_name = model.config._name_or_path
    instruct = "Instruct" in model_name
    eos_token_ids = [tokenizer.eos_token_id]  # Default EOS token
    stopping_criteria = []

    # Add Llama 3 specific stopping tokens
    if instruct:
        stop_tokens = ["<|begin_of_text|>", "<|start_header_id|>", "<|end_header_id|>", "<|eot_id|>"]
        #print(f"Special tokens in the tokenizer's vocabulary: {tokenizer.special_tokens_map}")
        for token in stop_tokens:
            if token in tokenizer.get_vocab():
                token_id = tokenizer.convert_tokens_to_ids(token)
                eos_token_ids.append(token_id)
                #print(f"Added {token} token (ID: {token_id}) to stopping criteria")
            else:
                print(f"WARNING: Token {token} not found in tokenizer's vocabulary")
    else:
        # We want to stop any time we see "User:" or "Assistant:" in the response, but they might not be single tokens.
        # So we'll just stop at the first token that matches "User" or "Assistant".
        stop_strings = ["User:", "Assistant:"]
        # XXX: this doesn't do _quite_ what we want, because it leaves the stop string in the response.
        stopping_criteria = [StopStringCriteria(tokenizer, stop_strings)]

    print(f"Using stopping token IDs: {eos_token_ids}")

    # Remove any existing hooks on the model
    remove_all_model_hooks(model, hook_fn_name="get_activations_hook_gwillen")

    # Add columns for top n logits and their probabilities
    test_dataset['top_n_tokens'] = None
    test_dataset['top_n_probs'] = None

    # Process test cases in batches
    for batch_start in tqdm(range(0, len(test_dataset), batch_size), desc="Running batched tests"):
        batch_end = min(batch_start + batch_size, len(test_dataset))
        batch_indices = list(range(batch_start, batch_end))
        batch = test_dataset.iloc[batch_indices]

        # Get prompts for this batch
        prompts = batch['prompt'].tolist()
        batch_test_ids = batch['id'].tolist()

        # Format prompts based on model type
        model_name = model.config._name_or_path
        formatted_prompts = []

        for prompt in prompts:
            formatted_prompt = format_prompt(prompt, instruct=instruct)
            formatted_prompts.append(formatted_prompt)

        # Tokenize all prompts in the batch
        batch_inputs = tokenizer(formatted_prompts, padding=True, return_tensors="pt").to(model.device)

        # Generate responses for the batch
        batch_responses = []
        batch_activations = []
        input_lengths = []

        # Process in smaller generation batches if needed (can help with memory)
        gen_batch_size = min(batch_size, gpu_batch_size)  # Adjust based on your GPU memory

        # XXX: right now this ends up doing only a single batch. If this actually goes back to being multiple batches, it might break something.
        for i in range(0, len(formatted_prompts), gen_batch_size):
            sub_batch_end = min(i + gen_batch_size, len(formatted_prompts))
            sub_batch_size = sub_batch_end - i
            sub_batch_indices = list(range(i, sub_batch_end))

            # Extract inputs for this generation sub-batch
            sub_batch_inputs = {
                'input_ids': batch_inputs.input_ids[sub_batch_indices],
                'attention_mask': batch_inputs.attention_mask[sub_batch_indices]
            }

            # Record input lengths for extracting only the generated text later
            sub_input_lengths = [len(ids) for ids in sub_batch_inputs['input_ids']]
            input_lengths.extend(sub_input_lengths)

            # Set up hooks to capture activations
            activation_dict = {}
            hooks = []

            layer_numbers = capture_layers
            if capture_layers is None:
                # Capture all transformer layers
                layer_numbers = list(range(model.config.num_hidden_layers))

            # Flag to track if we're in the initial pass or generation steps
            initial_pass_complete = False

            def get_activation(name):
                def get_activations_hook_gwillen(module, input, output):
                    # Move activations to CPU to save GPU memory
                    # Keep the batch dimension intact
                    shape = output[0].shape
                    if shape[1] == 1:
                        return  # only do prompt activations
                    else:
                        pass #print(f"Activation shape: {shape}; shape as list: {list(shape)}; shape[1]: {shape[1]}")
                    # Get only the last capture_activations_max activations
                    use_activations = output[0][:, -capture_activations_max:, :]
                    activation_dict[name] = use_activations.detach().cpu().float()
                    # Log about it
                    #print(f"Captured activation for {name} with shape {activation_dict[name].shape} in batch starting at {i}")
                return get_activations_hook_gwillen

            # Register hooks for the specified layers
            for layer_idx in layer_numbers:
                layer_name = f"layer_{layer_idx}"
                hook = get_activation(layer_name)
                layer = model.model.layers[layer_idx]
                hooks.append(layer.register_forward_hook(hook))
                # log about it
                #print(f"Registered hook for {layer_name} for batch starting at {i}")
            print(f"Registered hooks for batch starting at {i}")

            # Generate
            with torch.no_grad():
                outputs = model.generate(
                    **sub_batch_inputs,
                    max_new_tokens=512,
                    temperature=None,
                    top_p=None,
                    do_sample=False,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=eos_token_ids,
                    stopping_criteria=stopping_criteria,
                    output_scores=True,  # Enable output scores
                    return_dict_in_generate=True  # Return a dictionary with scores
                )

            # output is a dictionary with keys 'sequences' and 'scores'
            # sequences is a tensor of shape (batch_size, seq_len)
            # scores is a tensor of shape (batch_size, seq_len, vocab_size)
            show_shape(outputs, "generate outputs: ")

            # Extract logits for the first token of the response
            logits = outputs.scores[0]  # Get logits for the first generated token
            probs = torch.softmax(logits, dim=-1)  # Convert logits to probabilities
            top_n_probs, top_n_indices = torch.topk(probs, top_n)  # Get top n probabilities and their indices

            show_shape(top_n_indices, "top n indices: ")

            # Convert token indices to text
            top_n_tokens = [[tokenizer.decode(idx) for idx in indices] for indices in top_n_indices]
            show_shape(top_n_tokens, "top n tokens: ")
            show_shape(top_n_probs, "top n probs: ")

            # Process each output in the sub-batch
            for j, output in enumerate(outputs.sequences):
                input_length = sub_input_lengths[j]
                generated_ids = output[input_length:]
                generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True)
                batch_responses.append(generated_text)

                # Save top n tokens and their probabilities
                test_dataset.at[batch_indices[j], 'top_n_tokens'] = top_n_tokens[j]
                test_dataset.at[batch_indices[j], 'top_n_probs'] = top_n_probs[j].tolist()

            # Remove the hooks
            for hook in hooks:
                hook.remove()

            # Store activations
            for i in range(sub_batch_size):
                test_activations = {}  # for this test case, entry i in the batch
                # Since we're batching, we need to mask the activations with the attention mask
                attn_mask = sub_batch_inputs['attention_mask'][i]
                real_length = attn_mask.sum().item()
                if not (torch.all(attn_mask[-real_length:] == 1) and torch.all(attn_mask[:-real_length] == 0)):
                    print(f"Attention mask for test case {i} is not as expected: {attn_mask}")
                    print(f"We expected padding of zeroes, followed by (real_length) of ones. Real length: {real_length}")
                    assert False
                for layer_name, activations in activation_dict.items():
                    test_activations[layer_name] = activations[i, -real_length:, :]
                #batch_activations.append(test_activations)
                test_id = batch_test_ids[i]
                activation_data[test_id] = {
                    'prompt': prompts[i],
                    'activation_info': {
                        layer: tensor.shape for layer, tensor in test_activations.items()
                    }
                }
                activation_path = os.path.join(output_dir, f"activations_{test_id}.pt")
                print(f"Writing out activations: {activation_path}")
                torch.save(test_activations, activation_path)
                del test_activations  # Free up memory?

            # Free up memory
            del activation_dict

            # XXX... I'm not sure this is doing anything? But why not?
            torch.cuda.empty_cache()

        # For the base model, trim off our stopwords from the end of the response, as well as any preceding newlines.
        if not instruct:
            for i, response in enumerate(batch_responses):
                # Remove trailing stop words
                response = response.rstrip()  # XXX: why is this necessary?
                for stopword in stop_strings:
                    if response.endswith(stopword):
                        response = response[:response.rfind(stopword)]
                    else:
                        pass #print(f"Response {response} does not end with stopword {stopword}")
                batch_responses[i] = response.rstrip()

        # Store responses in the dataset
        for i, idx in enumerate(batch_indices):
            test_dataset.at[idx, 'response'] = batch_responses[i]

    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Save test results as CSV
    results_path = os.path.join(output_dir, f"test_results_{timestamp}.csv")
    test_dataset.to_csv(results_path, index=False)

    # Save test results as JSON for easier analysis
    json_path = os.path.join(output_dir, f"test_results_{timestamp}.json")
    test_dataset.to_json(json_path, orient='records')

    # Save activation metadata if captured
    activation_meta_path = os.path.join(output_dir, f"activation_metadata_{timestamp}.json")
    with open(activation_meta_path, 'w') as f:
        json.dump(activation_data, f, indent=2)

    return test_dataset

#%%
def main():
    """Main function to run the tests on both base and instruct models."""
    # Generate the test dataset
    print("Generating test dataset...")
    test_dataset = generate_simplified_test_dataset()
    print(f"Generated {len(test_dataset)} test cases")

    # Create timestamped output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_output_dir = f"results/simplified_test_{timestamp}"
    os.makedirs(base_output_dir, exist_ok=True)

    # Symlink the latest results to a fixed location
    latest_output_dir = "latest_results"
    if os.path.exists(latest_output_dir):  # XXX: this doesn't work if the link is dangling lol
        os.remove(latest_output_dir)
    os.symlink(base_output_dir, latest_output_dir, target_is_directory=True)

    # Save the test dataset
    test_dataset.to_csv(os.path.join(base_output_dir, "test_dataset.csv"), index=False)
    test_dataset.to_json(os.path.join(base_output_dir, "test_dataset.json"), orient='records')

    # Define models to test
    models_to_test = [
        {"name": "meta-llama/Llama-3.1-8B-Instruct", "is_base": False, "device_map": "cuda:0", "selected_layers": use_layers},
        {"name": "meta-llama/Llama-3.1-8B", "is_base": True, "device_map": "cuda:1", "selected_layers": use_layers}
    ]

    # Test each model
    for model_config in models_to_test:
        model_name = model_config["name"]
        is_base = model_config["is_base"]
        selected_layers = model_config["selected_layers"]
        device_map = model_config.get("device_map", "cuda")

        print(f"\nTesting model: {model_name}")

        # Create model-specific output directory
        model_dir_name = model_name.replace("/", "_")
        model_output_dir = os.path.join(base_output_dir, model_dir_name)

        # Create a symlink from either 'base' or 'instruct' to the model-specific output directory
        symlink_name = "base" if is_base else "instruct"
        symlink_path = os.path.join(base_output_dir, symlink_name)
        if os.path.exists(symlink_path):
            os.remove(symlink_path)
        os.symlink(model_dir_name, symlink_path, target_is_directory=True)

        try:
            # Setup model
            model, tokenizer = setup_model(model_name, base_model=is_base, device_map=device_map)

            # Run tests
            print(f"Running tests and capturing activations from layers {selected_layers}...")
            results = run_tests_batched(model, tokenizer, test_dataset.copy(), model_output_dir, capture_layers=selected_layers)

            print(f"Tests completed for {model_name}. Results saved to {model_output_dir}")

            # Clean up to free memory
            # del model
            # torch.cuda.empty_cache()

        except Exception as e:
            print(f"Error testing {model_name}: {str(e)}")
            import traceback
            traceback.print_exc()

    print(f"\nAll tests completed. Results saved to {base_output_dir}")

#%%
if __name__ == "__main__":
    main()

#%%

print("Done!")
#%%
