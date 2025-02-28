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

#%%
# Load environment variables
dotenv.load_dotenv()
my_hf_token = os.getenv("HF_TOKEN")

#use_model_name = "Qwen/Qwen2.5-7B-Instruct"
use_model_name = "meta-llama/Llama-3.1-8B-Instruct"

#%%
# Import our dataset generator
from quotation_test_dataset import generate_quotation_test_dataset

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

MODEL_CACHE = {}

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
def capture_activations(model, tokenizer, prompt, layer_numbers=None):
    """
    Capture activations from specific layers for a given prompt.

    Args:
        model: The loaded model
        tokenizer: The tokenizer
        prompt: The input prompt
        layer_numbers: List of layer indices to capture (None = all layers)

    Returns:
        dict: Mapping of layer indices to activation tensors
    """
    raise NotImplementedError("We don't use this right now.")
    # Format prompt as needed
    if "Instruct" in model.config._name_or_path:
        if "Llama-3" in model.config._name_or_path:
            ### XXX: bad bad bad
            formatted_prompt = f"<|begin_of_text|><|user|>\n{prompt}<|end_of_turn|>\n<|assistant|>\n"
        else:
            formatted_prompt = f"### Instruction:\n{prompt}\n\n### Response:\n"
    else:
        formatted_prompt = prompt

    # Tokenize the prompt
    inputs = tokenizer(formatted_prompt, return_tensors="pt").to(model.device)
    input_ids = inputs.input_ids

    # Set up hooks to capture activations
    activation_dict = {}
    hooks = []

    if layer_numbers is None:
        # Capture all transformer layers
        layer_numbers = list(range(model.config.num_hidden_layers))

    def get_activation(name):
        def hook(module, input, output):
            # Move activations to CPU to save GPU memory
            activation_dict[name] = output[0].detach().cpu().float()
        return hook

    # Register hooks for the specified layers
    for layer_idx in layer_numbers:
        layer_name = f"layer_{layer_idx}"
        hook = get_activation(layer_name)
        layer = model.model.layers[layer_idx]
        hooks.append(layer.register_forward_hook(hook))

    # Run a forward pass
    with torch.no_grad():
        outputs = model(input_ids=input_ids)

    # Remove the hooks
    for hook in hooks:
        hook.remove()

    return activation_dict

#%%
default_sys_prompt = "You are a helpful assistant. Follow the user's instructions. Be concise."

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

Sytem: {sys_prompt}

User: {prompt}

Assistant: """
    return formatted_prompt

#%%
def run_tests_batched(model, tokenizer, test_dataset, output_dir, batch_size=8, capture_layers=None):
    """
    Run tests on the model using batched processing for better GPU utilization.

    Args:
        model: The loaded model
        tokenizer: The tokenizer
        test_dataset: DataFrame containing test cases
        output_dir: Directory to save results
        batch_size: Number of examples to process simultaneously
        capture_layers: List of layer indices to capture activations from (None = no capture)

    Returns:
        DataFrame: The test dataset with response columns added
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Add response column to dataset
    test_dataset['response'] = None

    # Store activations separately if requested
    if capture_layers is not None:
        activation_data = {}

    # Set up stopping criteria for Llama 3 models
    model_name = model.config._name_or_path
    eos_token_ids = [tokenizer.eos_token_id]  # Default EOS token
    stopping_criteria = []

    # Add Llama 3 specific stopping tokens
    if "Instruct" in model_name:
        stop_tokens = ["<|begin_of_text|>", "<|start_header_id|>", "<|end_header_id|>", "<|eot_id|>"]
        # print out the list of special tokens in the tokenizer's vocabulary
        print(f"Special tokens in the tokenizer's vocabulary: {tokenizer.special_tokens_map}")
        for token in stop_tokens:
            if token in tokenizer.get_vocab():
                token_id = tokenizer.convert_tokens_to_ids(token)
                eos_token_ids.append(token_id)
                print(f"Added {token} token (ID: {token_id}) to stopping criteria")
            else:
                print(f"Token {token} not found in tokenizer's vocabulary")
    else:
        # We want to stop any time we see "User:" or "Assistant:" in the response, but they might not be single tokens.
        # So we'll just stop at the first token that matches "User" or "Assistant".
        stop_strings = ["User", "Assistant"]
        # XXX: this doesn't do _quite_ what we want, because it leaves the stop string in the response.
        stopping_criteria = [StopStringCriteria(tokenizer, stop_strings)]

    print(f"Using stopping token IDs: {eos_token_ids}")

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
            instruct = "Instruct" in model_name
            formatted_prompt = format_prompt(prompt, instruct=instruct)
            formatted_prompts.append(formatted_prompt)

        # Tokenize all prompts in the batch
        batch_inputs = tokenizer(formatted_prompts, padding=True, return_tensors="pt").to(model.device)

        # Generate responses for the batch
        batch_responses = []
        input_lengths = []

        # Process in smaller generation batches if needed (can help with memory)
        gen_batch_size = min(batch_size, 2)  # Adjust based on your GPU memory

        for i in range(0, len(formatted_prompts), gen_batch_size):
            sub_batch_end = min(i + gen_batch_size, len(formatted_prompts))
            sub_batch_indices = list(range(i, sub_batch_end))

            # Extract inputs for this generation sub-batch
            sub_batch_inputs = {
                'input_ids': batch_inputs.input_ids[sub_batch_indices],
                'attention_mask': batch_inputs.attention_mask[sub_batch_indices]
            }

            # Record input lengths for extracting only the generated text later
            sub_input_lengths = [len(ids) for ids in sub_batch_inputs['input_ids']]
            input_lengths.extend(sub_input_lengths)

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
                )

            # Process each output in the sub-batch
            for j, output in enumerate(outputs):
                input_length = sub_input_lengths[j]
                generated_ids = output[input_length:]
                generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True)
                batch_responses.append(generated_text)

        # Store responses in the dataset
        for i, idx in enumerate(batch_indices):
            test_dataset.at[idx, 'response'] = batch_responses[i]

        # Capture activations if requested
        if capture_layers is not None:
            # For activation capture, process each prompt individually
            # (since we need to attach hooks and capture per-example)
            for i, prompt in enumerate(prompts):
                test_id = batch_test_ids[i]
                formatted_prompt = formatted_prompts[i]

                # Capture activations
                activations = capture_activations(model, tokenizer, formatted_prompt, capture_layers)

                # Store activation metadata
                activation_data[test_id] = {
                    'prompt': prompt,
                    'activation_info': {
                        layer: tensor.shape for layer, tensor in activations.items()
                    }
                }

                # Save activations to disk directly to avoid OOM
                activation_path = os.path.join(output_dir, f"activations_{test_id}.pt")
                torch.save(activations, activation_path)

                # Clear GPU memory
                del activations
                torch.cuda.empty_cache()

    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Save test results as CSV
    results_path = os.path.join(output_dir, f"test_results_{timestamp}.csv")
    test_dataset.to_csv(results_path, index=False)

    # Save test results as JSON for easier analysis
    json_path = os.path.join(output_dir, f"test_results_{timestamp}.json")
    test_dataset.to_json(json_path, orient='records')

    # Save activation metadata if captured
    if capture_layers is not None:
        activation_meta_path = os.path.join(output_dir, f"activation_metadata_{timestamp}.json")
        with open(activation_meta_path, 'w') as f:
            json.dump(activation_data, f, indent=2)

    return test_dataset

#%%
"""
# Helper function to apply sharding across multiple GPUs
def get_sharded_model(model_name, num_gpus=None):
    " ""
    Load a model with explicit sharding across multiple GPUs.

    Args:
        model_name: Name of the model to load
        num_gpus: Number of GPUs to use (None = auto-detect)

    Returns:
        tuple: (model, tokenizer)
    " ""
    if num_gpus is None:
        num_gpus = torch.cuda.device_count()

    if num_gpus <= 1:
        # Fall back to regular loading
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            token=my_hf_token,
        )
    else:
        # Create custom device map to distribute across specific GPUs
        device_map = {}

        # For Llama models, try a simple layer-based sharding strategy
        if "Llama" in model_name:
            # Get model config to determine number of layers
            config = AutoConfig.from_pretrained(model_name, token=my_hf_token)
            num_layers = config.num_hidden_layers

            # Assign base components to first GPU
            device_map["model.embed_tokens"] = 0
            device_map["model.norm"] = num_gpus - 1  # Last GPU
            device_map["lm_head"] = num_gpus - 1     # Last GPU

            # Distribute layers across GPUs
            layers_per_gpu = num_layers // num_gpus
            remainder = num_layers % num_gpus

            # Assign layers to GPUs
            for i in range(num_layers):
                # Determine which GPU gets this layer
                if remainder > 0 and i >= (num_layers - remainder):
                    # Distribute remainder layers across GPUs
                    gpu_id = (i - (num_layers - remainder)) % num_gpus
                else:
                    gpu_id = min(i // layers_per_gpu, num_gpus - 1)

                device_map[f"model.layers.{i}"] = gpu_id
        else:
            # For other models, use auto device map
            device_map = "auto"

        # Load the model with the device map
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            device_map=device_map,
            token=my_hf_token,
        )

    tokenizer = AutoTokenizer.from_pretrained(model_name, token=my_hf_token, padding_side='left')
    if tokenizer.pad_token is None:
        # Set pad_token to eos_token for Llama tokenizers
        tokenizer.pad_token = tokenizer.eos_token
        print(f"Setting pad_token to eos_token: {tokenizer.eos_token}")
    print("Setting padding_side to 'left' for more efficient generation")
    return model, tokenizer
"""

#%%
def main():
    """Main function to run the tests on both base and instruct models."""
    # Generate the test dataset
    print("Generating test dataset...")
    test_dataset = generate_quotation_test_dataset()
    print(f"Generated {len(test_dataset)} test cases")

    # Create timestamped output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_output_dir = f"quotation_test_results_{timestamp}"
    os.makedirs(base_output_dir, exist_ok=True)

    # Save the test dataset
    test_dataset.to_csv(os.path.join(base_output_dir, "test_dataset.csv"), index=False)
    test_dataset.to_json(os.path.join(base_output_dir, "test_dataset.json"), orient='records')

    # Define models to test
    models_to_test = [
        {"name": "meta-llama/Llama-3.1-8B-Instruct", "is_base": False, "device_map": "cuda:0", "selected_layers": None}, # XXX [0, 8, 16, 24]},
        {"name": "meta-llama/Llama-3.1-8B", "is_base": True, "device_map": "cuda:1", "selected_layers": None}, # XXX [0, 8, 16, 24]}
    ]

    # Test each model
    for model_config in models_to_test:
        model_name = model_config["name"]
        is_base = model_config["is_base"]
        selected_layers = model_config["selected_layers"]
        device_map = model_config.get("device_map", "cuda")

        print(f"\nTesting model: {model_name}")

        # Create model-specific output directory
        model_dir_name = "base" if is_base else "instruct"
        model_output_dir = os.path.join(base_output_dir, model_dir_name)

        try:
            # Setup model
            model, tokenizer = setup_model(model_name, base_model=is_base, device_map=device_map)

            # Run tests
            print(f"Running tests and capturing activations from layers {selected_layers}...")
            results = run_tests_batched(model, tokenizer, test_dataset.copy(), model_output_dir, capture_layers=selected_layers)

            print(f"Tests completed for {model_name}. Results saved to {model_output_dir}")

            # Clean up to free memory
            del model
            torch.cuda.empty_cache()

        except Exception as e:
            print(f"Error testing {model_name}: {str(e)}")
            # print the exception e with a traceback:
            import traceback
            traceback.print_exc()

    print(f"\nAll tests completed. Results saved to {base_output_dir}")

#%%
if __name__ == "__main__":
    main()

# %%
