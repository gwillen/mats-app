#!/usr/bin/env python3

import os
import torch
import pandas as pd
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm.auto import tqdm
import json
import time
from datetime import datetime
from pathlib import Path

use_model_name = "Qwen/Qwen2.5-7B-Instruct"
#use_model_name = "meta-llama/Llama-3.1-8B-Instruct"

# Import our dataset generator
from quotation_test_dataset import generate_quotation_test_dataset

def setup_model(model_name=use_model_name, base_model=False):
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

    print(f"Loading model: {model_name}")

    # Load model with bfloat16 precision to save memory
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto"
    )

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    return model, tokenizer

def run_model(model, tokenizer, prompt, max_new_tokens=512, temperature=0):
    """
    Run the model on a single prompt and return the generated text.

    Args:
        model: The loaded model
        tokenizer: The tokenizer
        prompt: The input prompt
        max_new_tokens: Maximum number of tokens to generate
        temperature: Sampling temperature

    Returns:
        str: The generated text
    """
    # Format for instruct model if needed (model-specific formatting)
    if "Instruct" in model.config._name_or_path:
        if "Llama-3" in model.config._name_or_path:
            # Llama 3 Instruct format
            formatted_prompt = f"<|begin_of_text|><|user|>\n{prompt}<|end_of_turn|>\n<|assistant|>\n"
        else:
            # Generic instruct format
            formatted_prompt = f"### Instruction:\n{prompt}\n\n### Response:\n"
    else:
        # For base models, just use the raw prompt
        formatted_prompt = prompt

    inputs = tokenizer(formatted_prompt, return_tensors="pt").to(model.device)

    # Generate with minimal randomness for more consistent results
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            do_sample=temperature > 0,
        )

    # Extract only the newly generated tokens
    generated_ids = outputs[0][inputs.input_ids.shape[1]:]
    generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True)

    return generated_text

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
    # Format prompt as needed
    if "Instruct" in model.config._name_or_path:
        if "Llama-3" in model.config._name_or_path:
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

def run_tests(model, tokenizer, test_dataset, output_dir, capture_layers=None):
    """
    Run tests on the model using the test dataset and save results.

    Args:
        model: The loaded model
        tokenizer: The tokenizer
        test_dataset: DataFrame containing test cases
        output_dir: Directory to save results
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

    # Process each test case
    for idx, row in tqdm(test_dataset.iterrows(), total=len(test_dataset), desc="Running tests"):
        prompt = row['prompt']
        test_id = row['id']

        # Generate response
        response = run_model(model, tokenizer, prompt)
        test_dataset.at[idx, 'response'] = response

        # Capture activations if requested
        if capture_layers is not None:
            activations = capture_activations(model, tokenizer, prompt, capture_layers)
            activation_data[test_id] = {
                'prompt': prompt,
                'activation_info': {
                    layer: tensor.shape for layer, tensor in activations.items()
                }
            }

            # Save activations for this test case
            activation_path = os.path.join(output_dir, f"activations_{test_id}.pt")
            torch.save(activations, activation_path)

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
#        {"name": "meta-llama/Llama-3.1-8B-Instruct", "is_base": False, "selected_layers": [0, 8, 16, 24]},
#        {"name": "meta-llama/Llama-3.1-8B", "is_base": True, "selected_layers": [0, 8, 16, 24]}
        {"name": "Qwen/Qwen2.5-7B-Instruct", "is_base": False, "selected_layers": [0, 8, 16, 24]},
        {"name": "Qwen/Qwen2.5-7B", "is_base": True, "selected_layers": [0, 8, 16, 24]}
    ]

    # Test each model
    for model_config in models_to_test:
        model_name = model_config["name"]
        is_base = model_config["is_base"]
        selected_layers = model_config["selected_layers"]

        print(f"\nTesting model: {model_name}")

        # Create model-specific output directory
        model_dir_name = "base" if is_base else "instruct"
        model_output_dir = os.path.join(base_output_dir, model_dir_name)

        try:
            # Setup model
            model, tokenizer = setup_model(model_name, base_model=is_base)

            # Run tests
            print(f"Running tests and capturing activations from layers {selected_layers}...")
            results = run_tests(model, tokenizer, test_dataset.copy(), model_output_dir, capture_layers=selected_layers)

            print(f"Tests completed for {model_name}. Results saved to {model_output_dir}")

            # Clean up to free memory
            del model
            torch.cuda.empty_cache()

        except Exception as e:
            print(f"Error testing {model_name}: {str(e)}")

    print(f"\nAll tests completed. Results saved to {base_output_dir}")

if __name__ == "__main__":
    main()
