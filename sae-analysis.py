#!/usr/bin/env python3

import os
import torch
import numpy as np
import pandas as pd
import json
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from tqdm.auto import tqdm
import re
import requests
from huggingface_hub import hf_hub_download
from transformers import AutoModelForCausalLM, AutoTokenizer
import argparse
from datetime import datetime
import tempfile

class LlamaScopeSAE:
    """
    A wrapper for loading and using pretrained sparse autoencoders from Llama Scope.
    """
    def __init__(self, encoder_weights, decoder_weights, config=None, feature_stats=None):
        """
        Initialize a Llama Scope sparse autoencoder from pretrained weights.

        Args:
            encoder_weights: Path to encoder weights or preloaded weights
            decoder_weights: Path to decoder weights or preloaded weights
            config: SAE configuration (optional)
            feature_stats: Feature activation statistics (optional)
        """
        if isinstance(encoder_weights, str) and os.path.exists(encoder_weights):
            self.encoder_weights = torch.load(encoder_weights, map_location='cpu')
        else:
            self.encoder_weights = encoder_weights

        if isinstance(decoder_weights, str) and os.path.exists(decoder_weights):
            self.decoder_weights = torch.load(decoder_weights, map_location='cpu')
        else:
            self.decoder_weights = decoder_weights

        self.config = config
        self.feature_stats = feature_stats

        # Parse dimensions from the weights
        if isinstance(self.encoder_weights, dict) and 'weight' in self.encoder_weights:
            self.input_dim = self.encoder_weights['weight'].shape[1]
            self.dict_size = self.encoder_weights['weight'].shape[0]
        elif isinstance(self.encoder_weights, torch.Tensor):
            self.input_dim = self.encoder_weights.shape[1]
            self.dict_size = self.encoder_weights.shape[0]
        else:
            raise ValueError("Encoder weights format not recognized")

        # Build encoder and decoder modules
        self.encoder = torch.nn.Linear(self.input_dim, self.dict_size, bias=True)
        self.decoder = torch.nn.Linear(self.dict_size, self.input_dim, bias=True)

        # Load weights and biases
        self._load_weights()

        print(f"Initialized SAE with dictionary size {self.dict_size} and input dimension {self.input_dim}")

    def _load_weights(self):
        """Load weights into the encoder and decoder modules."""
        if isinstance(self.encoder_weights, dict):
            self.encoder.weight.data.copy_(self.encoder_weights['weight'])
            if 'bias' in self.encoder_weights:
                self.encoder.bias.data.copy_(self.encoder_weights['bias'])

            self.decoder.weight.data.copy_(self.decoder_weights['weight'])
            if 'bias' in self.decoder_weights:
                self.decoder.bias.data.copy_(self.decoder_weights['bias'])
        elif isinstance(self.encoder_weights, torch.Tensor):
            # Assuming the tensors are weights only without biases
            self.encoder.weight.data.copy_(self.encoder_weights)
            self.decoder.weight.data.copy_(self.decoder_weights)
        else:
            raise ValueError("Weight format not recognized")

    def encode(self, activations, top_k=None):
        """
        Encode model activations using the sparse autoencoder.

        Args:
            activations: Model activations tensor
            top_k: Number of features to keep active (optional)

        Returns:
            torch.Tensor: Sparse feature activations
        """
        with torch.no_grad():
            # Reshape activations if needed
            orig_shape = activations.shape
            if len(orig_shape) > 2:
                activations = activations.reshape(-1, activations.shape[-1])

            # Compute feature activations
            features = self.encoder(activations)

            # Apply ReLU
            features = torch.nn.functional.relu(features)

            # Keep only top-k features if specified
            if top_k is not None and top_k < self.dict_size:
                top_k_values, top_k_indices = torch.topk(features, top_k, dim=1)
                sparse_features = torch.zeros_like(features)
                sparse_features.scatter_(1, top_k_indices, top_k_values)
                features = sparse_features

            # Reshape back to original dimensions if needed
            if len(orig_shape) > 2:
                features = features.reshape(orig_shape[:-1] + (self.dict_size,))

            return features

    def decode(self, features):
        """
        Decode sparse features back to the original activation space.

        Args:
            features: Sparse feature activations

        Returns:
            torch.Tensor: Reconstructed activations
        """
        with torch.no_grad():
            # Reshape features if needed
            orig_shape = features.shape
            if len(orig_shape) > 2:
                features = features.reshape(-1, features.shape[-1])

            # Compute reconstructed activations
            reconstructed = self.decoder(features)

            # Reshape back to original dimensions if needed
            if len(orig_shape) > 2:
                reconstructed = reconstructed.reshape(orig_shape[:-1] + (self.input_dim,))

            return reconstructed

    def get_feature_stats(self):
        """Get feature activation statistics."""
        return self.feature_stats

def load_llama_scope_sae(layer, repo_id="fnlp/Llama3_1-8B-Base-LXR-8x", hf_token=None, cache_dir=None):
    """
    Download and load a Llama Scope SAE model for a specific layer.

    Args:
        layer: Layer number
        repo_id: Hugging Face repository ID
        hf_token: Hugging Face token
        cache_dir: Directory to cache downloaded files

    Returns:
        LlamaScopeSAE: Loaded SAE model
    """
    # Create layer directory name
    layer_dir = f"layer_{layer}"

    try:
        # Download encoder weights
        encoder_path = hf_hub_download(
            repo_id=repo_id,
            filename=f"{layer_dir}/encoder.bin",
            token=hf_token,
            cache_dir=cache_dir
        )

        # Download decoder weights
        decoder_path = hf_hub_download(
            repo_id=repo_id,
            filename=f"{layer_dir}/decoder.bin",
            token=hf_token,
            cache_dir=cache_dir
        )

        # Try to download config and feature stats
        try:
            config_path = hf_hub_download(
                repo_id=repo_id,
                filename=f"{layer_dir}/config.json",
                token=hf_token,
                cache_dir=cache_dir
            )
            with open(config_path, 'r') as f:
                config = json.load(f)
        except Exception as e:
            print(f"Warning: Could not load config for layer {layer}: {str(e)}")
            config = None

        try:
            feature_stats_path = hf_hub_download(
                repo_id=repo_id,
                filename=f"{layer_dir}/feature_acts.bin",
                token=hf_token,
                cache_dir=cache_dir
            )
            feature_stats = torch.load(feature_stats_path, map_location='cpu')
        except Exception as e:
            print(f"Warning: Could not load feature stats for layer {layer}: {str(e)}")
            feature_stats = None

        # Load encoder and decoder weights
        encoder_weights = torch.load(encoder_path, map_location='cpu')
        decoder_weights = torch.load(decoder_path, map_location='cpu')

        # Initialize SAE
        sae = LlamaScopeSAE(encoder_weights, decoder_weights, config, feature_stats)
        return sae

    except Exception as e:
        print(f"Error loading SAE for layer {layer}: {str(e)}")
        return None

def load_llama_scope_saes(layers, repo_id="fnlp/Llama3_1-8B-Base-LXR-8x", hf_token=None, cache_dir=None):
    """
    Load Llama Scope SAEs for multiple layers.

    Args:
        layers: List of layer indices
        repo_id: Hugging Face repository ID
        hf_token: Hugging Face token
        cache_dir: Directory to cache downloaded files

    Returns:
        dict: Dictionary mapping layer indices to SAE objects
    """
    saes = {}

    for layer in tqdm(layers, desc="Loading SAEs"):
        sae = load_llama_scope_sae(layer, repo_id, hf_token, cache_dir)
        if sae is not None:
            saes[layer] = sae

    return saes

def format_prompt(prompt, model_name):
    """Format a prompt for the specified model."""
    if "Instruct" in model_name:
        if "Meta-Llama-3" in model_name:
            return f"<|begin_of_text|><|user|>\n{prompt}<|end_of_turn|>\n<|assistant|>\n"
        else:
            return f"### Instruction:\n{prompt}\n\n### Response:\n"
    else:
        return prompt

def get_model_activations(model, tokenizer, prompt, layer_nums):
    """
    Get activations from specific layers for a given prompt.

    Args:
        model: The model
        tokenizer: The tokenizer
        prompt: The prompt
        layer_nums: List of layer indices

    Returns:
        dict: Dictionary mapping layer indices to activations
    """
    # Format prompt
    formatted_prompt = format_prompt(prompt, model.config._name_or_path)

    # Tokenize
    inputs = tokenizer(formatted_prompt, return_tensors="pt").to(model.device)

    # Setup hooks to capture activations
    activations = {}
    hooks = []

    def get_activation(layer_idx):
        def hook(module, input, output):
            # For LXR models, we want the input (residual stream before the layer)
            # Store only the first element (the residual stream)
            activations[layer_idx] = input[0].detach().cpu()
        return hook

    # Register hooks
    for layer_idx in layer_nums:
        layer = model.model.layers[layer_idx]
        hooks.append(layer.register_forward_hook(get_activation(layer_idx)))

    # Run forward pass
    with torch.no_grad():
        outputs = model(**inputs)

    # Remove hooks
    for hook in hooks:
        hook.remove()

    return activations

def analyze_with_saes(model, tokenizer, saes, test_dataset, output_dir, sample_size=5):
    """
    Analyze model activations using sparse autoencoders.

    Args:
        model: The model
        tokenizer: The tokenizer
        saes: Dictionary mapping layer indices to SAE objects
        test_dataset: Test dataset DataFrame
        output_dir: Directory to save results
        sample_size: Number of examples to analyze

    Returns:
        dict: Analysis results
    """
    print("Analyzing model activations with SAEs")

    # Create output directory
    sae_dir = os.path.join(output_dir, "sae_analysis")
    os.makedirs(sae_dir, exist_ok=True)

    # Group test cases by category
    direct_cases = test_dataset[test_dataset.subcategory == 'direct']
    quoted_cases = test_dataset[test_dataset.subcategory == 'quoted']

    # Sample examples
    sample_direct = direct_cases.sample(min(sample_size, len(direct_cases)))
    sample_quoted = quoted_cases.sample(min(sample_size, len(quoted_cases)))

    # Get layer numbers from SAEs
    layer_nums = sorted(list(saes.keys()))

    # Process direct prompts
    direct_features = {layer: [] for layer in layer_nums}
    for _, row in tqdm(sample_direct.iterrows(), desc="Processing direct prompts", total=len(sample_direct)):
        prompt = row['prompt']

        # Get activations
        activations = get_model_activations(model, tokenizer, prompt, layer_nums)

        # Encode with SAEs
        for layer_idx, sae in saes.items():
            if layer_idx in activations:
                features = sae.encode(activations[layer_idx])
                # Average across sequence dimension
                avg_features = features.mean(dim=1).squeeze()
                direct_features[layer_idx].append(avg_features)

    # Process quoted prompts
    quoted_features = {layer: [] for layer in layer_nums}
    for _, row in tqdm(sample_quoted.iterrows(), desc="Processing quoted prompts", total=len(sample_quoted)):
        prompt = row['prompt']

        # Get activations
        activations = get_model_activations(model, tokenizer, prompt, layer_nums)

        # Encode with SAEs
        for layer_idx, sae in saes.items():
            if layer_idx in activations:
                features = sae.encode(activations[layer_idx])
                # Average across sequence dimension
                avg_features = features.mean(dim=1).squeeze()
                quoted_features[layer_idx].append(avg_features)

    # Calculate average features for each type
    direct_avg = {layer: torch.stack(features).mean(dim=0) if features else None
                 for layer, features in direct_features.items()}

    quoted_avg = {layer: torch.stack(features).mean(dim=0) if features else None
                 for layer, features in quoted_features.items()}

    # Find features that differ the most between direct and quoted instructions
    analysis_results = {}

    for layer in layer_nums:
        if direct_avg[layer] is not None and quoted_avg[layer] is not None:
            # Calculate feature differences
            feat_diff = (direct_avg[layer] - quoted_avg[layer]).abs()

            # Find top differentiating features
            k = min(50, len(feat_diff))
            top_k_values, top_k_indices = torch.topk(feat_diff, k)

            # Store results
            analysis_results[f"layer_{layer}"] = {
                "top_features": {
                    "indices": top_k_indices.tolist(),
                    "difference_values": top_k_values.tolist(),
                    "direct_values": direct_avg[layer][top_k_indices].tolist(),
                    "quoted_values": quoted_avg[layer][top_k_indices].tolist()
                },
                "avg_diff": feat_diff.mean().item(),
                "max_diff": feat_diff.max().item(),
                "feature_histograms": {
                    "direct": direct_avg[layer].histc(bins=20).tolist(),
                    "quoted": quoted_avg[layer].histc(bins=20).tolist()
                }
            }

    # Generate visualizations
    plots_dir = os.path.join(sae_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)

    # Plot feature differences by layer
    plt.figure(figsize=(10, 6))
    layers = [int(layer.split('_')[1]) for layer in analysis_results.keys()]
    avg_diffs = [analysis_results[f"layer_{layer}"]["avg_diff"] for layer in layers]
    max_diffs = [analysis_results[f"layer_{layer}"]["max_diff"] for layer in layers]

    plt.plot(layers, avg_diffs, 'o-', label='Average Difference')
    plt.plot(layers, max_diffs, 'o-', label='Maximum Difference')
    plt.xlabel('Layer')
    plt.ylabel('Feature Difference')
    plt.title('Differences in SAE Feature Activations Between Direct and Quoted Instructions')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(plots_dir, 'feature_differences_by_layer.png'))

    # Plot top features for selected layers
    for layer in layers[:min(3, len(layers))]:  # Plot first 3 layers only
        plt.figure(figsize=(12, 6))

        # Get top feature indices and values
        top_indices = analysis_results[f"layer_{layer}"]["top_features"]["indices"][:20]  # Top 20
        direct_values = [analysis_results[f"layer_{layer}"]["top_features"]["direct_values"][i] for i in range(len(top_indices))]
        quoted_values = [analysis_results[f"layer_{layer}"]["top_features"]["quoted_values"][i] for i in range(len(top_indices))]

        x = range(len(top_indices))
        width = 0.35

        plt.bar([i - width/2 for i in x], direct_values, width, label='Direct')
        plt.bar([i + width/2 for i in x], quoted_values, width, label='Quoted')

        plt.xlabel('Feature Index')
        plt.ylabel('Average Activation')
        plt.title(f'Top Differentiating Features in Layer {layer}')
        plt.xticks(x, [str(idx) for idx in top_indices], rotation=45)
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, f'top_features_layer_{layer}.png'))

    # Save analysis results
    results_path = os.path.join(sae_dir, "sae_analysis_results.json")

    # Convert torch tensors to lists for JSON serialization
    serializable_results = {}
    for layer, layer_results in analysis_results.items():
        serializable_results[layer] = {
            k: (v.tolist() if isinstance(v, torch.Tensor) else v)
            for k, v in layer_results.items()
        }

    with open(results_path, 'w') as f:
        json.dump(serializable_results, f, indent=2)

    return analysis_results

def main():
    """Main function to run SAE analysis."""
    parser = argparse.ArgumentParser(description="Llama Scope SAE Analysis")

    parser.add_argument("--results-dir", type=str, required=True,
                       help="Directory containing test results")
    parser.add_argument("--model-type", type=str, choices=["instruct", "base"], default="instruct",
                       help="Model type to analyze")
    parser.add_argument("--repo-id", type=str, default="fnlp/Llama3_1-8B-Base-LXR-8x",
                       help="Hugging Face repo ID for SAE models")
    parser.add_argument("--layers", type=str, default="0,8,16,24,31",
                       help="Comma-separated list of layer indices to analyze")
    parser.add_argument("--sample-size", type=int, default=5,
                       help="Number of examples to sample for analysis")
    parser.add_argument("--hf-token", type=str, default=None,
                       help="Hugging Face token (or set HF_TOKEN env var)")
    parser.add_argument("--model-name", type=str, default=None,
                       help="Model name (default: automatically determined)")

    args = parser.parse_args()

    # Get Hugging Face token
    hf_token = args.hf_token or os.environ.get("HF_TOKEN")
    if not hf_token:
        print("WARNING: No Hugging Face token provided. Set with --hf-token or HF_TOKEN env var.")

    # Parse layers
    layers = [int(l) for l in args.layers.split(",")]

    # Load test dataset
    dataset_path = os.path.join(args.results_dir, "test_dataset.json")
    if os.path.exists(dataset_path):
        with open(dataset_path, 'r') as f:
            test_dataset = pd.DataFrame(json.load(f))
    else:
        print(f"Test dataset not found at {dataset_path}")
        return

    # Create temp directory for downloading SAE models
    with tempfile.TemporaryDirectory() as temp_dir:
        # Load SAEs
        print(f"Loading SAEs from {args.repo_id} for layers {layers}")
        saes = load_llama_scope_saes(layers, args.repo_id, hf_token, temp_dir)

        if not saes:
            print("No SAEs could be loaded. Exiting.")
            return

        # Determine model name
        if args.model_name:
            model_name = args.model_name
        else:
            if args.model_type == "instruct":
                model_name = "meta-llama/Meta-Llama-3.1-8B-Instruct"
            else:
                model_name = "meta-llama/Meta-Llama-3.1-8B"

        # Load model
        print(f"Loading model: {model_name}")
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            token=hf_token
        )

        tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            token=hf_token
        )

        # Fix padding token issue
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        # Set padding side to left for more efficient generation
        tokenizer.padding_side = 'left'

        # Analyze with SAEs
        model_dir = os.path.join(args.results_dir, args.model_type)
        analysis_results = analyze_with_saes(
            model, tokenizer, saes, test_dataset, model_dir, args.sample_size
        )

        print(f"SAE analysis complete. Results saved to {os.path.join(model_dir, 'sae_analysis')}")

if __name__ == "__main__":
    main()
