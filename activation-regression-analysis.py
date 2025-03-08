#!/usr/bin/env python3
#%%
import os
import torch
import numpy as np
import pandas as pd
import json
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score
from pathlib import Path
import argparse
from tqdm import tqdm
import glob
from utils import show_shape, get_hf_token
from transformers import AutoTokenizer

#%%
def load_tokenizer(model_name):
  my_hf_token = get_hf_token()
  tokenizer = AutoTokenizer.from_pretrained(model_name, token=my_hf_token, padding_side='left')
  if tokenizer.pad_token is None:
      # Set pad_token to eos_token for Llama tokenizers
      tokenizer.pad_token = tokenizer.eos_token
      print(f"Setting pad_token to eos_token: {tokenizer.eos_token}")
  return tokenizer

# Load tokenizers for both models if we didn't load them already
if 'tokenizers' not in globals():
  tokenizers = {}
  tokenizers['base'] = load_tokenizer("meta-llama/Llama-3.1-8B")
  tokenizers['instruct'] = load_tokenizer("meta-llama/Llama-3.1-8B-Instruct")

#%%
def find_most_recent_results(directory):
    """Find the most recent test_results file in the directory."""
    results_files = glob.glob(os.path.join(directory, "test_results_*.json"))
    if not results_files:
        # Look one directory up
        parent_dir = os.path.dirname(directory)
        results_files = glob.glob(os.path.join(parent_dir, "test_results_*.json"))

    if not results_files:
        raise FileNotFoundError(f"No test_results_*.json files found in {directory} or parent directory")

    # Sort by modification time, most recent first
    results_files.sort(key=os.path.getmtime, reverse=True)
    return results_files[0]

def load_test_results(results_file):
    """Load test results from the specified file."""
    with open(results_file, 'r') as f:
        test_results = json.load(f)

    # Create a dictionary for faster lookup by ID
    results_by_id = {result['id']: result for result in test_results}
    return results_by_id

def compute_token_probs(test_result):
    """
    Compute direct and quoted token probabilities from a test result entry.

    Args:
        test_result: Entry from test results JSON

    Returns:
        direct_prob, quoted_prob: Probabilities of the direct and quoted tokens
    """
    # Skip baseline cases
    if test_result['category'] == 'baseline':
        return None, None

    # Extract tokens and probabilities
    try:
        top_tokens = test_result['top_n_tokens']
        top_probs = test_result['top_n_probs']
    except KeyError:
        return None, None

    # Check for string encoded lists
    if isinstance(top_tokens, str):
        try:
            top_tokens = json.loads(top_tokens)
        except:
            return None, None

    if isinstance(top_probs, str):
        try:
            top_probs = json.loads(top_probs)
        except:
            return None, None

    # Get the expected good (direct) and bad (quoted) responses
    direct_token = test_result.get('good_response')
    quoted_token = test_result.get('bad_response')

    if not direct_token or not quoted_token:
        return None, None

    global model_type
    tokenizer = tokenizers[model_type]

    direct_token = tokenizer.convert_tokens_to_string([tokenizer.tokenize(direct_token)[0]])
    quoted_token = tokenizer.convert_tokens_to_string([tokenizer.tokenize(quoted_token)[0]])

    # Find the probabilities of these tokens
    direct_prob = 0.0
    quoted_prob = 0.0

    # Look for exact matches in top_tokens
    for token, prob in zip(top_tokens, top_probs):
        if token == direct_token:
            direct_prob = prob
        if token == quoted_token:
            quoted_prob = prob

    return direct_prob, quoted_prob

#%%
def analyze_activations_by_layer(activations_dir, results_file=None, output_dir=None, min_examples=2, graph_each=4):
    """
    Perform regression analysis by layer to find activation directions related to direct vs quoted instruction preferences.

    Args:
        activations_dir: Directory containing activation files
        results_file: Path to test results JSON file (if None, will search for most recent)
        output_dir: Directory to save analysis results
        min_examples: Minimum number of examples required to analyze a layer

    Returns:
        dict: Analysis results by layer
    """

    global analysis_model

    if output_dir is None:
        output_dir = os.path.join(os.path.dirname(activations_dir), "regression_analysis")

    os.makedirs(output_dir, exist_ok=True)

    # Create plots directory
    plots_dir = os.path.join(output_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)

    # Find and load test results
    if results_file is None:
        results_file = find_most_recent_results(activations_dir)

    print(f"Loading test results from: {results_file}")
    results_by_id = load_test_results(results_file)
    print(f"Loaded {len(results_by_id)} test results")

    # Find all activation files
    activation_files = [f for f in os.listdir(activations_dir) if f.startswith("activations_") and f.endswith(".pt")]
    if not activation_files:
        raise ValueError(f"No activation files found in {activations_dir}")

    print(f"Found {len(activation_files)} activation files")

    # Load a sample activation file to determine available layers
    sample_file = os.path.join(activations_dir, activation_files[0])
    sample_activation = torch.load(sample_file)

    # Extract layer names - assuming format like 'layer_0', 'layer_1', etc.
    layer_names = [name for name in sample_activation.keys()
                  if isinstance(name, str) and name.startswith("layer_") and not name.endswith("_input")]

    if not layer_names:
        # Try with _output suffix
        layer_names = [name for name in sample_activation.keys()
                      if isinstance(name, str) and name.startswith("layer_") and name.endswith("_output")]

    print(f"Found {len(layer_names)} layers: {', '.join(layer_names[:5])}...")

    # Dictionary to store results by layer
    layer_results = {}

    # Track cases by category for reporting
    category_counts = {}

    #print("LIMITING LAYERS FOR TESTING")
    #layer_names = layer_names[:2]

    # Analyze each layer
    layer_idx = 0
    for layer_name in tqdm(layer_names, desc="Analyzing layers"):
        print(f"\nProcessing {layer_name}...")

        # Collect data for this layer
        X = []  # Activations
        y = []  # Probability differences
        example_ids = []  # Keep track of which examples we use
        categories = []  # Keep track of categories

        # Process all activation files
        for act_file in tqdm(activation_files, desc=f"Processing {layer_name}", leave=False):
            try:
                # Extract ID from filename
                example_id = act_file.replace("activations_", "").replace(".pt", "")

                # Find corresponding test result
                if example_id not in results_by_id:
                    continue

                test_result = results_by_id[example_id]
                print(f"Processing {example_id} ({test_result['category']})")

                # Skip baseline cases
                if test_result['category'] == 'baseline':
                    continue

                # Get direct and quoted token probabilities
                direct_prob, quoted_prob = compute_token_probs(test_result)

                if direct_prob is None or quoted_prob is None:
                    continue

                # Load activations
                act_path = os.path.join(activations_dir, act_file)
                activation = torch.load(act_path)
                #print(f"ACTIVATION SHAPE:")
                #show_shape(activation)

                # Check if this layer exists in the file
                if layer_name not in activation:
                    continue

                # Calculate probability difference
                prob_diff = direct_prob - quoted_prob

                # Extract and process activations
                layer_act = activation[layer_name]
                #print(f"LAYER_ACT SHAPE:")
                #show_shape(layer_act)

                # Mean pooling across sequence length
                if len(layer_act.shape) >= 2:
                    layer_act = layer_act.mean(dim=0)
                #print(f"MEAN POOLED LAYER_ACT SHAPE:")
                #show_shape(layer_act)

                # Flatten to 1D if needed
                layer_act = layer_act.reshape(-1).cpu().numpy()

                # Add to datasets
                X.append(layer_act)
                y.append(prob_diff)
                example_ids.append(example_id)
                categories.append(test_result['category'])

                # Update category counts
                category = test_result['category']
                if category not in category_counts:
                    category_counts[category] = 0
                category_counts[category] += 1

            except Exception as e:
                print(f"Error processing {act_file}: {e}")

        # Skip if not enough data
        if len(X) < min_examples:
            print(f"Only {len(X)} examples for {layer_name}, skipping (need at least {min_examples})")
            continue

        show_shape(X, "X: ")
        show_shape(y, "y: ")

        X = np.array(X)
        y = np.array(y)

        print(f"Collected {len(X)} examples with shapes: X = {X.shape}, y = {y.shape}")
        print(f"Category distribution: {category_counts}")

        # Train ridge regression model
        regressor = Ridge(alpha=1.0)
        regressor.fit(X, y)

        # Evaluate
        y_pred = regressor.predict(X)
        r2 = r2_score(y, y_pred)

        print(f"{layer_name}: R² = {r2:.4f}")

        # Find top dimensions by coefficient magnitude
        coef_magnitudes = np.abs(regressor.coef_)
        top_idx = np.argsort(coef_magnitudes)[-100:][::-1]
        top_coef = regressor.coef_[top_idx]

        # Store results
        layer_results[layer_name] = {
            'r2': r2,
            'n_examples': len(X),
            'top_dimensions': top_idx.tolist(),
            'top_coefficients': top_coef.tolist(),
            'coefficient_norm': np.linalg.norm(regressor.coef_),
            'intercept': regressor.intercept_,
            'prediction_corr': np.corrcoef(y, y_pred)[0, 1]
        }

        if layer_idx % graph_each == 0:
            # Visualize predictions vs actual
            plt.figure(figsize=(10, 6))
            plt.scatter(y, y_pred, alpha=0.5)
            plt.plot([-1, 1], [-1, 1], 'k--', alpha=0.5)  # Diagonal line
            plt.xlabel('Actual Probability Difference (Direct - Quoted)')
            plt.ylabel('Predicted Probability Difference')
            plt.title(f'Layer {layer_name}: Regression Performance (R² = {r2:.4f}) for {analysis_model}')
            plt.grid(alpha=0.3)

            # Add annotations: green = direct wins, red = quoted wins
            for i, (actual, pred, ex_id) in enumerate(zip(y, y_pred, example_ids)):
                if i % max(1, len(y) // 20) == 0:  # Label every ~20 points
                    color = 'green' if actual > 0 else 'red'
                    plt.annotate(ex_id.split('_')[-1], (actual, pred), fontsize=8, color=color)

            plt.tight_layout()
            plt.savefig(os.path.join(plots_dir, f'{layer_name}_regression.png'))
            plt.close()

            # Visualize top coefficients
            plt.figure(figsize=(12, 6))
            plt.bar(range(20), top_coef[:20])
            plt.xlabel('Top Dimensions (by coefficient magnitude)')
            plt.ylabel('Coefficient Value')
            plt.title(f'Layer {layer_name}: Top 20 Dimensions for {analysis_model}')
            plt.xticks(range(20), top_idx[:20], rotation=90)
            plt.grid(axis='y', alpha=0.3)
            plt.tight_layout()
            plt.savefig(os.path.join(plots_dir, f'{layer_name}_top_dims.png'))
            plt.close()

        layer_idx += 1

    # Summarize layer results
    if layer_results:
        # Collect R² scores by layer
        layers = list(layer_results.keys())
        r2_scores = [layer_results[layer]['r2'] for layer in layers]

        # Plot R² by layer to show where the decision is most strongly encoded
        plt.figure(figsize=(14, 6))
        bars = plt.bar(range(len(layers)), r2_scores)

        # Color bars by R² value
        for i, bar in enumerate(bars):
            bar.set_color(plt.cm.viridis(r2_scores[i] / max(r2_scores)))

        plt.xlabel('Layer')
        plt.ylabel('R² Score')
        plt.title(f'Predictive Power of Each Layer for Direct vs. Quoted Decision for {analysis_model}')

        # Format layer names for readability
        layer_labels = [layer.replace('layer_', '').replace('_output', '') for layer in layers]
        plt.xticks(range(len(layers)), layer_labels, rotation=90)

        plt.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, 'layer_r2_scores.png'))
        plt.close()

        # Save numerical results as JSON
        results_path = os.path.join(output_dir, 'regression_results.json')

        print(f"Saving results to {results_path}")
        print(f"LAYER_RESULTS SHAPE:")
        show_shape(layer_results)

        # Make sure all values are serializable -- if they are numpy floats we have to call .item().
        serializable_results = {}

        for layer, results in layer_results.items():
            serializable_results[layer] = {}
            for k, v in results.items():
                if isinstance(v, np.generic):
                    serializable_results[layer][k] = v.item()
                else:
                    serializable_results[layer][k] = v

        print(f"SERIALIZABLE_RESULTS SHAPE:")
        show_shape(serializable_results)

        with open(results_path, 'w') as f:
            json.dump(serializable_results, f, indent=2)

        # Create a summary dataframe and save as CSV
        summary_data = []
        for layer in layers:
            summary_data.append({
                'layer': layer,
                'r2_score': layer_results[layer]['r2'],
                'coefficient_norm': layer_results[layer]['coefficient_norm'],
                'n_examples': layer_results[layer]['n_examples'],
                'prediction_corr': layer_results[layer]['prediction_corr']
            })

        summary_df = pd.DataFrame(summary_data)
        summary_df.to_csv(os.path.join(output_dir, 'layer_summary.csv'), index=False)

        # Print top 5 layers by R²
        print("\nTop 5 layers by R² score:")
        top_layers = sorted([(layer, layer_results[layer]['r2']) for layer in layers],
                           key=lambda x: x[1], reverse=True)[:5]

        for layer, r2 in top_layers:
            print(f"{layer}: R² = {r2:.4f}")

    else:
        print("No layers analyzed successfully")

    return layer_results

def analyze_activations_by_position(activations_dir, results_file=None, output_dir=None, min_examples=2, num_positions=999, graph_each=10):
    """
    Perform regression analysis by token position to find activation directions related to direct vs quoted instruction preferences.
    Analyzes positions counting backwards from the end of the prompt.

    Args:
        activations_dir: Directory containing activation files
        results_file: Path to test results JSON file (if None, will search for most recent)
        output_dir: Directory to save analysis results
        min_examples: Minimum number of examples required to analyze a position
        num_positions: Number of token positions to analyze from the end of the prompt

    Returns:
        dict: Analysis results by position
    """

    global analysis_model

    if output_dir is None:
        output_dir = os.path.join(os.path.dirname(activations_dir), "position_regression_analysis")

    os.makedirs(output_dir, exist_ok=True)

    # Create plots directory
    plots_dir = os.path.join(output_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)

    # Find and load test results
    if results_file is None:
        results_file = find_most_recent_results(activations_dir)

    print(f"Loading test results from: {results_file}")
    results_by_id = load_test_results(results_file)
    print(f"Loaded {len(results_by_id)} test results")

    # Find all activation files
    activation_files = [f for f in os.listdir(activations_dir) if f.startswith("activations_") and f.endswith(".pt")]
    if not activation_files:
        raise ValueError(f"No activation files found in {activations_dir}")

    print(f"Found {len(activation_files)} activation files")

    # Load a sample activation file to determine structure
    sample_file = os.path.join(activations_dir, activation_files[0])
    sample_activation = torch.load(sample_file)

    # Extract layer names - assuming format like 'layer_0', 'layer_1', etc.
    layer_names = [name for name in sample_activation.keys()
                  if isinstance(name, str) and name.startswith("layer_") and not name.endswith("_input")]

    if not layer_names:
        # Try with _output suffix
        layer_names = [name for name in sample_activation.keys()
                      if isinstance(name, str) and name.startswith("layer_") and name.endswith("_output")]

    print(f"Found {len(layer_names)} layers")

    # Initialize data structures for all positions
    position_data = {}
    for pos in range(1, num_positions + 1):
        neg_pos = -pos  # Convert to negative position (e.g., -1, -2, etc.)
        position_data[neg_pos] = {
            'X': [],
            'y': [],
            'example_ids': [],
            'categories': []
        }

    # Dictionary to store results by position
    position_results = {}

    # Track cases by category for reporting
    category_counts = {}

    # Process all activation files - load each file only once
    for act_file in tqdm(activation_files, desc="Processing activation files"):
        try:
            # Extract ID from filename
            example_id = act_file.replace("activations_", "").replace(".pt", "")

            # Find corresponding test result
            if example_id not in results_by_id:
                continue

            test_result = results_by_id[example_id]

            # Skip baseline cases
            if test_result['category'] == 'baseline':
                continue

            # Get direct and quoted token probabilities
            direct_prob, quoted_prob = compute_token_probs(test_result)

            if direct_prob is None or quoted_prob is None:
                print(f"Skipping {act_file} (missing probabilities)")
                continue

            # Calculate probability difference
            prob_diff = direct_prob - quoted_prob

            # Load activations - do this only once per file
            act_path = os.path.join(activations_dir, act_file)
            activation = torch.load(act_path)

            # Update category counts
            category = test_result['category']
            if category not in category_counts:
                category_counts[category] = 0
            category_counts[category] += 1

            # Find smallest sequence length across layers for this sample
            min_seq_length = float('inf')
            for layer_name in layer_names:
                if layer_name in activation:
                    layer_act = activation[layer_name]
                    if len(layer_act.shape) >= 2:  # Has sequence dimension
                        min_seq_length = min(min_seq_length, layer_act.shape[0])

            # Process all positions at once for this file
            for pos in range(1, min(num_positions + 1, min_seq_length + 1)):
                neg_pos = -pos  # Convert to positive index to negative

                # Collect activations at the specified position across all layers
                pos_activations = []

                for layer_name in layer_names:
                    if layer_name in activation:
                        layer_act = activation[layer_name]
                        # Check if we have sequence dimension
                        if len(layer_act.shape) >= 2:
                            # Get specified position from the end
                            pos_act = layer_act[neg_pos].reshape(-1).cpu().numpy()
                            pos_activations.append(pos_act)

                # Concatenate all layer activations for this position
                pos_act_concat = np.concatenate(pos_activations)

                # Add to datasets for this position
                position_data[neg_pos]['X'].append(pos_act_concat)
                position_data[neg_pos]['y'].append(prob_diff)
                position_data[neg_pos]['example_ids'].append(example_id)
                position_data[neg_pos]['categories'].append(category)

        except Exception as e:
            print(f"Error processing {act_file}: {e}")

    token_idx = 0
    # Now, analyze each position using the collected data
    for neg_pos, data in tqdm(position_data.items(), desc="Analyzing positions"):
        X = data['X']
        y = data['y']
        example_ids = data['example_ids']

        # Skip if not enough data
        if len(X) < min_examples:
            print(f"Only {len(X)} examples for position {neg_pos}, skipping (need at least {min_examples})")
            continue

        X = np.array(X)
        y = np.array(y)

        print(f"Position {neg_pos}: Collected {len(X)} examples with shapes: X = {X.shape}, y = {y.shape}")

        # Train ridge regression model
        regressor = Ridge(alpha=1.0)
        regressor.fit(X, y)

        # Evaluate
        y_pred = regressor.predict(X)
        r2 = r2_score(y, y_pred)

        print(f"Position {neg_pos}: R² = {r2:.4f}")

        # Find top dimensions by coefficient magnitude
        coef_magnitudes = np.abs(regressor.coef_)
        top_idx = np.argsort(coef_magnitudes)[-100:][::-1]
        top_coef = regressor.coef_[top_idx]

        # Store results
        position_results[str(neg_pos)] = {
            'r2': r2,
            'n_examples': len(X),
            'top_dimensions': top_idx.tolist(),
            'top_coefficients': top_coef.tolist(),
            'coefficient_norm': np.linalg.norm(regressor.coef_),
            'intercept': regressor.intercept_,
            'prediction_corr': np.corrcoef(y, y_pred)[0, 1]
        }

        if token_idx % graph_each == 0:
            # Visualize predictions vs actual
            plt.figure(figsize=(10, 6))
            plt.scatter(y, y_pred, alpha=0.5)
            plt.plot([-1, 1], [-1, 1], 'k--', alpha=0.5)  # Diagonal line
            plt.xlabel('Actual Probability Difference (Direct - Quoted)')
            plt.ylabel('Predicted Probability Difference')
            plt.title(f'Position {neg_pos}: Regression Performance (R² = {r2:.4f}) for {analysis_model}')
            plt.grid(alpha=0.3)

            # Add annotations: green = direct wins, red = quoted wins
            for i, (actual, pred, ex_id) in enumerate(zip(y, y_pred, example_ids)):
                if i % max(1, len(y) // 20) == 0:  # Label every ~20 points
                    color = 'green' if actual > 0 else 'red'
                    plt.annotate(ex_id.split('_')[-1], (actual, pred), fontsize=8, color=color)

            plt.tight_layout()
            plt.savefig(os.path.join(plots_dir, f'position_{neg_pos}_regression.png'))
            plt.close()

        token_idx += 1

    # Summarize position results
    if position_results:
        # Collect R² scores by position
        positions = [int(pos) for pos in position_results.keys()]
        positions.sort()  # Sort positions numerically
        r2_scores = [position_results[str(pos)]['r2'] for pos in positions]

        # Convert positions to strings for plotting
        position_labels = [str(pos) for pos in positions]

        # Plot R² by position
        plt.figure(figsize=(14, 6))
        bars = plt.bar(range(len(positions)), r2_scores)

        # Color bars by R² value
        for i, bar in enumerate(bars):
            bar.set_color(plt.cm.viridis(r2_scores[i] / max(r2_scores)))

        plt.xlabel('Position (counting backwards from end)')
        plt.ylabel('R² Score')
        plt.title(f'Predictive Power of Each Position for Direct vs. Quoted Decision for {analysis_model}')
        plt.xticks(range(len(positions)), position_labels, rotation=90)
        plt.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, 'position_r2_scores.png'))
        plt.close()

        # Save numerical results as JSON
        results_path = os.path.join(output_dir, 'position_regression_results.json')

        print(f"Saving results to {results_path}")
        print(f"POSITION_RESULTS SHAPE:")
        show_shape(position_results)

        # Make sure all values are serializable
        serializable_results = {}
        for pos, results in position_results.items():
            serializable_results[pos] = {}
            for k, v in results.items():
                if isinstance(v, np.generic):
                    serializable_results[pos][k] = v.item()
                else:
                    serializable_results[pos][k] = v

        with open(results_path, 'w') as f:
            json.dump(serializable_results, f, indent=2)

        # Create a summary dataframe and save as CSV
        summary_data = []
        for pos in positions:
            summary_data.append({
                'position': pos,
                'r2_score': position_results[str(pos)]['r2'],
                'coefficient_norm': position_results[str(pos)]['coefficient_norm'],
                'n_examples': position_results[str(pos)]['n_examples'],
                'prediction_corr': position_results[str(pos)]['prediction_corr']
            })

        summary_df = pd.DataFrame(summary_data)
        summary_df.to_csv(os.path.join(output_dir, 'position_summary.csv'), index=False)

        # Print top 5 positions by R²
        print("\nTop 5 positions by R² score:")
        top_positions = sorted([(pos, position_results[str(pos)]['r2']) for pos in positions],
                            key=lambda x: x[1], reverse=True)[:5]

        for pos, r2 in top_positions:
            print(f"Position {pos}: R² = {r2:.4f}")

    else:
        print("No positions analyzed successfully")

    return position_results

# Now a grid, both position and layer
def analyze_activations_by_grid(activations_dir, results_file=None, output_dir=None, min_examples=10, num_positions=20, act_file_prefix="activations_"):
    global analysis_model

    if output_dir is None:
        output_dir = os.path.join(os.path.dirname(activations_dir), "position_regression_analysis")

    os.makedirs(output_dir, exist_ok=True)

    # Create plots directory
    plots_dir = os.path.join(output_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)

    # Find and load test results
    if results_file is None:
        results_file = find_most_recent_results(activations_dir)

    print(f"Loading test results from: {results_file}")
    results_by_id = load_test_results(results_file)
    print(f"Loaded {len(results_by_id)} test results")

    # Find all activation files
    activation_files = [f for f in os.listdir(activations_dir) if f.startswith(act_file_prefix) and f.endswith(".pt")]
    if not activation_files:
        raise ValueError(f"No activation files found in {activations_dir}")

    print(f"Found {len(activation_files)} activation files")

    # Load a sample activation file to determine structure
    sample_file = os.path.join(activations_dir, activation_files[0])
    sample_activation = torch.load(sample_file)

    # Extract layer names - assuming format like 'layer_0', 'layer_1', etc.
    layer_names = [name for name in sample_activation.keys()
                  if isinstance(name, str) and name.startswith("layer_") and not name.endswith("_input")]

    if not layer_names:
        # Try with _output suffix
        layer_names = [name for name in sample_activation.keys()
                      if isinstance(name, str) and name.startswith("layer_") and name.endswith("_output")]

    print(f"Found {len(layer_names)} layers")

    grid_data = {}
    for layer_name in layer_names:
        grid_data[layer_name] = {}
        for pos in range(1, num_positions + 1):
            neg_pos = -pos
            grid_data[layer_name][neg_pos] = {
                'X': [],
                'y': [],
                'example_ids': [],
                'categories': []
            }

    # Dictionary to store grid results
    grid_results = {}

    # Track cases by category for reporting
    category_counts = {}

    # Process all activation files - load each file only once
    for act_file in tqdm(activation_files, desc="Processing activation files"):
        try:
            # Extract ID from filename
            example_id = act_file.replace("activations_", "").replace(".pt", "")

            # if the name contains 'baseline', skip it; we're going to skip it later anyway, don't load it.
            if 'baseline' in example_id:
                continue

            # Find corresponding test result
            if example_id not in results_by_id:
                continue

            test_result = results_by_id[example_id]

            # Skip baseline cases
            if test_result['category'] == 'baseline':
                continue

            # Get direct and quoted token probabilities
            direct_prob, quoted_prob = compute_token_probs(test_result)

            if direct_prob is None or quoted_prob is None:
                continue

            # Calculate probability difference
            prob_diff = direct_prob - quoted_prob

            # Load activations - do this only once per file
            act_path = os.path.join(activations_dir, act_file)
            activation = torch.load(act_path)

            # Update category counts
            category = test_result['category']
            if category not in category_counts:
                category_counts[category] = 0
            category_counts[category] += 1

            # Process for each layer and position
            for layer_name in layer_names:
                if layer_name not in activation:
                    continue

                layer_act = activation[layer_name]

                # Skip if no sequence dimension
                if len(layer_act.shape) < 2:
                    continue

                # Determine maximum position to analyze for this example
                seq_length = layer_act.shape[0]
                max_pos = min(num_positions, seq_length)

                # Process each position for this layer
                for pos in range(1, max_pos + 1):
                    neg_pos = -pos  # Convert to negative index

                    # Extract activations for this position
                    pos_act = layer_act[neg_pos].reshape(-1).cpu().numpy()

                    # Add to grid data
                    grid_data[layer_name][neg_pos]['X'].append(pos_act)
                    grid_data[layer_name][neg_pos]['y'].append(prob_diff)
                    grid_data[layer_name][neg_pos]['example_ids'].append(example_id)
                    grid_data[layer_name][neg_pos]['categories'].append(category)

        except Exception as e:
            print(f"Error processing {act_file}: {e}")

    # Initialize results dictionary structure
    for layer_name in layer_names:
        grid_results[layer_name] = {}

    # Now analyze each layer-position pair
    for layer_name in tqdm(layer_names, desc="Analyzing layers"):
        for neg_pos in tqdm([f"-{pos}" for pos in range(1, num_positions + 1)],
                            desc=f"Analyzing positions for {layer_name}", leave=False):
            neg_pos = int(neg_pos)  # Convert string back to integer

            # Skip if not enough examples
            if neg_pos not in grid_data[layer_name] or len(grid_data[layer_name][neg_pos]['X']) < min_examples:
                print(f"Only {len(grid_data[layer_name][neg_pos]['X'])} examples for {layer_name}, {neg_pos}, skipping")
                continue

            X = np.array(grid_data[layer_name][neg_pos]['X'])
            y = np.array(grid_data[layer_name][neg_pos]['y'])
            example_ids = grid_data[layer_name][neg_pos]['example_ids']

            # Train ridge regression model
            regressor = Ridge(alpha=1.0)
            regressor.fit(X, y)

            # Evaluate
            y_pred = regressor.predict(X)
            r2 = r2_score(y, y_pred)

            print(f"Layer {layer_name}, Position {neg_pos}: R² = {r2:.4f}")

            # Find top dimensions by coefficient magnitude
            coef_magnitudes = np.abs(regressor.coef_)
            top_idx = np.argsort(coef_magnitudes)[-100:][::-1]
            top_coef = regressor.coef_[top_idx]

            # Store results
            grid_results[layer_name][str(neg_pos)] = {
                'r2': r2,
                'n_examples': len(X),
                'top_dimensions': top_idx.tolist(),
                'top_coefficients': top_coef.tolist(),
                'coefficient_norm': np.linalg.norm(regressor.coef_),
                'intercept': regressor.intercept_,
                'prediction_corr': np.corrcoef(y, y_pred)[0, 1]
            }

            # We don't really need to plot all of these
            if False:
                # Create plots for this layer-position pair
                plt.figure(figsize=(10, 6))
                plt.scatter(y, y_pred, alpha=0.5)
                plt.plot([-1, 1], [-1, 1], 'k--', alpha=0.5)  # Diagonal line
                plt.xlabel('Actual Probability Difference (Direct - Quoted)')
                plt.ylabel('Predicted Probability Difference')
                plt.title(f'Layer {layer_name}, Position {neg_pos}: Regression (R² = {r2:.4f}) for {analysis_model} (testcases {act_file_prefix})')
                plt.grid(alpha=0.3)

                # Add annotations
                for i, (actual, pred, ex_id) in enumerate(zip(y, y_pred, example_ids)):
                    if i % max(1, len(y) // 20) == 0:  # Label every ~20 points
                        color = 'green' if actual > 0 else 'red'
                        plt.annotate(ex_id.split('_')[-1], (actual, pred), fontsize=8, color=color)

                plt.tight_layout()
                layer_name_safe = layer_name.replace('/', '_')
                plt.savefig(os.path.join(plots_dir, f'{layer_name_safe}_pos{abs(neg_pos)}_regression.png'))
                plt.close()

    # Save results to JSON file
    results_path = os.path.join(output_dir, 'grid_regression_results.json')
    print(f"Saving grid results to {results_path}")

    # Make results serializable
    serializable_results = {}
    for layer_name in grid_results:
        serializable_results[layer_name] = {}
        for pos, results in grid_results[layer_name].items():
            serializable_results[layer_name][pos] = {}
            for k, v in results.items():
                if isinstance(v, np.generic):
                    serializable_results[layer_name][pos][k] = v.item()
                else:
                    serializable_results[layer_name][pos][k] = v

    with open(results_path, 'w') as f:
        json.dump(serializable_results, f, indent=2)

    # Create heatmap of R² scores
    r2_matrix = []
    layer_indices = []
    position_indices = []

    # Collect all layers and positions that have results
    for layer_idx, layer_name in enumerate(layer_names):
        if layer_name in grid_results:
            positions = sorted([int(pos) for pos in grid_results[layer_name].keys()])
            if positions:
                layer_indices.append(layer_idx)
                for pos in positions:
                    if pos not in position_indices:
                        position_indices.append(pos)

    # Sort positions
    position_indices.sort()

    # Create R² matrix
    for layer_idx in layer_indices:
        layer_name = layer_names[layer_idx]
        r2_row = []
        for pos in position_indices:
            pos_str = str(pos)
            if pos_str in grid_results[layer_name]:
                r2_row.append(grid_results[layer_name][pos_str]['r2'])
            else:
                r2_row.append(np.nan)  # Use NaN for missing data
        r2_matrix.append(r2_row)

    # Create heatmap if we have data
    if r2_matrix and position_indices:
        plt.figure(figsize=(20, 12))

        # Create readable layer labels
        layer_labels = [layer_names[idx] for idx in layer_indices]
        position_labels = [str(abs(pos)) for pos in position_indices]

        # Create heatmap
        sns.heatmap(r2_matrix, cmap="viridis", # XXX ugly annot=True, fmt=".3f",
                   xticklabels=position_labels, yticklabels=layer_labels,
                   cbar_kws={'label': 'R² Score'})

        plt.title(f'Grid Analysis: R² Scores by Layer and Position for {analysis_model} (testcases {act_file_prefix})')
        plt.xlabel('Position (tokens from end)')
        plt.ylabel('Layer')
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, 'grid_r2_heatmap.png'), dpi=150)
        plt.close()

        # Also create a version with higher/lower cutoffs for better visualization
        plt.figure(figsize=(20, 12))
        r2_array = np.array(r2_matrix)
        vmin = max(0, np.nanpercentile(r2_array, 5))  # 5th percentile or 0, whichever is higher
        vmax = min(1, np.nanpercentile(r2_array, 95))  # 95th percentile or 1, whichever is lower

        sns.heatmap(r2_matrix, cmap="viridis", # XXX ugly annot=True, fmt=".3f",
                   xticklabels=position_labels, yticklabels=layer_labels,
                   cbar_kws={'label': 'R² Score'}, vmin=vmin, vmax=vmax)

        plt.title(f'Grid Analysis: R² Scores by Layer and Position (Normalized Scale) for {analysis_model} (testcases {act_file_prefix})')
        plt.xlabel('Position (tokens from end)')
        plt.ylabel('Layer')
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, 'grid_r2_heatmap_normalized.png'), dpi=150)
        plt.close()

    # Find top 10 layer-position pairs by R² score
    top_pairs = []
    for layer_name in grid_results:
        for pos, results in grid_results[layer_name].items():
            top_pairs.append((layer_name, pos, results['r2']))

    top_pairs.sort(key=lambda x: x[2], reverse=True)

    print("\nTop 10 layer-position pairs by R² score:")
    for layer_name, pos, r2 in top_pairs[:10]:
        print(f"Layer {layer_name}, Position {pos}: R² = {r2:.4f}")

    return grid_results

#%%

## Main

parser = argparse.ArgumentParser(description="Analyze model activations using regression")

parser.add_argument("--activations-dir", type=str, required=True,
                    help="Directory containing activation files")
parser.add_argument("--results-file", type=str, default=None,
                    help="Path to test results JSON file (if None, will search for most recent)")
parser.add_argument("--output-dir", type=str, default=None,
                    help="Directory to save analysis results")
parser.add_argument("--min-examples", type=int, default=10,
                    help="Minimum number of examples required to analyze a layer")

# XXX: Should really get this from the file so it can't go out of sync.
analysis_model = None
for model_type in ["base", "instruct"]:
    if model_type == "base":
        analysis_model = "Llama-3.1-8B"
    elif model_type == "instruct":
        analysis_model = "Llama-3.1-8B-instruct"

    # hacky: started this as a script but ended up running it as a notebook instead.
    args = parser.parse_args(["--activations-dir", f"latest_results/{model_type}"])

    # Create output directory if not specified
    if args.output_dir is None:
        from datetime import datetime
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.output_dir = f"results/actr_final_{timestamp}_{model_type}"

    os.makedirs(args.output_dir, exist_ok=True)
    os.system(f"ln -sf {args.output_dir} latest_actr")

    # Run analyses
    #layer_results = analyze_activations_by_layer(
    #    args.activations_dir,
    #    args.results_file,
    #    args.output_dir,
    #    args.min_examples
    #)

    # Set number of positions to analyze
    num_positions = 999 # analyze all tokens

    # Add position analysis
    position_results = analyze_activations_by_position(
        args.activations_dir,
        args.results_file,
        args.output_dir,
        args.min_examples,
        num_positions
    )

    # Add grid analysis
    grid_results = analyze_activations_by_grid(
        args.activations_dir,
        args.results_file,
        args.output_dir,
        args.min_examples,
        num_positions,
        act_file_prefix="activations_",
    )

    # Do one with just the information_extraction ones
    grid_results_2 = analyze_activations_by_grid(
        args.activations_dir,
        args.results_file,
        f"actr_final_{timestamp}_{model_type}-information_extraction",
        3, # args.min_examples,
        num_positions,
        act_file_prefix="activations_information_extraction_",
    )

    print(f"Analysis complete. Layer results and position results saved to {args.output_dir}")
#%%

print("Done!")

#%%
