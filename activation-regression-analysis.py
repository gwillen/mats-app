#!/usr/bin/env python3

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
    direct_token = test_result.get('grading_direct_token')
    quoted_token = test_result.get('grading_quoted_token')
    
    # Handle missing tokens
    if not direct_token or not quoted_token:
        # Try alternative fields
        direct_token = test_result.get('expected_direct_response')
        quoted_token = test_result.get('expected_quoted_response')
        
        if not direct_token or not quoted_token:
            return None, None
    
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

def analyze_activations_by_layer(activations_dir, results_file=None, output_dir=None, min_examples=10):
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
    
    # Analyze each layer
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
                
                # Check if this layer exists in the file
                if layer_name not in activation:
                    continue
                
                # Calculate probability difference
                prob_diff = direct_prob - quoted_prob
                
                # Extract and process activations
                layer_act = activation[layer_name]
                
                # Mean pooling across sequence length
                if len(layer_act.shape) > 2:
                    layer_act = layer_act.mean(dim=1)
                
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
        
        # Visualize predictions vs actual
        plt.figure(figsize=(10, 6))
        plt.scatter(y, y_pred, alpha=0.5)
        plt.plot([-1, 1], [-1, 1], 'k--', alpha=0.5)  # Diagonal line
        plt.xlabel('Actual Probability Difference (Direct - Quoted)')
        plt.ylabel('Predicted Probability Difference')
        plt.title(f'Layer {layer_name}: Regression Performance (R² = {r2:.4f})')
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
        plt.title(f'Layer {layer_name}: Top 20 Dimensions')
        plt.xticks(range(20), top_idx[:20], rotation=90)
        plt.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, f'{layer_name}_top_dims.png'))
        plt.close()
    
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
        plt.title('Predictive Power of Each Layer for Direct vs. Quoted Decision')
        
        # Format layer names for readability
        layer_labels = [layer.replace('layer_', '').replace('_output', '') for layer in layers]
        plt.xticks(range(len(layers)), layer_labels, rotation=90)
        
        plt.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, 'layer_r2_scores.png'))
        plt.close()
        
        # Save numerical results as JSON
        results_path = os.path.join(output_dir, 'regression_results.json')
        
        # Make sure all values are serializable
        serializable_results = {}
        for layer, results in layer_results.items():
            serializable_results[layer] = {k: v for k, v in results.items()}
        
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

def main():
    parser = argparse.ArgumentParser(description="Analyze model activations using regression")
    
    parser.add_argument("--activations-dir", type=str, required=True,
                        help="Directory containing activation files")
    parser.add_argument("--results-file", type=str, default=None,
                        help="Path to test results JSON file (if None, will search for most recent)")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Directory to save analysis results")
    parser.add_argument("--min-examples", type=int, default=10,
                        help="Minimum number of examples required to analyze a layer")
    
    args = parser.parse_args()
    
    # Create output directory if not specified
    if args.output_dir is None:
        from datetime import datetime
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.output_dir = f"activation_regression_{timestamp}"
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Run analysis
    analyze_activations_by_layer(
        args.activations_dir, 
        args.results_file,
        args.output_dir, 
        args.min_examples
    )
    
    print(f"Analysis complete. Results saved to {args.output_dir}")

if __name__ == "__main__":
    main()
