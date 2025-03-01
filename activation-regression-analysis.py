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

def analyze_activations_by_layer(activations_dir, output_dir, min_examples=10):
    """
    Perform regression analysis by layer to find activation directions related to direct vs quoted instruction preferences.
    
    Args:
        activations_dir: Directory containing activation files and metadata
        output_dir: Directory to save analysis results
        min_examples: Minimum number of examples required to analyze a layer
    
    Returns:
        dict: Analysis results by layer
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Create plots directory
    plots_dir = os.path.join(output_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)
    
    # Find all activation files
    activation_files = [f for f in os.listdir(activations_dir) if f.startswith("activations_") and f.endswith(".pt")]
    if not activation_files:
        raise ValueError(f"No activation files found in {activations_dir}")
    
    print(f"Found {len(activation_files)} activation files")
    
    # Load a sample activation file to determine available layers
    sample_file = os.path.join(activations_dir, activation_files[0])
    sample_activation = torch.load(sample_file)
    
    # Extract layer names
    layer_names = [name for name in sample_activation.keys() if name.startswith("layer_") and not name.endswith("_input")]
    
    print(f"Found {len(layer_names)} layers: {layer_names}")
    
    # Dictionary to store results by layer
    layer_results = {}
    
    # Analyze each layer
    for layer_name in tqdm(layer_names, desc="Analyzing layers"):
        print(f"\nProcessing {layer_name}...")
        
        # Collect data for this layer
        X = []  # Activations
        y = []  # Probability differences
        example_ids = []  # Keep track of which examples we use
        
        # Process all activation files
        for act_file in tqdm(activation_files, desc=f"Processing {layer_name}", leave=False):
            try:
                # Extract ID from filename
                example_id = act_file.replace("activations_", "").replace(".pt", "")
                
                # Load activations
                act_path = os.path.join(activations_dir, act_file)
                activation = torch.load(act_path)
                
                # Check if this layer exists in the file
                if layer_name not in activation:
                    continue
                
                # Load metadata
                meta_file = f"metadata_{example_id}.json"
                meta_path = os.path.join(activations_dir, meta_file)
                
                if not os.path.exists(meta_path):
                    print(f"Warning: No metadata found for {example_id}, skipping")
                    continue
                
                with open(meta_path, 'r') as f:
                    metadata = json.load(f)
                
                # Get direct and quoted token probabilities
                direct_prob = metadata.get('direct_token_prob')
                quoted_prob = metadata.get('quoted_token_prob')
                
                if direct_prob is None or quoted_prob is None:
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
                
            except Exception as e:
                print(f"Error processing {act_file}: {e}")
        
        # Skip if not enough data
        if len(X) < min_examples:
            print(f"Only {len(X)} examples for {layer_name}, skipping (need at least {min_examples})")
            continue
        
        X = np.array(X)
        y = np.array(y)
        
        print(f"Collected {len(X)} examples with shapes: X = {X.shape}, y = {y.shape}")
        
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
        plt.xlabel('Actual Probability Difference')
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
        layer_labels = [layer.replace('layer_', '') for layer in layers]
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
                        help="Directory containing activation files and metadata")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Directory to save analysis results")
    parser.add_argument("--min-examples", type=int, default=10,
                        help="Minimum number of examples required to analyze a layer")
    
    args = parser.parse_args()
    
    # Create output directory
    if args.output_dir is None:
        from datetime import datetime
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.output_dir = f"activation_regression_{timestamp}"
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Run analysis
    analyze_activations_by_layer(args.activations_dir, args.output_dir, args.min_examples)
    
    print(f"Analysis complete. Results saved to {args.output_dir}")

if __name__ == "__main__":
    main()
