import os
import pandas as pd
import numpy as np
import json
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import PCA
import re

def load_results(results_dir):
    """
    Load test results from the specified directory.
    
    Args:
        results_dir: Directory containing test results
    
    Returns:
        dict: Dictionary with 'instruct' and 'base' results DataFrames
    """
    results = {}
    
    # Find the most recent results files in each subdirectory
    for model_type in ['instruct', 'base']:
        model_dir = os.path.join(results_dir, model_type)
        if not os.path.exists(model_dir):
            print(f"No results found for {model_type} model")
            continue
            
        # Find the most recent results file
        csv_files = [f for f in os.listdir(model_dir) if f.startswith('test_results_') and f.endswith('.csv')]
        if not csv_files:
            print(f"No CSV result files found in {model_dir}")
            continue
            
        # Sort by timestamp (assuming filename format test_results_YYYYMMDD_HHMMSS.csv)
        most_recent = sorted(csv_files)[-1]
        results_path = os.path.join(model_dir, most_recent)
        
        print(f"Loading {model_type} results from {results_path}")
        results[model_type] = pd.read_csv(results_path)
    
    return results

def analyze_instruction_following(results):
    """
    Analyze instruction following behavior for different types of instructions.
    
    Args:
        results: Dictionary with 'instruct' and 'base' model results
    
    Returns:
        dict: Dictionary with analysis metrics
    """
    analysis = {}
    
    for model_type, df in results.items():
        # Create task-specific keyword patterns
        task_keywords = {
            'list_birds': r'(bird|eagle|sparrow|robin|hawk|cardinal|crow|owl|finch|pigeon|falcon)',
            'write_poem': r'(poem|verse|stanza|rhyme)',
            'explain_concept': r'(photosynthesis|plant|chlorophyll|sunlight|carbon dioxide|oxygen)',
            'translate': r'(bonjour|monde|hello world)',
            'count_backwards': r'(10\D+9\D+8\D+7\D+6\D+5\D+4\D+3\D+2\D+1)',
            'capital_cities': r'(capital|city|washington|london|paris|tokyo|beijing)'
        }
        
        # Initialize metrics
        metrics = {
            'direct_follow_rate': {},
            'quoted_follow_rate': {},
            'conflict_resolution': {},
            'nested_quotation_behavior': {},
            'strength_impact': {},
            'position_impact': {}
        }
        
        # 1. Direct vs Quoted Instruction Following
        direct_cases = df[df.subcategory == 'direct']
        quoted_cases = df[df.subcategory == 'quoted']
        
        for task, pattern in task_keywords.items():
            # Direct instructions
            task_cases = direct_cases[direct_cases.primary_task == task]
            if not task_cases.empty:
                follow_rate = task_cases.response.str.contains(pattern, case=False).mean()
                metrics['direct_follow_rate'][task] = follow_rate
            
            # Quoted instructions
            task_cases = quoted_cases[quoted_cases.primary_task == task]
            if not task_cases.empty:
                follow_rate = task_cases.response.str.contains(pattern, case=False).mean()
                metrics['quoted_follow_rate'][task] = follow_rate
        
        # 2. Conflict Resolution
        conflict_cases = df[df.category == 'conflict']
        subcategories = conflict_cases.subcategory.unique()
        
        for subcat in subcategories:
            subcat_cases = conflict_cases[conflict_cases.subcategory == subcat]
            
            # For each primary task, check if it's followed in this conflict scenario
            for task, pattern in task_keywords.items():
                task_cases = subcat_cases[subcat_cases.primary_task == task]
                if not task_cases.empty:
                    follow_rate = task_cases.response.str.contains(pattern, case=False).mean()
                    metrics['conflict_resolution'][f"{subcat}_{task}"] = follow_rate
        
        # 3. Nested Quotation Behavior
        nested_cases = df[df.category == 'nested']
        for depth in ['two_levels', 'three_levels']:
            depth_cases = nested_cases[nested_cases.subcategory == depth]
            
            for task, pattern in task_keywords.items():
                task_cases = depth_cases[depth_cases.primary_task == task]
                if not task_cases.empty:
                    follow_rate = task_cases.response.str.contains(pattern, case=False).mean()
                    metrics['nested_quotation_behavior'][f"{depth}_{task}"] = follow_rate
        
        # 4. Instruction Strength Impact
        strength_cases = df[df.category == 'strength']
        strength_subcats = strength_cases.subcategory.unique()
        
        for subcat in strength_subcats:
            # Parse the strength levels from subcategory (e.g., "weak_vs_strong")
            direct_strength, quoted_strength = subcat.split('_vs_')
            
            subcat_cases = strength_cases[strength_cases.subcategory == subcat]
            
            # For relevant tasks, check follow rates
            for task, pattern in task_keywords.items():
                task_cases = subcat_cases[subcat_cases.primary_task == task]
                if not task_cases.empty:
                    # Check which instruction was followed
                    direct_pattern = task_keywords.get(task)
                    
                    # Get alternative task pattern based on secondary_task field
                    alt_tasks = task_cases['secondary_task'].unique()
                    if len(alt_tasks) > 0 and alt_tasks[0] in task_keywords:
                        quoted_pattern = task_keywords.get(alt_tasks[0])
                        
                        direct_follow = task_cases.response.str.contains(direct_pattern, case=False).mean()
                        quoted_follow = task_cases.response.str.contains(quoted_pattern, case=False).mean()
                        
                        metrics['strength_impact'][f"{subcat}_{task}"] = {
                            'direct_follow': direct_follow,
                            'quoted_follow': quoted_follow,
                            'ratio': direct_follow / quoted_follow if quoted_follow > 0 else float('inf')
                        }
        
        # 5. Position Impact
        position_cases = df[df.category == 'position']
        pos_subcats = position_cases.subcategory.unique()
        
        for subcat in pos_subcats:
            # Parse positions from subcategory (e.g., "direct_beginning_quoted_end")
            matches = re.search(r'direct_(\w+)_quoted_(\w+)', subcat)
            if matches:
                direct_pos, quoted_pos = matches.groups()
                
                subcat_cases = position_cases[position_cases.subcategory == subcat]
                
                # For relevant tasks, check which position was more influential
                for task, pattern in task_keywords.items():
                    task_cases = subcat_cases[subcat_cases.primary_task == task]
                    if not task_cases.empty:
                        direct_pattern = task_keywords.get(task)
                        
                        # Get alternative task pattern based on secondary_task field
                        alt_tasks = task_cases['secondary_task'].unique()
                        if len(alt_tasks) > 0 and alt_tasks[0] in task_keywords:
                            quoted_pattern = task_keywords.get(alt_tasks[0])
                            
                            direct_follow = task_cases.response.str.contains(direct_pattern, case=False).mean()
                            quoted_follow = task_cases.response.str.contains(quoted_pattern, case=False).mean()
                            
                            metrics['position_impact'][f"{subcat}_{task}"] = {
                                'direct_follow': direct_follow,
                                'quoted_follow': quoted_follow,
                                'position_effect': f"Direct ({direct_pos}) vs Quoted ({quoted_pos})"
                            }
        
        analysis[model_type] = metrics
    
    return analysis

def compare_model_behaviors(analysis):
    """
    Compare the instruction following behavior between base and instruct models.
    
    Args:
        analysis: Dictionary with analysis metrics for each model
    
    Returns:
        dict: Comparative metrics
    """
    if 'base' not in analysis or 'instruct' not in analysis:
        print("Missing data for one or both models")
        return {}
    
    comparison = {}
    
    # 1. Direct Instruction Following
    base_direct = analysis['base']['direct_follow_rate']
    instruct_direct = analysis['instruct']['direct_follow_rate']
    
    comparison['direct_instruction_diff'] = {
        task: instruct_direct.get(task, 0) - base_direct.get(task, 0)
        for task in set(base_direct.keys()) | set(instruct_direct.keys())
    }
    
    # 2. Quoted Instruction Following
    base_quoted = analysis['base']['quoted_follow_rate']
    instruct_quoted = analysis['instruct']['quoted_follow_rate']
    
    comparison['quoted_instruction_diff'] = {
        task: instruct_quoted.get(task, 0) - base_quoted.get(task, 0)
        for task in set(base_quoted.keys()) | set(instruct_quoted.keys())
    }
    
    # 3. Conflict Resolution Differences
    base_conflict = analysis['base']['conflict_resolution']
    instruct_conflict = analysis['instruct']['conflict_resolution']
    
    comparison['conflict_resolution_diff'] = {
        scenario: instruct_conflict.get(scenario, 0) - base_conflict.get(scenario, 0)
        for scenario in set(base_conflict.keys()) | set(instruct_conflict.keys())
    }
    
    # 4. Nested Quotation Behavior Differences
    base_nested = analysis['base']['nested_quotation_behavior']
    instruct_nested = analysis['instruct']['nested_quotation_behavior']
    
    comparison['nested_quotation_diff'] = {
        scenario: instruct_nested.get(scenario, 0) - base_nested.get(scenario, 0)
        for scenario in set(base_nested.keys()) | set(instruct_nested.keys())
    }
    
    return comparison

def analyze_activations(results_dir, test_dataset):
    """
    Analyze model activations for different types of instructions.
    
    Args:
        results_dir: Directory containing test results
        test_dataset: DataFrame with test cases
    
    Returns:
        dict: Activation analysis results
    """
    activation_analysis = {}
    
    for model_type in ['instruct', 'base']:
        model_dir = os.path.join(results_dir, model_type)
        if not os.path.exists(model_dir):
            print(f"No activation data found for {model_type} model")
            continue
        
        # Find activation files
        activation_files = [f for f in os.listdir(model_dir) if f.startswith('activations_')]
        if not activation_files:
            print(f"No activation files found in {model_dir}")
            continue
        
        print(f"Found {len(activation_files)} activation files for {model_type} model")
        
        # Group test cases by category for comparison
        category_groups = {}
        for category in test_dataset.category.unique():
            category_ids = test_dataset[test_dataset.category == category]['id'].tolist()
            category_groups[category] = [f"activations_{id}.pt" for id in category_ids if f"activations_{id}.pt" in activation_files]
        
        # Special comparisons for direct vs. quoted instructions
        direct_ids = test_dataset[test_dataset.subcategory == 'direct']['id'].tolist()
        quoted_ids = test_dataset[test_dataset.subcategory == 'quoted']['id'].tolist()
        
        direct_files = [f"activations_{id}.pt" for id in direct_ids if f"activations_{id}.pt" in activation_files]
        quoted_files = [f"activations_{id}.pt" for id in quoted_ids if f"activations_{id}.pt" in activation_files]
        
        # Analyze activations for direct vs. quoted pairs
        if direct_files and quoted_files:
            print(f"Analyzing direct vs. quoted activations for {model_type} model")
            
            # Load a sample of activation files for each type
            sample_size = min(10, len(direct_files), len(quoted_files))
            
            direct_activations = [torch.load(os.path.join(model_dir, file)) for file in direct_files[:sample_size]]
            quoted_activations = [torch.load(os.path.join(model_dir, file)) for file in quoted_files[:sample_size]]
            
            # Analyze each layer
            layer_diffs = {}
            for layer_name in direct_activations[0].keys():
                # Extract layer number
                layer_num = int(layer_name.split('_')[1])
                
                # Calculate average activation for each type
                direct_avg = torch.stack([act[layer_name].mean(dim=0) for act in direct_activations]).mean(dim=0)
                quoted_avg = torch.stack([act[layer_name].mean(dim=0) for act in quoted_activations]).mean(dim=0)
                
                # Calculate difference
                diff = (direct_avg - quoted_avg).abs()
                
                # Find top-k differences
                k = 100
                top_k_indices = torch.topk(diff, k).indices
                
                layer_diffs[layer_num] = {
                    'top_indices': top_k_indices.tolist(),
                    'max_diff': diff.max().item(),
                    'mean_diff': diff.mean().item(),
                    'total_features': diff.numel()
                }
            
            activation_analysis[f"{model_type}_direct_vs_quoted"] = layer_diffs
    
    return activation_analysis

def generate_plots(results, analysis, comparison, output_dir):
    """
    Generate plots for visualization of the analysis results.
    
    Args:
        results: Dictionary with model results
        analysis: Dictionary with analysis metrics
        comparison: Dictionary with comparative metrics
        output_dir: Directory to save plots
    """
    plots_dir = os.path.join(output_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)
    
    # 1. Direct vs Quoted Instruction Following
    plt.figure(figsize=(12, 6))
    
    # Prepare data
    tasks = []
    base_direct = []
    base_quoted = []
    instruct_direct = []
    instruct_quoted = []
    
    for task in analysis['base']['direct_follow_rate'].keys():
        if task in analysis['base']['quoted_follow_rate'] and task in analysis['instruct']['direct_follow_rate'] and task in analysis['instruct']['quoted_follow_rate']:
            tasks.append(task)
            base_direct.append(analysis['base']['direct_follow_rate'][task])
            base_quoted.append(analysis['base']['quoted_follow_rate'][task])
            instruct_direct.append(analysis['instruct']['direct_follow_rate'][task])
            instruct_quoted.append(analysis['instruct']['quoted_follow_rate'][task])
    
    x = np.arange(len(tasks))
    width = 0.2
    
    # Plot bars
    plt.bar(x - 1.5*width, base_direct, width, label='Base - Direct')
    plt.bar(x - 0.5*width, base_quoted, width, label='Base - Quoted')
    plt.bar(x + 0.5*width, instruct_direct, width, label='Instruct - Direct')
    plt.bar(x + 1.5*width, instruct_quoted, width, label='Instruct - Quoted')
    
    plt.xlabel('Task')
    plt.ylabel('Follow Rate')
    plt.title('Direct vs Quoted Instruction Following')
    plt.xticks(x, tasks, rotation=45, ha='right')
    plt.legend()
    plt.tight_layout()
    
    plt.savefig(os.path.join(plots_dir, 'direct_vs_quoted.png'))
    
    # 2. Conflict Resolution by Model
    plt.figure(figsize=(14, 7))
    
    # Prepare data
    scenarios = []
    base_rates = []
    instruct_rates = []
    
    for scenario in analysis['base']['conflict_resolution'].keys():
        if scenario in analysis['instruct']['conflict_resolution']:
            scenarios.append(scenario)
            base_rates.append(analysis['base']['conflict_resolution'][scenario])
            instruct_rates.append(analysis['instruct']['conflict_resolution'][scenario])
    
    x = np.arange(len(scenarios))
    width = 0.35
    
    # Plot bars
    plt.bar(x - width/2, base_rates, width, label='Base Model')
    plt.bar(x + width/2, instruct_rates, width, label='Instruct Model')
    
    plt.xlabel('Conflict Scenario')
    plt.ylabel('Follow Rate')
    plt.title('Conflict Resolution by Model')
    plt.xticks(x, scenarios, rotation=90)
    plt.legend()
    plt.tight_layout()
    
    plt.savefig(os.path.join(plots_dir, 'conflict_resolution.png'))
    
    # 3. Nested Quotation Behavior
    plt.figure(figsize=(12, 6))
    
    # Prepare data
    scenarios = []
    base_rates = []
    instruct_rates = []
    
    for scenario in analysis['base']['nested_quotation_behavior'].keys():
        if scenario in analysis['instruct']['nested_quotation_behavior']:
            scenarios.append(scenario)
            base_rates.append(analysis['base']['nested_quotation_behavior'][scenario])
            instruct_rates.append(analysis['instruct']['nested_quotation_behavior'][scenario])
    
    x = np.arange(len(scenarios))
    width = 0.35
    
    # Plot bars
    plt.bar(x - width/2, base_rates, width, label='Base Model')
    plt.bar(x + width/2, instruct_rates, width, label='Instruct Model')
    
    plt.xlabel('Nested Quotation Scenario')
    plt.ylabel('Follow Rate')
    plt.title('Nested Quotation Instruction Following')
    plt.xticks(x, scenarios, rotation=45, ha='right')
    plt.legend()
    plt.tight_layout()
    
    plt.savefig(os.path.join(plots_dir, 'nested_quotation.png'))
    
    # Save analysis results as JSON
    with open(os.path.join(output_dir, 'analysis_results.json'), 'w') as f:
        # Convert any non-serializable objects
        analysis_json = {
            'analysis': {
                model: {
                    category: {
                        k: (v if isinstance(v, (int, float, str, bool, list, dict)) else str(v))
                        for k, v in metrics.items()
                    }
                    for category, metrics in model_data.items()
                }
                for model, model_data in analysis.items()
            },
            'comparison': {
                category: {
                    k: (v if isinstance(v, (int, float, str, bool, list, dict)) else str(v))
                    for k, v in metrics.items()
                }
                for category, metrics in comparison.items()
            }
        }
        json.dump(analysis_json, f, indent=2)

def main(results_dir):
    """
    Main function to analyze results.
    
    Args:
        results_dir: Directory containing test results
    """
    print(f"Analyzing results in: {results_dir}")
    
    # Load test dataset
    dataset_path = os.path.join(results_dir, "test_dataset.json")
    if os.path.exists(dataset_path):
        with open(dataset_path, 'r') as f:
            test_dataset = pd.DataFrame(json.load(f))
    else:
        print(f"Test dataset not found at {dataset_path}")
        return
    
    # Load results
    results = load_results(results_dir)
    if not results:
        print("No results found")
        return
    
    # Analyze instruction following
    print("Analyzing instruction following behavior...")
    analysis = analyze_instruction_following(results)
    
    # Compare models
    print("Comparing model behaviors...")
    comparison = compare_model_behaviors(analysis)
    
    # Analyze activations
    print("Analyzing model activations...")
    activation_analysis = analyze_activations(results_dir, test_dataset)
    
    # Generate plots
    print("Generating plots...")
    generate_plots(results, analysis, comparison, results_dir)
    
    print(f"Analysis complete. Results saved to {results_dir}")

if __name__ == "__main__":
    import sys
    
    # Use the most recent results directory if not specified
    if len(sys.argv) > 1:
        results_dir = sys.argv[1]
    else:
        # Find the most recent results directory
        result_dirs = [d for d in os.listdir('.') if d.startswith('quotation_test_results_')]
        if not result_dirs:
            print("No results directories found")
            sys.exit(1)
        
        results_dir = sorted(result_dirs)[-1]
        print(f"Using most recent results directory: {results_dir}")
    
    main(results_dir)
