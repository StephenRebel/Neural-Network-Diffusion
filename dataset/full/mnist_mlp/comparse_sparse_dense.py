import os
import json
import torch
import torch.nn as nn
from model import Model


def get_effective_sparsity_multi_threshold(model, thresholds):
    """
    Calculates sparsity for multiple thresholds simultaneously.
    
    Args:
        model: model checkpoint
        thresholds: List of float values (e.g., [1e-4, 1e-3, 1e-2, 1e-1])
    
    Returns:
        Dictionary mapping threshold -> sparsity percentage
    """
    # Collect all weights from Linear layers to which sparsity was applied
    all_weights = []
    for module in model.modules():
        if isinstance(module, nn.Linear):
            all_weights.append(module.weight.data.flatten())
    
    if not all_weights:
        return {t: 0.0 for t in thresholds}
    
    all_weights_tensor = torch.cat(all_weights)
    total_elements = all_weights_tensor.numel()
    
    # Calculate sparsity for each threshold
    sparsity_dict = {}
    for threshold in thresholds:
        zeros = torch.sum(torch.abs(all_weights_tensor) <= threshold).item()
        sparsity = (100.0 * zeros / total_elements) if total_elements > 0 else 0.0
        sparsity_dict[threshold] = sparsity
    
    return sparsity_dict


def process_checkpoints(ckpt_dir, json_path, thresholds):
    """
    Loads checkpoints and maps them to JSON progress data.
    Calculates sparsity across multiple thresholds for each checkpoint.
    """
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    progress = data['finetune_progress']
    results = {}
    
    # Ensure alphabetic/numeric order matches the progress tracking array
    files = sorted([f for f in os.listdir(ckpt_dir) if f.endswith('.pth')])
    
    model = Model()
    for idx, (progress_entry, filename) in enumerate(zip(progress, files)):
        ckpt_path = os.path.join(ckpt_dir, filename)
        state_dict = torch.load(ckpt_path, map_location='cpu', weights_only=True)
        model.load_state_dict(state_dict)
        
        # Calculate sparsity for all thresholds at once
        sparsity_multi = get_effective_sparsity_multi_threshold(model, thresholds)

        results[idx] = {
            "file": filename,
            "steps": progress_entry['steps'],
            "loss": progress_entry['loss'],
            "acc": progress_entry['acc'],
            "sparsity_by_threshold": sparsity_multi
        }
    return results, data['config']

def main():
    # Checkpoint paths
    dense_ckpt_dir = "./generated_dense"
    sparse_ckpt_dir = "./generated_sparse"
    # dense_ckpt_dir = "./checkpoint_dense"
    # sparse_ckpt_dir = "./checkpoint_sparse"
    dense_json = "./train_logs/mlp_finetune_report.json"
    sparse_json = "./train_logs/sparse_mlp_finetune_report.json"

    # Define thresholds to evaluate
    THRESHOLDS = [0, 1e-4, 1e-3, 1e-2, 1e-1]

    # Process both datasets
    dense_data, dense_cfg = process_checkpoints(dense_ckpt_dir, dense_json, thresholds=THRESHOLDS)
    sparse_data, sparse_cfg = process_checkpoints(sparse_ckpt_dir, sparse_json, thresholds=THRESHOLDS)

    # Compute averages for each threshold
    summary_by_threshold = {}
    for threshold in THRESHOLDS:
        avg_dense_sparse = sum(
            d['sparsity_by_threshold'][threshold] for d in dense_data.values()
        ) / len(dense_data)
        
        avg_sparse_sparse = sum(
            d['sparsity_by_threshold'][threshold] for d in sparse_data.values()
        ) / len(sparse_data)
        
        summary_by_threshold[threshold] = {
            "avg_sparsity_in_dense_generated": avg_dense_sparse,
            "avg_sparsity_in_sparse_generated": avg_sparse_sparse,
            "difference": avg_sparse_sparse - avg_dense_sparse
        }
    
    avg_acc_dense = sum(d['acc'] for d in dense_data.values()) / len(dense_data)
    avg_acc_sparse = sum(d['acc'] for d in sparse_data.values()) / len(sparse_data)

    report = {
        "comparison_summary": {
            "thresholds_evaluated": THRESHOLDS,
            "total_checkpoints_compared": len(sparse_data),
            "summary_by_threshold": summary_by_threshold,
            "avg_training_accuracy": {
                "dense": avg_acc_dense,
                "sparse": avg_acc_sparse
            }
        },
        "dense_config": dense_cfg,
        "sparse_config": sparse_cfg,
        "detailed_comparison": {
            i: {
                "file_dense": dense_data[i]["file"],
                "file_sparse": sparse_data[i]["file"],
                "steps": dense_data[i]["steps"],
                "loss_dense": dense_data[i]["loss"],
                "loss_sparse": sparse_data[i]["loss"],
                "acc_dense": dense_data[i]["acc"],
                "acc_sparse": sparse_data[i]["acc"],
                "sparsity_dense": dense_data[i]["sparsity_by_threshold"],
                "sparsity_sparse": sparse_data[i]["sparsity_by_threshold"]
            } 
            for i in range(len(sparse_data))
        }
    }

    # Update save path as needed for different comparisons
    output_path = "comparison_report.json"
    with open(output_path, "w") as f:
        json.dump(report, f, indent=2)
    
    # Print summary
    print("\n=== Unbiased Sparsity Report (Multi-Threshold Analysis) ===\n")
    print(f"Thresholds Evaluated: {THRESHOLDS}")
    print(f"Total Checkpoints Compared: {len(sparse_data)}\n")
    
    print(f"{'Threshold':<12} {'Dense Avg (%)':<15} {'Sparse Avg (%)':<15} {'Difference (%)':<15}")
    print("-" * 57)
    for threshold in THRESHOLDS:
        summary = summary_by_threshold[threshold]
        dense_avg = summary["avg_sparsity_in_dense_generated"]
        sparse_avg = summary["avg_sparsity_in_sparse_generated"]
        diff = summary["difference"]
        print(f"{threshold:<12.1e} {dense_avg:<15.2f} {sparse_avg:<15.2f} {diff:<15.2f}")
    
    print(f"\nReport saved to {output_path}")

if __name__ == "__main__":
    main()