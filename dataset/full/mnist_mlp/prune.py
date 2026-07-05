import os
import json
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.utils.prune as prune
from model import Model
from finetune import get_data_loaders, test


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_config():
    config_file = os.path.join(os.path.dirname(os.path.dirname(__file__)), "config.json")
    with open(config_file, "r") as f:
        additional_config = json.load(f)

    config = {
        "dataset_root": "from_additional_config",
        "batch_size": 128,
        "num_workers": 4,
        "sparsity_level": 0.5,
        "tag": os.path.basename(os.path.dirname(__file__)),
        "seed": 40
    }
    config.update(additional_config)
    return config


def report_sparsity(parameters_to_prune):
    sparsity_report = {}

    for module, name in parameters_to_prune:
        weight_tensor = getattr(module, name)
        sparsity_level = 100. * float(torch.sum(weight_tensor == 0)) / weight_tensor.nelement()
        sparsity_report[str(module)] = sparsity_level

        print(f"Sparsity in {module}: {sparsity_level:.2f}%")

    return sparsity_report


def main():
    config = get_config()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    set_seed(config['seed'])

    prune_report = {
        "config": {**config},
    }

    train_loader, test_loader = get_data_loaders(config)

    # Load trained dense model checkpoint
    model = Model()
    checkpoint_path = os.path.join(os.path.dirname(__file__), "checkpoint", "model.pth")
    state_dict = torch.load(checkpoint_path, map_location=device, weights_only=True)
    model.load_state_dict(state_dict)
    model.to(device)

    # Define the pruning parameters
    parameters_to_prune = []
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            parameters_to_prune.append((module, 'weight'))

    print("Initial test:")
    loss, acc, _, _ = test(model, test_loader, device)
    prune_report["initial_test"] = {"loss": loss, "accuracy": acc}
    prune_report["initial_sparsity"] = report_sparsity(parameters_to_prune)

    # Apply global unstructured pruning
    prune.global_unstructured(
        parameters_to_prune,
        pruning_method=prune.L1Unstructured,
        amount=config['sparsity_level'], 
    )

    print("After pruning:")
    loss, acc, _, _ = test(model, test_loader, device)
    prune_report["after_pruning_test"] = {"loss": loss, "accuracy": acc}
    prune_report["after_pruning_sparsity"] = report_sparsity(parameters_to_prune)


    # Save the pruned model checkpoint
    for module, name in parameters_to_prune:
        prune.remove(module, name)

    pruned_checkpoint_path = "pruned_pretrained.pth"
    torch.save(model.state_dict(), pruned_checkpoint_path)
    print(f"Pruned model saved to {pruned_checkpoint_path}")

    os.makedirs("./train_logs", exist_ok=True)
    with open(os.path.join("./train_logs", "mlp_finetune_report.json"), "w") as f:
        json.dump(prune_report, f, indent=2)

if __name__ == "__main__":
    main()