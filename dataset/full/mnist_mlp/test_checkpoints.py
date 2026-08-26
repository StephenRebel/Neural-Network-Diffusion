import json
import os
import sys
import torch

try:
    from finetune import *
except ImportError:
    from .finetune import *

# Directories or individual checkpoints supplied on command line.
test_paths = sys.argv[1:] if len(sys.argv) > 1 else ["./checkpoint"]


def get_checkpoint_files(path):
    """Return all checkpoint files contained in a directory or a single file."""
    if os.path.isfile(path):
        return [path]

    if os.path.isdir(path):
        return sorted(
            os.path.join(path, f)
            for f in os.listdir(path)
            if f.endswith(".pth")
        )

    raise FileNotFoundError(path)


if __name__ == "__main__":

    config = get_config()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _, test_loader = get_data_loaders(config)

    report = {}

    for path in test_paths:

        checkpoint_files = get_checkpoint_files(path)

        losses = []
        accuracies = []
        checkpoint_results = {}

        print(f"\nEvaluating {path}")

        for ckpt in checkpoint_files:

            state = torch.load(
                ckpt,
                map_location=device,
                weights_only=True,
            )

            model.load_state_dict(
                {k: v.to(torch.float32).to(device) for k, v in state.items()},
                strict=False,
            )

            loss, acc, _, _ = test(model, test_loader, device)

            losses.append(loss)
            accuracies.append(acc)

            checkpoint_results[os.path.basename(ckpt)] = {
                "loss": float(loss),
                "accuracy": float(acc),
            }

            print(
                f"  {os.path.basename(ckpt):30}"
                f" Loss={loss:.4f}"
                f" Acc={acc:.4f}"
            )

        report[os.path.basename(os.path.normpath(path))] = {
            "num_checkpoints": len(checkpoint_files),
            "average_loss": float(sum(losses) / len(losses)),
            "average_accuracy": float(sum(accuracies) / len(accuracies)),
            "checkpoints": checkpoint_results,
        }

    output_file = "evaluation_report.json"

    with open(output_file, "w") as f:
        json.dump(report, f, indent=4)

    print(f"\nSaved report to {output_file}")

    print("\nSummary")
    print("-" * 60)
    for name, stats in report.items():
        print(
            f"{name:25}"
            f" Loss={stats['average_loss']:.4f}"
            f" Acc={stats['average_accuracy']:.4f}"
            f" ({stats['num_checkpoints']} checkpoints)"
        )