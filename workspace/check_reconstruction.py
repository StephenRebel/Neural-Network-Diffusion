import sys, os
import torch
import torch.nn as nn
import importlib

# Setup paths
root = os.sep + os.sep.join(__file__.split(os.sep)[1:__file__.split(os.sep).index("Neural-Network-Diffusion")+1])
sys.path.append(root)
os.chdir(root)

# Load configuration and VAE
print('==> Loading configuration and VAE...')
item = importlib.import_module(f"workspace.{sys.argv[1]}.{sys.argv[2]}")
config = item.config
vae = item.vae

config["tag"] = config.get("tag") if config.get("tag") is not None else os.path.basename(item.__file__)[:-3]
generate_config = {
    "device": "cuda",
    "checkpoint": f"./checkpoint/{config['tag']}.pth",
}
config.update(generate_config)

# Load VAE weights
print('==> Building VAE model...')
diction = torch.load(config["checkpoint"], map_location="cpu", weights_only=True)
vae.load_state_dict(diction["vae"])
vae = vae.to(config["device"])
vae.eval()

# Instantiate dataset to get preprocessing pipeline (normalization stats, tokenization)
print('==> Setting up dataset preprocessing...')
divide_slice_length = 64
train_set = config["dataset"](dim_per_token=divide_slice_length,
                              granularity=0,
                              pe_granularity=0,
                              fill_value=0.)
print(f"Dataset sequence length: {train_set.sequence_length}")


def build_linear_container(state_dict):
    """Reconstruct real nn.Linear modules from a flat state_dict."""
    container = nn.Module()
    prefixes = []
    for key in state_dict:
        if key.endswith(".weight") and state_dict[key].dim() == 2:
            prefixes.append(key[: -len(".weight")])
    for i, prefix in enumerate(prefixes):
        weight = state_dict[prefix + ".weight"]
        bias_key = prefix + ".bias"
        bias = state_dict.get(bias_key)
        out_f, in_f = weight.shape
        linear = nn.Linear(in_f, out_f, bias=bias is not None)
        with torch.no_grad():
            linear.weight.copy_(weight)
            if bias is not None:
                linear.bias.copy_(bias)
        # attribute name doesn't matter for named_modules/isinstance to work
        container.add_module(f"linear_{i}__{prefix.replace('.', '_')}", linear)
    return container


def report_sparsity(parameters_to_prune):
    """Same logic as from checkpoint pruning reporting"""
    sparsity_report = {}
    total_zeros = 0
    total_elements = 0
    for module, name in parameters_to_prune:
        weight_tensor = getattr(module, name)
        sparsity_level = 100. * float(torch.sum(weight_tensor == 0)) / weight_tensor.nelement()
        sparsity_report[str(module)] = sparsity_level
        total_zeros += float(torch.sum(weight_tensor == 0))
        total_elements += weight_tensor.nelement()
        print(f"Sparsity in {module}: {sparsity_level:.2f}%")
    global_sparsity = 100. * total_zeros / total_elements
    sparsity_report["global_sparsity"] = global_sparsity
    print(f"Total Sparsity: {global_sparsity:.2f}%\n")
    return sparsity_report


# Load checkpoint, check sparsity
sparse_checkpoint_path = "./dataset/full/mnist_mlp/checkpoint/0002_acc0.9624_seed0040_mnist_mlp.pth"
# Could refactor to check many but fairly confident this is a point of failure

if not os.path.exists(sparse_checkpoint_path):
    print(f"Error: Sparse checkpoint not found: {sparse_checkpoint_path}")
    print("Please update 'sparse_checkpoint_path' to a valid sparse checkpoint file.")
    sys.exit(1)

print(f"Loading sparse checkpoint: {sparse_checkpoint_path}")
sparse_state_dict = torch.load(sparse_checkpoint_path, map_location="cpu", weights_only=True)

print("\n==> Original checkpoint sparsity:")
original_model = build_linear_container(sparse_state_dict)
original_parameters_to_prune = []
for name, module in original_model.named_modules():
    if isinstance(module, nn.Linear):
        original_parameters_to_prune.append((module, 'weight'))
original_sparsity_report = report_sparsity(original_parameters_to_prune)

# Preprocess, VAE encode/decode, postprocess, same way training/gen do
print("==> Applying dataset preprocessing pipeline...")
with torch.no_grad():
    param_tensor = train_set.preprocess(sparse_state_dict)
    print(f"After preprocess shape: {param_tensor.shape}")
    param_tensor = param_tensor.flatten(0, 1).unsqueeze(0).to(config["device"])
    print(f"Flattened to batch format: {param_tensor.shape}")

    print("\n==> Running VAE encode/decode...")
    reconstructed_tensor, _, mu, log_var = vae.encode_decode(
        input=param_tensor, use_var=True, manual_std=0.1
    )
    print(f"Reconstructed shape: {reconstructed_tensor.shape}")

    mse_loss = torch.nn.functional.mse_loss(param_tensor, reconstructed_tensor)
    l1_loss = torch.nn.functional.l1_loss(param_tensor, reconstructed_tensor)
    print(f"\n=== Reconstruction Quality (token space) ===")
    print(f"MSE Loss: {mse_loss.item():.6f}")
    print(f"L1 Loss:  {l1_loss.item():.6f}")

    # Convert the reconstructed tensor back into checkpoint form
    reconstructed_state_dict = train_set.postprocess(reconstructed_tensor.cpu())

# Save reconstructed checkpoint in the expected form, reload it and check sparsity
reconstructed_checkpoint_path = "./dataset/full/mnist_mlp/checkpoint/ensemble_reconstructed_test.pth"
torch.save(reconstructed_state_dict, reconstructed_checkpoint_path)
print(f"\nSaved reconstructed checkpoint to: {reconstructed_checkpoint_path}")

reloaded_state_dict = torch.load(reconstructed_checkpoint_path, map_location="cpu", weights_only=True)

print("\n==> Reconstructed checkpoint sparsity:")
reconstructed_model = build_linear_container(reloaded_state_dict)
reconstructed_parameters_to_prune = []
for name, module in reconstructed_model.named_modules():
    if isinstance(module, nn.Linear):
        reconstructed_parameters_to_prune.append((module, 'weight'))
reconstructed_sparsity_report = report_sparsity(reconstructed_parameters_to_prune)

print("=== Sparsity Comparison ===")
print(f"Original Global Sparsity:      {original_sparsity_report['global_sparsity']:.2f}%")
print(f"Reconstructed Global Sparsity: {reconstructed_sparsity_report['global_sparsity']:.2f}%")
print(f"Change:                        {reconstructed_sparsity_report['global_sparsity'] - original_sparsity_report['global_sparsity']:+.2f}%")

# Per-layer sparsity comparison
print("\n=== Per-Layer Sparsity Comparison ===")
original_by_prefix = {}
for name, module in original_model.named_modules():
    if isinstance(module, nn.Linear):
        # strip the "linear_i__" tag back to the real checkpoint prefix
        prefix = name.split("__", 1)[1].replace("_", ".") if "__" in name else name
        original_by_prefix[prefix] = module

reconstructed_by_prefix = {}
for name, module in reconstructed_model.named_modules():
    if isinstance(module, nn.Linear):
        prefix = name.split("__", 1)[1].replace("_", ".") if "__" in name else name
        reconstructed_by_prefix[prefix] = module

for prefix in original_by_prefix:
    orig_w = original_by_prefix[prefix].weight
    orig_sparsity = 100. * float(torch.sum(orig_w == 0)) / orig_w.nelement()
    if prefix in reconstructed_by_prefix:
        recon_w = reconstructed_by_prefix[prefix].weight
        recon_sparsity = 100. * float(torch.sum(recon_w == 0)) / recon_w.nelement()
        print(f"  {prefix}: original {orig_sparsity:.2f}% -> reconstructed {recon_sparsity:.2f}% "
              f"(change: {recon_sparsity - orig_sparsity:+.2f}%)")
    else:
        print(f"  {prefix}: original {orig_sparsity:.2f}% -> NOT FOUND in reconstructed checkpoint")

# Analyze sparsity leakage at originally-zero positions over multiple thresholds
print("\n=== Leakage at Original Zero Positions (multiple thresholds) ===")
thresholds = [0.0, 1e-4, 1e-3, 1e-2, 1e-1]

for prefix in original_by_prefix:
    if prefix not in reconstructed_by_prefix:
        continue
    orig_w = original_by_prefix[prefix].weight.detach()
    recon_w = reconstructed_by_prefix[prefix].weight.detach()
    if orig_w.shape != recon_w.shape:
        print(f"  {prefix}: shape mismatch {orig_w.shape} vs {recon_w.shape}, skipping")
        continue

    zero_mask = (orig_w == 0)
    num_zero_positions = int(zero_mask.sum())
    if num_zero_positions == 0:
        print(f"  {prefix}: no zero positions in original, skipping")
        continue

    leaked = recon_w[zero_mask]
    print(f"\n  {prefix} ({num_zero_positions} originally-zero positions):")
    print(f"    max |leak| = {leaked.abs().max().item():.6f}  "
          f"mean |leak| = {leaked.abs().mean().item():.6f}  "
          f"std = {leaked.std().item():.6f}")
    for t in thresholds:
        pct_within = 100. * float(torch.sum(leaked.abs() <= t)) / num_zero_positions
        print(f"    still within {t:g} of zero: {pct_within:.2f}%")

# Average leakage across all layers combined
print("\n=== Leakage at Original Zero Positions (all Linear layers combined) ===")
all_leaked = []
all_zero_count = 0
for prefix in original_by_prefix:
    if prefix not in reconstructed_by_prefix:
        continue
    orig_w = original_by_prefix[prefix].weight.detach()
    recon_w = reconstructed_by_prefix[prefix].weight.detach()
    if orig_w.shape != recon_w.shape:
        continue
    zero_mask = (orig_w == 0)
    all_zero_count += int(zero_mask.sum())
    all_leaked.append(recon_w[zero_mask])

if all_leaked:
    all_leaked = torch.cat(all_leaked)
    print(f"Total originally-zero positions: {all_zero_count}")
    print(f"max |leak| = {all_leaked.abs().max().item():.6f}  "
          f"mean |leak| = {all_leaked.abs().mean().item():.6f}  "
          f"std = {all_leaked.std().item():.6f}")
    for t in thresholds:
        pct_within = 100. * float(torch.sum(all_leaked.abs() <= t)) / all_zero_count
        label = "exact zero" if t == 0.0 else f"within {t:g}"
        print(f"  Still {label}: {pct_within:.2f}%")

print('\n==> VAE validation complete!')