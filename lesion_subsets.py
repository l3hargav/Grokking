import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset
import torch.nn as nn
import glob
import os

torch.manual_seed(42)
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
EQUAL_TOKEN = 97
EMBED_DIM = 64
N_HEADS = 4
FFN_DIM = 512

data = pd.read_csv("dataset.csv")
_, val_df = train_test_split(data, test_size=0.8, random_state=42)

def convert_tensor(df):
    x = torch.tensor(df[['a', 'b']].values, dtype=torch.long)
    eq = torch.full((len(x), 1), EQUAL_TOKEN)
    x = torch.cat([x, eq], dim=1)
    y = torch.tensor(df["output"].values, dtype=torch.long)
    return x, y

val_x, val_y = convert_tensor(val_df)
val_loader = DataLoader(TensorDataset(val_x, val_y), batch_size=512, shuffle=False)

class Transformer(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(98, EMBED_DIM)
        encoder = nn.TransformerEncoderLayer(
            d_model=EMBED_DIM, nhead=N_HEADS, dim_feedforward=FFN_DIM,
            batch_first=True, activation='gelu'
        )
        self.transformer = nn.TransformerEncoder(encoder, num_layers=1)
        self.norm = nn.LayerNorm(EMBED_DIM)
        self.fc = nn.Linear(EMBED_DIM, 97)

    def forward(self, x):
        x = self.embedding(x)
        x = self.transformer(x)
        x = self.norm(x)
        x = x[:, -1, :]
        x = self.fc(x)
        return x

def eval_accuracy(model):
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for x, y in val_loader:
            x, y = x.to(device), y.to(device)
            correct += (model(x).argmax(dim=1) == y).sum().item()
            total += y.size(0)
    return correct / total

def ablate_neurons(model, neuron_indices):
    layer = model.transformer.layers[0]
    indices = list(neuron_indices)

    def hook_fn(module, input, output):
        out = output.clone()
        out[:, :, indices] = 0.0
        return out

    handle = layer.linear1.register_forward_hook(hook_fn)
    return handle

def get_single_drop_ranking(model, baseline):
    drops = np.zeros(FFN_DIM)
    for i in range(FFN_DIM):
        handle = ablate_neurons(model, [i])
        acc = eval_accuracy(model)
        handle.remove()
        drops[i] = baseline - acc
    return drops

named_checkpoints = {
    "epoch_1000 (memorization)": "model_early_memorization.pth",
    "epoch_2500 (plateau)":      "model_deep_plateau.pth",
    "epoch_5000 (pre-grokking)": "model_pre_grokking.pth",
}
grokking_files = sorted(glob.glob("model_grokking_epoch_*.pth"))
if grokking_files:
    epoch_num = grokking_files[-1].replace("model_grokking_epoch_", "").replace(".pth", "")
    named_checkpoints[f"epoch_{epoch_num} (grokked)"] = grokking_files[-1]

checkpoints = {k: v for k, v in named_checkpoints.items() if os.path.exists(v)}
ckpt_names = list(checkpoints.keys())

k_values = [1, 2, 5, 10, 20, 30, 50, 75, 100, 150, 200, 256, 300, 400, 512]
N_RANDOM_TRIALS = 10  # how many random sets to average per k

print("=" * 60)
print(f"Neuron subset ablation: {len(k_values)} k-values × {len(checkpoints)} checkpoints")
print(f"Random baseline trials per k: {N_RANDOM_TRIALS}")
print("=" * 60)

top_k_accs = {c: {} for c in ckpt_names}
rand_k_accs = {c: {} for c in ckpt_names}
baselines = {}
all_single_drops = {}

for ckpt_name, ckpt_path in checkpoints.items():
    model = Transformer().to(device)
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    baseline = eval_accuracy(model)
    baselines[ckpt_name] = baseline
    print(f"\n── {ckpt_name} | baseline={baseline:.4f} ──")

    print("  Ranking neurons individually...")
    drops = get_single_drop_ranking(model, baseline)
    all_single_drops[ckpt_name] = drops
    ranked_neurons = np.argsort(drops)[::-1]  # highest drop first
    print(f"  Top neuron drop: {drops[ranked_neurons[0]]:.5f}  |  Top 10 mean drop: {drops[ranked_neurons[:10]].mean():.5f}")

    print("  Ablating top-k neurons...")
    for k in k_values:
        top_k = ranked_neurons[:k].tolist()
        handle = ablate_neurons(model, top_k)
        acc = eval_accuracy(model)
        handle.remove()
        top_k_accs[ckpt_name][k] = acc
        print(f"    k={k:4d} | top-k acc={acc:.4f}  drop={baseline - acc:.4f}")

    print("  Ablating random neuron sets...")
    rng = np.random.default_rng(42)
    for k in k_values:
        trial_accs = []
        for _ in range(N_RANDOM_TRIALS):
            random_set = rng.choice(FFN_DIM, size=k, replace=False).tolist()
            handle = ablate_neurons(model, random_set)
            acc = eval_accuracy(model)
            handle.remove()
            trial_accs.append(acc)
        rand_k_accs[ckpt_name][k] = np.mean(trial_accs)
    print(f"  Random ablation done.")


fig, axes = plt.subplots(2, 2, figsize=(13, 9), sharey=False)
axes = axes.flatten()

for i, ckpt_name in enumerate(ckpt_names):
    ax = axes[i]
    baseline = baselines[ckpt_name]

    top_k_drops = [baseline - top_k_accs[ckpt_name][k] for k in k_values]
    rand_k_drops = [baseline - rand_k_accs[ckpt_name][k] for k in k_values]

    ax.plot(k_values, top_k_drops, 'o-', color='crimson', label='Top-k (ranked)', linewidth=2)
    ax.plot(k_values, rand_k_drops, 's--', color='steelblue', label='Random-k (mean)', linewidth=2)
    ax.fill_between(k_values, top_k_drops, rand_k_drops, alpha=0.15, color='crimson')
    ax.set_title(f"{ckpt_name}\nbaseline={baseline:.3f}", fontsize=9)
    ax.set_xlabel("k (neurons ablated)")
    ax.set_ylabel("Accuracy Drop")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.set_xscale('log')

plt.suptitle("Top-k vs Random-k Neuron Ablation\n(gap = how structured the damage is)", fontsize=12)
plt.tight_layout()
plt.savefig("subset_topk_vs_random.png", dpi=150, bbox_inches='tight')
plt.clf()
print("\nSaved: subset_topk_vs_random.png")

fig, axes = plt.subplots(1, 2, figsize=(13, 5))

for ckpt_name in ckpt_names:
    baseline = baselines[ckpt_name]
    top_accs = [top_k_accs[ckpt_name][k] for k in k_values]
    rand_accs = [rand_k_accs[ckpt_name][k] for k in k_values]
    label = ckpt_name.split(' ')[0]  # just "epoch_XXXX"
    axes[0].plot(k_values, top_accs, 'o-', label=label, linewidth=1.5)
    axes[1].plot(k_values, rand_accs, 's--', label=label, linewidth=1.5)

for ax, title in zip(axes, ["Top-k Neuron Ablation", "Random-k Neuron Ablation (mean)"]):
    ax.set_xlabel("k (neurons ablated)")
    ax.set_ylabel("Validation Accuracy")
    ax.set_title(title, fontsize=11)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.set_xscale('log')
    ax.set_ylim(-0.05, 1.05)
    ax.axhline(1/97, color='gray', linestyle=':', alpha=0.5, label='chance')

plt.suptitle("Validation Accuracy After Neuron Subset Ablation\nAcross Training Stages", fontsize=12)
plt.tight_layout()
plt.savefig("subset_accuracy_curves.png", dpi=150, bbox_inches='tight')
plt.clf()
print("Saved: subset_accuracy_curves.png")

grokked_name = [c for c in ckpt_names if "grokked" in c][0]
baseline_g = baselines[grokked_name]

fig, axes = plt.subplots(1, 2, figsize=(13, 5))

fine_k = list(range(1, 51)) + list(range(50, 512, 10))
fine_k = sorted(set(fine_k))
model = Transformer().to(device)
model.load_state_dict(torch.load(checkpoints[grokked_name], map_location=device))
ranked = np.argsort(all_single_drops[grokked_name])[::-1]

fine_top_accs = []
fine_rand_accs = []
rng = np.random.default_rng(0)
print(f"\nFine-grained sweep on grokked checkpoint (k=1..512)...")
for k in fine_k:
    handle = ablate_neurons(model, ranked[:k].tolist())
    acc = eval_accuracy(model)
    handle.remove()
    fine_top_accs.append(acc)

    trials = []
    for _ in range(5):
        rs = rng.choice(FFN_DIM, size=k, replace=False).tolist()
        handle = ablate_neurons(model, rs)
        acc_r = eval_accuracy(model)
        handle.remove()
        trials.append(acc_r)
    fine_rand_accs.append(np.mean(trials))
    if k % 50 == 0 or k <= 10:
        print(f"  k={k:4d} | top-k={fine_top_accs[-1]:.4f}  rand={fine_rand_accs[-1]:.4f}")

axes[0].plot(fine_k, fine_top_accs, '-', color='crimson', label='Top-k', linewidth=2)
axes[0].plot(fine_k, fine_rand_accs, '--', color='steelblue', label='Random-k', linewidth=2)
axes[0].axhline(baseline_g, color='black', linestyle=':', label=f'Baseline ({baseline_g:.3f})')
axes[0].axhline(1/97, color='gray', linestyle=':', alpha=0.5, label='Chance (~0.010)')
axes[0].set_xlabel("k (neurons ablated)")
axes[0].set_ylabel("Validation Accuracy")
axes[0].set_title(f"Grokked Model — Fine-grained Ablation\n(how many neurons to break it?)", fontsize=10)
axes[0].legend(fontsize=8)
axes[0].grid(True, alpha=0.3)

# Right: single neuron drop distribution (histogram)
drops_g = all_single_drops[grokked_name]
axes[1].hist(drops_g, bins=50, color='steelblue', edgecolor='white', linewidth=0.5)
axes[1].axvline(drops_g.mean(), color='crimson', linestyle='--', label=f'Mean={drops_g.mean():.5f}')
axes[1].axvline(drops_g.max(), color='darkorange', linestyle='--', label=f'Max={drops_g.max():.5f}')
axes[1].set_xlabel("Single-Neuron Accuracy Drop")
axes[1].set_ylabel("Count")
axes[1].set_title("Distribution of Single-Neuron Drops\n(Grokked checkpoint)", fontsize=10)
axes[1].legend(fontsize=8)
axes[1].grid(True, alpha=0.3)

plt.suptitle(f"Grokked Model Neuron Analysis (baseline={baseline_g:.3f})", fontsize=12)
plt.tight_layout()
plt.savefig("grokked_deep_dive.png", dpi=150, bbox_inches='tight')
plt.clf()
print("Saved: grokked_deep_dive.png")

print("\n" + "=" * 60)
print("SUMMARY — Top-k ablation on grokked checkpoint")
print("=" * 60)
for k in k_values:
    acc = top_k_accs[grokked_name][k]
    rand = rand_k_accs[grokked_name][k]
    drop = baseline_g - acc
    gap = acc - rand  
    print(f"  k={k:4d} | top-k acc={acc:.4f}  rand acc={rand:.4f}  drop={drop:.4f}  gap(top-rand)={-gap:.4f}")