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
FFN_DIM = 512  # hidden layer size in the MLP

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

def ablate_neuron(model, neuron_idx):
    layer = model.transformer.layers[0]

    def hook_fn(module, input, output):
        out = output.clone()
        out[:, :, neuron_idx] = 0.0
        return out

    handle = layer.linear1.register_forward_hook(hook_fn)
    return handle

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
print(f"Running neuron lesion study: {FFN_DIM} neurons × {len(checkpoints)} checkpoints")
print(f"Total forward passes: {FFN_DIM * len(checkpoints)}")
print("=" * 60)

all_drops = {}
baselines = {}

for ckpt_name, ckpt_path in checkpoints.items():
    model = Transformer().to(device)
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    baseline = eval_accuracy(model)
    baselines[ckpt_name] = baseline
    print(f"\n{ckpt_name} | baseline={baseline:.3f}")

    drops = np.zeros(FFN_DIM)
    for neuron_idx in range(FFN_DIM):
        handle = ablate_neuron(model, neuron_idx)
        acc = eval_accuracy(model)
        handle.remove()
        drops[neuron_idx] = baseline - acc

        if neuron_idx % 100 == 0:
            print(f"  neuron {neuron_idx:03d}/{FFN_DIM} | max drop so far: {drops[:neuron_idx+1].max():.4f}")

    all_drops[ckpt_name] = drops
    top5_idx = np.argsort(drops)[::-1][:5]
    print(f"  Top 5 critical neurons: {top5_idx.tolist()}")
    print(f"  Top 5 drops: {drops[top5_idx].tolist()}")

np.save("neuron_drops.npy", {k: v for k, v in all_drops.items()})
print("\nSaved raw drops to neuron_drops.npy")

ckpt_names = list(checkpoints.keys())
n_ckpts = len(ckpt_names)

# PLOTS/VISUALIZATIONS
fig, axes = plt.subplots(n_ckpts, 1, figsize=(14, 3 * n_ckpts), sharex=False)
if n_ckpts == 1:
    axes = [axes]

for i, ckpt_name in enumerate(ckpt_names):
    drops = all_drops[ckpt_name]
    ax = axes[i]
    colors = ['red' if d > 0.01 else 'steelblue' for d in drops]
    ax.bar(range(FFN_DIM), drops, color=colors, width=1.0, alpha=0.8)
    ax.set_title(f"{ckpt_name} | baseline={baselines[ckpt_name]:.3f} | "
                 f"neurons with drop>0.01: {(drops > 0.01).sum()}", fontsize=10)
    ax.set_ylabel("Accuracy Drop")
    ax.set_xlim(0, FFN_DIM)
    n_critical = (drops > 0.01).sum()
    ax.axhline(0.01, color='red', linestyle='--', alpha=0.5, label='threshold=0.01')
    if i == n_ckpts - 1:
        ax.set_xlabel("Neuron Index")

plt.suptitle("Per-Neuron Accuracy Drop Across Training Stages\n(red bars = drop > 0.01)", fontsize=12)
plt.tight_layout()
plt.savefig("neuron_lesion_bars.png", dpi=150, bbox_inches='tight')
plt.clf()
print("Saved: neuron_lesion_bars.png")

drop_matrix = np.stack([all_drops[c] for c in ckpt_names], axis=0)  # (n_ckpts, 512)
max_drop_per_neuron = drop_matrix.max(axis=0)
top_n = 50
top_neuron_idx = np.argsort(max_drop_per_neuron)[::-1][:top_n]
top_neuron_idx_sorted = np.sort(top_neuron_idx)  
fig, ax = plt.subplots(figsize=(18, 4))
im = ax.imshow(
    drop_matrix[:, top_neuron_idx_sorted],
    cmap='RdYlGn_r', aspect='auto',
    vmin=0, vmax=drop_matrix.max()
)
ax.set_yticks(range(n_ckpts))
ax.set_yticklabels(ckpt_names, fontsize=9)
ax.set_xticks(range(top_n))
ax.set_xticklabels([str(i) for i in top_neuron_idx_sorted], fontsize=7, rotation=90)
ax.set_xlabel("Neuron Index (top 50 by max drop across checkpoints)")
ax.set_title("Neuron Criticality Heatmap — Top 50 Neurons\n(red = high accuracy drop when ablated)", fontsize=11)
plt.colorbar(im, ax=ax, label="Accuracy Drop")
plt.tight_layout()
plt.savefig("neuron_lesion_heatmap.png", dpi=150, bbox_inches='tight')
plt.clf()
print("Saved: neuron_lesion_heatmap.png")

if grokking_files:
    grokked_name = [k for k in ckpt_names if "grokked" in k][0]
    drops_grokked = all_drops[grokked_name]
    sorted_drops = np.sort(drops_grokked)[::-1]
    cumulative = np.cumsum(sorted_drops)

    fig, axes = plt.subplots(1, 2, figsize=(13, 4))

    # Left: sorted drop per neuron
    axes[0].plot(sorted_drops, color='crimson')
    axes[0].axhline(0.01, color='gray', linestyle='--', label='drop=0.01 threshold')
    axes[0].fill_between(range(FFN_DIM), sorted_drops, alpha=0.2, color='crimson')
    axes[0].set_xlabel("Neuron rank (most → least critical)")
    axes[0].set_ylabel("Accuracy Drop")
    axes[0].set_title("Neuron Importance Spectrum\n(Grokked checkpoint)", fontsize=11)
    axes[0].legend()
    n_above = (drops_grokked > 0.01).sum()
    axes[0].set_xlim(0, FFN_DIM)
    axes[0].annotate(f"{n_above} neurons\nabove threshold",
                     xy=(n_above, 0.01), xytext=(n_above + 30, 0.02),
                     arrowprops=dict(arrowstyle='->', color='black'), fontsize=9)

    total = cumulative[-1]
    axes[1].plot(cumulative / total * 100, color='steelblue')
    axes[1].axhline(80, color='gray', linestyle='--', label='80% of total damage')
    axes[1].axhline(95, color='orange', linestyle='--', label='95% of total damage')
    axes[1].set_xlabel("Number of neurons ablated (ranked)")
    axes[1].set_ylabel("Cumulative % of total accuracy damage")
    axes[1].set_title("Cumulative Damage Curve\n(Grokked checkpoint)", fontsize=11)
    axes[1].legend(fontsize=9)
    axes[1].set_xlim(0, FFN_DIM)

    for threshold, color in [(80, 'gray'), (95, 'orange')]:
        idx = np.searchsorted(cumulative / total * 100, threshold)
        axes[1].annotate(f"n={idx}", xy=(idx, threshold),
                         xytext=(idx + 20, threshold - 5),
                         arrowprops=dict(arrowstyle='->', color=color), fontsize=9)

    plt.tight_layout()
    plt.savefig("neuron_importance_spectrum.png", dpi=150, bbox_inches='tight')
    plt.clf()
    print("Saved: neuron_importance_spectrum.png")

THRESHOLD = 0.01
fig, ax = plt.subplots(figsize=(8, 4))
for ckpt_name in ckpt_names:
    drops = all_drops[ckpt_name]
    critical_mask = drops > THRESHOLD
    ax.plot(range(FFN_DIM), critical_mask.astype(float) * baselines[ckpt_name],
            alpha=0.6, label=ckpt_name, linewidth=0.8)

ax.set_xlabel("Neuron Index")
ax.set_ylabel("Baseline acc (scaled) × critical")
ax.set_title("Which Neurons Are Critical at Each Stage?\n(spike = neuron is critical, height = baseline accuracy)", fontsize=10)
ax.legend(fontsize=8)
plt.tight_layout()
plt.savefig("neuron_critical_overlap.png", dpi=150, bbox_inches='tight')
plt.clf()
print("Saved: neuron_critical_overlap.png")

print("\n" + "=" * 60)
print("SUMMARY")
print("=" * 60)
for ckpt_name in ckpt_names:
    drops = all_drops[ckpt_name]
    n_critical = (drops > 0.01).sum()
    top1 = np.argmax(drops)
    print(f"{ckpt_name:35s} baseline={baselines[ckpt_name]:.3f}  "
          f"critical neurons (drop>0.01): {n_critical:3d}/512  "
          f"top neuron: #{top1} (drop={drops[top1]:.4f})")