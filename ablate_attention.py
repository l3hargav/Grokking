import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset
import torch.nn as nn
import glob
import os

torch.manual_seed(42)
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
p = 97
EQUAL_TOKEN = 97
EMBED_DIM = 64
N_HEADS = 4
HEAD_DIM = EMBED_DIM // N_HEADS  # 16

data = pd.read_csv("dataset.csv")
_, val_df = train_test_split(data, test_size=0.8, random_state=42)

def convert_tensor(df):
    x = torch.tensor(df[['a', 'b']].values, dtype=torch.long)
    eq = torch.full((len(x), 1), EQUAL_TOKEN)
    x = torch.cat([x, eq], dim=1)
    y = torch.tensor(df["output"].values, dtype=torch.long)
    return x, y

val_x, val_y = convert_tensor(val_df)
val_dataset = TensorDataset(val_x, val_y)
val_loader = DataLoader(val_dataset, batch_size=512, shuffle=False)

class Transformer(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(98, EMBED_DIM)
        encoder = nn.TransformerEncoderLayer(
            d_model=EMBED_DIM, nhead=N_HEADS, dim_feedforward=512,
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
            logits = model(x)
            correct += (logits.argmax(dim=1) == y).sum().item()
            total += y.size(0)
    return correct / total


_hook_handle = None

def ablate_head(model, head_idx):
    layer = model.transformer.layers[0]

    def hook_fn(module, input, output):
        attn_out, attn_weights = output
        attn_out = attn_out.clone()
        start = head_idx * HEAD_DIM
        end = start + HEAD_DIM
        attn_out[:, :, start:end] = 0.0
        return (attn_out, attn_weights)

    handle = layer.self_attn.register_forward_hook(hook_fn)
    return handle

def ablate_multiple_heads(model, head_indices):
    handles = []
    for h in head_indices:
        handles.append(ablate_head(model, h))
    return handles

named_checkpoints = {
    "epoch_1000\n(memorization)":   "model_early_memorization.pth",
    "epoch_2500\n(plateau)":        "model_deep_plateau.pth",
    "epoch_5000\n(pre-grokking)":   "model_pre_grokking.pth",
}

grokking_files = sorted(glob.glob("model_grokking_epoch_*.pth"))
if grokking_files:
    grokking_path = grokking_files[-1]
    epoch_num = grokking_path.replace("model_grokking_epoch_", "").replace(".pth", "")
    named_checkpoints[f"epoch_{epoch_num}\n(grokked)"] = grokking_path
    print(f"Found grokking checkpoint: {grokking_path}")
else:
    print("WARNING: No model_grokking_epoch_*.pth found. Running without grokked checkpoint.")

checkpoints = {name: path for name, path in named_checkpoints.items() if os.path.exists(path)}
missing = [path for path in named_checkpoints.values() if not os.path.exists(path)]
if missing:
    print(f"WARNING: Missing checkpoints: {missing}")

print(f"\nRunning lesion study on {len(checkpoints)} checkpoints × {N_HEADS} heads")
print("=" * 60)

results = {}
baseline_accs = {}

for ckpt_name, ckpt_path in checkpoints.items():
    model = Transformer().to(device)
    model.load_state_dict(torch.load(ckpt_path, map_location=device))

    baseline = eval_accuracy(model)
    baseline_accs[ckpt_name] = baseline
    print(f"\n{ckpt_name.replace(chr(10), ' ')} | baseline val acc: {baseline:.3f}")

    head_accs = {}
    for head_idx in range(N_HEADS):
        handle = ablate_head(model, head_idx)
        acc = eval_accuracy(model)
        handle.remove()
        head_accs[head_idx] = acc
        drop = baseline - acc
        print(f"  ablate head {head_idx}: acc={acc:.3f}  drop={drop:+.3f}")

    results[ckpt_name] = head_accs

ckpt_names = list(checkpoints.keys())
n_ckpts = len(ckpt_names)

acc_matrix = np.array([[results[c][h] for h in range(N_HEADS)] for c in ckpt_names])
drop_matrix = np.array([
    [baseline_accs[c] - results[c][h] for h in range(N_HEADS)]
    for c in ckpt_names
])
baseline_arr = np.array([baseline_accs[c] for c in ckpt_names])

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

im = axes[0].imshow(drop_matrix, cmap='RdYlGn_r', aspect='auto',
                    vmin=0, vmax=max(drop_matrix.max(), 0.05))
axes[0].set_xticks(range(N_HEADS))
axes[0].set_xticklabels([f"Head {i}" for i in range(N_HEADS)])
axes[0].set_yticks(range(n_ckpts))
axes[0].set_yticklabels([c.replace('\n', ' ') for c in ckpt_names], fontsize=9)
axes[0].set_title("Validation Accuracy DROP per Head Ablation\n(red = more critical)", fontsize=11)
axes[0].set_xlabel("Ablated Head")
plt.colorbar(im, ax=axes[0], label="Accuracy Drop")

for i in range(n_ckpts):
    for j in range(N_HEADS):
        axes[0].text(j, i, f"{drop_matrix[i, j]:.3f}", ha='center', va='center',
                     fontsize=9, color='black')

for h in range(N_HEADS):
    axes[1].plot(range(n_ckpts), acc_matrix[:, h], marker='o', label=f'Head {h}')
axes[1].plot(range(n_ckpts), baseline_arr, 'k--', marker='s', label='Baseline', linewidth=2)
axes[1].set_xticks(range(n_ckpts))
axes[1].set_xticklabels([c.replace('\n', '\n') for c in ckpt_names], fontsize=8)
axes[1].set_ylabel("Validation Accuracy")
axes[1].set_title("Val Accuracy After Single Head Ablation\nAcross Training Stages", fontsize=11)
axes[1].legend(fontsize=9)
axes[1].set_ylim(-0.05, 1.05)
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("lesion_single_head.png", dpi=150, bbox_inches='tight')
plt.clf()
print("\nSaved: lesion_single_head.png")

if grokking_files:
    print("\nRunning pairwise head ablation on grokked checkpoint...")
    from itertools import combinations

    model = Transformer().to(device)
    model.load_state_dict(torch.load(grokking_files[-1], map_location=device))
    baseline_grokked = eval_accuracy(model)

    pair_results = {}
    for h1, h2 in combinations(range(N_HEADS), 2):
        handles = ablate_multiple_heads(model, [h1, h2])
        acc = eval_accuracy(model)
        for h in handles:
            h.remove()
        pair_results[(h1, h2)] = acc
        print(f"  ablate heads ({h1},{h2}): acc={acc:.3f}  drop={baseline_grokked - acc:+.3f}")

    single_results = {}
    for h in range(N_HEADS):
        handle = ablate_head(model, h)
        acc = eval_accuracy(model)
        handle.remove()
        single_results[h] = acc

    pair_matrix = np.full((N_HEADS, N_HEADS), np.nan)
    for h in range(N_HEADS):
        pair_matrix[h, h] = single_results[h]
    for (h1, h2), acc in pair_results.items():
        pair_matrix[h1, h2] = acc
        pair_matrix[h2, h1] = acc

    fig, ax = plt.subplots(figsize=(7, 6))
    masked = np.ma.masked_invalid(pair_matrix)
    im = ax.imshow(masked, cmap='RdYlGn', vmin=0, vmax=1, aspect='auto')
    ax.set_xticks(range(N_HEADS))
    ax.set_xticklabels([f"Head {i}" for i in range(N_HEADS)])
    ax.set_yticks(range(N_HEADS))
    ax.set_yticklabels([f"Head {i}" for i in range(N_HEADS)])
    ax.set_title(f"Pairwise Head Ablation — Grokked Checkpoint\n(diagonal = single ablation, baseline={baseline_grokked:.3f})", fontsize=10)
    plt.colorbar(im, ax=ax, label="Val Accuracy After Ablation")
    for i in range(N_HEADS):
        for j in range(N_HEADS):
            if not np.isnan(pair_matrix[i, j]):
                ax.text(j, i, f"{pair_matrix[i, j]:.3f}", ha='center', va='center', fontsize=10)
    plt.tight_layout()
    plt.savefig("lesion_pairwise_grokked.png", dpi=150, bbox_inches='tight')
    plt.clf()
    print("Saved: lesion_pairwise_grokked.png")

print("\n" + "=" * 60)
print("SUMMARY")
print("=" * 60)
for ckpt_name in ckpt_names:
    b = baseline_accs[ckpt_name]
    drops = {h: baseline_accs[ckpt_name] - results[ckpt_name][h] for h in range(N_HEADS)}
    most_critical = max(drops, key=drops.get)
    print(f"{ckpt_name.replace(chr(10), ' '):35s} baseline={b:.3f}  most critical: Head {most_critical} (drop={drops[most_critical]:.3f})")