from flask import Flask, render_template, request, jsonify
import torch
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset
import torch.nn as nn
import glob
import os

app = Flask(__name__)

EQUAL_TOKEN = 97
EMBED_DIM = 64
N_HEADS = 4
FFN_DIM = 512
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"Using device: {device}")

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

def load_val_data():
    data = pd.read_csv("dataset.csv")
    _, val_df = train_test_split(data, test_size=0.8, random_state=42)
    x = torch.tensor(val_df[['a', 'b']].values, dtype=torch.long)
    eq = torch.full((len(x), 1), EQUAL_TOKEN)
    x = torch.cat([x, eq], dim=1)
    y = torch.tensor(val_df["output"].values, dtype=torch.long)
    return DataLoader(TensorDataset(x, y), batch_size=512, shuffle=False)

print("Loading val data...")
val_loader = load_val_data()

print("Loading checkpoints...")
checkpoint_files = {
    "epoch_1000 (memorization)": "model_early_memorization.pth",
    "epoch_2500 (plateau)":      "model_deep_plateau.pth",
    "epoch_5000 (pre-grokking)": "model_pre_grokking.pth",
}
grokking_files = sorted(glob.glob("model_grokking_epoch_*.pth"))
if grokking_files:
    epoch_num = grokking_files[-1].replace("model_grokking_epoch_", "").replace(".pth", "")
    checkpoint_files[f"epoch_{epoch_num} (grokked)"] = grokking_files[-1]

# Load all models into memory
models = {}
baselines = {}
for name, path in checkpoint_files.items():
    if os.path.exists(path):
        m = Transformer().to(device)
        m.load_state_dict(torch.load(path, map_location=device))
        m.eval()
        models[name] = m
        print(f"  Loaded: {name}")

neuron_scores = {}
if os.path.exists("neuron_drops.npy"):
    raw = np.load("neuron_drops.npy", allow_pickle=True).item()
    for ckpt_name, drops in raw.items():
        neuron_scores[ckpt_name] = drops.tolist()
    print("Loaded neuron importance scores from neuron_drops.npy")
else:
    print("WARNING: neuron_drops.npy not found — importance coloring disabled")
    for name in models:
        neuron_scores[name] = [0.0] * FFN_DIM

def eval_accuracy(model, ablate_neurons=None, ablate_heads=None):
    handles = []

    if ablate_neurons:
        layer = model.transformer.layers[0]
        indices = list(ablate_neurons)
        def neuron_hook(module, input, output):
            out = output.clone()
            out[:, :, indices] = 0.0
            return out
        handles.append(layer.linear1.register_forward_hook(neuron_hook))

    if ablate_heads:
        layer = model.transformer.layers[0]
        head_dim = EMBED_DIM // N_HEADS
        head_list = list(ablate_heads)
        def head_hook(module, input, output):
            attn_out, attn_weights = output
            attn_out = attn_out.clone()
            for h in head_list:
                attn_out[:, :, h * head_dim:(h + 1) * head_dim] = 0.0
            return (attn_out, attn_weights)
        handles.append(layer.self_attn.register_forward_hook(head_hook))

    model.eval()
    correct = total = 0
    with torch.no_grad():
        for x, y in val_loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            correct += (logits.argmax(dim=1) == y).sum().item()
            total += y.size(0)

    for h in handles:
        h.remove()

    return correct / total

print("Computing baselines...")
for name, model in models.items():
    baselines[name] = eval_accuracy(model)
    print(f"  {name}: {baselines[name]:.4f}")

checkpoint_names = list(models.keys())
print(f"\nReady! Loaded {len(models)} checkpoints.")

@app.route("/")
def index():
    return render_template("index.html",
                           checkpoints=checkpoint_names,
                           baselines=baselines,
                           neuron_scores=neuron_scores,
                           ffn_dim=FFN_DIM,
                           n_heads=N_HEADS)

@app.route("/ablate", methods=["POST"])
def ablate():
    data = request.get_json()
    ckpt_name = data.get("checkpoint")
    neuron_indices = data.get("neurons", [])
    head_indices = data.get("heads", [])

    if ckpt_name not in models:
        return jsonify({"error": f"Unknown checkpoint: {ckpt_name}"}), 400

    model = models[ckpt_name]
    baseline = baselines[ckpt_name]

    acc = eval_accuracy(
        model,
        ablate_neurons=neuron_indices if neuron_indices else None,
        ablate_heads=head_indices if head_indices else None
    )

    return jsonify({
        "checkpoint": ckpt_name,
        "baseline": round(baseline, 4),
        "ablated_accuracy": round(acc, 4),
        "drop": round(baseline - acc, 4),
        "n_neurons_ablated": len(neuron_indices),
        "n_heads_ablated": len(head_indices),
    })

@app.route("/top_k", methods=["POST"])
def top_k():
    data = request.get_json()
    ckpt_name = data.get("checkpoint")
    k = int(data.get("k", 10))

    if ckpt_name not in neuron_scores:
        return jsonify({"error": "No scores for checkpoint"}), 400

    scores = np.array(neuron_scores[ckpt_name])
    ranked = np.argsort(scores)[::-1][:k].tolist()
    return jsonify({"indices": ranked, "k": k})

if __name__ == "__main__":
    app.run(debug=False, port=5050)
