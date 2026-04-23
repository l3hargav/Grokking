import torch
import pandas as pd
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
import torch.nn as nn

torch.manual_seed(42)

device = torch.device("mps")

data = pd.read_csv("dataset.csv")

print(data.head())
print(len(data))

# Check the data once
p = 97
assert((data['a'] + data['b']) % p == data['output']).all()


# Split data
# According to papers, the best way to split is 30/70 for training/validation, grokking is a phenomenon that mostly happens with a small training sample.

train_df, val_df = train_test_split(data, test_size=0.8, random_state=42)

EQUAL_TOKEN = 97

def convert_tensor(df):
    x = torch.tensor(df[['a', 'b']].values, dtype=torch.long)
    eq = torch.full((len(x), 1), EQUAL_TOKEN)
    x = torch.cat([x, eq], dim=1)
    y = torch.tensor(df["output"].values, dtype=torch.long)
    return x, y

train_x, train_y = convert_tensor(train_df)
val_x, val_y = convert_tensor(val_df)


train_dataset = TensorDataset(train_x, train_y)
val_dataset = TensorDataset(val_x, val_y)

train_loader = DataLoader(train_dataset, batch_size=len(train_dataset), shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=512, shuffle=False)


class Transformer(nn.Module):
    def __init__(self):
        super().__init__()
        vocab_size = 98   # 0-96 for numbers, 97 for equal sign
        embedding_dim = 64
        self.embedding = nn.Embedding(vocab_size, embedding_dim)

        encoder = nn.TransformerEncoderLayer(d_model=embedding_dim, nhead=4, dim_feedforward=512, batch_first=True, activation='gelu') # Papers use GeLU activation

        self.transformer = nn.TransformerEncoder(encoder, num_layers=1)
        self.norm = nn.LayerNorm(embedding_dim)

        self.fc = nn.Linear(embedding_dim, 97)

    def forward(self, x):
        x = self.embedding(x)
        x = self.transformer(x)
        x = self.norm(x)
        x = x[:, -1, :]
        x = self.fc(x)
        return x
    

model = Transformer().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.5)

def accuracy(logits, y):
    preds = logits.argmax(dim=1)
    return (preds == y).float().mean().item()

def compute_weight_norm(model):
    return sum(p.norm().item()**2 for p in model.parameters()) ** 0.5

i = 0
train_losses = []
val_losses = []
train_accs = []
val_accs = []
grokking_epoch = None
weight_norms = []
for epoch in range(30000):
    model.train()
    total_loss = 0
    total_acc = 0
    c = 0
    for x, y in train_loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        loss = criterion(logits, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        acc = accuracy(logits, y)
        total_acc += acc
        c += 1
    
    train_loss = total_loss / c
    train_acc = total_acc / c
    train_losses.append(train_loss)
    train_accs.append(train_acc)

    model.eval()

    total_val_loss = 0
    total_val_acc = 0
    val_c = 0

    with torch.no_grad():
        for x, y in val_loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            acc = accuracy(logits, y)
            loss = criterion(logits, y)
            total_val_loss += loss.item()
            total_val_acc += acc
            val_c += 1
    val_loss = total_val_loss / val_c
    val_acc = total_val_acc / val_c
    val_losses.append(val_loss)
    val_accs.append(val_acc)
    wn = compute_weight_norm(model)
    weight_norms.append(wn)

    if val_acc > 0.95:
        grokking_epoch = epoch
        break

    if epoch % 100 == 0: 
        print("Epoch:", epoch)
        print("Training Accuracy:", train_acc)
        print("Validation Accuracy:", val_acc)

n = len(train_losses)
metrics_df = pd.DataFrame({
    'epoch':        range(n),
    'train_loss':   train_losses,
    'val_loss':     val_losses,
    'train_acc':    train_accs,
    'val_acc':      val_accs,
    'weight_norm':  weight_norms,
})
metrics_df.to_csv("training_metrics.csv", index=False)
print("Saved training_metrics.csv")
 
 
fig, ax1 = plt.subplots(figsize=(10, 5))
 
ax1.plot(weight_norms, color='steelblue', linewidth=1.5, label='Weight Norm')
ax1.set_xlabel('Epochs')
ax1.set_ylabel('L2 Weight Norm', color='steelblue')
ax1.tick_params(axis='y', labelcolor='steelblue')
 
ax2 = ax1.twinx()
ax2.plot(val_accs, color='orange', linewidth=1.5, linestyle='--', label='Val Accuracy')
ax2.set_ylabel('Validation Accuracy', color='orange')
ax2.tick_params(axis='y', labelcolor='orange')
ax2.set_ylim(-0.05, 1.05)
 
if grokking_epoch:
    ax1.axvline(grokking_epoch, color='red', linestyle='--', alpha=0.6, label=f'Grokking (epoch {grokking_epoch})')
 
lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left')
 
plt.title('Weight Norm Trajectory vs Validation Accuracy\n(Circuit Efficiency Hypothesis)')
plt.tight_layout()
plt.savefig("weight_norm_trajectory.png", dpi=150)
plt.clf()
print("Saved weight_norm_trajectory.png")