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

#print(train_x.shape)
#print(train_y.shape)

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

i = 0
train_losses = []
val_losses = []
train_accs = []
val_accs = []
model_prev_state = None
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

    if epoch == 5000:
        torch.save(model.state_dict(), "model_pre_grokking.pth")
    
    if val_acc > 0.95:
        torch.save(model.state_dict(), f"model_grokking_epoch_{epoch}.pth")
        torch.save(model_prev_state, f"model_pre_grokking_epoch_{epoch}.pth")
        print("Grokking achieved at epoch", epoch)
        print("Training Accuracy:", train_acc)
        print("Validation Accuracy:", val_acc)
        break

    if epoch % 100 == 0: 
        print("Epoch:", epoch)
        print("Training Accuracy:", train_acc)
        print("Validation Accuracy:", val_acc)
    model_prev_state = {k: v.clone() for k, v in model.state_dict().items()}

torch.save(model.state_dict(), "model.pth")
    
plt.plot(train_losses, label='Train Loss')
plt.plot(val_losses, label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Training and Validation Loss')
plt.legend()
plt.savefig("loss.png")
plt.clf()


plt.plot(train_accs, label='Train Accuracy')
plt.plot(val_accs, label='Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.title('Training and Validation Accuracy')
plt.legend()
plt.savefig("accuracy.png")
plt.clf()