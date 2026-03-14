import torch
import pandas as pd
from sklearn.model_selection import train_test_split

data = pd.read_csv("dataset.csv")

print(data.head())
print(len(data))

# Check the data once
p = 97
assert((data['a'] + data['b']) % p == data['output']).all()


# Split data
# According to papers, the best way to split is 30/70 for training/validation, grokking is a phenomenon that mostly happens with a small training sample.

train_df, val_df = train_test_split(data, test_size=0.7, random_state=42)
