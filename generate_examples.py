'''
Code to generate examples of the modular arithmetic examples used in training the transformer.
(a + b) mod p 
Where:
    p = 97 =>  From papers (p = prime)
    a, b = positive integers
'''

import torch
import random
import pandas as pd

p = 97 

data = []
for a in range(p):
    for b in range(p):
        data.append([a, b, (a+b) % p])

random.shuffle(data)

a = []
b = []
out = []
for sample in data:
    a_ = sample[0]
    b_ = sample[1]
    out_ = sample[2]
    a.append(a_)
    b.append(b_)
    out.append(out_)

df = pd.DataFrame({'a': a, 'b': b, 'output': out})

print(df.head())

df.to_csv("dataset.csv", index=False)

