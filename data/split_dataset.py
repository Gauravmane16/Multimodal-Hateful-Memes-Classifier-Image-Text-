import json
import random

# Load the flat list from splits.json
with open('data/splits.json', 'r') as f:
    data = json.load(f)

# Shuffle the data for randomness
random.seed(42)
random.shuffle(data)

total = len(data)
train_end = int(0.7 * total)
val_end = int(0.85 * total)

train = data[:train_end]
val = data[train_end:val_end]
test = data[val_end:]

with open('data/train.json', 'w') as f:
    json.dump(train, f, indent=2)
with open('data/val.json', 'w') as f:
    json.dump(val, f, indent=2)
with open('data/test.json', 'w') as f:
    json.dump(test, f, indent=2)

print(f"Train: {len(train)}, Val: {len(val)}, Test: {len(test)}")
