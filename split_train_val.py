import random

input_file = 'train_list.txt'
train_file = 'train_list.txt'
val_file = 'validation_list.txt'

with open(input_file, 'r') as f:
    lines = [line for line in f if line.strip()]

random.shuffle(lines)
n_total = len(lines)
n_train = int(0.8 * n_total)
train_lines = lines[:n_train]
val_lines = lines[n_train:]

with open(train_file, 'w') as f:
    for line in train_lines:
        f.write(line)
        if not line.endswith('\n'):
            f.write('\n')

with open(val_file, 'w') as f:
    for line in val_lines:
        f.write(line)
        if not line.endswith('\n'):
            f.write('\n')

print(f"Total baris train_list.txt (train): {len(train_lines)}")
print(f"Total baris validation_list.txt (val): {len(val_lines)}") 