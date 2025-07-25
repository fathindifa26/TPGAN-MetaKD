import os

IMG_DIR      = "dataset/128x128"
OUTPUT_FILE  = "train_list.txt"
VALID_EXTS   = {'.png', '.jpg', '.jpeg'}
FRONTAL_CODE = '_051_'

count = 0
with open(OUTPUT_FILE, 'w') as f:
    for root, _, files in os.walk(IMG_DIR):
        for fname in sorted(files):
            ext = os.path.splitext(fname)[1].lower()
            # skip non‐image and skip frontal (_051_)
            if ext in VALID_EXTS and FRONTAL_CODE not in fname:
                rel_path = os.path.join(root, fname).replace('\\','/')
                f.write(rel_path + '\n')
                count += 1

print(f"berhasil membuat {OUTPUT_FILE} dengan {count} entri non-frontal")