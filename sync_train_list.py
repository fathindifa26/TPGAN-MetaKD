import os

PATCH_TYPES = ["left_eye", "right_eye", "nose", "mouth"]
PATCH_DIR = "dataset/patch"
TRAIN_LIST = "train_list.txt"
OUTPUT_LIST = "train_list_synced.txt"
IMG_DIR = "dataset/128x128"

# Ambil nama file dari setiap baris train_list.txt
with open(TRAIN_LIST, 'r') as f:
    lines = [line.strip() for line in f if line.strip()]

synced_lines = []
for line in lines:
    filename = os.path.basename(line)
    # Buat nama frontal: ganti bagian kedua dari belakang dengan '051'
    parts = filename.split('_')
    if len(parts) < 3:
        continue
    parts[-2] = '051'
    frontal_filename = '_'.join(parts)
    frontal_path = os.path.join(IMG_DIR, frontal_filename)
    # Cek file frontal ada
    if not os.path.isfile(frontal_path):
        continue
    # Cek patch frontal ada semua
    all_patch_exist = True
    for patch in PATCH_TYPES:
        patch_frontal_path = os.path.join(PATCH_DIR, patch, frontal_filename)
        if not os.path.isfile(patch_frontal_path):
            all_patch_exist = False
            break
    if all_patch_exist:
        synced_lines.append(line)

# Simpan hasil yang sudah sinkron
with open(OUTPUT_LIST, 'w') as f:
    for line in synced_lines:
        f.write(line + '\n')

print(f"Selesai! {len(synced_lines)} dari {len(lines)} data sudah sinkron (dengan frontal & patch frontal) dan disimpan di {OUTPUT_LIST}") 