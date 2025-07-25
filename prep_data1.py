import os
import shutil
from collections import defaultdict
from tqdm import tqdm

# Direktori sumber dan tujuan
src_dir = 'dataset/datasets/Multi_Pie/HR_128'
dst_dir = 'dataset/datasets/Multi_Pie/fix'

os.makedirs(dst_dir, exist_ok=True)

# 1) Kumpulkan dulu
groups = defaultdict(list)
for fname in os.listdir(src_dir):
    if not fname.lower().endswith(('.png', '.jpg', '.jpeg')):
        continue
    parts = fname.split('_')
    ID = parts[0]
    kode_pose = parts[3]
    groups[(ID, kode_pose)].append(fname)

print(f"[INFO] Ketemu {len(groups)} grup (ID, pose) berbeda")

# 2) Loop per grup dengan progress bar
for (ID, kode_pose), files in tqdm(groups.items(), desc='Grup', unit='grp'):
    files.sort()
    # Untuk tiap file dalam grup, beri nama baru dan pindah
    for idx, fname in enumerate(tqdm(files, desc=f'{ID}_{kode_pose}', leave=False), start=1):
        ext       = os.path.splitext(fname)[1]
        new_name  = f"{ID}_{kode_pose}_{idx:03d}{ext}"
        src_path  = os.path.join(src_dir, fname)
        dst_path  = os.path.join(dst_dir, new_name)

        try:
            # Lebih cepat jika di-drive sama:
            os.rename(src_path, dst_path)
            # Jika perlu copy saja, uncomment baris di bawah dan comment os.rename:
            # shutil.copy2(src_path, dst_path)
        except Exception as e:
            print(f"[ERROR] Gagal memindah {fname}: {e}")

print("[DONE] Semua file selesai diproses ke", dst_dir)
