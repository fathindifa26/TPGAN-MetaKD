import os

IMG_DIR = "dataset/128x128"
VALID_EXTS = {'.png', '.jpg', '.jpeg'}

id_set = set()
for fname in os.listdir(IMG_DIR):
    ext = os.path.splitext(fname)[1].lower()
    if ext in VALID_EXTS:
        identity = fname.split('_')[0]    # Misal: '01234_abc.jpg' → '01234'
        id_set.add(identity)

print(f"Jumlah identitas unik (berdasarkan nama file): {len(id_set)}")
print("Contoh ID:", list(id_set)[:10])
