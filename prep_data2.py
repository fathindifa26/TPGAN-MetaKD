import os
import numpy as np
from PIL import Image
from tqdm import tqdm
import face_alignment

# ====== Konfigurasi Direktori ======
IMG_DIR     = 'dataset/datasets/Multi_Pie/fix'   # input: nama file sudah {ID}_{kode_pose}_{idx}.png
LM_DIR      = 'dataset/landmarks'               # simpan .txt landmark
PATCH_DIR   = 'dataset/patch'                   # simpan crop patch
OUT128_DIR  = 'dataset/128x128'                 # simpan resize 128x128
OUT64_DIR   = 'dataset/64x64'                   # simpan resize 64x64
OUT32_DIR   = 'dataset/32x32'                   # simpan resize 32x32

# Buat folder output
os.makedirs(LM_DIR,      exist_ok=True)
os.makedirs(OUT128_DIR,  exist_ok=True)
os.makedirs(OUT64_DIR,   exist_ok=True)
os.makedirs(OUT32_DIR,   exist_ok=True)
for part in ['left_eye','right_eye','nose','mouth']:
    os.makedirs(os.path.join(PATCH_DIR, part), exist_ok=True)

# ====== Landmark Detector ======
fa = face_alignment.FaceAlignment(
    face_alignment.LandmarksType.TWO_D, flip_input=False, device='cpu'
)

# ====== Ukuran Patch ======
PATCH_SIZE = {
    'left_eye':  (40, 40),
    'right_eye': (40, 40),
    'nose':      (40, 32),
    'mouth':     (48, 32),
}

# ====== Proses Semua Gambar ======
image_list = sorted([
    f for f in os.listdir(IMG_DIR)
    if f.lower().endswith(('.png','.jpg','.jpeg'))
])

for fname in tqdm(image_list, desc='Processing Multi_Pie/fix'):
    img_path = os.path.join(IMG_DIR, fname)
    img = Image.open(img_path).convert('RGB')
    img_np = np.array(img)

    # — Landmark Detection —
    try:
        landmarks = fa.get_landmarks(img_np)
    except Exception as e:
        print(f"[!] Error on {fname}: {e}")
        continue
    if landmarks is None:
        print(f"[!] No landmark: {fname}")
        continue

    landmark = landmarks[0]
    # simpan .txt landmark
    lm_path = os.path.join(LM_DIR, fname.rsplit('.',1)[0] + '.txt')
    np.savetxt(lm_path, landmark, fmt='%.6f')

    # — Resize ke 128×128, 64×64 & 32×32 —
    img.resize((128,128), Image.LANCZOS).save(os.path.join(OUT128_DIR, fname))
    img.resize((64,64),   Image.LANCZOS).save(os.path.join(OUT64_DIR, fname))
    img.resize((32,32),   Image.LANCZOS).save(os.path.join(OUT32_DIR, fname))

    # — Crop Patch berdasarkan rata-rata landmark —
    le = landmark[36:42].mean(axis=0)  # left eye
    re = landmark[42:48].mean(axis=0)  # right eye
    no = landmark[27:36].mean(axis=0)  # nose
    mo = landmark[48:68].mean(axis=0)  # mouth
    centers = {'left_eye':le, 'right_eye':re, 'nose':no, 'mouth':mo}

    for part, (cx,cy) in centers.items():
        w, h = PATCH_SIZE[part]
        x1, y1 = int(cx - w//2), int(cy - h//2)
        x2, y2 = x1 + w,     y1 + h
        cropped = img.crop((x1,y1,x2,y2))
        cropped.save(os.path.join(PATCH_DIR, part, fname))

print("✅ Selesai memproses dataset Multi_Pie/fix dengan ukuran 128×128, 64×64, 32×32 dan crop patch.")
