import argparse
import os
import glob
import random
import importlib
import torch
import numpy as np
from torch.autograd import Variable
from PIL import Image
import torchvision.transforms as transforms
from tqdm import tqdm
import onnxruntime as ort
import csv
from models.network import Generator, GeneratorStudent
from utils.utils import *
import time

# --- Set seed agar hasil eksperimen konsisten ---
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# --- Argumen ---
def parse_args():
    parser = argparse.ArgumentParser(description="Unified TP-GAN Tester (Teacher/Student)")
    parser.add_argument("--input_list")
    parser.add_argument('--resume_model', help='resume_model dirname (angka saja, misal 0 atau 3)')
    parser.add_argument('--epoch', type=int, default=None, help='(Student only) epoch ke berapa, untuk path save/try_{}/epoch_{}/')
    parser.add_argument("--subdir", help='output_dir = save/$resume_model/test/$subdir')
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--max_images', type=int, default=5000, help='Jumlah maksimal gambar yang diproses')
    parser.add_argument('--mode', choices=['teacher', 'student'], default='teacher', help='Pilih mode: teacher (Generator) atau student (GeneratorStudent)')
    return parser.parse_args()

# --- Helper: get frontal path dari profile path ---
def get_frontal_path(profile_path):
    fname = os.path.basename(profile_path)
    parts = fname.split('_')
    parts[1] = '051'  # Ganti kode pose ke 051
    frontal_fname = '_'.join(parts)
    return os.path.join(os.path.dirname(profile_path), frontal_fname)

# --- Fungsi untuk membaca landmark dari file (diadaptasi dari test_KD.py dan test.py) ---
def clean_landmark_string_from_file(path):
    coords = []
    with open(path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            # Pisahkan berdasarkan spasi atau koma
            parts = line.replace(',', ' ').split()
            if len(parts) == 2:
                x, y = parts
                coords.extend([x, y])
    return ' '.join(coords)

# --- Fungsi untuk membuat patch local dari gambar dan landmark (diadaptasi dari test_KD.py/test.py) ---
def process(img, landmarks_5pts):
    batch = {}
    name = ['left_eye','right_eye','nose','mouth']
    patch_size = {
        'left_eye':(40,40),
        'right_eye':(40,40),
        'nose':(40,32),
        'mouth':(48,32),
    }
    landmarks_5pts[3,0] =  (landmarks_5pts[3,0] + landmarks_5pts[4,0]) / 2.0
    landmarks_5pts[3,1] = (landmarks_5pts[3,1] + landmarks_5pts[4,1]) / 2.0
    for i in range(4):
        x = int(np.floor(landmarks_5pts[i,0]))
        y = int(np.floor(landmarks_5pts[i,1]))
        patch = img.crop((
            x - patch_size[name[i]][0]//2 + 1,
            y - patch_size[name[i]][1]//2 + 1,
            x + patch_size[name[i]][0]//2 + 1,
            y + patch_size[name[i]][1]//2 + 1
        ))
        patch = patch.convert('RGB')  # pastikan patch RGB
        batch[name[i]] = patch
    return batch

# --- Custom Dataset ---
from torch.utils.data import Dataset
class CustomTestDataset(Dataset):
    def __init__(self, img_list):
        self.img_list = img_list
    def __len__(self):
        return len(self.img_list)
    def __getitem__(self, idx):
        batch = {}
        img_path = self.img_list[idx]
        img_base = os.path.splitext(os.path.basename(img_path))[0]
        frontal_path = get_frontal_path(img_path)
        landmark_dir = 'dataset/landmarks'
        lm_path = os.path.join(landmark_dir, img_base + '.txt')
        img = Image.open(img_path).convert('RGB')
        lm = clean_landmark_string_from_file(lm_path)
        lm = np.array(lm.split(' '), np.float32).reshape(-1,2)
        for i in range(lm.shape[0]):
            lm[i][0] *= 128/img.width
            lm[i][1] *= 128/img.height
        img = img.resize((128,128), Image.LANCZOS)
        batch_profile = process(img, lm)
        batch_profile['img'] = img
        batch_profile['img64'] = img.resize((64,64), Image.LANCZOS)
        batch_profile['img32'] = batch_profile['img64'].resize((32,32), Image.LANCZOS)
        if os.path.exists(frontal_path):
            img_frontal = Image.open(frontal_path).convert('RGB').resize((128,128), Image.LANCZOS)
        else:
            img_frontal = img
        batch['img'] = batch_profile['img']
        batch['img64'] = batch_profile['img64']
        batch['img32'] = batch_profile['img32']
        batch['left_eye'] = batch_profile['left_eye']
        batch['right_eye'] = batch_profile['right_eye']
        batch['nose'] = batch_profile['nose']
        batch['mouth'] = batch_profile['mouth']
        to_tensor = transforms.ToTensor()
        for k in batch:
            batch[k] = to_tensor(batch[k])
            batch[k] = batch[k] * 2.0 - 1.0
        batch['img_frontal'] = to_tensor(img_frontal) * 2.0 - 1.0
        return batch

# --- ONNX ArcFace ---
onnx_path = 'model.onnx'  # Ganti path jika perlu
# Pastikan ONNX dijalankan di GPU jika tersedia
providers = ['CUDAExecutionProvider', 'CPUExecutionProvider'] if ort.get_device() == 'GPU' else ['CPUExecutionProvider']
try:
    ort_session = ort.InferenceSession(onnx_path, providers=providers)
    if 'CUDAExecutionProvider' in ort_session.get_providers():
        print('[INFO] ONNX ArcFace dijalankan di GPU (CUDA)')
    else:
        print('[WARNING] ONNX ArcFace dijalankan di CPU, proses akan lebih lambat')
except Exception as e:
    print(f'[ERROR] Gagal inisialisasi ONNX ArcFace: {e}')
    exit(1)

def extract_onnx_embedding(img_pil):
    img = img_pil.resize((112,112))
    img_np = np.asarray(img).astype(np.float32)
    img_np = (img_np - 127.5) / 128.0
    img_np = np.transpose(img_np, (2, 0, 1))
    img_np = np.expand_dims(img_np, axis=0)
    input_name = ort_session.get_inputs()[0].name
    emb = ort_session.run(None, {input_name: img_np})[0]
    return emb.squeeze(0)

if __name__ == "__main__":
    args = parse_args()
    # --- Pilih config dan path resume sesuai mode ---
    if args.mode == 'teacher':
        # Teacher: save/try_{}/, config/config.py
        resume_dir = os.path.join('save', f"try_{args.resume_model}")
        config_path = 'config/config.py'
        config_module = 'config.config'
    else:
        # Student: save/try_{}/epoch_{}/, config/config_distillation.py
        if args.epoch is None:
            raise ValueError('Untuk mode student, --epoch harus diisi!')
        resume_dir = os.path.join('save', f"try_{args.resume_model}", f"epoch_{args.epoch}")
        config_path = 'config/config_distillation.py'
        config_module = 'config.config_distillation'

    # --- Load config ---
    spec = importlib.util.spec_from_file_location("train_config", config_path)
    train_config = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(train_config)

    # --- Init output dir ---
    os.makedirs(os.path.join(resume_dir, 'test', args.subdir, 'single'), exist_ok=True)
    os.makedirs(os.path.join(resume_dir, 'test', args.subdir, 'grid'), exist_ok=True)

    # --- Load data list ---
    img_list = open(args.input_list, 'r').read().split('\n')
    if img_list[-1] == '':
        img_list.pop()
    filtered_img_list = []
    for img_path in img_list:
        img_base = os.path.splitext(os.path.basename(img_path))[0]
        lm_path = os.path.join('dataset/landmarks', img_base + '.txt')
        frontal_path = get_frontal_path(img_path)
        if not os.path.exists(lm_path):
            print(f"[WARNING] Landmark tidak ditemukan untuk gambar: {img_path} | landmark: {lm_path}")
            continue
        if not os.path.exists(frontal_path):
            print(f"[WARNING] Frontal tidak ditemukan untuk gambar: {img_path} | frontal: {frontal_path}")
            continue
        filtered_img_list.append(img_path)
    if len(filtered_img_list) > args.max_images:
        filtered_img_list = random.sample(filtered_img_list, args.max_images)
    img_list = filtered_img_list

    # --- DataLoader ---
    dataloader = torch.utils.data.DataLoader(CustomTestDataset(img_list), batch_size=args.batch_size, shuffle=False,
                                             num_workers=0, pin_memory=True)

    # --- Load model sesuai mode ---
    if args.mode == 'teacher':
        G = Generator(
            zdim=train_config.G['zdim'],
            use_batchnorm=train_config.G['use_batchnorm'],
            use_residual_block=train_config.G['use_residual_block'],
            num_classes=train_config.G['num_classes']
        ).cuda()
        resume_model(G, resume_dir)
    else:
        G = GeneratorStudent(
            zdim=train_config.G['zdim'],
            num_classes=train_config.G['num_classes'],
            use_batchnorm=train_config.G['use_batchnorm'],
            use_residual_block=train_config.G['use_residual_block'],
            fm_mult=train_config.G.get('fm_mult', 0.75)
        ).cuda()
        resume_model(G, resume_dir)
    set_requires_grad(G, False)

    # --- Kelompokkan gambar per derajat (berdasarkan kode pose di nama file) ---
    DERAJAT_MAP = {
        '051': '0',
        '050': '30',
        '041': '60',
        '010': '90'
    }
    img_by_angle = {v: [] for v in DERAJAT_MAP.values()}
    for path in img_list:
        fname = os.path.basename(path)
        kode = fname.split('_')[1]
        if kode in DERAJAT_MAP:
            derajat = DERAJAT_MAP[kode]
            img_by_angle[derajat].append(path)

    # --- Ekstrak embedding gallery (frontal) untuk ONNX ArcFace ---
    gallery_embeddings_onnx = {}
    for path in tqdm(img_list, desc='Ekstrak embedding gallery'):
        frontal_path = get_frontal_path(path)
        if not os.path.exists(frontal_path):
            continue
        label = int(os.path.basename(frontal_path).split('_')[0])
        img = Image.open(frontal_path).convert('RGB')
        emb = extract_onnx_embedding(img)
        gallery_embeddings_onnx[label] = emb
    if not gallery_embeddings_onnx:
        print("[WARNING] Tidak ditemukan gallery embedding frontal untuk ArcFace ONNX. Evaluasi identitas akan dilewati.")

    # --- Loop per derajat: FID, IS, accuracy top-1 ---
    for derajat, imglist in img_by_angle.items():
        if not imglist:
            print(f"[INFO] Tidak ada gambar untuk derajat {derajat}")
            continue
        dataloader = torch.utils.data.DataLoader(CustomTestDataset(imglist), batch_size=args.batch_size, shuffle=False,
                                                 num_workers=0, pin_memory=True)
        all_fake_imgs = []
        all_real_imgs = []
        all_labels = []
        for step, batch in enumerate(tqdm(dataloader, desc=f'Testing {derajat} deg', unit='batch')):
            for k in ['img','img64','img32','left_eye','right_eye','nose','mouth']:
                batch[k] = Variable(batch[k].cuda(non_blocking=True))
            z = Variable(torch.FloatTensor(np.random.uniform(-1, 1, (len(batch['img']), train_config.G['zdim']))).cuda())
            outputs = G(
                batch['img'], batch['img64'], batch['img32'], batch['left_eye'], batch['right_eye'], batch['nose'],
                batch['mouth'], z, use_dropout=False, return_features=True)
            img128_fake = outputs[0]
            for i in range(img128_fake.shape[0]):
                all_fake_imgs.append(img128_fake[i].detach().cpu())
                all_real_imgs.append(batch['img_frontal'][i].detach().cpu())
                fname = os.path.basename(imglist[step * args.batch_size + i])
                label = int(fname.split('_')[0])
                all_labels.append(label)
        # --- Simpan fake & real ke folder sementara per derajat ---
        tmp_fake_dir = os.path.join(resume_dir, 'test', args.subdir, f'fid_fake_{derajat}')
        tmp_real_dir = os.path.join(resume_dir, 'test', args.subdir, f'fid_real_{derajat}')
        tmp_input_dir = os.path.join(resume_dir, 'test', args.subdir, f'input_{derajat}')
        os.makedirs(tmp_fake_dir, exist_ok=True)
        os.makedirs(tmp_real_dir, exist_ok=True)
        os.makedirs(tmp_input_dir, exist_ok=True)
        for idx, img in enumerate(all_fake_imgs):
            img_vis = (img + 1) / 2
            img_pil = transforms.ToPILImage()(img_vis)
            img_pil.save(os.path.join(tmp_fake_dir, f'{idx:05d}.png'))
        for idx, img in enumerate(all_real_imgs):
            img_vis = (img + 1) / 2
            img_pil = transforms.ToPILImage()(img_vis)
            img_pil.save(os.path.join(tmp_real_dir, f'{idx:05d}.png'))
        # --- Simpan input profile ke folder input_{derajat} ---
        for idx, img_path in enumerate(imglist):
            img_input = Image.open(img_path).convert('RGB').resize((128,128), Image.LANCZOS)
            img_input.save(os.path.join(tmp_input_dir, f'{idx:05d}.png'))
        # --- FID & IS per derajat ---
        fid_score_val = '-'
        is_score_val = '-'
        try:
            from pytorch_fid import fid_score
            fid_score_val = fid_score.calculate_fid_given_paths([tmp_real_dir, tmp_fake_dir], batch_size=32, device='cuda' if torch.cuda.is_available() else 'cpu', dims=2048)
        except Exception:
            pass
        try:
            import torch_fidelity
            metrics = torch_fidelity.calculate_metrics(input1=tmp_real_dir, input2=tmp_fake_dir, cuda=torch.cuda.is_available(), isc=True, fid=True, kid=False, verbose=False)
            fid_score_val = metrics['frechet_inception_distance']
            is_score_val = metrics['inception_score_mean']
        except Exception:
            pass
        print(f"{derajat} derajat FID: {fid_score_val}")
        print(f"{derajat} derajat IS: {is_score_val}")
        # --- Accuracy top-1 per derajat (fake TP-GAN) ---
        all_fake_embs = []
        for img in all_fake_imgs:
            img_input_pil = transforms.ToPILImage()((img + 1) / 2)
            emb_fake = extract_onnx_embedding(img_input_pil)
            all_fake_embs.append(emb_fake)
        all_fake_preds = []
        for emb_fake in all_fake_embs:
            sims = []
            for label_g, emb_g in gallery_embeddings_onnx.items():
                sim = np.dot(emb_fake, emb_g) / (np.linalg.norm(emb_fake) * np.linalg.norm(emb_g))
                sims.append((sim, label_g))
            if sims:
                pred_label = max(sims, key=lambda x: x[0])[1]
            else:
                pred_label = -1
            all_fake_preds.append(pred_label)
        if gallery_embeddings_onnx:
            correct_fake = sum([p == l for p, l in zip(all_fake_preds, all_labels) if p != -1])
            total_fake = sum([p != -1 for p in all_fake_preds])
            acc_fake = 100.0 * correct_fake / total_fake if total_fake > 0 else 0.0
            print(f"{derajat} derajat acc (ONNX ArcFace): {acc_fake:.2f}%")
        else:
            print(f"{derajat} derajat acc (ONNX ArcFace): - (gallery tidak ditemukan)")

    # --- Kumpulkan semua fake, real, dan label untuk evaluasi full ---
    all_fake_imgs_full = []
    all_real_imgs_full = []
    all_labels_full = []
    t_start = time.time()
    for derajat, imglist in img_by_angle.items():
        if not imglist:
            continue
        dataloader = torch.utils.data.DataLoader(CustomTestDataset(imglist), batch_size=args.batch_size, shuffle=False,
                                                 num_workers=0, pin_memory=True)
        for step, batch in enumerate(dataloader):
            for k in ['img','img64','img32','left_eye','right_eye','nose','mouth']:
                batch[k] = Variable(batch[k].cuda(non_blocking=True))
            z = Variable(torch.FloatTensor(np.random.uniform(-1, 1, (len(batch['img']), train_config.G['zdim']))).cuda())
            outputs = G(
                batch['img'], batch['img64'], batch['img32'], batch['left_eye'], batch['right_eye'], batch['nose'],
                batch['mouth'], z, use_dropout=False, return_features=True)
            img128_fake = outputs[0]
            for i in range(img128_fake.shape[0]):
                all_fake_imgs_full.append(img128_fake[i].detach().cpu())
                all_real_imgs_full.append(batch['img_frontal'][i].detach().cpu())
                fname = os.path.basename(imglist[step * args.batch_size + i])
                label = int(fname.split('_')[0])
                all_labels_full.append(label)
    t_end = time.time()
    # --- Simpan fake & real ke folder sementara untuk FID/IS full ---
    tmp_fake_dir_full = os.path.join(resume_dir, 'test', args.subdir, 'fid_fake_full')
    tmp_real_dir_full = os.path.join(resume_dir, 'test', args.subdir, 'fid_real_full')
    os.makedirs(tmp_fake_dir_full, exist_ok=True)
    os.makedirs(tmp_real_dir_full, exist_ok=True)
    for idx, img in enumerate(all_fake_imgs_full):
        img_vis = (img + 1) / 2
        img_pil = transforms.ToPILImage()(img_vis)
        img_pil.save(os.path.join(tmp_fake_dir_full, f'{idx:05d}.png'))
    for idx, img in enumerate(all_real_imgs_full):
        img_vis = (img + 1) / 2
        img_pil = transforms.ToPILImage()(img_vis)
        img_pil.save(os.path.join(tmp_real_dir_full, f'{idx:05d}.png'))
    # --- FID & IS full ---
    fid_score_val = '-'
    is_score_val = '-'
    try:
        from pytorch_fid import fid_score
        fid_score_val = fid_score.calculate_fid_given_paths([tmp_real_dir_full, tmp_fake_dir_full], batch_size=32, device='cuda' if torch.cuda.is_available() else 'cpu', dims=2048)
    except Exception:
        pass
    try:
        import torch_fidelity
        metrics = torch_fidelity.calculate_metrics(input1=tmp_real_dir_full, input2=tmp_fake_dir_full, cuda=torch.cuda.is_available(), isc=True, fid=True, kid=False, verbose=False)
        fid_score_val = metrics['frechet_inception_distance']
        is_score_val = metrics['inception_score_mean']
    except Exception:
        pass
    print(f"FULL validation FID: {fid_score_val}")
    print(f"FULL validation IS: {is_score_val}")
    # --- Accuracy top-1 full (fake TP-GAN) ---
    all_fake_embs_full = []
    for img in all_fake_imgs_full:
        img_input_pil = transforms.ToPILImage()((img + 1) / 2)
        emb_fake = extract_onnx_embedding(img_input_pil)
        all_fake_embs_full.append(emb_fake)
    all_fake_preds_full = []
    for emb_fake in all_fake_embs_full:
        sims = []
        for label_g, emb_g in gallery_embeddings_onnx.items():
            sim = np.dot(emb_fake, emb_g) / (np.linalg.norm(emb_fake) * np.linalg.norm(emb_g))
            sims.append((sim, label_g))
        if sims:
            pred_label = max(sims, key=lambda x: x[0])[1]
        else:
            pred_label = -1
        all_fake_preds_full.append(pred_label)
    if gallery_embeddings_onnx:
        correct_fake = sum([p == l for p, l in zip(all_fake_preds_full, all_labels_full) if p != -1])
        total_fake = sum([p != -1 for p in all_fake_preds_full])
        acc_fake = 100.0 * correct_fake / total_fake if total_fake > 0 else 0.0
        print(f"Accuracy top-1 (FULL validation, fake TP-GAN, ONNX ArcFace): {acc_fake:.2f}%")
    else:
        print(f"Accuracy top-1 (FULL validation, fake TP-GAN, ONNX ArcFace): - (gallery tidak ditemukan)")

    # --- Output statistik model dan speed untuk evaluasi full ---
    total_params = sum(p.numel() for p in G.parameters())
    trainable_params = sum(p.numel() for p in G.parameters() if p.requires_grad)
    print(f"[INFO] Jumlah total parameter (Generator): {total_params:,}")
    print(f"[INFO] Jumlah parameter trainable (Generator): {trainable_params:,}")
    # Hitung latency dan FPS untuk evaluasi full
    total_time = t_end - t_start
    num_samples = len(all_fake_imgs_full)
    latency_ms = (total_time / num_samples) * 1000 if num_samples > 0 else 0
    fps = num_samples / total_time if total_time > 0 else 0
    print(f"[INFO] Inference selesai untuk {num_samples} gambar dalam {total_time:.2f} detik.")
    print(f"[INFO] Latency rata-rata: {latency_ms:.2f} ms/gambar")
    print(f"[INFO] FPS: {fps:.2f}")

    # --- Selesai ---
    print("[INFO] Evaluasi selesai.")

# --- Komentar ---
# Jalankan dengan:
# python test_unified.py --input_list validation_list.txt --resume_model 0 --subdir eksp1 --mode teacher
# python test_unified.py --input_list validation_list.txt --resume_model 3 --epoch 5 --subdir eksp1 --mode student
# Untuk student, wajib isi --epoch
# Model ONNX ArcFace harus tersedia di path yang sesuai
# Output: FID, IS, dan accuracy top-1 identitas seluruh validation 