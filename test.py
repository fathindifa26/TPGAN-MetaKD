from data.data import TestDataset
from models.network import Discriminator, Generator
from utils.utils import *
import importlib
import argparse
import torch
import numpy as np
import time
from torch.autograd import Variable
from models import feature_extract_network
from PIL import Image
import torchvision.transforms as transforms
import os
import glob
import csv
import sys
import random
try:
    from ptflops import get_model_complexity_info
    ptflops_available = True
except ImportError:
    ptflops_available = False
try:
    import psutil
    psutil_available = True
except ImportError:
    psutil_available = False
try:
    from pytorch_fid import fid_score
    fid_available = True
except ImportError:
    fid_available = False
try:
    import torch_fidelity
    torch_fidelity_available = True
except ImportError:
    torch_fidelity_available = False
from torch.utils.data import Dataset
from math import floor
from tqdm import tqdm
import onnxruntime as ort

test_time = False

# python test.py --input_list validation_list.txt --resume_model save/try_0 --subdir eksp1 --batch_size 2
def parse_args():
    parser = argparse.ArgumentParser(description="bicubic")
    parser.add_argument("--input_list")
    parser.add_argument('--resume_model', help='resume_model dirname')
    parser.add_argument("--subdir", help='output_dir = save/$resume_model/test/$subdir')
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--max_images', type=int, default=5000, help='Jumlah maksimal gambar yang diproses')
    flag_parser = parser.add_mutually_exclusive_group(required=False)
    flag_parser.add_argument("--folder", dest='folder', action="store_true")
    flag_parser.add_argument("--nofolder", dest='folder', action='store_false')
    parser.set_defaults(folder=True)
    args = parser.parse_args()
    return args


def init_dir(args):
    os.system('mkdir -p {}'.format('/'.join([args.resume_model, 'test', args.subdir, 'single'])))
    os.system('mkdir -p {}'.format('/'.join([args.resume_model, 'test', args.subdir, 'grid'])))


def clean_landmark_string(s):
    s = s.replace('\n', ' ').replace(',', ' ')
    s = ' '.join([v for v in s.split() if v.strip() != ''])
    return s


def is_valid_landmark(lm_str, expected_len=60, debug_path=None):
    vals = [v for v in lm_str.split() if v.strip() != '']
    if len(vals) != expected_len:
        if debug_path:
            print(f"[DEBUG] Landmark jumlah angka tidak sesuai ({len(vals)}) di: {debug_path}")
            print(f"[DEBUG] Isi landmark: {lm_str}")
        return False
    try:
        arr = np.array(vals, dtype=np.float32)
        if np.isnan(arr).any():
            if debug_path:
                print(f"[DEBUG] Landmark mengandung NaN di: {debug_path}")
                print(f"[DEBUG] Isi landmark: {lm_str}")
            return False
    except Exception as e:
        if debug_path:
            print(f"[DEBUG] Landmark error parsing di: {debug_path} | error: {e}")
            print(f"[DEBUG] Isi landmark: {lm_str}")
        return False
    return True


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
        x = floor(landmarks_5pts[i,0])
        y = floor(landmarks_5pts[i,1])
        patch = img.crop((
            x - patch_size[name[i]][0]//2 + 1,
            y - patch_size[name[i]][1]//2 + 1,
            x + patch_size[name[i]][0]//2 + 1,
            y + patch_size[name[i]][1]//2 + 1
        ))
        patch = patch.convert('RGB')  # pastikan patch RGB
        batch[name[i]] = patch
    return batch


def get_frontal_path(profile_path):
    fname = os.path.basename(profile_path)
    parts = fname.split('_')
    parts[1] = '051'  # Ganti kode pose ke 051
    frontal_fname = '_'.join(parts)
    return os.path.join(os.path.dirname(profile_path), frontal_fname)


class CustomTestDataset(Dataset):
    def __init__(self, img_list):
        self.img_list = img_list
    def __len__(self):
        return len(self.img_list)
    def __getitem__(self, idx):
        batch = {}
        img_path = self.img_list[idx]
        img_base = os.path.splitext(os.path.basename(img_path))[0]
        # Path frontal
        frontal_path = get_frontal_path(img_path)
        # Path landmark
        landmark_dir = 'dataset/landmarks'
        lm_path = os.path.join(landmark_dir, img_base + '.txt')
        # Baca gambar dan landmark
        img = Image.open(img_path).convert('RGB')
        lm = clean_landmark_string_from_file(lm_path)
        lm = np.array(lm.split(' '), np.float32).reshape(-1,2)
        # Normalisasi ke ukuran 128x128
        for i in range(lm.shape[0]):
            lm[i][0] *= 128/img.width
            lm[i][1] *= 128/img.height
        img = img.resize((128,128), Image.LANCZOS)
        batch_profile = process(img, lm)
        batch_profile['img'] = img
        batch_profile['img64'] = img.resize((64,64), Image.LANCZOS)
        batch_profile['img32'] = batch_profile['img64'].resize((32,32), Image.LANCZOS)
        # Frontal
        if os.path.exists(frontal_path):
            img_frontal = Image.open(frontal_path).convert('RGB').resize((128,128), Image.LANCZOS)
        else:
            img_frontal = img  # fallback, meski sebaiknya warning
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


onnx_path = 'model.onnx'
ort_session = ort.InferenceSession(onnx_path)

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
    init_dir(args)
    img_list = open(args.input_list, 'r').read().split('\n')
    if img_list[-1] == '':
        img_list.pop()
    # Filter hanya gambar yang landmark dan frontal-nya ada
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
    # Sampling 5000 foto secara acak jika lebih dari 5000
    if len(filtered_img_list) > args.max_images:
        filtered_img_list = random.sample(filtered_img_list, args.max_images)
    img_list = filtered_img_list

    # input
    train_config = importlib.import_module('.'.join([*args.resume_model.split('/'), 'config']))
    dataloader = torch.utils.data.DataLoader(CustomTestDataset(img_list), batch_size=args.batch_size, shuffle=False,
                                             num_workers=0, pin_memory=True)

    G = Generator(zdim=train_config.G['zdim'], use_batchnorm=train_config.G['use_batchnorm'],
                  use_residual_block=train_config.G['use_residual_block'],
                  num_classes=train_config.G['num_classes']).cuda()
    if args.resume_model is not None:
        resume_model(G, args.resume_model)
    set_requires_grad(G, False)

    # --- Jumlah Parameter ---
    total_params = sum(p.numel() for p in G.parameters())
    trainable_params = sum(p.numel() for p in G.parameters() if p.requires_grad)
    print(f"[INFO] Jumlah total parameter (Generator): {total_params:,}")
    print(f"[INFO] Jumlah parameter trainable (Generator): {trainable_params:,}")

    # --- Model File Size (MB) ---
    gen_ckpts = sorted(
        [f for f in glob.glob(os.path.join(args.resume_model, '*.pth')) if os.path.basename(f).startswith('Generator')],
        key=os.path.getmtime, reverse=True
    )
    if gen_ckpts:
        model_file = gen_ckpts[0]
        model_size_mb = os.path.getsize(model_file) / (1024*1024)
        print(f"[INFO] Ukuran file model Generator terakhir: {model_size_mb:.2f} MB ({os.path.basename(model_file)})")
    else:
        model_file = None
        model_size_mb = 0
        print("[INFO] Tidak ditemukan file Generator .pth di folder checkpoint.")

    # --- FLOPs (jika ptflops tersedia) ---
    print("[INFO] Lewati perhitungan FLOPs (ptflops) karena model membutuhkan banyak input.")
    flops_str = '-'

    # --- Inference & Speed-up ---
    all_fake_imgs = []
    all_real_imgs = []
    t_start = time.time()
    if psutil_available:
        process = psutil.Process(os.getpid())
        ram_start = process.memory_info().rss
    torch.cuda.reset_peak_memory_stats()
    for step, batch in enumerate(tqdm(dataloader, desc='Testing', unit='batch')):
        for k in ['img','img64','img32','left_eye','right_eye','nose','mouth']:
            batch[k] = Variable(batch[k].cuda(non_blocking=True))
        z = Variable(torch.FloatTensor(np.random.uniform(-1, 1, (len(batch['img']), train_config.G['zdim']))).cuda())
        outputs = G(
            batch['img'], batch['img64'], batch['img32'], batch['left_eye'], batch['right_eye'], batch['nose'],
            batch['mouth'], z, use_dropout=False, return_features=True)
        img128_fake = outputs[0]
        for i in range(img128_fake.shape[0]):
            img_name = img_list[step * args.batch_size + i].split('/')[-1]
            img128_fake_vis = (img128_fake[i].data.cpu() + 1) / 2
            save_image(img128_fake_vis, '/'.join([args.resume_model, 'test', args.subdir, 'single', img_name]))
            all_fake_imgs.append(img128_fake[i].detach().cpu())
            # Real frontal untuk FID
            real_frontal_vis = (batch['img_frontal'][i].data.cpu() + 1) / 2
            all_real_imgs.append(batch['img_frontal'][i].detach().cpu())
    t_end = time.time()
    total_time = t_end - t_start
    num_samples = len(all_fake_imgs)
    latency_ms = (total_time / num_samples) * 1000 if num_samples > 0 else 0
    fps = num_samples / total_time if total_time > 0 else 0
    print(f"[INFO] Inference selesai untuk {num_samples} gambar dalam {total_time:.2f} detik.")
    print(f"[INFO] Latency rata-rata: {latency_ms:.2f} ms/gambar")
    print(f"[INFO] FPS: {fps:.2f}")

    # --- Peak Memory Usage ---
    if psutil_available:
        ram_end = process.memory_info().rss
        peak_ram = max(ram_start, ram_end) / (1024*1024)
        print(f"[INFO] Peak RAM usage: {peak_ram:.2f} MB")
    else:
        peak_ram = '-'
        print("[INFO] psutil tidak ditemukan. Lewati perhitungan RAM.")
    if torch.cuda.is_available():
        peak_vram = torch.cuda.max_memory_allocated() / (1024*1024)
        print(f"[INFO] Peak VRAM usage: {peak_vram:.2f} MB")
    else:
        peak_vram = '-'
        print("[INFO] CUDA tidak tersedia. Lewati perhitungan VRAM.")

    # --- Simpan gambar fake & real ke folder sementara untuk FID/IS ---
    tmp_fake_dir = os.path.join(args.resume_model, 'test', args.subdir, 'fid_fake')
    tmp_real_dir = os.path.join(args.resume_model, 'test', args.subdir, 'fid_real')
    os.makedirs(tmp_fake_dir, exist_ok=True)
    os.makedirs(tmp_real_dir, exist_ok=True)
    for idx, img in enumerate(all_fake_imgs):
        img_vis = (img + 1) / 2
        img_pil = transforms.ToPILImage()(img_vis)
        img_pil.save(os.path.join(tmp_fake_dir, f'{idx:05d}.png'))
    for idx, img in enumerate(all_real_imgs):
        img_vis = (img + 1) / 2
        img_pil = transforms.ToPILImage()(img_vis)
        img_pil.save(os.path.join(tmp_real_dir, f'{idx:05d}.png'))

    # --- FID & IS ---
    fid_score_val = '-'
    is_score_val = '-'
    if fid_available:
        print("[INFO] Menghitung FID dengan pytorch-fid...")
        fid_score_val = fid_score.calculate_fid_given_paths([tmp_real_dir, tmp_fake_dir], batch_size=32, device='cuda' if torch.cuda.is_available() else 'cpu', dims=2048)
        print(f"[INFO] FID: {fid_score_val:.4f}")
    if torch_fidelity_available:
        print("[INFO] Menghitung FID & IS dengan torch-fidelity...")
        metrics = torch_fidelity.calculate_metrics(input1=tmp_real_dir, input2=tmp_fake_dir, cuda=torch.cuda.is_available(), isc=True, fid=True, kid=False, verbose=False)
        fid_score_val = metrics['frechet_inception_distance']
        is_score_val = metrics['inception_score_mean']
        print(f"[INFO] FID: {fid_score_val:.4f}")
        print(f"[INFO] IS: {is_score_val:.4f}")
    else:
        print("[INFO] Library FID/IS tidak ditemukan. Silakan install pytorch-fid atau torch-fidelity.")

    # --- Ekstrak embedding gallery (frontal) untuk ONNX ArcFace ---
    gallery_embeddings_onnx = {}
    for path in img_list:  # img_list dari validation_list.txt
        frontal_path = get_frontal_path(path)
        if not os.path.exists(frontal_path):
            continue  # skip jika frontal tidak ada
        label = int(os.path.basename(frontal_path).split('_')[0])
        img = Image.open(frontal_path).convert('RGB')
        emb = extract_onnx_embedding(img)
        gallery_embeddings_onnx[label] = emb
    if not gallery_embeddings_onnx:
        print("[WARNING] Tidak ditemukan gallery embedding frontal untuk ArcFace ONNX. Evaluasi identitas akan dilewati.")

    # --- Loop per derajat ---
    for derajat, imglist in img_by_angle.items():
        if not imglist:
            print(f"[INFO] Tidak ada gambar untuk derajat {derajat}")
            continue
        dataloader = torch.utils.data.DataLoader(CustomTestDataset(imglist), batch_size=args.batch_size, shuffle=False,
                                                 num_workers=0, pin_memory=True)
        all_fake_imgs = []
        all_real_imgs = []
        all_real_input_imgs = []
        all_labels = []
        all_onnx_preds = []
        all_input_preds = []
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
                all_real_input_imgs.append(batch['img'][i].detach().cpu())
                # --- ONNX ArcFace embedding & matching (fake) ---
                fname = os.path.basename(imglist[step * args.batch_size + i])
                label = int(fname.split('_')[0])
                all_labels.append(label)
                img_input = (img128_fake[i].detach().cpu() + 1) / 2
                img_input_pil = transforms.ToPILImage()(img_input)
                emb_fake = extract_onnx_embedding(img_input_pil)
                sims = []
                for label_g, emb_g in gallery_embeddings_onnx.items():
                    sim = np.dot(emb_fake, emb_g) / (np.linalg.norm(emb_fake) * np.linalg.norm(emb_g))
                    sims.append((sim, label_g))
                if sims:
                    pred_label_onnx = max(sims, key=lambda x: x[0])[1]
                else:
                    pred_label_onnx = -1
                all_onnx_preds.append(pred_label_onnx)
                # --- ONNX ArcFace embedding & matching (input asli) ---
                img_input_real_pil = transforms.ToPILImage()(batch['img'][i].detach().cpu())
                emb_input = extract_onnx_embedding(img_input_real_pil)
                sims_input = []
                for label_g, emb_g in gallery_embeddings_onnx.items():
                    sim = np.dot(emb_input, emb_g) / (np.linalg.norm(emb_input) * np.linalg.norm(emb_g))
                    sims_input.append((sim, label_g))
                if sims_input:
                    pred_label_input = max(sims_input, key=lambda x: x[0])[1]
                else:
                    pred_label_input = -1
                all_input_preds.append(pred_label_input)
        # --- FID & IS ---
        # ... existing code ...
        # --- Accuracy ---
        if gallery_embeddings_onnx:
            correct_onnx = sum([p == l for p, l in zip(all_onnx_preds, all_labels) if p != -1])
            total_onnx = sum([p != -1 for p in all_onnx_preds])
            acc_onnx = 100.0 * correct_onnx / total_onnx if total_onnx > 0 else 0.0
            print(f"{derajat} derajat acc (ONNX ArcFace): {acc_onnx:.2f}%")
            correct_input = sum([p == l for p, l in zip(all_input_preds, all_labels) if p != -1])
            total_input = sum([p != -1 for p in all_input_preds])
            acc_input = 100.0 * correct_input / total_input if total_input > 0 else 0.0
            print(f"{derajat} derajat acc (Input Asli, ONNX ArcFace): {acc_input:.2f}%")
        else:
            print(f"{derajat} derajat acc (ONNX ArcFace): - (gallery tidak ditemukan)")
            print(f"{derajat} derajat acc (Input Asli, ONNX ArcFace): - (gallery tidak ditemukan)")

    # --- Simpan hasil evaluasi ke CSV ---
    csv_path = os.path.join(args.resume_model, 'test', args.subdir, 'evaluasi.csv')
    fieldnames = [
        'total_params', 'trainable_params', 'model_file', 'model_size_mb', 'FLOPs',
        'total_time_s', 'latency_ms', 'fps', 'peak_ram_mb', 'peak_vram_mb', 'FID', 'IS'
    ]
    data = {
        'total_params': total_params,
        'trainable_params': trainable_params,
        'model_file': os.path.basename(model_file) if model_file else '-',
        'model_size_mb': f"{model_size_mb:.2f}",
        'FLOPs': flops_str,
        'total_time_s': f"{total_time:.2f}",
        'latency_ms': f"{latency_ms:.2f}",
        'fps': f"{fps:.2f}",
        'peak_ram_mb': f"{peak_ram:.2f}" if peak_ram != '-' else '-',
        'peak_vram_mb': f"{peak_vram:.2f}" if peak_vram != '-' else '-',
        'FID': f"{fid_score_val:.4f}" if isinstance(fid_score_val, float) or isinstance(fid_score_val, np.floating) else fid_score_val,
        'IS': f"{is_score_val:.4f}" if isinstance(is_score_val, float) or isinstance(is_score_val, np.floating) else is_score_val,
    }
    with open(csv_path, 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerow(data)
    print(f"[INFO] Hasil evaluasi disimpan di {csv_path}")
