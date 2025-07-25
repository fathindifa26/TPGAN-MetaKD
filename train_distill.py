# train_distill.py
import torch
from torchvision import transforms
import config.config_distillation as config
from data.data import TrainDataset
import numpy as np
from models.network import Discriminator, Generator, GeneratorStudent, DiscriminatorStudent
from torch.autograd import Variable
import time
from logs.log import TensorBoardX
from utils.utils import *
from models import feature_extract_network
import importlib
import torch.nn.functional as F
import pandas as pd
import os
from datetime import datetime
from models.meta_kd import MetaKDOptimizer
import torch.autograd
from torch.cuda.amp import autocast, GradScaler  # AMP untuk mixed precision
import torch.nn as nn

if not torch.cuda.is_available():
    torch.Tensor.cuda = lambda self, *args, **kwargs: self
    torch.nn.Module.cuda = lambda self, *args, **kwargs: self
    torch.Tensor.pin_memory = lambda self: self

test_time = False

torch.autograd.set_detect_anomaly(True)  # Enable anomaly detection

class ProjectionHead(torch.nn.Module):
    def __init__(self, in_dim, hidden_dim=256, out_dim=128):
        super().__init__()
        self.layer1 = torch.nn.Linear(in_dim, hidden_dim)
        self.layer2 = torch.nn.Linear(hidden_dim, out_dim)
        
    def forward(self, x):
        # Flatten features if needed
        if len(x.shape) > 2:
            x = x.view(x.size(0), -1)  # Flatten to [batch_size, features]
        x = F.relu(self.layer1(x))
        x = self.layer2(x)
        return F.normalize(x, dim=1)

def feature_loss(t_features, s_features):
    """Compute feature-based distillation loss"""
    losses = []
    for t_feat, s_feat in zip(t_features, s_features):
        # Ensure same spatial dimensions if needed
        if t_feat.shape[-2:] != s_feat.shape[-2:]:
            t_feat = F.interpolate(t_feat, size=s_feat.shape[-2:], mode='bilinear', align_corners=False)
        losses.append(F.mse_loss(s_feat, t_feat))
    return sum(losses)

def info_nce_loss(teacher_proj, student_proj, temperature=0.5):
    """Compute contrastive loss between teacher and student embeddings"""
    # Clone inputs to avoid inplace operations
    z1 = F.normalize(teacher_proj.clone(), dim=1)
    z2 = F.normalize(student_proj.clone(), dim=1)
    
    N = z1.shape[0]
    sim = torch.mm(z1, z2.T) / temperature
    
    negative_mask = ~torch.eye(N, dtype=bool, device=sim.device)
    positive = sim.diag()
    
    numerator = torch.exp(positive)
    denominator = torch.sum(negative_mask.float() * torch.exp(sim), dim=1)
    
    loss = -torch.log(numerator / denominator)
    return loss.mean()

class CMPDisLoss(torch.nn.Module):
    def __init__(self):
        super(CMPDisLoss, self).__init__()
        self.criterion = torch.nn.L1Loss()
        self.weights = [1.0/32, 1.0/16, 1.0/8, 1.0/4, 1.0]
    def forward(self, real_list, fake_list):
        loss = 0
        j = 0
        for i in range(1, len(real_list), 2):
            # Ambil channel teacher sebanyak channel student
            t = real_list[i]
            s = fake_list[i]
            c = s.shape[1]
            t = t[:, :c, :, :]
            loss += self.weights[j] * self.criterion(s, t)
            j += 1
        return loss

if __name__ == "__main__":
    # Initialize CSV logger
    tb = TensorBoardX(config_filename_list=["config/config_distillation.py"])
    csv_path = os.path.join(tb.path, 'training_log.csv')
    csv_columns = ['epoch', 'step', 'G_loss', 'D_loss', 'kd_d_loss', 'gan_d_loss', 'gp_loss', 
                  'pixelwise_loss', 'local_loss', 'symmetry_loss', 
                  'adv_G_loss', 'tv_loss', 'ce_loss',
                  'kd_s_loss', 'gan_s_loss', 'kd_feat_loss', 'kd_feat_D_loss', 'kd_feat_intermediate_loss']
    
    if not os.path.exists(csv_path):
        pd.DataFrame(columns=csv_columns).to_csv(csv_path, index=False)

    # --- DataLoader ---
    img_list = open(config.train['img_list'], 'r').read().split('\n')
    img_list.pop()
    dataloader = torch.utils.data.DataLoader(
        TrainDataset(img_list),
        batch_size=config.train['batch_size'],
        shuffle=True,
        num_workers=8,
        pin_memory=True
    )

    # --- Teacher (frozen) ---
    teacher = torch.nn.DataParallel(
        Generator(
            zdim=config.G['zdim'],
            use_batchnorm=config.G['use_batchnorm'],
            use_residual_block=config.G['use_residual_block'],
            num_classes=config.G['num_classes']
        )
    ).cuda()
    if config.train['resume_model']:
        resume_model(teacher, config.train['resume_model'], strict=False)
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad = False

    # --- Discriminator Teacher (frozen, for feature loss) ---
    D_teacher = torch.nn.DataParallel(
        Discriminator(
            use_batchnorm=config.D['use_batchnorm'],
            fm_mult=1.0 # Teacher is full size
        )
    ).cuda()
    if config.train['resume_model']:
        resume_model(D_teacher, config.train['resume_model'], strict=False)
    D_teacher.eval()
    for p in D_teacher.parameters():
        p.requires_grad = False

    # --- Student (light) ---
    student = torch.nn.DataParallel(
        GeneratorStudent(
            zdim=config.G['zdim'],
            num_classes=config.G['num_classes'],
            use_batchnorm=config.G['use_batchnorm'],
            use_residual_block=config.G['use_residual_block'],
            fm_mult=config.G['fm_mult']
        )
    ).cuda()

    # --- Load weight dari teacher ke student (jika ada checkpoint) ---
    # Path otomatis dari config
    teacher_gen_ckpt = config.train.get('teacher_generator_ckpt', None)
    if teacher_gen_ckpt:
        load_teacher_to_student(student.module, teacher_gen_ckpt)

    # --- Discriminator (Student) ---
    D = torch.nn.DataParallel(
        DiscriminatorStudent(
            use_batchnorm=config.D['use_batchnorm'],
            fm_mult=config.D['fm_mult']
        )
    ).cuda()

    # --- Load weight dari teacher ke D student (jika ada checkpoint) ---
    teacher_disc_ckpt = config.train.get('teacher_discriminator_ckpt', None)
    if teacher_disc_ckpt:
        load_teacher_to_student(D.module, teacher_disc_ckpt)

    # --- Optimizers ---
    optimizer_S = torch.optim.Adam(
        student.parameters(),
        lr=config.train['learning_rate']
    )
    optimizer_D = torch.optim.Adam(
        D.parameters(),
        lr=config.train['learning_rate']
    )

    # --- Loss functions ---
    mse = torch.nn.MSELoss().cuda()
    l1_loss = torch.nn.L1Loss().cuda()
    cross_entropy = torch.nn.CrossEntropyLoss().cuda()

    # --- Feature Extractor (frozen) ---
    pretrain_config = importlib.import_module(
        '.'.join([*config.feature_extract_model['resume'].split('/'), 'pretrain_config'])
    )
    model_name = pretrain_config.stem['model_name']
    kwargs = pretrain_config.stem.copy()
    kwargs.pop('model_name')
    feature_extract_model = eval('feature_extract_network.' + model_name)(**kwargs)
    resume_model(feature_extract_model, config.feature_extract_model['resume'], strict=False)
    feature_extract_model = torch.nn.DataParallel(feature_extract_model).cuda()
    feature_extract_model.eval()
    for param in feature_extract_model.parameters():
        param.requires_grad = False

    # Log singkat memastikan EfficientNet sudah load pretrained dari timm
    if 'efficientnet' in model_name:
        print(f"[INFO] EfficientNet {model_name} loaded (pretrained timm/ImageNet, tanpa .pth jika file tidak ditemukan)")

    # --- Projection heads (frozen) ---
    # Get feature dimension from a test forward pass
    with torch.no_grad():
        dummy_input = torch.randn(1, 3, 128, 128).cuda()
        features, _ = feature_extract_model(dummy_input)
        feature_dim = features.view(1, -1).shape[1]
        print(f"Feature dimension detected: {feature_dim}")

    teacher_projection = ProjectionHead(in_dim=feature_dim).cuda()
    student_projection = ProjectionHead(in_dim=feature_dim).cuda()
    # Freeze projection heads
    teacher_projection.eval()
    for p in teacher_projection.parameters():
        p.requires_grad = False
    for p in student_projection.parameters():
        p.requires_grad = False

    # --- Resume Training (jika ada) ---
    start_epoch = 0
    resume_path = config.train.get('resume_distill_from')
    if resume_path and os.path.exists(resume_path):
        print(f"--- Resuming training from checkpoint: {resume_path} ---")
        # 1. Resume model student dan discriminator
        resume_model(student, resume_path, strict=False)
        resume_model(D, resume_path, strict=False)

        # 2. Resume optimizer dan dapatkan epoch terakhir
        last_epoch = resume_optimizer(optimizer_S, student, resume_path)
        resume_optimizer(optimizer_D, D, resume_path) # Resume optimizer D juga

        if last_epoch > -1:
            start_epoch = last_epoch + 1
            print(f"--- Resuming from epoch {start_epoch} ---")
        else:
            print("--- Optimizer checkpoint not found, starting from epoch 0 ---")
    else:
        if resume_path:
            print(f"--- Resume path not found: {resume_path}. Starting from scratch. ---")

    # --- Logging interval (log_interval: jumlah log per epoch, hanya saat update G) ---
    total_updateG = len(dataloader) // config.train['n_critic']
    log_intervals = [
        round(total_updateG * i / config.train['log_interval'])
        for i in range(1, config.train['log_interval'] + 1)
    ]
    log_updateG_steps = sorted(set((n * config.train['n_critic']) - 1 for n in log_intervals))

    # --- Training loop ---
    t = time.time()  # Initialize time variable for throughput calculation
    best_loss = float('inf')
    
    # Initialize MetaKD optimizer if enabled
    meta_kd = None
    if config.train.get('use_metakd', False):
        meta_kd = MetaKDOptimizer(
            num_losses=4,
            hidden_dim=config.metakd.get('hidden_dim', 64),
            lr=config.metakd.get('learning_rate', 1e-4),
            activation=config.metakd.get('activation', 'sigmoid')
        ).cuda()
    
    cmpdis_loss_fn = CMPDisLoss().cuda()

    # --- Inisialisasi statistik loss (di luar loop epoch, sebelum training loop) ---
    alpha_EMA = config.metakd.get('alpha_EMA', 0.9)
    use_EMA = config.metakd.get('is_EMA', True)
    use_rel = config.metakd.get('is_relative_improvement', True)
    ema_kd_feat, ema_pix, ema_pix_local = None, None, None
    prev_kd_feat, prev_pix, prev_pix_local = None, None, None

    # --- Logging MetaKD weights ---
    metakd_weights = []  # List untuk simpan [w_feat, w_pix, w_pix_local, w_feat_intermediate] per step
    metakd_csv_path = os.path.join(tb.path, 'metakd_weights.csv')
    if not os.path.exists(metakd_csv_path):
        pd.DataFrame(columns=['epoch', 'w_feat', 'w_pix', 'w_pix_local', 'w_feat_intermediate']).to_csv(metakd_csv_path, index=False)

    # Inisialisasi scaler hanya jika AMP aktif
    scaler_S = GradScaler('cuda') if config.train.get('use_amp', True) else None  # AMP scaler untuk student
    scaler_D = GradScaler('cuda') if config.train.get('use_amp', True) else None  # AMP scaler untuk discriminator

    # === Adapter untuk intermediate feature distillation generator ===
    adapters = {}
    if config.loss.get('enable_kd_feat_intermediate', False):
        kd_layers = config.loss['kd_feat_intermediate_layers']
        adapters = {}
        for key, idxs in kd_layers.items():
            adapters[key] = nn.ModuleList()
            for i in idxs:
                adapters[key].append(None)
        adapters = {k: v.cuda() for k, v in adapters.items()}
    adapters_initialized = False

    for epoch in range(start_epoch, config.train['num_epochs']):
        epoch_losses = {col: [] for col in csv_columns[2:]}  # Initialize epoch loss tracking
        
        # --- Tambahkan: epoch_warmup untuk MetaKD ---
        epoch_warmup_metakd = config.metakd.get('epoch_warmup_metakd', 3)
        use_metakd_now = config.train.get('use_metakd', False) and (epoch >= epoch_warmup_metakd)
        
        for step, batch in enumerate(dataloader):
            # Move tensors to GPU and create copies to avoid inplace ops
            batch_gpu = {}
            for k in batch:
                batch_gpu[k] = Variable(batch[k].cuda(non_blocking=True).clone(), requires_grad=False)
            z = Variable(torch.randn(len(batch['img']), config.G['zdim'])).cuda()

            # ---- Teacher forward ----
            if config.loss.get('enable_kd_feat_intermediate', False):
                with torch.no_grad():
                    t128, t64, t32, t_encoder_predict, t_local_vision, t_le, t_re, t_nose, t_mouth, t_local_input, t_feats = teacher(
                        batch_gpu['img'].clone(), batch_gpu['img64'].clone(), batch_gpu['img32'].clone(),
                        batch_gpu['left_eye'].clone(), batch_gpu['right_eye'].clone(),
                        batch_gpu['nose'].clone(), batch_gpu['mouth'].clone(),
                        z.clone(), use_dropout=False, return_features=True
                    )
            else:
                with torch.no_grad():
                    t128, t64, t32, t_encoder_predict, t_local_vision, t_le, t_re, t_nose, t_mouth, t_local_input = teacher(
                        batch_gpu['img'].clone(), batch_gpu['img64'].clone(), batch_gpu['img32'].clone(),
                        batch_gpu['left_eye'].clone(), batch_gpu['right_eye'].clone(),
                        batch_gpu['nose'].clone(), batch_gpu['mouth'].clone(),
                        z.clone(), use_dropout=False
                    )
                    t_feats = None
            # Detach and clone teacher outputs
            t128, t64, t32 = t128.detach().clone(), t64.detach().clone(), t32.detach().clone()
            t_local_vision = t_local_vision.detach().clone()
            t_le, t_re = t_le.detach().clone(), t_re.detach().clone()
            t_nose, t_mouth = t_nose.detach().clone(), t_mouth.detach().clone()

            # ---- Student forward ----
            if config.loss.get('enable_kd_feat_intermediate', False):
                s128, s64, s32, s_encoder_predict, s_local_vision, s_le, s_re, s_nose, s_mouth, s_local_input, s_feats = student(
                    batch_gpu['img'].clone(), batch_gpu['img64'].clone(), batch_gpu['img32'].clone(),
                    batch_gpu['left_eye'].clone(), batch_gpu['right_eye'].clone(),
                    batch_gpu['nose'].clone(), batch_gpu['mouth'].clone(),
                    z.clone(), use_dropout=True, return_features=True
                )
            else:
                s128, s64, s32, s_encoder_predict, s_local_vision, s_le, s_re, s_nose, s_mouth, s_local_input = student(
                    batch_gpu['img'].clone(), batch_gpu['img64'].clone(), batch_gpu['img32'].clone(),
                    batch_gpu['left_eye'].clone(), batch_gpu['right_eye'].clone(),
                    batch_gpu['nose'].clone(), batch_gpu['mouth'].clone(),
                    z.clone(), use_dropout=True
                )
                s_feats = None

            # ---- Update D ----
            set_requires_grad(D, True)
            # 1. Distillation Discriminator Loss (L_KD_D)
            kd_d_loss = -torch.mean(D(t128.detach())[0]) + torch.mean(D(s128.detach())[0])
            # 1b. Feature distillation loss for D (kd_feat_D_loss)
            with torch.no_grad():
                _, h_list_teacher_D_real = D_teacher(batch_gpu['img_frontal'])
                _, h_list_teacher_D_fake = D_teacher(s128.detach())
            _, h_list_student_D_real = D(batch_gpu['img_frontal'])
            _, h_list_student_D_fake = D(s128.detach())
            kd_feat_D_loss_real = cmpdis_loss_fn(h_list_teacher_D_real, h_list_student_D_real)
            kd_feat_D_loss_fake = cmpdis_loss_fn(h_list_teacher_D_fake, h_list_student_D_fake)
            kd_feat_D_loss = (kd_feat_D_loss_real + kd_feat_D_loss_fake) / 2.0
            # 2. L_GAN_D (Adversarial Loss from Real Images)
            gan_d_loss = -torch.mean(D(batch_gpu['img_frontal'])[0]) + torch.mean(D(s128.detach())[0])
            # Gradient penalty untuk L_GAN_D
            alpha = torch.rand(batch_gpu['img_frontal'].size(0), 1, 1, 1).cuda()
            interp = Variable(alpha * s128.detach().clone() + (1 - alpha) * batch_gpu['img_frontal'].clone(),
                              requires_grad=True)
            out = D(interp)[0]
            grad = torch.autograd.grad(
                outputs=out, inputs=interp,
                grad_outputs=torch.ones_like(out),
                retain_graph=True, create_graph=True
            )[0].view(out.size(0), -1)
            gp_loss = torch.mean((grad.norm(2, 1) - 1) ** 2)
            # Total discriminator loss: loss_D = loss_KD_D + λ * loss_GAN_D + λ_feat_D * kd_feat_D_loss
            L_D = kd_d_loss + config.loss['weight_gan_d'] * gan_d_loss + config.loss['weight_gradient_penalty'] * gp_loss + config.loss['weight_kd_feat_D'] * kd_feat_D_loss

            optimizer_D.zero_grad()
            if scaler_D is not None:
                scaler_D.scale(L_D).backward(retain_graph=True)  # AMP backward
                scaler_D.unscale_(optimizer_D)
                torch.nn.utils.clip_grad_norm_(D.parameters(), max_norm=5.0)
                scaler_D.step(optimizer_D)  # AMP step
                scaler_D.update()  # AMP update
            else:
                L_D.backward(retain_graph=True)
                torch.nn.utils.clip_grad_norm_(D.parameters(), max_norm=5.0)
                optimizer_D.step()

            # ---- Update student (Generator) ----
            set_requires_grad(D, False)
            if (step + 1) % config.train['n_critic'] == 0:
                with autocast(enabled=config.train.get('use_amp', True)):
                    # 1. Adversarial loss (lama, dinonaktifkan via config)
                    adv_G_loss = -torch.mean(D(s128)[0])
                    # 2. Adversarial Distillation Loss (L_KD_S)
                    kd_s_loss = -torch.mean(D(s128)[0]) + torch.mean(D(t128.detach())[0])
                    # 3. L_GAN_S (Adversarial Loss from Real Images)
                    gan_s_loss = -torch.mean(D(s128)[0]) + torch.mean(D(batch_gpu['img_frontal'])[0])
                    # 4. Pixel-wise losses
                    pixelwise_128_loss = l1_loss(s128, batch_gpu['img_frontal'])
                    pixelwise_64_loss = l1_loss(s64, batch_gpu['img64_frontal'])
                    pixelwise_32_loss = l1_loss(s32, batch_gpu['img32_frontal'])
                    pixelwise_loss = pixelwise_128_loss + pixelwise_64_loss + pixelwise_32_loss
                    # 5. Local component losses
                    eyel_loss = l1_loss(s_le, batch_gpu['left_eye_frontal'])
                    eyer_loss = l1_loss(s_re, batch_gpu['right_eye_frontal'])
                    nose_loss = l1_loss(s_nose, batch_gpu['nose_frontal'])
                    mouth_loss = l1_loss(s_mouth, batch_gpu['mouth_frontal'])
                    pixelwise_local_loss = eyel_loss + eyer_loss + nose_loss + mouth_loss
                    # 6. Symmetry loss
                    inv_idx128 = torch.arange(s128.size()[3] - 1, -1, -1).long().cuda()
                    s128_flip = s128.index_select(3, Variable(inv_idx128))
                    inv_idx64 = torch.arange(s64.size()[3] - 1, -1, -1).long().cuda()
                    s64_flip = s64.index_select(3, Variable(inv_idx64))
                    inv_idx32 = torch.arange(s32.size()[3] - 1, -1, -1).long().cuda()
                    s32_flip = s32.index_select(3, Variable(inv_idx32))
                    symmetry_loss = (
                        l1_loss(s128, s128_flip) +
                        l1_loss(s64, s64_flip) +
                        l1_loss(s32, s32_flip)
                    )
                    # 7. Total variation loss
                    tv_loss = torch.mean(torch.abs(s128[:, :, :-1, :] - s128[:, :, 1:, :])) + \
                             torch.mean(torch.abs(s128[:, :, :, :-1] - s128[:, :, :, 1:]))
                    # 8. Cross entropy loss
                    cross_entropy_loss = cross_entropy(s_encoder_predict, batch_gpu['label'])
                    
                    # Dapatkan h_list dari D_teacher (yang sudah di-load di luar loop) dan D_student
                    with torch.no_grad():
                        _, h_list_teacher = D_teacher(t128.detach())
                    _, h_list_student = D(s128)
                    kd_feat_loss = cmpdis_loss_fn(h_list_teacher, h_list_student)
                    # 9. Intermediate feature distillation loss
                    kd_feat_intermediate_loss = 0.0
                    if config.loss.get('enable_kd_feat_intermediate', False):
                        kd_layers = config.loss['kd_feat_intermediate_layers']
                        if not adapters_initialized:
                            for key, idxs in kd_layers.items():
                                for i, idx in enumerate(idxs):
                                    s_f = s_feats[key][i]
                                    t_f = t_feats[key][i]
                                    adapters[key][i] = nn.Conv2d(s_f.shape[1], t_f.shape[1], kernel_size=1).cuda()
                            adapters_initialized = True
                        for key, idxs in kd_layers.items():
                            for i, idx in enumerate(idxs):
                                s_f = s_feats[key][i]
                                t_f = t_feats[key][i]
                                mapped_s_f = adapters[key][i](s_f)
                                # Resize spatial jika perlu
                                if mapped_s_f.shape[-2:] != t_f.shape[-2:]:
                                    mapped_s_f = F.interpolate(mapped_s_f, size=t_f.shape[-2:], mode='bilinear', align_corners=False)
                                kd_feat_intermediate_loss = kd_feat_intermediate_loss + F.mse_loss(mapped_s_f, t_f)
                    # --- MetaKD: dynamic loss weighting ---
                    if use_metakd_now:
                        # Ambil semua loss utama
                        cur_kd_feat = kd_feat_loss.detach()
                        cur_pix = l1_loss(s128, t128).detach()
                        cur_pix_local = l1_loss(s_le, t_le).detach()
                        if config.loss.get('enable_kd_feat_intermediate', False):
                            cur_kd_feat_intermediate = kd_feat_intermediate_loss.detach() if isinstance(kd_feat_intermediate_loss, torch.Tensor) else torch.tensor(kd_feat_intermediate_loss, device=cur_kd_feat.device)
                        else:
                            cur_kd_feat_intermediate = torch.tensor(0.0, device=cur_kd_feat.device)
                        # EMA
                        if ema_kd_feat is None:
                            ema_kd_feat = cur_kd_feat
                            ema_pix = cur_pix
                            ema_pix_local = cur_pix_local
                            ema_kd_feat_intermediate = cur_kd_feat_intermediate
                        else:
                            ema_kd_feat = alpha_EMA * ema_kd_feat + (1 - alpha_EMA) * cur_kd_feat
                            ema_pix = alpha_EMA * ema_pix + (1 - alpha_EMA) * cur_pix
                            ema_pix_local = alpha_EMA * ema_pix_local + (1 - alpha_EMA) * cur_pix_local
                            ema_kd_feat_intermediate = alpha_EMA * ema_kd_feat_intermediate + (1 - alpha_EMA) * cur_kd_feat_intermediate
                        # Relative improvement
                        if prev_kd_feat is None:
                            rel_kd_feat = torch.zeros_like(cur_kd_feat)
                            rel_pix = torch.zeros_like(cur_pix)
                            rel_pix_local = torch.zeros_like(cur_pix_local)
                            rel_kd_feat_intermediate = torch.zeros_like(cur_kd_feat_intermediate)
                        else:
                            rel_kd_feat = (prev_kd_feat - cur_kd_feat) / (prev_kd_feat + 1e-8)
                            rel_pix = (prev_pix - cur_pix) / (prev_pix + 1e-8)
                            rel_pix_local = (prev_pix_local - cur_pix_local) / (prev_pix_local + 1e-8)
                            rel_kd_feat_intermediate = (prev_kd_feat_intermediate - cur_kd_feat_intermediate) / (prev_kd_feat_intermediate + 1e-8)
                        prev_kd_feat = cur_kd_feat
                        prev_pix = cur_pix
                        prev_pix_local = cur_pix_local
                        prev_kd_feat_intermediate = cur_kd_feat_intermediate
                        # Gabungkan statistik sesuai config
                        meta_input_list = [cur_kd_feat.unsqueeze(0), cur_pix.unsqueeze(0), cur_pix_local.unsqueeze(0), cur_kd_feat_intermediate.unsqueeze(0)]
                        if use_EMA:
                            meta_input_list += [ema_kd_feat.unsqueeze(0), ema_pix.unsqueeze(0), ema_pix_local.unsqueeze(0), ema_kd_feat_intermediate.unsqueeze(0)]
                        if use_rel:
                            meta_input_list += [rel_kd_feat.unsqueeze(0), rel_pix.unsqueeze(0), rel_pix_local.unsqueeze(0), rel_kd_feat_intermediate.unsqueeze(0)]
                        meta_input = torch.cat(meta_input_list)
                        meta_out = meta_kd(meta_input)
                        w_feat = meta_out[0]
                        w_pix = 1 + 9 * meta_out[1]
                        w_pix_local = 1 + 9 * meta_out[2]
                        w_feat_intermediate = meta_out[3]
                        # Logging per step
                        metakd_weights.append([w_feat.item(), w_pix.item(), w_pix_local.item(), w_feat_intermediate.item()])
                    elif config.train.get('use_metakd', False):
                        # Warmup: pakai config dulu
                        w_feat = config.loss['weight_kd_feat']
                        w_pix = config.loss['weight_kd_pix']
                        w_pix_local = config.loss['weight_kd_pix_local']
                        w_feat_intermediate = config.loss['weight_kd_feat_intermediate']
                        # Tambahan: log juga bobot config ke metakd_weights agar tetap tercatat di CSV
                        metakd_weights.append([
                            float(w_feat), float(w_pix), float(w_pix_local), float(w_feat_intermediate)
                        ])
                    else:
                        w_feat = config.loss['weight_kd_feat']
                        w_pix = config.loss['weight_kd_pix']
                        w_pix_local = config.loss['weight_kd_pix_local']
                        w_feat_intermediate = config.loss['weight_kd_feat_intermediate']
                    # Total loss dengan retain_graph
                    L_syn = (
                        config.loss['weight_pixelwise'] * pixelwise_loss +
                        config.loss['weight_pixelwise_local'] * pixelwise_local_loss +
                        w_pix * l1_loss(s128, t128) +
                        w_pix_local * l1_loss(s_le, t_le) +
                        config.loss['weight_symmetry'] * symmetry_loss +
                        config.loss['weight_total_varation'] * tv_loss
                    )
                    L_G = L_syn + config.loss['weight_cross_entropy'] * cross_entropy_loss + \
                          config.loss['lambda_gan'] * kd_s_loss \
                          + 0.2 * config.loss['lambda_gan'] * gan_s_loss \
                          + w_feat * kd_feat_loss \
                          + w_feat_intermediate * kd_feat_intermediate_loss

                optimizer_S.zero_grad()
                if scaler_S is not None:
                    scaler_S.scale(L_G).backward(retain_graph=True)  # AMP backward
                    scaler_S.unscale_(optimizer_S)
                    torch.nn.utils.clip_grad_norm_(student.parameters(), max_norm=5.0)
                    scaler_S.step(optimizer_S)  # AMP step
                    scaler_S.update()  # AMP update
                else:
                    L_G.backward(retain_graph=True)
                    torch.nn.utils.clip_grad_norm_(student.parameters(), max_norm=5.0)
                    optimizer_S.step()

                # Collect losses for CSV logging
                current_losses = {
                    'G_loss': L_G.item(),
                    'D_loss': L_D.item(),
                    'kd_d_loss': kd_d_loss.item(),
                    'gan_d_loss': gan_d_loss.item(),
                    'gp_loss': gp_loss.item(),
                    'pixelwise_loss': pixelwise_loss.item(),
                    'local_loss': pixelwise_local_loss.item(),
                    'symmetry_loss': symmetry_loss.item(),
                    'adv_G_loss': adv_G_loss.item(),
                    'tv_loss': tv_loss.item(),
                    'ce_loss': cross_entropy_loss.item(),
                    'kd_s_loss': kd_s_loss.item(),
                    'gan_s_loss': gan_s_loss.item(),
                    'kd_feat_loss': kd_feat_loss.item(),
                    'kd_feat_D_loss': kd_feat_D_loss.item(),
                    'kd_feat_intermediate_loss': float(kd_feat_intermediate_loss) if config.loss.get('enable_kd_feat_intermediate', False) else 0.0
                }
                # Add losses to epoch tracking
                for k, v in current_losses.items():
                    epoch_losses[k].append(v)

                # ---- Logging (gabungan D & G, hanya saat update G dan step in log_updateG_steps) ----
                if (step + 1) % config.train['n_critic'] == 0:
                    updateG_idx = (step + 1) // config.train['n_critic']
                    if step in log_updateG_steps:
                        new_t = time.time()
                        speed = config.train['batch_size'] * config.train['n_critic'] / (new_t-t) if (new_t-t) > 0 else 0.0
                        print(
                            f"epoch {epoch} , step {step}/{len(dataloader)} , "
                            f"kd_D_loss {kd_d_loss.item():.3f} , gan_D_loss {gan_d_loss.item():.3f} , "
                            f"gp_loss {gp_loss.item():.3f} , "
                            f"G_loss {L_G.item():.3f} , "
                            f"L_KD_S {kd_s_loss.item():.3f} , L_GAN_S {gan_s_loss.item():.3f} , "
                            f"kd_pix {l1_loss(s128, t128).item():.3f} , kd_pix_local {l1_loss(s_le, t_le).item():.3f} , "
                            f"kd_feat {kd_feat_loss.item():.3f} , "
                            f"kd_feat_D {kd_feat_D_loss.item():.3f} , "
                            f"kd_feat_intermediate {float(kd_feat_intermediate_loss) if config.loss.get('enable_kd_feat_intermediate', False) else 0.0:.3f} , "
                            f"{speed:.1f} imgs/s"
                        )
                        # Logging ke TensorBoard untuk D
                        global_step = epoch * len(dataloader) + step
                        tb.add_scalar('loss/D/kd_distillation', kd_d_loss.item(), global_step, 'train')
                        tb.add_scalar('loss/D/gan_adversarial', gan_d_loss.item(), global_step, 'train')
                        tb.add_scalar('loss/D/gradient_penalty', gp_loss.item(), global_step, 'train')
                        tb.add_scalar('loss/D/total', L_D.item(), global_step, 'train')
                        tb.add_scalar('loss/D/kd_feat_D', kd_feat_D_loss.item(), global_step, 'train')
                        # Logging ke TensorBoard untuk G dan images
                        tb.add_scalar('loss/G/adversarial', adv_G_loss.item(), global_step, 'train')
                        tb.add_scalar('loss/G/pixelwise', pixelwise_loss.item(), global_step, 'train')
                        tb.add_scalar('loss/G/local', pixelwise_local_loss.item(), global_step, 'train')
                        tb.add_scalar('loss/G/symmetry', symmetry_loss.item(), global_step, 'train')
                        tb.add_scalar('loss/G/total_variation', tv_loss.item(), global_step, 'train')
                        tb.add_scalar('loss/G/cross_entropy', cross_entropy_loss.item(), global_step, 'train')
                        tb.add_scalar('loss/G/total', L_G.item(), global_step, 'train')
                        tb.add_scalar('loss/G/kd_feat_intermediate', float(kd_feat_intermediate_loss) if config.loss.get('enable_kd_feat_intermediate', False) else 0.0, global_step, 'train')
                        # Log images
                        tb.add_image_grid("grid/predict", 4, s128.data.float()/2.+0.5,
                                          global_step, 'train')
                        tb.add_image_grid("grid/teacher", 4, t128.data.float()/2.+0.5,
                                          global_step, 'train')
                        tb.add_image_grid("grid/frontal", 4, batch_gpu['img_frontal'].data.float()/2.+0.5,
                                          global_step, 'train')
                        tb.add_image_grid("grid/profile", 4, batch_gpu['img'].data.float()/2.+0.5,
                                          global_step, 'train')
                        tb.add_image_grid("grid/local", 4, s_local_vision.data.float()/2.+0.5,
                                          global_step, 'train')
                        tb.add_image_grid("grid/local_input", 4, s_local_input.data.float()/2.+0.5,
                                          global_step, 'train')
                        t = new_t

        # End of epoch processing
        # Calculate average losses for the epoch
        avg_losses = {k: (sum(v)/len(v) if len(v) > 0 else 0.0) for k, v in epoch_losses.items()}
        
        # Log to CSV
        df = pd.DataFrame([{
            'epoch': epoch,
            'step': -1,  # -1 indicates epoch average
            **avg_losses
        }])
        df.to_csv(csv_path, mode='a', header=False, index=False)
        
        # Save models at specific intervals
        if (epoch + 1) % config.train.get('save_interval', 5) == 0 or \
           avg_losses['G_loss'] < best_loss:
            save_dir = os.path.join(tb.path, f'epoch_{epoch}')
            os.makedirs(save_dir, exist_ok=True)
            
            # Save models
            save_model(student, save_dir, epoch)
            save_model(D, save_dir, epoch)
            save_optimizer(optimizer_S, student, save_dir, epoch)
            save_optimizer(optimizer_D, D, save_dir, epoch)
            
            # Update best loss if needed
            if avg_losses['G_loss'] < best_loss:
                best_loss = avg_losses['G_loss']
                print(f"New best model saved at epoch {epoch}")
            
            print(f"Models saved at {save_dir}")

        # Logging rata-rata MetaKD weights per epoch
        if config.train.get('use_metakd', False) and len(metakd_weights) > 0:
            avg_w_feat = sum([w[0] for w in metakd_weights]) / len(metakd_weights)
            avg_w_pix = sum([w[1] for w in metakd_weights]) / len(metakd_weights)
            avg_w_pix_local = sum([w[2] for w in metakd_weights]) / len(metakd_weights)
            avg_w_feat_intermediate = sum([w[3] for w in metakd_weights]) / len(metakd_weights)
            # Simpan ke CSV
            df = pd.DataFrame([{'epoch': epoch, 'w_feat': avg_w_feat, 'w_pix': avg_w_pix, 'w_pix_local': avg_w_pix_local, 'w_feat_intermediate': avg_w_feat_intermediate}])
            df.to_csv(metakd_csv_path, mode='a', header=False, index=False)
            # Log ke TensorBoard
            tb.add_scalar('metakd/w_feat', avg_w_feat, epoch, 'train')
            tb.add_scalar('metakd/w_pix', avg_w_pix, epoch, 'train')
            tb.add_scalar('metakd/w_pix_local', avg_w_pix_local, epoch, 'train')
            tb.add_scalar('metakd/w_feat_intermediate', avg_w_feat_intermediate, epoch, 'train')
            # Reset list untuk epoch berikutnya
            metakd_weights = []
