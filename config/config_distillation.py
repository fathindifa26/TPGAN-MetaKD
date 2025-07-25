# config_distillation.py

# === Data & training ===
train = {}
train['img_list']        = './train_list.txt'
train['use_amp']         = False  # Aktifkan atau nonaktifkan AMP (mixed precision) dari config
train['learning_rate']   = 1e-4
train['num_epochs']      = 200
train['batch_size']      = 32 
train['log_interval']    = 100  # berapa kali print log setiap epoch
train['save_interval']   = 1

train['resume_model']    = 'save/try_68' # load teacher
train['resume_optimizer']= 'save/try_68' # ini nggak perlu nggak papa
train['resume_distill_from'] = None # Path untuk resume training, contoh: 'save/try_79'

train['n_critic'] = 5 # Jumlah update D sebelum update G (default 1, bisa diatur) -> TinyGAN 5 kali

# === Teacher checkpoint untuk transfer ke student ===
# train['teacher_generator_ckpt'] = 'save/try_68/Generator_epoch128.pth'  # contoh: 'save/try_1/Generator_epoch199.pth'
train['teacher_generator_ckpt'] = None
train['teacher_discriminator_ckpt'] = 'save/try_68/Discriminator_epoch128.pth'  # contoh: 'save/try_1/Discriminator_epoch199.pth'


# === Generator (Teacher & Student) ===
G = {}
G['zdim']               = 64
G['use_residual_block'] = True # di paper True
G['use_batchnorm']      = False
G['num_classes']        = 347
G['fm_mult']            = 0.5  # Compression factor for student model

# === Discriminator ===
D = {}
D['use_batchnorm'] = False
D['fm_mult']       = 0.5 # Compression factor for student discriminator

# === Loss weights ===
loss = {}
# Basic losses (same as original)
loss['weight_gradient_penalty'] = 10
loss['weight_pixelwise'] = 0 # disable digantikan oleh weight_kd_pix
loss['weight_pixelwise_local'] = 0 # disable digantikan oleh weight_kd_pix_local
loss['weight_symmetry'] = 0 # dari 3e-1 di disable
loss['weight_adv_G'] = 0 # disable
loss['weight_total_varation'] = 0 # disable
loss['weight_cross_entropy'] = 1e1

# === Generator Distillation Loss weights ===
loss['lambda_gan'] = 0.01  # Bobot untuk L_KD_S dan L_GAN_S pada generator
loss['weight_kd_pix'] = 1.0  # Default sama dengan pixelwise -> L_KD_pix
loss['weight_kd_pix_local'] = 3.0 # Default sama dengan pixelwise_local -> L_KD_pix_local
loss['weight_kd_feat'] = 1.0  # Bobot untuk L_KD_feat (CMPDisLoss) -> selalu 1.0 memang yang utama
# Jadi L_G = lambda_gan * L_KD_S + lambda_gan * 0.2 L_GAN_S + weight_kd_pix * L_KD_pix + weight_kd_pix_local * L_KD_pix_local + weight_kd_feat * L_KD_feat

# === Generator Intermediate Feature Distillation ===
loss['enable_kd_feat_intermediate'] = True  # Aktifkan distillation intermediate feature generator
# Layer yang diambil: global pathway conv1-4, local pathway conv1-3
loss['kd_feat_intermediate_layers'] = {
    'global': [0, 1, 2, 3],  # index conv1, conv2, conv3, conv4
    'left_eye': [0, 1, 2],
    'right_eye': [0, 1, 2],
    'nose': [0, 1, 2],
    'mouth': [0, 1, 2],
}
loss['weight_kd_feat_intermediate'] = 1.0  # Bobot loss intermediate feature distillation

# === Discriminator Distillation Loss weights ===
loss['weight_kd_feat_D'] = 1.0  # Bobot untuk feature distillation loss pada D student (CMPDisLoss)
loss['weight_gan_d'] = 0.2 # Weight untuk L_GAN_D (adversarial loss from real images) -> di paper TinyGAN 0.2
# Weight untuk L_KD_D (adversarial loss dari teacher) sudah pasti 1.0
# Jadi L_D = L_KD_D + weight_gan_d * L_GAN_D

# === Feature extractor (frozen) ===
feature_extract_model = {}
# ini harus folder yang berisi:
#   - checkpoint(.pth)
#   - pretrain_config.py
feature_extract_model['resume'] = 'feature_extractor_models/efficientnet'

# === MetaKD ===
train['use_metakd']     = True  # Whether to use MetaKD for dynamic loss weighting
# MetaKD parameters (used when train['use_metakd'] is True)
metakd = {}
metakd['activation'] = 'sigmoid'  # Pilihan: 'sigmoid' atau 'softmax' untuk output MetaKD
metakd['hidden_dim'] = 64          # Hidden dimension for weight prediction network
metakd['learning_rate'] = 1e-4     # Learning rate for meta optimizer
metakd['momentum'] = 0.9           # Momentum for loss statistics
metakd['alpha_EMA'] = 0.9          # Alpha smoothing untuk EMA loss (statistik input meta-net)
metakd['is_EMA'] = True            # Apakah statistik EMA loss dipakai sebagai input meta-net
metakd['is_relative_improvement'] = True  # Apakah statistik relative improvement dipakai sebagai input meta-net
metakd['epoch_warmup_metakd'] = 2  # Jumlah epoch awal pakai config loss sebelum MetaKD aktif
# metakd['is_norm_gradient'] = True # -> jika mau menerapkan juga (input meta-net)

