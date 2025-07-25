# config/config.py
train =  {}
train['img_list'] = './train_list.txt'
train['learning_rate'] = 5e-5  # 1e-4 -> 5e-5 (biar makin halus di akhir)
train['num_epochs'] = 400 
train['batch_size'] = 8 # dari 8 ke 32 untuk GPU A100
train['log_step'] = 1000
train['resume_model'] = None
train['resume_optimizer'] = None

G = {}
G['zdim'] = 64
G['use_residual_block'] = True # sebelumnya false, di paper True
G['use_batchnorm'] = False
G['num_classes'] = 347

D = {}
D['use_batchnorm'] = False

loss = {}
loss['weight_gradient_penalty'] = 10
loss['weight_128'] = 1.0
loss['weight_64'] = 1.0
loss['weight_32'] = 1.5
loss['weight_pixelwise'] = 3.0  # dari 1.0 → 2.0 -> 3.0 (ngilangin noise makin tinggi)
loss['weight_pixelwise_local'] = 3.0

loss['weight_symmetry'] = 3e-1
loss['weight_adv_G'] = 2e-3 # 1. dari 1e-3 → 2e-3 -> 5e-3 (langsung tajam) -> turunin 2e-3 
loss['weight_identity_preserving'] = 3e1
loss['weight_total_varation'] = 1e-2 # dari 1e-3 -> 5e-3 -> 1e-2 (diperbesar ngilangin noise)
loss['weight_cross_entropy'] = 1e1

feature_extract_model = {}
feature_extract_model['resume'] =  'feature_extractor_models/efficientnet'
# Alternatif untuk training yang lebih cepat:
# feature_extract_model['resume'] =  'feature_extractor_models/mobilenetv2'  # Lebih ringan
# feature_extract_model['resume'] =  'feature_extractor_models/lightcnnv4'   # Lebih ringan
