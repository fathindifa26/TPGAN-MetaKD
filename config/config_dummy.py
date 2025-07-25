# config_dummy.py

train = {}
train['img_list'] = './train_list.txt'  # path ke list file training
train['learning_rate'] = 1e-4
train['num_epochs'] = 10                # cukup pendek untuk eksperimen cepat
train['batch_size'] = 2                 # kecil agar bisa jalan di CPU
train['log_step'] = 1                   # log setiap step untuk debug cepat
train['resume_model'] = None
train['resume_optimizer'] = None

G = {}
G['zdim'] = 64
G['use_residual_block'] = False         # nonaktifkan untuk efisiensi
G['use_batchnorm'] = False              # nonaktifkan agar ringan
G['num_classes'] = 6                    # jumlah identitas di dataset kecil

D = {}
D['use_batchnorm'] = False              # efisien di CPU

loss = {}
loss['weight_gradient_penalty'] = 10
loss['weight_128'] = 1.0
loss['weight_64'] = 1.0
loss['weight_32'] = 1.5
loss['weight_pixelwise'] = 1.0
loss['weight_pixelwise_local'] = 3.0
loss['weight_symmetry'] = 0.3           # masih diset tapi belum digunakan di dummy
loss['weight_adv_G'] = 1e-3
loss['weight_identity_preserving'] = 30 # tidak dipakai di dummy
loss['weight_total_varation'] = 1e-3
loss['weight_cross_entropy'] = 10       # tidak dipakai di dummy

# dummy config untuk menjaga struktur sama dengan aslinya
feature_extract_model = {}
feature_extract_model['resume'] = None  # tidak dipakai di dummy
