test_time = False

num_classes = 347

train = {}
train['train_img_list'] = 'pretrain_train.list'
train['val_img_list'] = 'pretrain_val.list'

train['batch_size'] = 128
train['num_epochs'] = 35
train['log_step'] = 100

train['optimizer'] = 'SGD'
train['learning_rate'] = 1e-1
train['momentum'] = 0.9
train['nesterov'] = True
train['warmup_length'] = 0
train['learning_rate_decay'] = 1.0
train['auto_adjust_lr'] = False

train['sub_dir'] = 'feature_extract_model'
train['pretrained'] = None
train['resume'] = None
train['resume_optimizer'] = None
train['resume_epoch'] = None  #None means the last epoch

train['use_lr_scheduler'] = True
train['lr_scheduler_milestones'] = [10,20,30]

stem = {}
model_name = ['efficientnet_b0', 'efficientnet_b1', 'efficientnet_b2', 'efficientnet_b3', 'efficientnet_b4']
stem['model_name'] = 'efficientnet_b0'  # Default ke B0 (paling ringan)
stem['num_classes'] = num_classes

assert stem['model_name'] in model_name

# EfficientNet specific parameters
if stem['model_name'].startswith('efficientnet'):
    stem['dropout'] = 0.5
    stem['pretrained'] = True  # Gunakan pretrained dari timm

loss = {}
loss['weight_l2_reg'] = 1e-4 