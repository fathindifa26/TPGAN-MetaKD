import torch
from torchvision import transforms
import config.config as config
from data.data import TrainDataset
import numpy as np
from skimage.io import imsave
from models.network import Discriminator, Generator
from torch.autograd import Variable
import time
from logs.log import TensorBoardX
from utils.utils import *
from models import feature_extract_network
import importlib
from torch.cuda.amp import autocast, GradScaler

if not torch.cuda.is_available():
    torch.Tensor.cuda = lambda self, *args, **kwargs: self
    torch.nn.Module.cuda = lambda self, *args, **kwargs: self
    torch.Tensor.pin_memory = lambda self: self

test_time = False

if __name__ == "__main__":
    img_list = open(config.train['img_list'], 'r').read().split('\n')
    img_list.pop()

    # input
    dataloader = torch.utils.data.DataLoader(
        TrainDataset(img_list),
        batch_size=config.train['batch_size'],
        shuffle=True,
        num_workers=16,
        pin_memory=True,
        prefetch_factor=2,
        persistent_workers=True
    )

    G = torch.nn.DataParallel(
        Generator(
            zdim=config.G['zdim'],
            use_batchnorm=config.G['use_batchnorm'],
            use_residual_block=config.G['use_residual_block'],
            num_classes=config.G['num_classes']
        )
    ).cuda()
    D = torch.nn.DataParallel(
        Discriminator(use_batchnorm=config.D['use_batchnorm'])
    ).cuda()

    optimizer_G = torch.optim.Adam(
        filter(lambda p: p.requires_grad, G.parameters()), lr=1e-4
    )
    optimizer_D = torch.optim.Adam(
        filter(lambda p: p.requires_grad, D.parameters()), lr=1e-4
    )

    last_epoch = -1
    if config.train['resume_model'] is not None:
        e1 = resume_model(G, config.train['resume_model'])
        e2 = resume_model(D, config.train['resume_model'])
        assert e1 == e2
        last_epoch = e1
    if config.train['resume_optimizer'] is not None:
        e3 = resume_optimizer(optimizer_G, G, config.train['resume_optimizer'])
        e4 = resume_optimizer(optimizer_D, D, config.train['resume_optimizer'])
        assert e1 == e2 and e2 == e3 and e3 == e4
        last_epoch = e1

        
    tb = TensorBoardX(config_filename_list=["config/config.py"])

    pretrain_config = importlib.import_module(
        '.'.join([*config.feature_extract_model['resume'].split('/'), 'pretrain_config'])
    )
    model_name = pretrain_config.stem['model_name']
    kwargs = pretrain_config.stem.copy()
    kwargs.pop('model_name')
    feature_extract_model = eval('feature_extract_network.' + model_name)(**kwargs)

    # Log singkat memastikan EfficientNet sudah load pretrained dari timm
    if 'efficientnet' in model_name:
        print(f"[INFO] EfficientNet {model_name} loaded (pretrained timm/ImageNet, tanpa .pth jika file tidak ditemukan)")

    resume_model(feature_extract_model, config.feature_extract_model['resume'], strict=False)
    feature_extract_model = torch.nn.DataParallel(feature_extract_model).cuda()

    l1_loss = torch.nn.L1Loss().cuda()
    mse = torch.nn.MSELoss().cuda()
    cross_entropy = torch.nn.CrossEntropyLoss().cuda()

    for param in feature_extract_model.parameters():
        param.requires_grad = False

    # Optimasi: set feature extractor ke eval mode dan disable gradient computation
    feature_extract_model.eval()
    
    t = time.time()
    if test_time:
        tt = time.time()

    scaler_G = GradScaler()
    scaler_D = GradScaler()

    for epoch in range(last_epoch + 1, config.train['num_epochs']):
        for step, batch in enumerate(dataloader):
            if test_time:
                print("step :", step)
                t_pre = time.time()
                print("preprocess time :", t_pre - tt)
                tt = t_pre

            # 1) Pindahkan batch ke GPU
            for k in batch:
                batch[k] = Variable(batch[k].cuda(non_blocking=True), requires_grad=False)

            # 2) Sampling noise dan forward G
            z = Variable(
                torch.FloatTensor(
                    np.random.uniform(-1, 1, (len(batch['img']), config.G['zdim']))
                ).cuda()
            )
            if test_time:
                t_mv_to_cuda = time.time()
                print("mv_to_cuda time :", t_mv_to_cuda - tt)
                tt = t_mv_to_cuda

            img128_fake, img64_fake, img32_fake, G_encoder_outputs, \
            local_predict, le_fake, re_fake, nose_fake, mouth_fake, local_input = G(
                batch['img'], batch['img64'], batch['img32'],
                batch['left_eye'], batch['right_eye'], batch['nose'],
                batch['mouth'], z, use_dropout=True
            )
            if test_time:
                t_forward_G = time.time()
                print("forward_G time :", t_forward_G - tt)
                tt = t_forward_G

            # 3) Update D
            set_requires_grad(D, True)
            optimizer_D.zero_grad()
            with autocast():
                adv_D_loss = -torch.mean(D(batch['img_frontal'])) + torch.mean(D(img128_fake.detach()))
                # gradient penalty
                alpha = torch.rand(
                    batch['img_frontal'].shape[0], 1, 1, 1,
                    device=batch['img_frontal'].device
                ).expand_as(batch['img_frontal']).clone()
                interpolated_x = Variable(
                    alpha * img128_fake.detach()
                    + (1.0 - alpha) * batch['img_frontal'].detach(),
                    requires_grad=True
                )
                out = D(interpolated_x)
                grad_outputs = torch.ones_like(out)
                dxdD = torch.autograd.grad(
                    outputs=out,
                    inputs=interpolated_x,
                    grad_outputs=grad_outputs,
                    retain_graph=True,
                    create_graph=True
                )[0].view(out.shape[0], -1)
                gp_loss = ((dxdD.norm(2, dim=1) - 1) ** 2).mean()
                L_D = adv_D_loss + config.loss['weight_gradient_penalty'] * gp_loss
            scaler_D.scale(L_D).backward()
            scaler_D.step(optimizer_D)
            scaler_D.update()
            set_requires_grad(D, False)

            # 4) Update G
            optimizer_G.zero_grad()
            with autocast():
                adv_G_loss = -torch.mean(D(img128_fake))
                pixelwise_128_loss = l1_loss(img128_fake, batch['img_frontal'])
                pixelwise_64_loss = l1_loss(img64_fake, batch['img64_frontal'])
                pixelwise_32_loss = l1_loss(img32_fake, batch['img32_frontal'])
                pixelwise_loss = (
                    config.loss['weight_128'] * pixelwise_128_loss
                    + config.loss['weight_64'] * pixelwise_64_loss
                    + config.loss['weight_32'] * pixelwise_32_loss
                )
                eyel_loss = l1_loss(le_fake, batch['left_eye_frontal'])
                eyer_loss = l1_loss(re_fake, batch['right_eye_frontal'])
                nose_loss = l1_loss(nose_fake, batch['nose_frontal'])
                mouth_loss = l1_loss(mouth_fake, batch['mouth_frontal'])
                pixelwise_local_loss = eyel_loss + eyer_loss + nose_loss + mouth_loss
                inv_idx128 = torch.arange(img128_fake.size()[3] - 1, -1, -1).long().cuda()
                img128_fake_flip = img128_fake.index_select(3, Variable(inv_idx128)).detach_()
                inv_idx64 = torch.arange(img64_fake.size()[3] - 1, -1, -1).long().cuda()
                img64_fake_flip = img64_fake.index_select(3, Variable(inv_idx64)).detach_()
                inv_idx32 = torch.arange(img32_fake.size()[3] - 1, -1, -1).long().cuda()
                img32_fake_flip = img32_fake.index_select(3, Variable(inv_idx32)).detach_()
                symmetry_128_loss = l1_loss(img128_fake, img128_fake_flip)
                symmetry_64_loss = l1_loss(img64_fake, img64_fake_flip)
                symmetry_32_loss = l1_loss(img32_fake, img32_fake_flip)
                symmetry_loss = (
                    config.loss['weight_128'] * symmetry_128_loss
                    + config.loss['weight_64'] * symmetry_64_loss
                    + config.loss['weight_32'] * symmetry_32_loss
                )
                
                # Optimasi: Gunakan torch.no_grad() untuk feature extraction
                with torch.no_grad():
                    feature_frontal, fc_frontal = feature_extract_model(batch['img_frontal'])
                    feature_predict, fc_predict = feature_extract_model(img128_fake)
                
                ip_loss = mse(feature_predict, feature_frontal.detach())
                tv_loss = torch.mean(
                    torch.abs(img128_fake[:, :, :-1, :] - img128_fake[:, :, 1:, :])
                ) + torch.mean(
                    torch.abs(img128_fake[:, :, :, :-1] - img128_fake[:, :, :, 1:])
                )
                cross_entropy_loss = cross_entropy(G_encoder_outputs, batch['label'])
                L_syn = (
                    config.loss['weight_pixelwise'] * pixelwise_loss
                    + config.loss['weight_pixelwise_local'] * pixelwise_local_loss
                    + config.loss['weight_symmetry'] * symmetry_loss
                    + config.loss['weight_adv_G'] * adv_G_loss
                    + config.loss['weight_identity_preserving'] * ip_loss
                    + config.loss['weight_total_varation'] * tv_loss
                )
                L_G = L_syn + config.loss['weight_cross_entropy'] * cross_entropy_loss
            scaler_G.scale(L_G).backward()
            scaler_G.step(optimizer_G)
            scaler_G.update()

            # 5) Logging & save
            if step % config.train['log_step'] == 0:
                new_t = time.time()
                print(
                    f"epoch {epoch} , step {step}/{len(dataloader)} , "
                    f"adv_D_loss {adv_D_loss.item():.3f} , gp_loss {gp_loss.item():.3f} , "
                    f"G_loss {L_G.item():.3f} , pixelwise {pixelwise_loss.item():.3f} , "
                    f"local {pixelwise_local_loss.item():.3f} , symmetry {symmetry_loss.item():.3f} , "
                    f"adv_G {adv_G_loss.item():.3f} , ip {ip_loss.item():.3f} , tv {tv_loss.item():.3f} , "
                    f"ce {cross_entropy_loss.item():.3f} , {config.train['log_step']*config.train['batch_size']/(new_t-t):.1f} imgs/s"
                )
                tb.add_image_grid("grid/predict", 4, img128_fake.data.float()/2.+0.5,
                                  epoch*len(dataloader)+step, 'train')
                tb.add_image_grid("grid/frontal", 4, batch['img_frontal'].data.float()/2.+0.5,
                                  epoch*len(dataloader)+step, 'train')
                tb.add_image_grid("grid/profile", 4, batch['img'].data.float()/2.+0.5,
                                  epoch*len(dataloader)+step, 'train')
                tb.add_image_grid("grid/local", 4, local_predict.data.float()/2.+0.5,
                                  epoch*len(dataloader)+step, 'train')
                tb.add_image_grid("grid/local_input",4, local_input.data.float()/2.+0.5,
                                  epoch*len(dataloader)+step,'train')
                t = new_t

        save_model(G, tb.path, epoch)
        save_model(D, tb.path, epoch)
        save_optimizer(optimizer_G, G, tb.path, epoch)
        save_optimizer(optimizer_D, D, tb.path, epoch)
        print(f"Save done at {tb.path}")
