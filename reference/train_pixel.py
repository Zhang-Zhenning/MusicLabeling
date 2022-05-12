import os
import sys
import argparse
import random
from typing import Final
import numpy as np
from numpy.core.fromnumeric import repeat
from numpy.lib.npyio import savez_compressed
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
from torch.autograd import Variable
import torch.nn.functional as F
from utils import DataLoader, DataLoaderNPY, DataLoaderX
from datetime import datetime
from compute_LR import store_NLL, compute_NLL
from sklearn import metrics
from scipy import interpolate
from tqdm import tqdm
import scipy.optimize
from sklearn.metrics import f1_score
import matplotlib.pyplot as plt


def KL_div(mu,logvar,reduction = 'avg'):
    mu = mu.view(mu.size(0),mu.size(1))
    logvar = logvar.view(logvar.size(0), logvar.size(1))
    # a = torch.FloatTensor([2]).to(device)
    if reduction == 'sum':
        return -0.5 * torch.nansum(1 + logvar - mu.pow(2) - logvar.exp()) 
    else:
        KL = -0.5 * torch.nansum(1 + logvar - mu.pow(2) - logvar.exp(),1)
        # max_kl = 5e4
        # KL = torch.clip(KL, max=max_kl) # limit the maximum KL
        return KL

def perturb(x, mu,device):
    b,c,h,w = x.size()
    mask = torch.rand(b,c,h,w)<mu
    mask = mask.float().to(device)
    noise = torch.FloatTensor(x.size()).random_(0, 256).to(device)
    x = x*255
    perturbed_x = ((1-mask)*x + mask*noise)/255.
    return perturbed_x

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    # torch.backends.cudnn.deterministic = True

def val(netE, netG, val_batch, repeat=200):
    netE.eval()
    netG.eval()
    NLL_val = []

    for i, (xi, file_name) in tqdm(enumerate(val_batch)):
        x = xi.expand(repeat,-1,-1,-1).contiguous()
        weights_agg  = []
        with torch.no_grad():
            for _ in range(5):
                
                x = x.to(device)
        
                [z,mu,logvar] = netE(x)
                recon = netG(z)
                mu = mu.view(mu.size(0),mu.size(1)) # 200,100,1,1 -> 200,100
                logvar = logvar.view(logvar.size(0), logvar.size(1))
                z = z.view(z.size(0),z.size(1))
                weights = store_NLL(x, recon, mu, logvar, z, opt.repeat, opt.weight_switch, opt.h, opt.w, opt.high_freq, opt.low_freq)
            
                weights_agg.append(weights)
            
            weights_agg = torch.stack(weights_agg).view(-1) 
            
            NLL_loss_before = compute_NLL(weights_agg) 
            NLL_val = np.append(NLL_val, NLL_loss_before.detach().cpu().numpy())
        # if i>100:break

    return NLL_val

def score(NLL_indist, NLL_ood):
    combined = np.concatenate((NLL_indist, NLL_ood))
    label_1 = np.ones(len(NLL_indist))
    label_2 = np.zeros(len(NLL_ood))
    label = np.concatenate((label_1, label_2))
    '''eer'''
    fpr, tpr, thresholds = metrics.roc_curve(label, combined, pos_label=0)
    rocauc = metrics.auc(fpr, tpr)

    def func(x):
        return 1. - x - interpolate.interp1d(fpr, tpr)(x) # 1-roc-x,减去x是为了把x,y相等的点转化为y=0，然后通过求root的方法求出
    
    x_dis = np.arange(0, 1, 0.01)
    y_dis = func(x_dis)
    final_eer =scipy.optimize.brentq(func, 0, 1)

    final_threshold = interpolate.interp1d(fpr,thresholds)(final_eer)
    new_label = label.copy()

    for i in range(len(new_label)):
        if new_label[i] == 1:
            new_label[i] = 0
        else:
            new_label[i] = 1

    new_combine = np.zeros(combined.shape)
    for idx in range(len(new_combine)):
        if combined[idx] >= final_threshold:
            new_combine[idx] = 1

    final_f1_score = f1_score(new_label,new_combine)

    return rocauc, final_eer, final_f1_score

def plot(ll_indist, ll_ood, rocauc, final_eer, final_f1_score, epoch, save_path):
    ll_indist = -ll_indist
    ll_ood = -ll_ood
    os.makedirs(save_path, exist_ok=True)
    min_ll = -50000
    max_ll = max(ll_indist.max(), ll_ood.max())
    bins_ll = np.linspace(min_ll, max_ll, 100)

    plt.hist(ll_indist, bins_ll, alpha=0.5, label='In-distribution')
    plt.hist(ll_ood, bins_ll, alpha=0.5, label='OOD')

    plt.legend(loc='upper right')
    plt.title(f'NLL  AUC: {round(rocauc,4)} EER: {round(final_eer, 4)} F1: {round(final_f1_score, 4)}')
    plt.savefig(save_path + "train_valid_{}.png".format(epoch))
    plt.clf()

if __name__=="__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--dataroot', default='/home/lxf/Documents/xinfeng', help='path to dataset')
    parser.add_argument('--workers', type=int, help='number of data loading workers', default=8)
    parser.add_argument('--batchSize', type=int, default=64, help='input batch size')
    parser.add_argument('--imageSize', type=int, default=64, help='the height / width of the input image to network') # 这个可能要用256*256替代，原本是32
    parser.add_argument('--nc', type=int, default=1, help='input image channels')
    parser.add_argument('--nz', type=int, default=32, help='size of the latent z vector')
    parser.add_argument('--ngf', type=int, default=16, help = 'hidden channel size')
    parser.add_argument('--niter', type=int, default=200, help='number of epochs to train for')
    parser.add_argument('--lr', type=float, default=3e-4, help='learning rate')

    parser.add_argument('--beta1', type=float, default=0.9, help='beta1 for adam. default=0.9')
    parser.add_argument('--beta', type=float, default=2, help='beta for beta-vae')

    parser.add_argument('--ngpu'  , type=int, default=2, help='number of GPUs to use')
    parser.add_argument('--gpus', type=str, default='1', help='gpus')
    parser.add_argument('--experiment', default="models/dolphin-split", help='Where to store samples and models')
    parser.add_argument('--perturbed', action='store_true', default=False, help='Whether to train on perturbed data, used for comparing with likelihood ratio by Ren et al.')
    parser.add_argument('--ratio', type=float, default=0.2, help='ratio for perturbation of data, see Ren et al.')

    parser.add_argument('--h', type=int, default=64, help='height of input images')
    parser.add_argument('--w', type=int, default=64, help='width of input images')
    parser.add_argument('--c', type=int, default=1, help='channel of input images')
    parser.add_argument('--t_length', type=int, default=1, help='length of the frame sequences')
    parser.add_argument('--exp_dir', type=str, default='log', help='directory of log')
    parser.add_argument('--record', type=str, default='True', help='whether log is needed')
    parser.add_argument('--valid_mode', type=str, default='False', help='enable valid mode or not.')
    parser.add_argument('--high_freq', type=float, default=0.45, help='the weight of high freq in loss')
    parser.add_argument('--low_freq', type=float, default=0.35, help='the weight of low freq in loss')
    parser.add_argument('--weight_switch', type=str, default='False', help='whether diy weight of loss')
    parser.add_argument('--tune_mode', type=str, default='False', help='fine-tuning mode or training mode')
    parser.add_argument('--npy', action='store_true', default=False, help='whether use npy instead of png')
    parser.add_argument('--recl', type=str, default='ce', help='use ce or mse as reconstruction loss')
    parser.add_argument('--keepsize', action='store_true', default=False, help='whether keep size before entering the network')
    parser.add_argument('--sample_rate', type=int, default=256, help='sample_rate of the value')
    parser.add_argument('--rm_content', type=str, default="False", help='remove confusing content like NiHaoYoYo')
    parser.add_argument('--low_freq_repeat', default='False', help="repeat low freq sub-bands")
    parser.add_argument('--train_sample', type=float, default=1, help="sample rate of train set")

    opt = parser.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = opt.gpus
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    opt.ngpu = len(opt.gpus)
    opt.manualSeed = 1234 # random.randint(1, 10000) # fix seed
    setup_seed(opt.manualSeed)
    cudnn.benchmark = True

    assert opt.recl=='ce' or opt.recl=='mse', 'the reconstruction loss is invalid'
    if opt.low_freq_repeat == 'True': print('Attention: Low frequency will be repeated about 10 times')
    if opt.tune_mode == 'True':
        import models.DCGAN_VAE_pixel_tune as DVAE
    elif opt.recl == 'mse':
        if opt.keepsize == False:
            import models.DCGAN_VAE_pixel_mse as DVAE
        if opt.keepsize == True:
            import models.DCGAN_VAE_pixel_mse_keepsize as DVAE
    elif opt.recl == 'ce':
        if opt.keepsize == False:
            import models.DCGAN_VAE_pixel as DVAE
        if opt.keepsize == True:
            import models.DCGAN_VAE_pixel_ce_keepsize as DVAE

    now = '{}'.format(datetime.now().strftime("%m%d_%H%M"))
    if opt.record == 'True':
        log_dir = os.path.join('./exp', 'img-lab2-kaggle',
            '_'.join((opt.exp_dir, 'h'+str(opt.h), 'w'+str(opt.w), 'bs'+str(opt.batchSize), 'nz'+str(opt.nz), 'ngf'+str(opt.ngf), 'lr'+str(opt.lr))),
            now)
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
        # Recording Logs Start!
        orig_stdout = sys.stdout
        f = open(os.path.join(log_dir, f'log_{now}.txt'),'w')
        sys.stdout= f

    if opt.experiment is None:
        raise RuntimeError('model path undefined!')

    if opt.low_freq_repeat == 'True': print('Attention: Low frequency will be repeated about 10 times')
    print("Random Seed: ", opt.manualSeed)
    print(f'是否开启频域分段attention:{opt.weight_switch}, 是否开启验证模式:{opt.valid_mode}')

    '''定义训练集'''
    # train_folder = opt.dataroot+"/dataset/VCTK/img_VCTK/stft_1_5s_35db_dns_nfft256_winlen256_hop64/train-player"
    if opt.npy == False:
        train_folder = os.path.join( 
            opt.dataroot, 
            "oppo-defense/kaggle_dataset/fluent_speech_commands_dataset/img_kaggle/0809_Kaggle_test3_stft_1_5s_35db_nfft256_winlen256_hop64_cmvn"
        )
        print("train_folder:", train_folder)
        
        print(vars(opt))
        # Loading dataset
        train_dataset = DataLoader(train_folder, transforms.Compose([
                    transforms.ToTensor(),
                    ]), 
                    low_freq_repeat = opt.low_freq_repeat,
                    resize_height=opt.h, resize_width=opt.w, time_step=opt.t_length-1, img_channel=opt.c, keepsize=opt.keepsize, sample_rate=opt.sample_rate)
        # train_dataset2 = DataLoaderNPY(train_folder2, transforms.Compose([
        #             transforms.ToTensor(),          
        #             ]), resize_height=opt.h, resize_width=opt.w, time_step=opt.t_length-1, img_channel=opt.c)
        # train_dataset += train_dataset2
        if opt.keepsize==True:
            opt.h = train_dataset[0][0].shape[1]
            opt.w = train_dataset[0][0].shape[2]
    elif opt.npy == True:    
        train_folder = opt.dataroot+"/oppo-defense/img_dolphin/img_kaggle/0809_Kaggle_test3_stft_1_5s_35db_nfft256_winlen256_hop64_cmvn"
        print("train_folder:", train_folder)
        
        print(vars(opt))
        # Loading dataset
        train_dataset = DataLoaderNPY(train_folder, transforms.Compose([
                    transforms.ToTensor(),          
                    ]),
                    low_freq_repeat = opt.low_freq_repeat,
                    resize_height=opt.h, resize_width=opt.w, time_step=opt.t_length-1, img_channel=opt.c, keepsize=opt.keepsize, sample_rate=opt.sample_rate)
        if opt.keepsize==True:
            opt.h = train_dataset[0][0].shape[1]
            opt.w = train_dataset[0][0].shape[2]

    train_size = len(train_dataset)
    num_samples = int(train_size * opt.train_sample)
    train_dataset, _ = torch.utils.data.random_split(train_dataset, [num_samples, train_size-num_samples])
    train_size = len(train_dataset)
    print(train_size)
    train_batch = torch.utils.data.DataLoader(train_dataset, batch_size = opt.batchSize, 
                              shuffle=True, num_workers=opt.workers)

    '''定义验证集'''
    if opt.valid_mode == 'True':
        valid_indist = opt.dataroot+"/oppo-defense/img_dolphin/img_usslab2/image_stft_1_5s_35db_dns_before/valid-player"
        valid_ood = opt.dataroot+"/oppo-defense/img_dolphin/img_usslab4/image_stft_1_5s_35db_dns_before/valid-player"
        print(f'valid_indist: {valid_indist}, valid_ood: {valid_ood}')
        valid_indist_dataset = DataLoader(valid_indist, transforms.Compose([
                    transforms.ToTensor(),          
                    ]),
                    rm_content = opt.rm_content, low_freq_repeat = opt.low_freq_repeat,
                    resize_height=opt.h, resize_width=opt.w, time_step=opt.t_length-1, img_channel=opt.c)
        valid_ood_dataset = DataLoader(valid_ood, transforms.Compose([
                    transforms.ToTensor(),          
                    ]), 
                    rm_content = opt.rm_content, low_freq_repeat = opt.low_freq_repeat,
                    resize_height=opt.h, resize_width=opt.w, time_step=opt.t_length-1, img_channel=opt.c)
        valid_indist_batch = torch.utils.data.DataLoader(valid_indist_dataset, batch_size = 1, 
                              shuffle=False, num_workers=0)
        valid_ood_batch = torch.utils.data.DataLoader(valid_ood_dataset, batch_size = 1, 
                              shuffle=False, num_workers=0)
        
    ngpu = int(opt.ngpu)
    nz = int(opt.nz)
    ngf = int(opt.ngf)
    nc = int(opt.nc)
    sample_rate = opt.sample_rate

    # custom weights initialization called on netG and netD
    def weights_init(m):
        classname = m.__class__.__name__
        if classname.find('Conv') != -1:
            m.weight.data.normal_(0.0, 0.02)
        elif classname.find('BatchNorm') != -1:
            m.weight.data.normal_(1.0, 0.02)
            m.bias.data.fill_(0)

    netG = DVAE.DCGAN_G(opt.h,opt.w, nz, nc, ngf, ngpu, sample_rate = sample_rate)
    print(netG)
    print('Total params: %.2fM' % (sum(p.numel() for p in netG.parameters()) / 1048576.0))
    netG.apply(weights_init)

    netE = DVAE.Encoder(opt.h,opt.w, nz, nc, ngf, ngpu)
    print(netE)
    print('Total params: %.2fM' % (sum(p.numel() for p in netE.parameters()) / 1048576.0))
    netE.apply(weights_init)

    
    netE.to(device) # why in the code, there're some classification questions? I doubt that 
    netG.to(device)

    # setup optimizer
    
    optimizer1 = optim.Adam(netE.parameters(), lr=opt.lr, weight_decay = 3e-5)
    optimizer2 = optim.Adam(netG.parameters(), lr=opt.lr, weight_decay = 3e-5)

    netE.train()
    netG.train()

    if opt.recl == 'ce':
        loss_fn = nn.CrossEntropyLoss(reduction = 'none')
    elif opt.recl == 'mse':
        loss_fn = nn.MSELoss(reduction = 'none')
    rec_l = []
    kl = []
    for epoch in range(opt.niter):
        for i, (x, file_name) in enumerate(train_batch):
            x = x.to(device) # shape 8*1*256*256
            if opt.perturbed:
                x = perturb(x, opt.ratio, device)
            b = x.size(0)
            if opt.recl == 'ce':
                target = Variable(x.data.view(-1) * (sample_rate-1)).long()   # 知道原因了，这个target是负数了，应该是正数才对。像MNIST中的数据都是正数，最小是0
            elif opt.recl == 'mse':
                target = Variable(x.data.view(-1))
            [z,mu,logvar] = netE(x)
            recon = netG(z)
            
            recon = recon.contiguous()
            if opt.recl == 'ce':
                recon = recon.view(-1,sample_rate)
            elif opt.recl == 'mse':
                recon = recon.view(-1)


            # print("recon size is:",recon.size())
 # -------------------split the recl into three different-weighted parts---------------         
            if opt.weight_switch == 'True':       
                recl = loss_fn(recon,target)
                recl = torch.nansum(recl.view(b, opt.h, -1), 2)
                header_weight = torch.ones(b, 30) * opt.high_freq
                body_weight = torch.ones(b, opt.h - 30 - 1) * (1-opt.high_freq-opt.low_freq)
                tail_weight = torch.ones(b, 1) * opt.low_freq
                weight = torch.cat((header_weight,body_weight,tail_weight),1).to(device)
                weight.requires_grad = False
                recl = recl * weight
                recl = torch.nansum(recl) / b
            
            else:
                recl = loss_fn(recon,target)
                recl = torch.nansum(recl) / b


            kld = KL_div(mu,logvar)
            loss =  recl + opt.beta*kld.mean()
            
            optimizer1.zero_grad()
            optimizer2.zero_grad()
            total_loss = loss
            loss.backward(retain_graph=True)
            
            optimizer1.step()
            optimizer2.step()
            rec_l.append(recl.detach().item())
            kl.append(kld.mean().detach().item())

        
        print('epoch:{} recon:{} kl:{}'.format(epoch,np.mean(rec_l),np.mean(kl)))
        if opt.record == 'True':
            save_path = f'{log_dir}/{opt.experiment}/'
            os.makedirs(save_path, exist_ok=True)
            # if epoch>120 and (epoch+1)%10==0:
            if opt.valid_mode == 'True':
                NLL_indist = val(netE, netG, valid_indist_batch, repeat=200)
                NLL_ood = val(netE, netG, valid_ood_batch, repeat=200)
                rocauc, final_eer, final_f1_score = score(NLL_indist, NLL_ood)
                print(f'rocauc:{round(rocauc,4)}, final_eer:{round(final_eer,4)}, final_f1_score:{round(final_f1_score,4)}')
                plot(NLL_indist, NLL_ood, rocauc, final_eer, final_f1_score, epoch, f'{log_dir}/valid_figures/')
            
            if epoch>120 and (epoch+1)%10==0:
                torch.save(netG.state_dict(), save_path+f'netG_pixel_{epoch+1}.pth')
                torch.save(netE.state_dict(), save_path+f'netE_pixel_{epoch+1}.pth')
    # Recording Logs End!
    if opt.record == 'True':
        sys.stdout = orig_stdout
        f.close()
