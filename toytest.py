from model import *

import os
import sys
import argparse
from xmlrpc.client import boolean
from utils import MyDataset
import torch
import torchvision.transforms as transforms
from datetime import datetime
import numpy as np
import random
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
from model import BasicModel
from utils import *
import torch.optim as optim
import random


def addDim(in_tensor):
    return torch.unsqueeze(in_tensor, dim=1)


def init_weights(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
          m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
          m.weight.data.normal_(1.0, 0.02)
          m.bias.data.fill_(0)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--workers', type=int,
                        help='number of data loading workers', default=0)
    parser.add_argument('--batchSize', type=int,
                        default=20, help='input batch size')
    parser.add_argument('--in_channel', type=int, default=1,
                        help='input channel')
    parser.add_argument('--output_size', type=int, default=18,
                        help='output vector length')
    parser.add_argument('--epoch', type=int, default=100,
                        help='number of epochs to train for')
    parser.add_argument('--lr', type=float, default=3e-4, help='learning rate')

    parser.add_argument('--h', type=int, default=256,
                        help='height of input images')
    parser.add_argument('--w', type=int, default=256,
                        help='width of input images')
    parser.add_argument('--record', type=str, default='True',
                        help='whether log is needed')
    parser.add_argument('--gpu', type=str, default='1', help='current gpu')
    parser.add_argument('--seed', type=int, default=1234, help='random seed')
    parser.add_argument('--train', type=boolean, default=True,help='whether we train in this round')
    parser.add_argument('--log', type=boolean, default=True,help='whether we logger in this round')
    parser.add_argument('--testMusicPath', type=str, default="./dataset/who-bargain.WAV",help='the music we would like to test')
    parser.add_argument('--resultPath', type=str,
                        default="./loggers/eval_result", help='the path we would like to store the eval result')
    parser.add_argument('--evalModelPath', type=str,
                        default="./loggers/h256_w256_bs20_in_channel1_epo100_lr0.0003/0512_0147/models/BasicModel_20.pth", help='the model we would like to use')

    # --------------------------------------parse config-----------------------------------
    commands = parser.parse_args()

    # setup random seed
    torch.manual_seed(commands.seed)
    torch.cuda.manual_seed_all(commands.seed)
    np.random.seed(commands.seed)
    random.seed(commands.seed)

    # setup device info
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    cudnn.benchmark = True

    types = load_obj("./tools/types")
    commands.output_size = len(types)

    blose = nn.BCELoss()

    # ------------------------------------setup dataloader---------------------------------
    if commands.train == True:
        musics = load_obj("./dataset/2_musics")
        labels = load_obj("./dataset/2_labels")
        num_musics = len(musics)

        train_ratio = 0.8
        ran = random.sample(range(0, num_musics), int(train_ratio*num_musics))

        train_labels, train_musics, test_labels,test_musics = [],[],[],[]
        for i in range(num_musics):
            if i in ran:
                train_labels.append(labels[i][1])
                train_musics.append(musics[i][1])
            else:
                test_labels.append(labels[i][1])
                test_musics.append(musics[i][1])
        del musics
        del labels

        train_dataset = MyDataset(train_musics, train_labels,  transforms.Compose([
            transforms.ToTensor(), ]), commands.h, commands.w)
        train_dataloader = DataLoader(train_dataset, batch_size=commands.batchSize, shuffle=False,num_workers=commands.workers)

        test_dataset = MyDataset(test_musics, test_labels,  transforms.Compose([
            transforms.ToTensor(), ]), commands.h, commands.w)
        test_dataloader = DataLoader(
            test_dataset, batch_size=commands.batchSize, shuffle=False, num_workers=commands.workers)
        
        l = BasicModel(256,256,1,97)
        i = 1
        loses = []
        for cur_data,cur_label in train_dataloader:
            i -= 1
            pred = l(cur_data)
            
            total_loss = None
            for j in range(cur_data.size()[0]):
                k = blose(pred[j],cur_label[j])
                if total_loss == None:
                    total_loss = k
                else:
                    total_loss += k
            loses.append(total_loss)
        
        print(loses)








