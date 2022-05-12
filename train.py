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
    return torch.unsqueeze(in_tensor,dim=1)


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
    parser.add_argument('--train',type=boolean,default=False,help='whether we train in this round')
    parser.add_argument('--log',type=boolean,default=True,help='whether we logger in this round')
    parser.add_argument('--testMusicPath',type=str,default="./dataset/who-bargain.WAV",help='the music we would like to test')
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
  
    

    
    # ------------------------------------setup dataloader---------------------------------
    if commands.train == True:
        musics = load_obj("./dataset/0_musics")
        labels = load_obj("./dataset/0_labels")
        num_musics = len(musics)

        train_ratio = 0.8
        ran = random.sample(range(0, num_musics),int(train_ratio*num_musics) )

        train_labels,train_musics,test_labels,test_musics = [],[],[],[]
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
        train_dataloader = DataLoader(train_dataset,batch_size=commands.batchSize,shuffle=False,num_workers=commands.workers)
    
        test_dataset = MyDataset(test_musics, test_labels,  transforms.Compose([
            transforms.ToTensor(), ]), commands.h, commands.w)
        test_dataloader = DataLoader(
            test_dataset, batch_size=commands.batchSize, shuffle=False, num_workers=commands.workers)
    

    
    # ---------------------------------------Training--------------------------------------
    
    if commands.train == True:
        
        now = '{}'.format(datetime.now().strftime("%m%d_%H%M"))

        model = BasicModel(commands.h,commands.w,commands.in_channel,commands.output_size)
        optimizer = optim.Adam(model.parameters(), lr=commands.lr, weight_decay=3e-5)
        
        # start logger
        if commands.log ==True:
            orig_stdout = sys.stdout
            log_dir = os.path.join('./loggers',
                                   '_'.join(('h'+str(commands.h), 'w'+str(commands.w), 'bs'+str(
                                       commands.batchSize), 'in_channel'+str(commands.in_channel), 'epo'+str(commands.epoch),'lr'+str(commands.lr))),
                                   now)
        
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
        
        orig_stdout = sys.stdout
        f = open(os.path.join(log_dir, f'log.txt'),'w')
        sys.stdout = f


        # print basic info
        print("---------------------------TRIANING MODE----------------------------")
        print("cur device is: ",device)
        print(model)
        # print('Total param: %.2fM' % (sum(p.numel() for p in model.parameters()) / 1048576.0))

        # model initialization
        model.apply(init_weights)
        model.to(device)
        model.train()
        
        # start training
    
        print("---------------------------START TRIANING----------------------------")
        for cur_epoch in range(commands.epoch):
            total_loss = 0
            num_loss = 0
            for idx,(cur_data,cur_label) in enumerate(train_dataloader):
                cur_data = cur_data.to(device)
                
                cur_label = torch.squeeze(cur_label)
                cur_label = cur_label.to(device)

                predict_output = model(cur_data)

                cur_loss = model.loss(torch.squeeze(predict_output).float(),torch.squeeze(cur_label).float())
                optimizer.zero_grad()
                cur_loss.backward(retain_graph=True)
                optimizer.step()

                total_loss += cur_loss.detach().item()
                num_loss += 1

            
            print('epoch:{0} trainLoss:{1}'.format(cur_epoch,total_loss/num_loss))
            model_dir = os.path.join(log_dir, 'models')
            if not os.path.exists(model_dir):
                   os.makedirs(model_dir)
            torch.save(model, os.path.join(model_dir, f'BasicModel_{cur_epoch+1}.pth'))

            if cur_epoch % 10 == 0:
                test_loss = 0
                test_num = 0
                # now we need to test the intermediate result
                for idx,(cur_data,cur_label) in enumerate(test_dataloader):
                    cur_data = cur_data.to(device)
                    cur_label = torch.squeeze(cur_label)
                    cur_label = cur_label.to(device)

                    predict_output = model(cur_data)

                    cur_loss = model.loss(torch.squeeze(
                        predict_output).float(), torch.squeeze(cur_label).float())
                    test_loss += cur_loss.detach().item()
                    test_num += 1
                
                print('TESTING: epoch:{0} testLoss:{1}'.format(cur_epoch,test_loss/test_num))



                
        
        if commands.log == True:
            sys.stdout = orig_stdout

    
    # ---------------------------------------Testing--------------------------------------
    else:
        model = torch.load(commands.evalModelPath)
        device = "cpu"
        model.to(device)
        eval_music,eval_sr = librosa.load(commands.testMusicPath)
        eval_dataset = MyDataset([eval_music], [1],  transforms.Compose([
            transforms.ToTensor(), ]), commands.h, commands.w)
        eval_dataloader = DataLoader(
            eval_dataset, batch_size=commands.batchSize, shuffle=False, num_workers=commands.workers)

        # setup testing logger
        if commands.log == True:
            now = '{}'.format(datetime.now().strftime("%m%d_%H%M"))
            orig_stdout = sys.stdout

            f = open(os.path.join(commands.resultPath, f'{now}_result.txt'), 'w')
            sys.stdout = f

        
        print("-----------------------------TESTING MODE------------------------------")
        print(device)
        print(model)

        # start evaluating
        for (eval_batch,eval_label) in eval_dataloader:
            with torch.no_grad():
                eval_batch = eval_batch.to(device)
                eval_result = torch.squeeze(model(eval_batch)).detach().cpu().numpy()

                print(eval_result)

        print("----------------------------TESTING FINISH-----------------------------")

        if commands.log == True:
            sys.stdout = orig_stdout
        





        







