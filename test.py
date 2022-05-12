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
from utils import *
import random



if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    
    parser.add_argument('--h', type=int, default=256,
                        help='height of input images')
    parser.add_argument('--w', type=int, default=256,
                        help='width of input images')


    parser.add_argument('--train', type=boolean, default=False,help='whether we train in this round')
    parser.add_argument('--log', type=boolean, default=True,help='whether we logger in this round')
    parser.add_argument('--testMusicPath', type=str, default="./dataset/who-bargain.WAV",help='the music we would like to test')
    parser.add_argument('--resultPath', type=str,
                        default="./loggers/eval_result", help='the path we would like to store the eval result')
    parser.add_argument('--evalModelPath', type=str,
                        default="./loggers/h256_w256_bs20_in_channel1_epo100_lr0.0003/0512_0147/models/BasicModel_20.pth", help='the model we would like to use')

    # --------------------------------------parse config-----------------------------------
    commands = parser.parse_args()

    
    commands.output_size = 97

    # ------------------------------------setup dataloader---------------------------------
    

    # ---------------------------------------Testing--------------------------------------
   
    model = torch.load(commands.evalModelPath)
    device = "cpu"
    model.to(device)
    eval_music, eval_sr = librosa.load(commands.testMusicPath)
    eval_dataset = MyDataset([eval_music], [1],  transforms.Compose([
        transforms.ToTensor(), ]), commands.h, commands.w)
    eval_dataloader = DataLoader(
        eval_dataset, batch_size=1, shuffle=False, num_workers=0)

    # setup testing logger
    if commands.log == True:
        now = '{}'.format(datetime.now().strftime("%m%d_%H%M"))
        orig_stdout = sys.stdout

        f = open(os.path.join(commands.resultPath,
                    f'{now}_result.txt'), 'w')
        sys.stdout = f

    print("-----------------------------TESTING MODE------------------------------")

    # start evaluating
    for (eval_batch, eval_label) in eval_dataloader:
        with torch.no_grad():
            eval_batch = eval_batch.to(device)
            eval_result = torch.squeeze(
                model(eval_batch)).detach().cpu().numpy()

            print(eval_result)

    print("----------------------------TESTING FINISH-----------------------------")

    if commands.log == True:
        sys.stdout = orig_stdout
