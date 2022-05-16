from cProfile import label
import numpy as np
from collections import defaultdict
import os
from glob import glob
import cv2
import torch.utils.data as data
import librosa
import torch
import pickle


def save_obj(obj, name):
    with open(name, 'wb') as f:
        pickle.dump(obj, f)


def load_obj(name):
    with open(name, 'rb') as f:
        return pickle.load(f)

def process_labels():
    f = open("./CAL500_labels.txt","r")
    song_dict = defaultdict(list)
    type_list = []
    for line in f:
        song = line.split("\t")[0]
        if "\n" in line.split("\t")[-1]:
            type = line.split("\t")[-1][:-1]
        else:
            type = line.split("\t")[-1]
        song_dict[song].append(type)
        if type not in type_list:
            type_list.append(type)
    
    num_types = len(type_list)
    label_dict = defaultdict(list)

    for cur_song in song_dict:
        cur_types = song_dict[cur_song]
        cur_labels = [0 for i in range(num_types)]
        for cur_type in cur_types:
            idx = type_list.index(cur_type)
            cur_labels[idx] = 1
        label_dict[cur_song] = cur_labels
    save_obj(song_dict,"./tools/song_dict")
    save_obj(label_dict,"./tools/label_dict")
    save_obj(type_list,"./tools/types")

def prepare_musics_and_labels(label_d):
    musics = []
    labels = []
    sampleRates = []
    i = 300
    for file in glob('./dataset/*WAV'):
        i -= 1
        if i <= 0:
            break
        m1,sr1 = librosa.load(file)
        song_name = file.split("\\")[-1]
        song_name = song_name.split(".WAV")[0]
        label = label_d[song_name]
        if label ==[]:
            print("warning:wrong dict!")
        musics.append([song_name,m1])
        sampleRates.append([song_name,sr1])
        labels.append([song_name,label])
    
    save_obj(musics,"./dataset/3_musics")
    save_obj(labels,"./dataset/3_labels")
    save_obj(sampleRates,"./dataset/3_sampleRates")


def rectify_songs():
    for file in glob('./dataset/*'):
        k = file.split("\\")[-1]
        name1 = k.split(".WAV")[0]
        name1 = name1[:-2] + ".WAV"
        # print(name1)
        new_name = os.path.join("./dataset",name1)
        os.rename(file,new_name)

    
        


class MyDataset(data.Dataset):
    def __init__(self,raw_musics,raw_labels,transform,resize_height,resize_width,n_fft=256,hop_length=64,sample_rate=16000,win_len=256):
        self.raw_musics = raw_musics
        self.labels = raw_labels
        self.transform = transform
        self.resize_height = resize_height
        self.resize_width = resize_width
        self.stft_musics = []
        self.n_fft = n_fft
        self.win_len = win_len
        self.hop_len = hop_length
        self.sample_rate = sample_rate
        
        # transform all the music into STFT format
        self.transformSTFT()
        
    def normalizePic(self,inputPic):
        return (inputPic - inputPic.mean(axis=0)) / (inputPic.std(axis=0) + 2e-12)

    def transformSTFT(self):
        for curMusic in self.raw_musics:
            # get the STFT result
            curPic = librosa.stft(curMusic,n_fft=self.n_fft,hop_length=self.hop_len,win_length=self.win_len,window=np.hamming)
            magnitude,phase = librosa.magphase(curPic)
            # normalize the magnitude of STFT
            curPic = self.normalizePic(magnitude.T)
            # normalize to 0-255
            stdPic = (curPic - curPic.min() + 10e-6) / (curPic.max()-curPic.min()+ 10e-6)
            curPic = stdPic * (255 - 0) + 0
            
            curPic = cv2.resize(curPic,(self.resize_height,self.resize_width))
            self.stft_musics.append(curPic)
        
    def transformOneSTFT(self,curMusic):
            # get the STFT result
            curPic = librosa.stft(curMusic, n_fft=self.n_fft, hop_length=self.hop_len,
                                  win_length=self.win_len, window=np.hamming)
            magnitude, phase = librosa.magphase(curPic)
            # normalize the magnitude of STFT
            curPic = self.normalizePic(magnitude.T)
            # normalize to 0-255
            stdPic = (curPic - curPic.min() + 10e-6) / \
                (curPic.max()-curPic.min() + 10e-6)
            curPic = stdPic * (255 - 0) + 0

            curPic = cv2.resize(
                curPic, (self.resize_height, self.resize_width))
            return curPic
    
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, index: int):
        curLabel = self.labels[index]
        curImage = self.stft_musics[index]
        if self.transform is not None:
            curImage = self.transform(curImage)
            curLabel = torch.from_numpy(np.array(curLabel))
            curLabel = torch.eye(2)[curLabel.long(), :]

        else:
            curImage = torch.from_numpy(curImage)
            curLabel = torch.from_numpy(np.array(curLabel))
            curLabel = torch.eye(2)[curLabel.long(), :]            
        return curImage,curLabel
    





