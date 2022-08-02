import os
import glob
import torch
import random
import librosa
import numpy as np
# from config import ex
import torchaudio.transforms as T
from torch.utils.data import Dataset, DataLoader


class Libri_lpc_data_orig(Dataset):
    
    # @ex.capture
    def __init__(self, task = 'train', chunks=1): # 28539
        
        # training - 28539
        # testing - 2703
        # One "chunk" corresponds to 2400 samples, 15 frames
        self.task = task
        if self.task == 'train':
            path = '/media/sdb1/Data/librispeech/train-clean-100/*/*/*.wav'
            self.feature_folder = '/media/sdb1/Data/libri_lpc_pt/train/'
        elif self.task == 'val':
            path = '/media/sdb1/Data/librispeech/dev-clean/*/*/*.wav'
            self.feature_folder = '/media/sdb1/Data/libri_lpc_pt/val/'
        
        self.files = glob.glob(path)
        self.chunks = chunks
        
        self.maxi = 24.1
        
        print('Using original data')
        
    def __len__(self):
        
        return len(self.files)
    
    def __getitem__(self, idx):
        
        eps = 1e-10
    
        file_path = self.files[idx] # /media/sdb1/Data/librispeech/train-clean-100/103/1240/103-1240-0000.wav
        sample_name = file_path.split('/')[-1][:-4] # 103-1240-0000
        feature_path = self.feature_folder + sample_name + '_features.pt'
        
        # in_data = torch.load(file_path) 
        in_data, sr = librosa.load(file_path, sr=None)
        in_data = in_data/max(np.abs(in_data).max(),eps) * 0.999 # (2400,)
        
        nb_frames = len(in_data) // 2400
        
        features = torch.load(feature_path) # (19, 36)
        features = features[:nb_frames]

        if self.chunks == 0: # pass all the data to dataloader
            self.chunks = nb_frames
            
        while nb_frames < self.chunks:
            in_data = np.hstack((in_data, in_data))
            features = torch.vstack((features, features))
            nb_frames *= 2
        
        # print(in_data.shape)
        in_data = torch.tensor(np.reshape(in_data[:len(in_data) // 2400 * 2400], (-1, 2400)))
        
        if self.task == 'train':
            i = np.random.choice(nb_frames-self.chunks) if nb_frames > self.chunks else 0
        elif self.task == 'val':
            i = nb_frames-self.chunks if nb_frames > self.chunks else 0
        
        while 1:
            x = torch.reshape(in_data[i:i+self.chunks], (self.chunks*2400,))
            mid_feat = torch.reshape(features[i:i+self.chunks, 2:-2, :], 
                                     (self.chunks*15, -1)) # [n*15, 36]
            feat = torch.cat([features[i,:2, :], mid_feat, features[i+self.chunks-1,-2:, :]], 0)
            if torch.abs(x).max() == 0 or torch.sum(torch.isnan(feat)) != 0:
                i = np.random.choice(nb_frames-self.chunks)
            else:
                break
                
        nm_feat = feat / self.maxi
        
        x = x.unsqueeze(0) #(1, 2400)
        # feat - (N, 36)
        
        return x, feat, nm_feat
        
        
        
        
        
         
        