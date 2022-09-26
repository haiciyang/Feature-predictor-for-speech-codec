import os
import glob
import torch
import random
import librosa
import numpy as np
# from config import ex

import torchaudio.transforms as T
from torch.utils.data import Dataset, DataLoader

from ceps2lpc_vct import ceps2lpc_v


class Libri_lpc_data_retrain(Dataset):
    
    # @ex.capture
    def __init__(self): # 28539
        
        '''
        # training - 28539
        # testing - 2703
        # One "chunk" corresponds to 2400 samples, 15 frames
        # qtz: 
            -1: return all unquantized features
             0: return unquantized ceps features and quantized pitches
             1: return all quantized features     
        '''
        
        self.maxi = 24.1
        
        self.feat_path = '/media/sdb1/haici/libri_qtz_ft/1_2048_large_17'
        self.sample_path = '/media/sdb1/Data/librispeech/train-clean-100/*/*/*.wav'
        # self.orig_feat_path = '/media/sdb1/Data/libri_lpc_qtz_pt/train/'

        self.files = glob.glob(self.feat_path+'/*')

        print('Using original data')
        
    def __len__(self):
        
        return len(self.files)
    
    def __getitem__(self, idx):
        
        eps = 1e-10
        
        feat_path = self.files[idx]
        sample_name = feat_path.split('/')[-1][:-4]
        # n1, n2, n3 = sample_name.split('-')
        
        # sample_path = '/media/sdb1/Data/librispeech/train-clean-100/{}/{}/{}.wav'.format(n1, n2, sample_name)
        sample_path = '/media/sdb1/Data/libri_lpc_pt/train/{}_in_data.pt'.format(sample_name)

        features = torch.tensor(np.load(feat_path)) # (10, 19, 36)
        features = torch.reshape(features[:,2:-2, :], (-1, 36)) # (150, 36)
        
        # -- Load data -- 
        # in_data, sr = librosa.load(sample_path, sr=None)
        in_data = torch.load(sample_path) # (nb_frames, 2400, 1)
        in_data = in_data/max(np.abs(in_data).max(),eps) * 0.999 # (2400,)
        
        i = 5
        x = torch.reshape(in_data[i:i+10], (10*2400,))
        x = x.unsqueeze(0) #(1, 2400*n)
        
        return sample_name, x, features
        
        
        
        
        
         
        