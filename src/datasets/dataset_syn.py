import os
import glob
import torch
import random
import librosa
import numpy as np
# from config import ex
import torchaudio.transforms as T
from torch.utils.data import Dataset, DataLoader


class Libri_lpc_data_syn(Dataset):
    
    # @ex.capture
    def __init__(self, chunks=1): # 28539
        
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
        self.chunks = chunks
        

        path = '/data/hy17/librispeech/librispeech/dev-clean/*/*/*.wav'
        
        self.files = glob.glob(path)
        
        self.feature_qtz_folder = '/data/hy17/librispeech/libri_lpc_qtz_pt/val/'
        self.feature_folder = '/data/hy17/librispeech/libri_lpc_pt/val/'

        print('Using original data')
        
    def __len__(self):
        
        return len(self.files)
    
    def __getitem__(self, idx):
        
        eps = 1e-10
    
        file_path = self.files[idx] # /media/sdb1/Data/librispeech/train-clean-100/103/1240/103-1240-0000.wav
        sample_name = file_path.split('/')[-1][:-4] # 103-1240-0000
        
        # -- Load data -- 
        in_data, sr = librosa.load(file_path, sr=None)
        in_data = in_data/max(np.abs(in_data).max(),eps) * 0.999 # (2400,)
        
        nb_frames = len(in_data) // 2400
        
        # -- Load features --
        qtz_feat_path = self.feature_qtz_folder + sample_name + '_features.pt'
        feat_path = self.feature_folder + sample_name + '_features.pt'
            
        qtz_features = torch.load(qtz_feat_path) # (19, 36)
        qtz_features = qtz_features[:nb_frames]
        
        features = torch.load(feat_path) # (19, 36)
        features = features[:nb_frames]
        
        features[:,:,-2:] = qtz_features[:,:,-2:]
        # features[:,:,0] = qtz_features[:,:,0]
        

        if self.chunks == 0: # pass all the data to dataloader
            self.chunks = nb_frames
            
        while nb_frames < self.chunks:
            in_data = np.hstack((in_data, in_data))
            qtz_features = torch.vstack((qtz_features, qtz_features))
            features = torch.vstack((features, features))
            nb_frames *= 2
        
        # print(in_data.shape)
        in_data = torch.tensor(np.reshape(in_data[:len(in_data) // 2400 * 2400], (-1, 2400)))
        
        i = nb_frames-self.chunks if nb_frames > self.chunks else 0
        

        x = torch.reshape(in_data[i:i+self.chunks], (self.chunks*2400,))
        feat = torch.reshape(features[i:i+self.chunks, 2:-2, :], 
                                 (self.chunks*15, -1)) # [n*15, 36]
        qtz_feat = torch.reshape(qtz_features[i:i+self.chunks, 2:-2, :], 
                                 (self.chunks*15, -1)) # [n*15, 36]
                
        nm_feat = feat / self.maxi
        
        x = x.unsqueeze(0) #(1, 2400)
        # feat - (N, 36)
        
        return sample_name, x, nm_feat, qtz_feat
        
        
        
        
        
         
        