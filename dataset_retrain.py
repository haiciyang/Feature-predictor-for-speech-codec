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
        
        self.feat_path = '/media/sdb1/haici/libri_qtz_ft/1_2048_large_17_new'
        self.sample_path = '/media/sdb1/Data/librispeech/train-clean-100/*/*/*.wav'
        # self.orig_feat_path = '/media/sdb1/Data/libri_lpc_qtz_pt/train/'

        self.files = glob.glob(self.feat_path+'/*')
        
        # self.files_f = glob.glob(self.feat_path+'/*')

        print('Using original data')
        
    def __len__(self):
        
        return len(self.files)
    
    def __getitem__(self, idx):
        
        eps = 1e-10
        
        feat_path = self.files[idx]
        sample_name = feat_path.split('/')[-1][:-4]
        n1, n2, n3 = sample_name.split('-')
        
        sample_path = '/media/sdb1/Data/librispeech/train-clean-100/{}/{}/{}.wav'.format(n1, n2, sample_name)

        # -- Load data -- 
        in_data, sr = librosa.load(sample_path, sr=None)
        in_data = in_data/max(np.abs(in_data).max(),eps) * 0.999 # (2400,)
        
        nb_frames = len(in_data) // 2400
        
        # -- Load features --
        
#         orig_feat_path = self.orig_feat_path + sample_name + '_features.pt'
#         # feat_path = self.feat_path + sample_name + '.pt'
            
#         orig_feat = torch.load(orig_feat_path) # (19, 36)
#         orig_feat = orig_feat[:nb_frames]
        
        features = torch.tensor(np.load(feat_path)) # (1, 150, 36)
        features = features[:,:,:-16] * self.maxi
        e, lpc_c, rc = ceps2lpc_v((features).reshape(-1, features.shape[-1]).cpu())
        features = torch.cat((features.cpu(), lpc_c.unsqueeze(0)), -1)
        
        
        while nb_frames < 10:
            in_data = np.hstack((in_data, in_data))

        in_data = torch.tensor(np.reshape(in_data[:len(in_data) // 2400 * 2400], (-1, 2400)))
        
        i = nb_frames-10 if nb_frames > 10 else 0
        
        x = torch.reshape(in_data[i:i+10], (10*2400,))
#         # o_feat = torch.reshape(orig_feat[i:i+10, 2:-2, :], 
#         #                          (10*15, -1)) # [n*15, 36]
        
                
#         nm_feat = o_feat / self.maxi
        
        x = x.unsqueeze(0) #(1, 2400)
        # feat - (N, 36)
        
        return sample_name, x, features[0]
        
        
        
        
        
         
        