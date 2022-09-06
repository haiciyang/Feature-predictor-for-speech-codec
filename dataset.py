import os
import glob
import torch
import random
import numpy as np
# from config import ex
import torchaudio.transforms as T
from torch.utils.data import Dataset, DataLoader


class Libri_lpc_data(Dataset):
    
    # @ex.capture
    def __init__(self, task = 'train', chunks=1): # 28539
        
        # training - 28539
        # testing - 2703
        
        if task == 'train':
            path = '/media/sdb1/Data/libri_lpc_pt/train/*_in_data.pt'
        elif task == 'val':
            path = '/media/sdb1/Data/libri_lpc_pt/val/*_in_data.pt'
        
        self.task = task
        self.files = glob.glob(path)
        self.chunks = chunks
        
        print('Using processed data')
        
    def __len__(self):
        
        return len(self.files)
    
    def __getitem__(self, idx):
        
        eps = 1e-10
        
        f = self.files[idx][:-11] 
        
        in_data = torch.load(f + '_in_data.pt')  # (nb_frames, 2400, 1)
        # out_data = torch.load(f + '_out_data.pt') # (nb_frames, 2400, 1)
        features = torch.load(f + '_features.pt') # (nb_frames, 19, 36)
        
        # Not normalize the data if using mu-law
        # print('Normalizing data')
        max_d = torch.abs(in_data).max()#, torch.abs(out_data).max())
        in_data = in_data/(max_d+eps) * 0.999 # (2400, 1)
        # out_data = out_data/(max_d+eps) * 0.999 # (2400, 1)
           
        nb_frames = min(len(in_data), len(features))
        
        in_data = in_data[:nb_frames]
        
        if self.task == 'val':
            np.random.seed(0)
        i = np.random.choice(nb_frames-self.chunks)
        
        while 1:
            x = torch.reshape(in_data[i:i+self.chunks], (1, self.chunks*2400))
            # y = torch.reshape(out_data[i:i+self.chunks], (1, self.chunks*2400))
            mid_feat = torch.reshape(features[i:i+self.chunks, 2:-2, :], 
                                     (self.chunks*15, -1)) # [n*15, 36]
            feat = torch.cat([features[i,:2, :], mid_feat, features[i+self.chunks-1,-2:, :]], 0)
            # if torch.abs(x).max() == 0 or torch.abs(y).max() == 0 or torch.sum(torch.isnan(feat)) != 0:
            if torch.abs(x).max() == 0 or torch.sum(torch.isnan(feat)) != 0:
                i = np.random.choice(nb_frames-self.chunks)
            else:
                break
                
        # print(x.shape)
        # print(feat.shape)
        # return x, y, feat
        return x, feat
        
        
        
        
        
         
        