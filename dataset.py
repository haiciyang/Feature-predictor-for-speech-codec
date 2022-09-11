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
    def __init__(self, task = 'train', chunks=1, qtz=0): # 28539
        
        # training - 28539
        # testing - 2703
        
        self.maxi = 24.1
        self.task = task
        self.chunks = chunks
        self.qtz = qtz
        
        if task == 'train':
            path = '/media/sdb1/Data/libri_lpc_pt/train/*_in_data.pt'
        elif task == 'val':
            path = '/media/sdb1/Data/libri_lpc_pt/val/*_in_data.pt'
        
        self.feature_qtz_folder = '/media/sdb1/Data/libri_lpc_qtz_pt/{}/'.format(task)
        self.feature_folder = '/media/sdb1/Data/libri_lpc_pt/{}/'.format(task)
        
        self.files = glob.glob(path)


        print('Using processed data')
        
    def __len__(self):
        
        return len(self.files)
    
    def __getitem__(self, idx):
        
        eps = 1e-10
        
        f = self.files[idx] 
        sample_name = f.split('/')[-1][:-11] # 103-1240-0000

        in_data = torch.load(f)  # (nb_frames, 2400, 1)
        # features = torch.load(f + '_features.pt') # (nb_frames, 19, 36)
        
        if self.qtz == 1:
            feature_path = self.feature_qtz_folder + sample_name + '_features.pt'
        else: 
            feature_path = self.feature_folder + sample_name + '_features.pt'
            
        features = torch.load(feature_path) # (19, 36)
        
#         nb_frames = min(len(in_data), len(features))
        
        i = 5
        in_data = in_data[i:i+self.chunks]
        features = features[i:i+self.chunks]

        if self.qtz == 0:
            qtz_pitch = torch.load(self.feature_qtz_folder + sample_name + '_features.pt')
            features[:,:,-2:] = qtz_pitch[i:i+self.chunks, :, -2:]
        
        # max_d = torch.abs(in_data).max()#, torch.abs(out_data).max())
        # in_data = in_data/(max_d+eps) * 0.999 # (2400, 1)
        # out_data = out_data/(max_d+eps) * 0.999 # (2400, 1)
           

        # if self.task == 'val':
        #     np.random.seed(0)
        # i = np.random.choice(nb_frames-self.chunks)
        

        # while 1:
        x = torch.reshape(in_data, (1, self.chunks*2400))
        # y = torch.reshape(out_data[i:i+self.chunks], (1, self.chunks*2400))
        mid_feat = torch.reshape(features[:, 2:-2, :], 
                                 (self.chunks*15, -1)) # [n*15, 36]
        feat = torch.cat([features[0,:2, :], mid_feat, features[-1,-2:, :]], 0)
            # if torch.abs(x).max() == 0 or torch.abs(y).max() == 0 or torch.sum(torch.isnan(feat)) != 0:
            # if torch.abs(x).max() == 0 or torch.sum(torch.isnan(feat)) != 0:
            #     i = np.random.choice(nb_frames-self.chunks)
            # else:
            #     break
            
        nm_feat = feat / self.maxi

        return sample_name, x, feat, nm_feat
        
        
        
        
        
         
        