import os
import glob
import torch
import numpy as np
from tqdm import tqdm

folder = '/media/sdb1/haici/libri_qtz_ft/1_2048_large_17/*'
f_list = glob.glob(folder)

for f in tqdm(f_list):
    
    name = f.split('/')[-1]
    name = name[2:-5]

    s = torch.load(f).cpu().data.numpy()
    np.save('/media/sdb1/haici/libri_qtz_ft/1_2048_large_17_new/'+name+'.npy', s)
    
    
    







