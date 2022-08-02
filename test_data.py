import os
import glob
import torch
import random
import librosa
import numpy as np
# from config import ex
import torchaudio.transforms as T
from torch.utils.data import Dataset, DataLoader
from dataset_orig import Libri_lpc_data_orig

test_data = Libri_lpc_data_orig('val', 0)

ll = []
for i, (x, feat, nm_feat) in enumerate(test_data):
    
    if i == 10:
        torch.save(x, 'example.pt')
        break
    # ll.append(l)
    
# print(max(ll), min(ll))
    
    