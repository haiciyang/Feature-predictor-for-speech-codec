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

from ceps2lpc_sc import ceps2lpc_s
from ceps2lpc_vct import ceps2lpc_v


te_data = Libri_lpc_data_orig('train', chunks=5)

x, feat, nm_feat = te_data[0]

feat = torch.tensor([ 8.6534, -8.6220,  0.2948,  3.6121, -0.1741, -0.0941,  0.4711,  0.5865,
         0.3292,  0.3319, -0.6430,  0.0478, -0.1304, -0.1146, -0.2340, -0.6239,
        -0.2399,  0.3739,  0.7100, -0.1695,  1.4345,  2.6830,  3.4697,  4.1100,
         4.0251,  3.6048,  2.6411,  1.5679,  0.4696, -0.3163, -0.8315, -0.9335,
        -0.8098, -0.5357, -0.2867, -0.1045], requires_grad=True)

# feat_ = feat.detach().clone()

feat_ = feat[None, :]#.detach().clone()

e, lpc_v, rc = ceps2lpc_v(feat_)
# x = ceps2lpc_v(feat_)

torch.autograd.set_detect_anomaly(True)

loss = torch.sum(lpc_v)

print(loss)

loss.backward()

print(feat.grad)

# print(x_s)
# print(x_v[0])

# print(lpc_s)
# print(lpc_v.shape)
# print(lpc_v[0])