
import os
import gc
import json
import time
import librosa
import numpy as np
from tqdm import tqdm
from matplotlib import pyplot as plt

import torch
import torch.nn as nn
from torch import optim
from torchaudio import transforms
from torch.distributions.normal import Normal
from torch.utils.data import Dataset, DataLoader

import utils
from wavenet import Wavenet
from wavernn import Wavernn
from dataset import Libri_lpc_data
from ceps2lpc_vct import ceps2lpc_v
from dataset import Libri_lpc_data
from modules import ExponentialMovingAverage, GaussianLoss

from config import ex
from sacred import Experiment

torch.backends.cudnn.benchmark = True
np.set_printoptions(precision=4)

use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")

MAXI = 24.1

@ex.capture
def build_rnn(cfg):
    model = Wavernn(in_features = 20, 
                    gru_units1 = cfg['gru_units1'],
                    gru_units2 = cfg['gru_units2'],
                    attn_units = cfg['attn_units'],
                    bidirectional = cfg['bidirectional'],
                    rnn_layers = cfg['rnn_layers'],
                    fc_units = cfg['fc_units'],).to(device)
    return model


@ex.automain
def run(cfg, model_label): 
    
    
    # ---- LOAD DATASETS -------
    
    task = 'train' 
    qtz_dataset = Libri_lpc_data(task, chunks=10, qtz=0) # return unquantized ceps and qunatized pitch
    # dataset = Libri_lpc_data_orig(task, chunks=10, qtz=-1) # return unquantized features
    # all_dataset = Libri_lpc_data_orig(task, chunks=10, qtz=1) # return quantized features
    data_loader = DataLoader(qtz_dataset, batch_size=1, shuffle=False, num_workers=4, pin_memory=True)

    # Build model
    model_f = build_rnn().to(device)
    model_f.eval()
        
    transfer_model_path = 'saved_models/{}/{}_{}.pth'.format(str(cfg['transfer_model_f']), str(cfg['transfer_model_f']), str(cfg['transfer_epoch_f']))
    print("Load checkpoint from: {}".format(transfer_model_path))
    model_f.load_state_dict(torch.load(transfer_model_path))
    
    
    original = []
    pred = []

    # path = '/data/hy17/haici/libri_qtz_ft/{}'.format(cfg['cb_path'].split('/')[-1][17:-4])
    path = '/data/hy17/librispeech/libri_qtz_ft/{}'.format(cfg['cb_path'].split('/')[-1][17:-4])

    print('Saving quantized features at:', path)
    
    if not os.path.exists(path):
        os.mkdir(path)
    
    k = 0
    # for (qtz_inp, inp, all_inp) in tqdm(zip(qtz_dataset, dataset, all_dataset)):
    for sample_name, x, c, nm_c in tqdm(data_loader):
        
        if k < 5000:
            k += 1
            continue
            
        
#         # Calculate the average error between features and LPCNet original quantized features
#         orig_nm_c = torch.unsqueeze(inp[-1], 0)[:, 2:-2, :-16].to(torch.float).to(device)
#         qtz_nm_c = torch.unsqueeze(all_inp[-1], 0)[:, 2:-2, :-16].to(torch.float).to(device)
        
#         ori = torch.sum((qtz_nm_c[:, 1:, :] - orig_nm_c[:, 1:, :]) ** 2)
#         original.append(ori.cpu().data.numpy())
        
        
#         # Feature predictive coding
#         c = torch.unsqueeze(qtz_inp[-2], 0)
#         nm_c = torch.unsqueeze(qtz_inp[-1], 0)
        
        # nm_c - (10, 19, 36)
        
        if cfg['normalize']:
            feat = nm_c[:, :, :-16].to(torch.float).to(device) # (batch_size, seq_length, ndims)
        else:
            feat = c[:, :, :-16].to(torch.float).to(device) # (batch_size, seq_length, ndims)
        
        mask = torch.ones(nm_c.shape[1])
        # mask[:nm_c.shape[1]//2*2] = torch.tensor((1,0)).repeat(nm_c.shape[1]//2)

        feat_in, r, r_qtz = model_f.encoder(feat=feat, n_dim=cfg['code_dim'], mask = mask)     
        
        # feat_in, r, r_qtz = model_f.encoder(feat=feat, n_dim=cfg['code_dim'], mask = mask) # (bt, seq_length, n_dim)
        feat_in *= MAXI
        
        e, lpc_c, rc = ceps2lpc_v(feat_in.reshape(-1, feat_in.shape[-1]).cpu()) # lpc_c - (N*(L-1), 16)
        
        all_features = torch.cat((feat_in.cpu(), lpc_c.unsqueeze(0)), -1).data.numpy()
        
        sizeof = all_features.strides[-1]
        all_features = np.lib.stride_tricks.as_strided(
                    all_features.flatten(), 
                    shape=(10, 19, 36),
                    strides=(15*36*sizeof, 36*sizeof, sizeof)) #(10, 19, 36)
        
        np.save('{}/{}.npy'.format(path, sample_name[0]), all_features)
        # fake()
        
        k += 1
        
        
        # torch.save(all_features, '{}/{}.pt'.format(path, sample_name[0]))
        
        
#         torch.save(nm_c[:,2:-2, :18], 'samples/nm_feat.pt')
#         torch.save(feat_in, 'samples/feat_in.pt')
#         torch.save(r, 'samples/r.pt')
#         torch.save(r_qtz, 'samples/r_qtz.pt')

#         pre = torch.sum((feat_in[:, 1:, :] - orig_nm_c[:, 1:, :]) ** 2)
#         pred.append(pre.cpu().data.numpy())
        
        # print('({:2f}, {:2f}), ({:2f}, {:2f})'.format(ori.cpu().data, pre.cpu().data, np.mean(original), np.mean(pred)))
        
        # break
        
    # np.save('1_1024_original.npy', original)
    # np.save('1_2048_18_pred.npy', pred)
        
        # feat_out = model_f.decoder(r=r, inp=inp)
        
        
        

        