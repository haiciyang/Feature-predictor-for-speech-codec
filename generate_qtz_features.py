
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
from dataset_orig import Libri_lpc_data_orig
from modules import ExponentialMovingAverage, GaussianLoss

from config import ex
from sacred import Experiment

torch.backends.cudnn.benchmark = True
np.set_printoptions(precision=4)

use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")

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
    qtz_dataset = Libri_lpc_data_orig(task, chunks=10, qtz=0) # return unquantized ceps and qunatized pitch
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

    path = '/media/sdb1/haici/libri_qtz_ft/{}'.format(cfg['cb_path'].split('/')[-1][17:-4])

    print('Saving quantized features at:', path)
    
    if not os.path.exists(path):
        os.mkdir(path)
    
    # for (qtz_inp, inp, all_inp) in tqdm(zip(qtz_dataset, dataset, all_dataset)):
    for sample_name, x, c, nm_c in tqdm(data_loader):
        
#         # Calculate the average error between features and LPCNet original quantized features
#         orig_nm_c = torch.unsqueeze(inp[-1], 0)[:, 2:-2, :-16].to(torch.float).to(device)
#         qtz_nm_c = torch.unsqueeze(all_inp[-1], 0)[:, 2:-2, :-16].to(torch.float).to(device)
        
#         ori = torch.sum((qtz_nm_c[:, 1:, :] - orig_nm_c[:, 1:, :]) ** 2)
#         original.append(ori.cpu().data.numpy())
        
        
#         # Feature predictive coding
#         c = torch.unsqueeze(qtz_inp[-2], 0)
#         nm_c = torch.unsqueeze(qtz_inp[-1], 0)
        
        
        if cfg['normalize']:
            feat = nm_c[:, 2:-2, :-16].to(torch.float).to(device) # (batch_size, seq_length, ndims)
        else:
            feat = c[:, 2:-2, :-16].to(torch.float).to(device) # (batch_size, seq_length, ndims)
        feat_in, r, r_qtz = model_f.encoder(feat=feat, n_dim=cfg['code_dim']) # (bt, seq_length, n_dim)
        # mask = torch.repeat(torch.tensor(0,1), len(r)
        # r_qtz = 
        # feat_out = model_f.decoder(r=r_qtz, feat=feat)
                            
        
        e, lpc_c, rc = ceps2lpc_v(feat_in.reshape(-1, feat_in.shape[-1]).cpu()) # lpc_c - (N*(L-1), 16)
        
        all_features = torch.cat((feat_in.cpu(), lpc_c.unsqueeze(0)), -1)

        
        torch.save(all_features, '{}/{}.pt'.format(path, sample_name[0]))
        
        
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
        
        
        

        
