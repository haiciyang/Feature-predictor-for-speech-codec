
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
from models.wavernn import Wavernn
from datasets.dataset import Libri_lpc_data
from ceps2lpc.ceps2lpc_vct import ceps2lpc_v
from datasets.dataset import Libri_lpc_data
from quantization.vq_func import vq_quantize, scl_quantize
from models.modules import ExponentialMovingAverage, GaussianLoss

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


def enc_features(cfg, model_f, sample_name, c, nm_c, mask, l1, l2, vq_quantize, scl_quantize, qtz):
    
    if cfg['normalize']:
        feat = nm_c[:, :, :-16].to(torch.float).to(device) # (batch_size, seq_length, ndims)
    else:
        feat = c[:, :, :-16].to(torch.float).to(device) # (batch_size, seq_length, ndims)
    
    # ind1, ind2, cb_t = None
    feat_in, r, r_qtz, r_bl, ind1, ind2, cb_t = model_f.encoder(cfg = cfg, feat=feat, mask = mask, l1=l1, l2=l2, vq_quantize=vq_quantize, scl_quantize=scl_quantize, qtz=qtz)  
    
    # feat_in, r, r_qtz, r_bl, scl_mask, vct_mask, cb_t = model_f.mask_enc(feat=feat, cfg=cfg, vq_quantize=vq_quantize, scl_quantize=scl_quantize, qtz=qtz) 
    
    feat_in *= MAXI

    e, lpc_c, rc = ceps2lpc_v(feat_in.reshape(-1, feat_in.shape[-1]).cpu()) # lpc_c - (N*(L-1), 16)
    all_features = torch.cat((feat_in.cpu(), lpc_c.unsqueeze(0)), -1).data.numpy()

    sizeof = all_features.strides[-1]
    all_features = np.lib.stride_tricks.as_strided(
                all_features.flatten(), 
                shape=(10, 19, 36),
                strides=(15*36*sizeof, 36*sizeof, sizeof)) #(10, 19, 36)

    return all_features, r.cpu().data.numpy(), r_bl.cpu().data.numpy(), r_qtz.cpu().data.numpy(), ind1, ind2, cb_t  

def dec_features(model_f, nm_c, r_qtz):
    
    feat = nm_c[:, :, :-16].to(torch.float).to(device) # (batch_size, seq_length, ndims)

    feat_out = model_f.decoder(r=r_qtz, feat=nm_c) 
    
    feat_out *= MAXI

    e, lpc_c, rc = ceps2lpc_v(feat_in.reshape(-1, feat_in.shape[-1]).cpu()) # lpc_c - (N*(L-1), 16)
    all_features = torch.cat((feat_in.cpu(), lpc_c.unsqueeze(0)), -1).data.numpy()

    sizeof = all_features.strides[-1]
    all_features = np.lib.stride_tricks.as_strided(
                all_features.flatten(), 
                shape=(10, 19, 36),
                strides=(15*36*sizeof, 36*sizeof, sizeof)) #(10, 19, 36)

    return all_features


def cal_entropy(cb):
    

    cb /= np.sum(cb)
    ent = np.sum(- cb * np.log2(cb + 1e-20))
    
    
    return ent


@ex.automain
def run(cfg, model_label): 
    
    
    # ---- LOAD DATASETS -------
    
    task = 'train' 
    qtz_dataset = Libri_lpc_data(task='train', chunks=10, qtz=0) # return unquantized ceps and 
    train_data_loader = DataLoader(qtz_dataset, batch_size=1, shuffle=False, num_workers=4, pin_memory=True)
    
    # qtz_dataset = Libri_lpc_data(task='val', chunks=10, qtz=0) # return unquantized ceps and 
    # val_data_loader = DataLoader(qtz_dataset, batch_size=1, shuffle=False, num_workers=4, pin_memory=True)
    
    
    # dataset = Libri_lpc_data('val', chunks=10, qtz=-1) # return unquantized features
    # all_dataset = Libri_lpc_data('val', chunks=10, qtz=1) # return quantized features

    # Build model
    model_f = build_rnn().to(device)
    model_f.eval()
    
    transfer_model = str(cfg['transfer_model_f'])#[1:] 
    transfer_model_path = '../saved_models/{}/{}_{}.pth'.format(transfer_model, transfer_model, str(cfg['transfer_epoch_f']))
    print("Load checkpoint from: {}".format(transfer_model_path))
    model_f.load_state_dict(torch.load(transfer_model_path))
    
    original = []
    pred = []
    
    note = cfg['note']
    l1 = cfg['l1']
    l2 = cfg['l2']

    path = '/data/hy17/librispeech/libri_qtz_ft/{}{}'.format(cfg['cb_path'].split('/')[-1][17:-4], note)

    print('Saving quantized features at:', path)
    
    if not os.path.exists(path):
        os.mkdir(path)
    if not os.path.exists(path+'/train/'):
        os.mkdir(path+'/train/')
    # if not os.path.exists(path+'/val/'):
    #     os.mkdir(path+'/val/')
        
    # k = 0
#     for sample_name, x, c, nm_c in tqdm(val_data_loader):
        
#         mask = torch.ones(nm_c.shape[1])
#         # mask[:nm_c.shape[1]//2*2] = torch.tensor((1,0)).repeat(nm_c.shape[1]//2)
        
#         all_features = generate_features(cfg, model_f, sample_name, c, nm_c, mask)
#         np.save('{}/{}.npy'.format(path+'/val/', sample_name[0]), all_features)
        
#         k += 1
#         if k == 1000:
#             break     
    ind1_all = 0
    ind2_all = 0
    
    
    cb_tot = [0,0,0,0,0]
    
    k = 0
    for sample_name, x, c, nm_c in tqdm(train_data_loader):
        
#         if k < 2000:
#             continue  
        # print(sample_name)
        mask = None
        # mask = torch.ones(nm_c.shape[1])
        # mask[:nm_c.shape[1]//2*2] = torch.tensor((1,0)).repeat(nm_c.shape[1]//2)
        
        
        all_features, r, r_bl, r_qtz, ind1, ind2, cb_t = enc_features(cfg, model_f, sample_name, c, nm_c, mask, l1, l2, vq_quantize, scl_quantize, cfg['qtz'])
        
        # all_features, cb_t = enc_features(cfg, model_f, sample_name, c, nm_c, mask, l1, l2, vq_quantize, scl_quantize, cfg['qtz'])
        
#         ind1_all += ind1
#         ind2_all += ind2
        
        cb_tot = [cb_t[i] + cb_tot[i] for i in range(len(cb_tot))]
        
        # print([cal_entropy(cb_tot[i]) for i in range(len(cb_tot))])
        
        # print(np.sum(cb_t2))
        
        # np.save(path+'/train/{}.npy'.format(sample_name[0]), all_features)
        # np.save(path+'/train/{}_res.npy'.format(sample_name[0]), r)
        # np.save(path+'/train/{}_res_bl.npy'.format(sample_name[0]), r_bl)
        
        # break
        
        
        k += 1
        if k == 1000:
            break
            
    # print(ind1_all/k, ind2_all/k)
    print([cal_entropy(cb_tot[i]) for i in range(len(cb_tot))])
    

        

        
        
        

        
