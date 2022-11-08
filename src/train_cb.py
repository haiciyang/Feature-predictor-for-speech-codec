'''
Learn codebooks for the cepstrum vector quantization

'''
import os
import gc
import json
import time
import math
import librosa
import numpy as np
from tqdm import tqdm
from sklearn.cluster import KMeans
from matplotlib import pyplot as plt

import torch
import torch.nn as nn
from torch import optim
from torchaudio import transforms
from torch.nn.utils.rnn import pad_sequence
from torch.distributions.normal import Normal
from torch.utils.data import Dataset, DataLoader

from torch.nn.utils.rnn import pad_packed_sequence
from torch.nn.utils.rnn import pack_padded_sequence

import utils
from quantization.cb_func import *
from quantization.vq_func import vq_quantize, scl_quantize
from models.wavernn import Wavernn
from datasets.dataset import Libri_lpc_data
from datasets.dataset_orig import Libri_lpc_data_orig
from models.modules import ExponentialMovingAverage, GaussianLoss

 

def build_model(cfg):
    model = Wavernn(in_features = 20, 
                    gru_units1 = cfg['gru_units1'],
                    gru_units2 = cfg['gru_units2'],
                    fc_units = cfg['fc_units'],
                    attn_units = cfg['attn_units'],
                    bidirectional = cfg['bidirectional'],
                    packing = cfg['packing']).to('cuda')
    return model
    

if __name__ == '__main__':
    
    device = 'cuda'
    
    model_label = time.strftime("%m%d_%H%M%S")
    
    cfg = {
        'debugging': False,
        'chunks': 10, 
        'batch_size': 100,
        'padding': False,
        'packing': False,
        'normalize': True, 
        
        'gru_units1': 384,
        'gru_units2': 128,
        'fc_units': 18, 
        'attn_units': 20,
        'rnn_layers': 2, 
        'bidirectional':False,
        
        # data
        'orig': True,
        'total_secs': 1,
        'sr': 16000,
        'n_sample_seg': 2400,
        'normalize': True,
        
        # 'cb_path': '../codebooks/ceps_vq_codebook_mask_30.npy',
        'cb_path': '',
        'n_entries': [512], 
        'stages': 1,
        'code_dims': 17, 
        'train_bl': True, 
        'scl_clusters': 256, 
        'scl_clusters_bl': 16, 
        # 'scl_cb_path': '../codebooks/scalar_center_256.npy',
        
        'note': 'mask30_bl',
        
        
        # 'transfer_model': '0722_001326',
        # 'transfer_model': '1006_174335',
        'transfer_model': '1009_171447',
        # 'transfer_model': '0714_125944',
        'epoch': '1560'
        # 'epoch': '2665'
        # 'epoch': '2000'
    }
    
    # 3s speech
    model_label = str(cfg['transfer_model'])
    
    path = '../saved_models/'+ model_label + '/' + model_label + '_'+ str(cfg['epoch']) +'.pth'
    
    # if not os.path.exists('samples/'+model_label):
    #     os.mkdir('samples/'+model_label)
    
    model = build_model(cfg)
    print("Load checkpoint from: {}".format(path))
    model.load_state_dict(torch.load(path))
    model.to(device)

    model.eval()
    
    
    length = cfg['total_secs']*cfg['sr']
    tot_chunks = length//cfg['n_sample_seg']
    
    # load training data
    if cfg['orig']:
        train_dataset = Libri_lpc_data_orig('train', tot_chunks) 
    else:
        train_dataset = Libri_lpc_data('train', tot_chunks)
        
    train_loader = DataLoader(train_dataset, batch_size=5000, shuffle=False, num_workers=0, pin_memory=True)
    
    if cfg['cb_path']:
        codebook = np.load(cfg['cb_path'])
    else:
        codebook = []
        for i in range(cfg['stages']):
            codebook.append(np.zeros((cfg['n_entries'][i], cfg['code_dims'])))
    
    
    l1 = 0.09
    l2 = 0.28
    
    scl_res = []
    scl_res_bl = []
    
    print('training:','../codebooks/ceps_vq_codebook_{}.npy'.format(cfg['note']))
    
    for batch_idx, inp in enumerate(train_loader):
    # for inp in tqdm(train_loader):
        
        # Encoder
        
        x, c, nm_c = inp[1], inp[2], inp[3]
        # x, c, nm_c = inp[0], inp[1], inp[2]

        if cfg['padding']:
            c_lens = inp[3]

        if cfg['normalize']:
            feat = nm_c[:, 2:-2, :-16].to(torch.float).to(device) # (batch_size, seq_length, ndims)
        else:
            feat = c[:, 2:-2, :-16].to(torch.float).to(device) # (batch_size, seq_length, ndims)


        # Generate training residual
#         feat_in = feat + 0.01 * (torch.rand(feat.shape)-0.5).to(device)/2
#         f_out, _, _ = model(feat_in)# inputs previous frames; predicts i+1th frame
#         r = feat_in[:,1:,:18] - f_out[:,:-1, :]
#         r = torch.reshape(r[:,:,-cfg['code_dims']:], (-1, cfg['code_dims'])).cpu().data.numpy() # (9000, 18)
#         r = np.array([r[i] for i in range(len(r)) if sum(abs(r[i])) > l2])
        
        
        # Generate synthesis residual
#         c_out, r, r_qtz, r_bl, l1_ratio, l2_ratio, cb = model.encoder(
#             cfg=cfg, feat=feat, mask=None, l1=l1, l2=l2, qtz=0)
        
        c_in, r_orig, r, r_bl, _, _, _ = model.mask_enc(
            cfg=cfg, feat=feat, vq_quantize=None, scl_quantize=None, qtz=False) 
        

        # Train scalar codebook
        scl_res.extend(np.array([k for k in r[:,:,0].cpu().data.numpy().flatten() if k != 0]))
        scl_res_bl.extend(np.array([k for k in r_bl[:,:,0].cpu().data.numpy().flatten() if k != 0]))

        # print(r.shape, r_bl.shape)
    
        if not cfg['train_bl']:
            # Code above threshold
            r = torch.reshape(r[:,:,-cfg['code_dims']:], (-1, cfg['code_dims'])).cpu().data.numpy() # (B*L, n_dim)
        elif cfg['train_bl']:
            # Code below threshold
            r = torch.reshape(r_bl[:,:,-cfg['code_dims']:], (-1, cfg['code_dims'])).cpu().data.numpy() # (B*L, n_dim)
        
        r = np.array([r[i] for i in range(len(r)) if sum(abs(r[i])) != 0])


        
        print('Finish residual calculating of epoch {}'.format(batch_idx))

        if batch_idx == 0 and not cfg['cb_path']:
        
            for i in range(cfg['stages']):

                codebook[i] = vq_train(r, codebook[i], cfg['n_entries'][i]) # (nb_entries, ndims)

                qr = quantize(codebook[i], r)
                r = qr - r

                if i == 2:
                    print('Epoch: {}, Err: {}'.format(batch_idx, np.sum(r*r)))
        
        else:
            for i in range(cfg['stages']):
                for _ in range(10):
                    codebook[i] = update(r, codebook[i], cfg['n_entries'][i]) # (nb_entries, ndims)
    
                qr = quantize(codebook[i], r)
                r = qr - r
                
                if i == 2:
                    print('Epoch: {}, Err: {}'.format(batch_idx, np.sum(r*r)))
    

    np.save('../codebooks/ceps_vq_codebook_{}.npy'.format(cfg['note']), codebook)
    
#     kmeans = KMeans(n_clusters=cfg['scl_clusters'], random_state=0).fit(np.array(scl_res).flatten()[:,None])
#     codes = kmeans.cluster_centers_
#     np.save('../codebooks/scalar_center_{}_{}_tr.npy'.format(str(cfg['scl_clusters']), cfg['note']), codes)
    
    
#     kmeans = KMeans(n_clusters=cfg['scl_clusters_bl'], random_state=0).fit(np.array(scl_res_bl).flatten()[:,None])
#     codes = kmeans.cluster_centers_
#     np.save('../codebooks/scalar_center_{}_{}_bl_tr.npy'.format(str(cfg['scl_clusters_bl']), cfg['note']), codes)
        
        
    
    
    
    
    
    
    
        
        

        

    
    