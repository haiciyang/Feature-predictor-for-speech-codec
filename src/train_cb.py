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
        
        # 'cb_path': '../codebooks/ceps_vq_codebook_2_1024_17.npy',
        'cb_path': '',
        'n_entries': [1024, 1024], 
        'stages': 2,
        'code_dims': 17, 
        'scl_cb_path': '../codebooks/scalar_center_256.npy',
        
        'note': '2_1024',
        
        
        'transfer_model': '0722_001326',
        # 'transfer_model': '0714_125944',
        'epoch': '4000'
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
        
    train_loader = DataLoader(train_dataset, batch_size=700, shuffle=False, num_workers=0, pin_memory=True)
    
    if cfg['cb_path']:
        codebook = np.load(cfg['cb_path'])
    else:
        codebook = []
        for i in range(cfg['stages']):
            codebook.append(np.zeros((cfg['n_entries'][i], cfg['code_dims'])))
    
    l1 = 0.02
    l2 = 0.05
    
    
    for batch_idx, inp in enumerate(train_loader):
        
        # Encoder
        
        x, c, nm_c = inp[1], inp[2], inp[3]
        # x, c, nm_c = inp[0], inp[1], inp[2]

        if cfg['padding']:
            c_lens = inp[3]

        if cfg['normalize']:
            feat = nm_c[:, 2:-2, :-16].to(torch.float).to(device) # (batch_size, seq_length, ndims)
        else:
            feat = c[:, 2:-2, :-16].to(torch.float).to(device) # (batch_size, seq_length, ndims)


        c_in = torch.zeros(feat.shape).to(device) # (B, L, C)

        # Use residual from the model with no quantization
        feat_in = feat + 0.01 * (torch.rand(feat.shape)-0.5).to(device)/2
        f_out, _, _ = model(feat_in)# inputs previous frames; predicts i+1th frame
        r = feat_in[:,1:,:18] - f_out[:,:-1, :]
        r = torch.reshape(r[:,:,-cfg['code_dims']:], (-1, cfg['code_dims'])).cpu().data.numpy() # (9000, 18)
        
        # Use the quantization output
        # batch for feat is 1
#         r = []
#         for i in tqdm(range(len(feat))):
#             feat_sub = feat[i:i+1]
#             mask = torch.ones(nm_c.shape[1])
#             mask[:nm_c.shape[1]//2*2] = torch.tensor((1,0)).repeat(nm_c.shape[1]//2)
#             c_in, r_sub, r_qtz = model.encoder(cfg=cfg, feat=feat_sub, n_dim=cfg['code_dims'],mask = mask, l1=l1, l2=l2,  vq_quantize=vq_quantize, scl_quantize=scl_quantize, qtz=0) 

#             r_sub = torch.cat([r_sub[:,i,-cfg['code_dims']:] for i in range(r_sub.shape[1]) if torch.sum(r_sub[:,i,-cfg['code_dims']:]) == 0], 0).cpu().data.numpy() # (45, 17)
#             r.append(r_sub)
        
#         r = np.vstack(r)
        
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
                for _ in range(20):
                    codebook[i] = update(r, codebook[i], cfg['n_entries'][i]) # (nb_entries, ndims)
    
                qr = quantize(codebook[i], r)
                r = qr - r
                
                if i == 2:
                    print('Epoch: {}, Err: {}'.format(batch_idx, np.sum(r*r)))
    

    np.save('../codebooks/ceps_vq_codebook_{}.npy'.format(cfg['note']), codebook)
        
        
        
    
    
    
    
    
    
    
    
        
        

        

    
    