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
from wavernn import Wavernn
from dataset import Libri_lpc_data
from dataset_orig import Libri_lpc_data_orig
from modules import ExponentialMovingAverage, GaussianLoss


def vq_train(data, codebook, nb_entries):
    
    # data - (nb_vectors, ndims)

    ndims = data.shape[1]
    
    codebook[0] = np.mean(data, 0)

    e = 1
    while e < nb_entries:
        
        # split
        codebook[e, :] = codebook[0, :]
        delta = .001 * (np.random.rand(e, ndims) / 2)
        codebook[:e, :] += delta
        e += 1

        # update
        for _ in range(4):  
            codebook[:e, :] = update(data, codebook[:e, :], e)

    test = np.sum(codebook, -1) # (nb_entries, )
    
    for _ in range(20):
        codebook = update(data, codebook, nb_entries)
        
    return codebook

def find_nearest(data, codebook):
    
    # data - (nb_vectors, ndims)
    # codebook - (nb_entries, ndims)
    
    data = np.expand_dims(data, axis=0) # (1, nb_vectors, ndims)
    codebook = np.expand_dims(codebook, axis=1) # (nb_entries, 1, ndims)
    
    dist = np.sum((data - codebook) ** 2, -1) # (nb_entries, nb_vectors)
    
    min_index = np.argmin(dist, 0) # (nb_vectors,)
    
    return min_index
        
    
def update(data, codebook, nb_entries_tmp):
    
    nb_vectors = data.shape[0]
    ndims = data.shape[1]
    
    count = np.zeros((nb_entries_tmp,1))
    
    min_index = find_nearest(data, codebook)
    
    codebook = np.zeros((nb_entries_tmp, ndims))
    
    for i in range(nb_vectors):
        
        n = min_index[i]
        count[n] += 1
        codebook[n] += data[i]
    
    codebook /= count + 1e-20
    
    # for i in range(nb_entries_tmp):
    #     if sum(codebook_upd[i]) != 0:
    #         codebook[i] = codebook_upd[i]

    w2 = np.sum((count/nb_vectors)**2)

    print('{} - min: {}, max: {}, small: {}, error: {}'\
          .format(nb_entries_tmp, min(count), max(count), sum(count == 0), w2))
    # print(np.mean(codebook, 0))

    return codebook
    
        
def quantize(codebook, data):
    
    # codebook - (nb_entries, ndims)
    # data - (nb_vectors, ndims)
    
    # find nearest
    min_index = find_nearest(data, codebook)
    qdata = np.array([codebook[i] for i in min_index])
    
    return qdata
    

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
        'learning_rate': 0.0001,
        'epochs': 5000,
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
        
        'nb_entries': [2048, 2048], 
        'stages': 2,
        'n_dims': 17, 
        
        'note': '2_2048_17',
        
        
        'transfer_model': '0722_001326',
        # 'transfer_model': '0714_125944',
        'epoch': '4000'
    }
    
    # 3s speech
    model_label = str(cfg['transfer_model'])
    
    path = 'saved_models/'+ model_label + '/' + model_label + '_'+ str(cfg['epoch']) +'.pth'
    
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
    
    codebook = []
    for i in range(cfg['stages']):
        codebook.append(np.zeros((cfg['nb_entries'][i], cfg['n_dims'])))
    
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
        c_in[:,:,-2:] = feat[:,:,-2:]
        r = torch.zeros(feat.shape[0], feat.shape[1], 18).to(device)
        f = torch.zeros(feat.shape[0], feat.shape[1], 18).to(device)
        
        for i in range(c_in.shape[1]-1):

            f_out = model(c_in[:, :i+1, :])[:,-1, :] # inputs previous frames; predicts i+1th frame
            f[:, i+1, :] = f_out
            
            r_s = feat[:,i+1,:18] - f_out
            r[:,i+1,:] = r_s.clone() 
            c_in[:,i+1,:-2] = f_out.clone() + r_s.clone() + 0.01 * (torch.rand(r_s.shape)-0.5).to(device)/2
            #  (Add error ranging from -0.0025 - 0.0025)
            
        
        r = torch.reshape(r[:,:,-cfg['n_dims']:], (-1, cfg['n_dims'])).cpu().data.numpy() # (9000, 18)
        
        print('Finishe residual calculating of epoch {}'.format(batch_idx))

        if batch_idx == 0:
        
            for i in range(cfg['stages']):

                codebook[i] = vq_train(r, codebook[i], cfg['nb_entries'][i]) # (nb_entries, ndims)

                qr = quantize(codebook[i], r)
                r = qr - r

                if i == 2:
                    print('Epoch: {}, Err: {}'.format(batch_idx, np.sum(r*r)))
        else:
            
            for i in range(cfg['stages']):
                for _ in range(20):
                    codebook[i] = update(r, codebook[i], cfg['nb_entries'][i]) # (nb_entries, ndims)
    
                qr = quantize(codebook[i], r)
                r = qr - r
                
                if i == 2:
                    print('Epoch: {}, Err: {}'.format(batch_idx, np.sum(r*r)))

    np.save('ceps_vq_codebook_{}.npy'.format(cfg['note']), codebook)
        
        
        
    
    
    
    
    
    
    
    
        
        

        

    
    