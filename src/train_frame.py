
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
from models.wavernn import Wavernn
# from models.wavernn_para import Wavernn_para
from datasets.dataset import Libri_lpc_data
from datasets.dataset_orig import Libri_lpc_data_orig
from models.modules import ExponentialMovingAverage, GaussianLoss

# from config_frame import ex
# from sacred import Experiment

torch.backends.cudnn.benchmark = True
np.set_printoptions(precision=4)

use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")

gaussianloss = GaussianLoss()
mseloss = nn.MSELoss()
crossentropy = nn.CrossEntropyLoss()

MAXI = 24.1


def gaussian_loss(mean, log_std, y):
    # print(mean.shape, log_std.shape, y.shape)
    
    log_probs = 0.5 * (- math.log(2.0 * math.pi) - 2. * log_std - torch.pow(y - mean, 2) * torch.exp((-2.0 * log_std)))
    
    return - log_probs.squeeze().mean()
    

def train(model, optimizer, train_loader, epoch, model_label, padding, packing, fc_units, normalize, keep_rate, debugging):

    model.train()
    
    epoch_loss = 0.
    start_time = time.time()
    
    exc_hat = None
    
    for batch_idx, (sample_name, x, c, nm_c) in enumerate(train_loader):
        
        if batch_idx >10 and model.scale < 100:
            model.scale += 5
        
        if normalize:
            feat = nm_c[:, 2:-2, :-16].to(torch.float).to(device) # (B, L, C)
        else:
            feat = c[:, 2:-2, :-16].to(torch.float).to(device) # (B, L, C)
        
        if packing:
            inp = pack_padded_sequence(feat, c_lens, batch_first=True, enforce_sorted=False)
        else:
            inp = feat
 
        if batch_idx <= 10:
            feat_out, _, _ = model(inp) # (B, L, C)
            loss = mseloss(feat_out[:,:-1,:], feat[:,1:,:fc_units])
            
        else:
            feat_out, r_orig, r, r_bl, scl_mask, vct_mask = model.mask_enc(inp)        
            loss = mseloss(feat_out[:,:-1,:fc_units], feat[:,1:,:fc_units]) + (torch.mean(scl_mask)-keep_rate)**2 + (torch.mean(vct_mask)-keep_rate)**2
            
        # feat_mid, feat_out, _, _,_ = model(inp)
        # loss = 2 * mseloss(feat_out, feat[:,:,:fc_units])\
        # + mseloss(feat_mid[:,:-1,:], feat[:,1:,:fc_units])
        
        optimizer.zero_grad()
        loss.backward()        
        optimizer.step()
    
        epoch_loss += loss.item()

        if batch_idx == 0 and epoch % 20 == 0:
            
            if not os.path.exists('../samples/'+model_label):
                os.mkdir('../samples/'+model_label)
           
            
            # feat_out = nm_feat_out * MAXI
            plt.imshow(feat_out[0,:,:].detach().cpu().numpy(), origin='lower', aspect='auto')
            plt.imshow(feat_out[0, :-1, :].detach().cpu().numpy(), origin='lower', aspect='auto')
            plt.colorbar()
            plt.savefig('../samples/{}/feat_out_{}.jpg'.format(model_label, epoch))
            plt.clf()
            
            # plt.imshow((feat[0, 1:-1, :fc_units]).detach().cpu().numpy(), origin='lower', aspect='auto')
            plt.imshow((feat[0, :, :fc_units]).detach().cpu().numpy(), origin='lower', aspect='auto')
            # plt.imshow((feat[0, 1:, :fc_units]).detach().cpu().numpy(), origin='lower', aspect='auto')
            # plt.imshow((feat[0, 1:, :fc_units]-feat[0, :-1,:fc_units]).detach().cpu().numpy(), origin='lower', aspect='auto')
            plt.colorbar()
            plt.savefig('../samples/{}/feat_{}.jpg'.format(model_label, epoch))
            plt.clf()

             
        if debugging:
            break

    return epoch_loss

def evaluate(model, test_loader, padding, packing, fc_units, normalize, keep_rate, debugging):

    model.eval()
    epoch_loss = 0.
    
    for batch_idx, (sample_name, x, c, nm_c)  in enumerate(test_loader):
        
        if normalize:
            feat = nm_c[:, 2:-2, :-16].to(torch.float).to(device) # (B, L, C)
        else:
            feat = c[:, 2:-2, :-16].to(torch.float).to(device) # (B, L, C)

        if packing:
            inp = pack_padded_sequence(feat, c_lens, batch_first=True, enforce_sorted=False)
        else:
            inp = feat
            
            
        if batch_idx <= 10:
            feat_out, _, _ = model(inp) # (B, L, C)
            loss = mseloss(feat_out[:,:-1,:], feat[:,1:,:fc_units])
            
        else:
            feat_out, r_orig, r, r_bl, scl_mask, vct_mask = model.mask_enc(inp)        
            loss = mseloss(feat_out[:,:-1,:fc_units], feat[:,1:,:fc_units]) + (torch.mean(scl_mask)-keep_rate)**2 + (torch.mean(vct_mask)-keep_rate)**2
        
        # feat_out, _, _ = model(inp) # (B, L, C)
        # loss = mseloss(feat_out[:,:-1,:], feat[:,1:,:fc_units])
        
#         feat_mid, feat_out, _, _,_ = model(inp)
        
#         loss = 2 * mseloss(feat_out, feat[:,:,:fc_units])\
#         # + mseloss(feat_mid[:,:-1,:], feat[:,1:,:fc_units])

        
        epoch_loss += loss.item()
        
        if debugging:
            break
        
    return epoch_loss

def pad_collate(batch):
    
    """
    Args : batch(tuple) : List of tuples / (x, c)  x : list of (T,) c : list of (T, D)
    Returns : Tuple of batch / Network inputs waveform samples x (B, 1, T), padded cepstrum coefficients (B, 36, L*), length list of the frames 
            
    """
    
    xl = [pair[0] for pair in batch]
    cl = [pair[1] for pair in batch]
    nm_cl = [pair[2] for pair in batch]
    
    c_lens = np.array([len(c)-4 for c in cl])
 
    cl_pad = pad_sequence(cl, batch_first=True, padding_value=0)
    nm_cl_pad = pad_sequence(nm_cl, batch_first=True, padding_value=0)

    return xl, cl_pad, nm_cl_pad, c_lens


if __name__ == '__main__':
    
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
        'attn_units': 128,
        'rnn_layers': 2, 
        'bidirectional':False,
        'keep_rate': 0.3, 
        
        
        # 'transfer_model': None,
        'transfer_model': '0722_001326',
        'transfer_epoch': '4000'
    }
    
    # ----- Wirte and print the hyper-parameters -------
    result_path = '../results/'+ model_label +'.txt'
    
    if not cfg['debugging']:
        with open(result_path, 'a+') as file:
            file.write(model_label+'\n')
            for items in cfg:
                print(items, cfg[items]);
                file.write('%s %s\n'%(items, cfg[items]));
            file.flush()
        print(model_label)
        
    # ---- LOAD DATASETS -------
    train_dataset = Libri_lpc_data_orig('train', cfg['chunks'])
    test_dataset = Libri_lpc_data_orig('val', cfg['chunks'])
    
    if cfg['padding']:
        train_loader = DataLoader(train_dataset, batch_size=cfg['batch_size'], shuffle=True, num_workers=4, pin_memory=True, collate_fn=pad_collate)
        test_loader = DataLoader(test_dataset, batch_size=cfg['batch_size'], shuffle=True, num_workers=4, pin_memory=True, collate_fn=pad_collate)
    else:
        train_loader = DataLoader(train_dataset, batch_size=cfg['batch_size'], shuffle=True, num_workers=4, pin_memory=True)
        test_loader = DataLoader(test_dataset, batch_size=cfg['batch_size'], shuffle=True, num_workers=4, pin_memory=True)
    
    model = Wavernn(in_features = 20, 
                    gru_units1 = cfg['gru_units1'],
                    gru_units2 = cfg['gru_units2'],
                    fc_units = cfg['fc_units'],
                    attn_units = cfg['attn_units'],
                    bidirectional = cfg['bidirectional'],
                    packing = cfg['packing']
                   ).to(device)
    
    if cfg['transfer_model'] is not None:
        
        transfer_model_path = '../saved_models/{}/{}_{}.pth'.format(str(cfg['transfer_model']), str(cfg['transfer_model']), str(cfg['transfer_epoch']))
        print("Load checkpoint from: {}".format(transfer_model_path))
        model.load_state_dict(torch.load(transfer_model_path), strict=False)

    optimizer = optim.Adam(model.parameters(), lr=cfg['learning_rate'])

    global_step = 0
    
    # transform = transforms.MelSpectrogram(sample_rate=16000, n_fft=1024, hop_length=256, n_mels=cfg['n_mels'], f_min=125, f_max=7600).to(device)
    
    min_loss = float('inf')
    for epoch in range(cfg['epochs']):
        
        start = time.time()
        
        train_epoch_loss = train(model, optimizer, train_loader, epoch, model_label, cfg['padding'], cfg['packing'], cfg['fc_units'], cfg['normalize'], cfg['keep_rate'], cfg['debugging'])
        
        with torch.no_grad():
            test_epoch_loss = evaluate(model, test_loader, cfg['padding'], cfg['packing'], cfg['fc_units'], cfg['normalize'], cfg['keep_rate'], cfg['debugging'])
            
        end = time.time()
        duration = end-start
        
        # batch_id = None; When logging epoch results, we don't need batch_id
        min_loss = utils.checkpoint(debugging = cfg['debugging'], 
                                    epoch = epoch, 
                                    batch_id = None, 
                                    duration = duration, 
                                    model_label = model_label, 
                                    state_dict = model.state_dict(), 
                                    train_loss = train_epoch_loss, 
                                    valid_loss = test_epoch_loss, 
                                    min_loss = min_loss)

        
