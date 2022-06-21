
import os
import gc
import json
import time
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

# from config_frame import ex
# from sacred import Experiment

torch.backends.cudnn.benchmark = True
np.set_printoptions(precision=4)

use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")

gaussianloss = GaussianLoss()
mseloss = nn.MSELoss()
crossentropy = nn.CrossEntropyLoss()


def train(model, optimizer, train_loader, epoch, model_label, padding, debugging):

    model.train()
    
    epoch_loss = 0.
    start_time = time.time()
    display_step = 5000
    display_loss = 0.
    display_start = time.time()
    
    exc_hat = None
    
    for batch_idx, inp in enumerate(train_loader):

        x, c = inp[0], inp[1]
        if padding:
            c_lens = inp[2]
        
        feat = c[:, 2:-2, :-16].to(torch.float).to(device) # (B, L, C)

        feat_p = pack_padded_sequence(feat, c_lens, batch_first=True, enforce_sorted=False)
        
        feat_out = model(feat_p)
       
        loss = mseloss(feat_out[:,:-1,:], feat[:,1:,:])
        
        optimizer.zero_grad()
        loss.backward()        
        optimizer.step()

        epoch_loss += loss.item()
        display_loss += loss.item()
        
        if batch_idx == 0:

            if not os.path.exists('samples/'+model_label):
                os.mkdir('samples/'+model_label)
            plt.imshow(feat_out[0,:,:].detach().cpu().numpy(), origin='lower', aspect='auto')
            plt.savefig('samples/{}/feat_out_{}.jpg'.format(model_label, epoch))
            plt.clf()
            
            plt.imshow(feat[0,:,:].detach().cpu().numpy(), origin='lower', aspect='auto')
            plt.savefig('samples/{}/feat_{}.jpg'.format(model_label, epoch))
            plt.clf()
        
        if batch_idx % display_step == 0 and batch_idx != 0:
            display_end = time.time()
            duration = display_end - display_start
            display_start = time.time()
            utils.checkpoint(False, epoch, batch_idx, duration, model_label, None, display_loss/display_step, None, None)
            display_loss = 0.
             
        if debugging:
            break

    return epoch_loss / len(train_loader)

def evaluate(model, test_loader, padding, debugging):

    model.eval()
    epoch_loss = 0.
    
    for batch_idx, inp in enumerate(test_loader):
        
        x, c = inp[0], inp[1]
        if padding:
            c_lens = inp[2]
        
        feat = c[:, 2:-2, :-16].to(torch.float).to(device) # (B, L, C)

        feat_p = pack_padded_sequence(feat, c_lens, batch_first=True, enforce_sorted=False)
        
        feat_out = model(feat_p)
    
        loss = mseloss(feat_out[:,:-1,:], feat[:,1:,:])
        
        epoch_loss += loss.item()
        
        if debugging:
            break
        
    return epoch_loss / len(test_loader)

def pad_collate(batch):
    
    """
    Args : batch(tuple) : List of tuples / (x, c)  x : list of (T,) c : list of (T, D)
    Returns : Tuple of batch / Network inputs waveform samples x (B, 1, T), padded cepstrum coefficients (B, 36, L*), length list of the frames 
            
    """
    
    xl = [pair[0] for pair in batch]
    cl = [pair[1] for pair in batch]
    
    c_lens = np.array([len(c)-4 for c in cl])
 
    cl_pad = pad_sequence(cl, batch_first=True, padding_value=0)

    return xl, cl_pad, c_lens


if __name__ == '__main__':
    
    model_label = time.strftime("%m%d_%H%M%S")
    
    cfg = {
        'debugging': False,
        'chunks': 0,
        'batch_size': 100,
        'learning_rate': 0.001,
        'epochs': 5000,
        'padding': True,
        
        'transfer_model': None,
        # 'transfer_model': '0620_005910',
        # 'transfer_epoch': '99'
    }
    
    # ----- Wirte and print the hyper-parameters -------
    result_path = 'results/'+ model_label +'_rnn.txt'
    
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
    
    model = Wavernn(in_features=20, out_features=20, num_layers=2, fc_unit=20)
    model.to(device)
    
    if cfg['transfer_model'] is not None:
        
        transfer_model_path = 'saved_models/{}/{}_{}.pth'.format(str(cfg['transfer_model']), str(cfg['transfer_model']), str(cfg['transfer_epoch']))
        print("Load checkpoint from: {}".format(transfer_model_path))
        model.load_state_dict(torch.load(transfer_model_path))

    optimizer = optim.Adam(model.parameters(), lr=cfg['learning_rate'])

    global_step = 0
    
    # transform = transforms.MelSpectrogram(sample_rate=16000, n_fft=1024, hop_length=256, n_mels=cfg['n_mels'], f_min=125, f_max=7600).to(device)
    
    min_loss = float('inf')
    for epoch in range(cfg['epochs']):
        
        start = time.time()
        
        train_epoch_loss = train(model, optimizer, train_loader, epoch, model_label, cfg['padding'], cfg['debugging'])
        
        with torch.no_grad():
            test_epoch_loss = evaluate(model, test_loader, cfg['padding'], cfg['debugging'])
            
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

        
