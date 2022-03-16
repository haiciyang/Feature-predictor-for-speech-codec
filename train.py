
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
from torch.distributions.normal import Normal
from torch.utils.data import Dataset, DataLoader

import utils
from wavenet import Wavenet
from dataset import Libri_lpc_data
# from dataset_orig import Libri_lpc_data
from modules import ExponentialMovingAverage, GaussianLoss

from config import ex
from sacred import Experiment

torch.backends.cudnn.benchmark = True
np.set_printoptions(precision=4)

use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")

criterion = GaussianLoss()
mseloss = nn.MSELoss()

@ex.capture
def build_model(cfg):
    model = Wavenet(out_channels=2,
                    num_blocks=cfg['num_blocks'],
                    num_layers=cfg['num_layers'],
                    residual_channels=cfg['residual_channels'],
                    gate_channels=cfg['gate_channels'],
                    skip_channels=cfg['skip_channels'],
                    kernel_size=cfg['kernel_size'],
                    cin_channels=cfg['cin_channels'],
                    upsample_scales=[10, 16],
                    local=cfg['local'])
    return model


# def clone_as_averaged_model(model, ema):
#     assert ema is not None
#     averaged_model = build_model()
#     averaged_model.to(device)
#     averaged_model.load_state_dict(model.state_dict())

#     for name, param in averaged_model.named_parameters():
#         if name in ema.shadow:
#             param.data = ema.shadow[name].clone().data
#     return averaged_model

def mel_spec(x, transform):
    
    reference = 20.0
    min_db = -100.0
    
    feat = transform(x)[:,0, :,:-1]
    feat = 20 * torch.log10(torch.maximum(torch.tensor(1e-4), feat)) - reference
    feat = torch.clip((feat - min_db) / (-min_db), 0, 1)
    
    return feat


def train(model, optimizer, train_loader, epoch, model_label, debugging, cin_channels, transform):

    epoch_loss = 0.
    start_time = time.time()
    display_step = 500
    display_loss = 0.
    display_start = time.time()
    
    for batch_idx, (x, y, c) in enumerate(train_loader):
        
        # x - torch.Size([3, 1, 2400])
        # y - torch.Size([3, 1, 2400])
        # c - torch.Size([3, 19, 36])
        
        if torch.sum(torch.isnan(x)) != 0:
            print('x_in')
        
        # if torch.sum(torch.isnan(y)) != 0:
        #     print('y_in')
        
        x = x.to(torch.float).to(device)
        y = y.to(torch.float).to(device)
        
#         feat = mel_spec(x, transform)    
#         x = x[:,:,:11776] # 46*16*16
#         y = y[:,:,:11776] # 46*16*16
        
        
        if cin_channels == 20:
            feat = torch.transpose(c[:, 2:-2, :-16], 1,2).to(torch.float).to(device) # (bt, 20, 15)
        else:
            feat = torch.transpose(c[:, 2:-2, :], 1,2).to(torch.float).to(device) # (bt, 20, 15)
        lpc = c[:, 2:-2, -16:].to(torch.float).to(device) # (bt, 15, 16) 
        
        pred = utils.lpc_pred(x=x, lpc=lpc) # (bt, 1, 2400)
        
        if torch.sum(torch.isnan(c)) != 0:
            print('cf')

        # y_hat = pred + exc_hat
        exc = y - pred
        # exc = x[:,:,1:] - pred[:,:,:-1]
        
        optimizer.zero_grad()
        exc_hat = model(exc, feat) # (bt, 2, 2400)
        # y_hat = model(x, feat) # (bt, 2, 2400)
        
        # exc_hat = exc_hat[:,:,:-1]
        loss = criterion(exc_hat[:,:,:-1], exc[:,:,1:], size_average=True)
        # loss = criterion(y_hat, y, size_average=True)
        # loss = criterion(y_hat[:,:,:-1], x[:,:,1:], size_average=True)
        loss.backward()
        
        # fake()
        
        nn.utils.clip_grad_norm_(model.parameters(), 10.)
        
        optimizer.step()

        epoch_loss += loss.item()
        display_loss += loss.item()
        
        if batch_idx % display_step == 0 and batch_idx != 0:
            display_end = time.time()
            duration = display_end - display_start
            display_start = time.time()
            utils.checkpoint(False, epoch, batch_idx, duration, model_label, None, display_loss/display_step, None, None)
            display_loss = 0.
            
            
        if batch_idx == 0:
            mean_y_hat = exc_hat[0, 0, :].detach().cpu().numpy()
            plot_y = exc[0,0,:].detach().cpu().numpy()
            
            if not os.path.exists('samples/'+model_label):
                os.mkdir('samples/'+model_label)
            p_hat = plt.plot(mean_y_hat)
            plt.savefig('samples/{}/y_out_{}.jpg'.format(model_label, epoch))
            plt.clf()
            p = plt.plot(plot_y)
            plt.savefig('samples/{}/y_{}.jpg'.format(model_label, epoch))
            plt.clf()
            
        if debugging:
            break

    return epoch_loss / len(train_loader)

def evaluate(model, test_loader, debugging, cin_channels, transform):

    model.eval()
    epoch_loss = 0.
    for batch_idx, (x, y, c) in enumerate(test_loader):
        
        x = x.to(torch.float).to(device)
        y = y.to(torch.float).to(device)
       
        
        feat = torch.transpose(c[:, 2:-2, :], 1,2).to(torch.float).to(device) # (bt, 20, 15)
        if cin_channels == 20:
            feat = torch.transpose(c[:, 2:-2, :-16], 1,2).to(torch.float).to(device) # (bt, 20, 15)
        else:
            feat = torch.transpose(c[:, 2:-2, :], 1,2).to(torch.float).to(device) # (bt, 20, 15)
        lpc = c[:, 2:-2, -16:].to(torch.float).to(device) # (bt, 15, 16) 
        
        pred = utils.lpc_pred(x=x, lpc=lpc) # (bt, 1, 2400)
            
        # y_hat = pred + exc_hat
        exc = y - pred
        # exc = x[:,:,1:] - pred[:,:,:-1]
        
        # y_hat = model(x, feat) # (bt, 2, 2400)
        exc_hat = model(exc, feat) # (bt, 2, 2400)
        
        nn.utils.clip_grad_norm_(model.parameters(), 10.)
        
        # exc_hat = exc_hat[:,:,:-1]
        loss = criterion(exc_hat[:,:,:-1], exc[:,:,1:], size_average=True)

        # loss = criterion(y_hat, y, size_average=True)
        epoch_loss += loss
        
        if debugging:
            break
        
    return epoch_loss / len(test_loader)


@ex.automain
def run(cfg, model_label): 
    
    # ----- Wirte and print the hyper-parameters -------
    result_path = 'results/'+ model_label +'.txt'
    if not cfg['debugging']:
        with open(result_path, 'a+') as file:
            file.write(model_label+'\n')
            for items in cfg:
                print(items, cfg[items]);
                file.write('%s %s\n'%(items, cfg[items]));
            file.flush()
        print(model_label)
        
    # ---- LOAD DATASETS -------
    train_dataset = Libri_lpc_data('train', cfg['chunks'])
    test_dataset = Libri_lpc_data('val', cfg['chunks'])
    train_loader = DataLoader(train_dataset, batch_size=cfg['batch_size'], shuffle=True, num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=cfg['batch_size'], shuffle=True, num_workers=4, pin_memory=True)
    
    model = build_model()
    model.to(device)
    
    if cfg['transfer_model'] is not None:
        
        transfer_model_path = 'saved_models/{}/{}_{}.pth'.format(str(cfg['transfer_model']), str(cfg['transfer_model']), str(cfg['transfer_epoch']))
        print("Load checkpoint from: {}".format(transfer_model_path))
        model.load_state_dict(torch.load(transfer_model_path))

    optimizer = optim.Adam(model.parameters(), lr=cfg['learning_rate'])

    global_step = 0
    
    transform = transforms.MelSpectrogram(sample_rate=16000, n_fft=1024, hop_length=256, n_mels=cfg['n_mels'], f_min=125, f_max=7600).to(device)
    
    min_loss = float('inf')
    for epoch in range(cfg['epochs']):
        
        start = time.time()
        
        train_epoch_loss = train(model, optimizer, train_loader, epoch, model_label, cfg['debugging'], cfg['cin_channels'], transform)
        
        with torch.no_grad():
            test_epoch_loss = evaluate(model, test_loader, cfg['debugging'], cfg['cin_channels'], transform)
            
        end = time.time()
        duration = end-start
        
        # batch_id = None; When logging epoch results, we don't need batch_id
        min_loss = utils.checkpoint(cfg['debugging'], epoch, None, duration, model_label, model.state_dict(), train_epoch_loss, test_epoch_loss, min_loss)

        
