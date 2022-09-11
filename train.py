
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
from ceps2lpc_vct import ceps2lpc_v
from dataset_orig import Libri_lpc_data_orig
from dataset_retrain import Libri_lpc_data_retrain
from modules import ExponentialMovingAverage, GaussianLoss

from config import ex
from sacred import Experiment

torch.backends.cudnn.benchmark = True
np.set_printoptions(precision=4)

use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")

gaussianloss = GaussianLoss()
mseloss = nn.MSELoss()
crossentropy = nn.CrossEntropyLoss()

@ex.capture
def build_model(cfg):
    model = Wavenet(out_channels=cfg['out_channels'],
                    num_blocks=cfg['num_blocks'],
                    num_layers=cfg['num_layers'],
                    inp_channels=cfg['inp_channels'],
                    residual_channels=cfg['residual_channels'],
                    gate_channels=cfg['gate_channels'],
                    skip_channels=cfg['skip_channels'],
                    kernel_size=cfg['kernel_size'],
                    cin_channels=cfg['cin_channels']+64,
                    cout_channels=cfg['cout_channels'],
                    upsample_scales=[10, 16],
                    local=cfg['local'],
                    fat_upsampler=cfg['fat_upsampler'])
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

# def mel_spec(x, transform):
    
#     reference = 20.0
#     min_db = -100.0
    
#     feat = transform(x)[:,0, :,:-1]
#     feat = 20 * torch.log10(torch.maximum(torch.tensor(1e-4), feat)) - reference
#     feat = torch.clip((feat - min_db) / (-min_db), 0, 1)
    
#     return feat


def sample_mu_prob(p, feat):
    
    feat = np.repeat(feat[0, 19, :], 160) # (bt, 1, L)
    p *= np.power(p, np.maximum(0, 1.5*feat - .5))
    p = p/(1e-18 + np.sum(p))
    #Cut off the tail of the remaining distribution
    p = np.maximum(p-0.002, 0).astype('float64')
    p = p/(1e-8 + np.sum(p)) # (256, L)

    exc_out_u = np.argmax(p, 0)
    
    assert p.shape[-1] == exc_out_u.shape[-1]
    
    return exc_out_u

def train(model, optimizer, train_loader, epoch, model_label, debugging, cin_channels, inp_channels, stft_loss):

    epoch_loss = 0.
    start_time = time.time()
    display_step = 100
    display_loss = 0.
    display_start = time.time()
    
    exc_hat = None
    
    for batch_idx, (_, x, c) in enumerate(train_loader):
        
        # x - torch.Size([3, 1, 2400*n])
        # c - torch.Size([3, 15*n, 36])
        
        x = x.to(torch.float).to(device)

        # y = y.to(torch.float).to(device)
        # e, lpc_c, rc = ceps2lpc_v(c[:, 2:-2, :].reshape(-1, c.shape[-1]))
        # lpc = lpc_c.reshape(c.shape[0], c.shape[1]-4, 16)
        # lpc = torch.tensor(lpc).to(torch.float).to(device)


        if cin_channels == 20:
            feat = torch.transpose(c[:, :, :-16], 1,2).to(torch.float).to(device) # (bt, 20, 15)
        else:
            feat = torch.transpose(c, 1,2).to(torch.float).to(device) # (bt, 20, 15)
        
        lpc = c[:, :, -16:].to(torch.float).to(device) # (bt, 15, 16) 
        periods = (.1 + 50*c[:,:,18:19]+100).to(torch.int32).to(device)
        
        pred = utils.lpc_pred(x=x, lpc=lpc) # (bt, 1, 2400)
        exc = x - torch.roll(pred,shifts=1,dims=2) #(bt, 1, L) at i
        
        # x_i, exc_i, pred_i+1
        if inp_channels == 1:
            # inp = utils.l2u(x)
            inp = x
        elif inp_channels == 3:
            # inp = torch.cat((utils.l2u(x), utils.l2u(exc), utils.l2u(pred).to(device)), 1) # (bt, 3, n*2400)
            inp = torch.cat((x, exc, pred.to(device)), 1) # (bt, 3, n*2400)
        
        optimizer.zero_grad()
        exc_dist = model(inp, periods, feat) # (bt, 2, 2400) exc_hat_i+1
        
        loss1 = gaussianloss(exc_dist[:,:,:-1], exc[:,:,1:], size_average=True)
        loss2 = 0
        
        loss = loss1 + loss2

        loss.backward()
        
        nn.utils.clip_grad_norm_(model.parameters(), 10.)
        
        optimizer.step()

        epoch_loss += loss.item()
        display_loss += loss.item()
        
        if batch_idx == 0:

            if not os.path.exists('samples/'+model_label):
                os.mkdir('samples/'+model_label)
            
            exc_out = utils.sample_from_gaussian(exc_dist[0:1, :, :])
            plt.plot(exc_out[0,:,0].detach().cpu().numpy())
            plt.savefig('samples/{}/exc_out_{}.jpg'.format(model_label, epoch))
            plt.clf()
            
            plt.plot(exc[0,0,:].detach().cpu().numpy())
            plt.savefig('samples/{}/exc_{}.jpg'.format(model_label, epoch))
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

def evaluate(model, test_loader, debugging, cin_channels, inp_channels, stft_loss):

    model.eval()
    epoch_loss = 0.
    for batch_idx, (_, x, c) in enumerate(test_loader):
        
        x = x.to(torch.float).to(device)
        
        if cin_channels == 20:
            feat = torch.transpose(c[:, :, :-16], 1,2).to(torch.float).to(device) # (bt, 20, 15)
        else:
            feat = torch.transpose(c[:, :, :], 1,2).to(torch.float).to(device) # (bt, 20, 15)
        lpc = c[:, :, -16:].to(torch.float).to(device) # (bt, 15, 16) 
        
        periods = (.1 + 50*c[:, :,18:19]+100).to(torch.int32).to(device)
        
        pred = utils.lpc_pred(x=x, lpc=lpc) # (bt, 1, 2400)
        exc = x - torch.roll(pred,shifts=1,dims=2) #(bt, 1, L) at i
        
        if inp_channels == 1:
            inp = x
        elif inp_channels == 3:
            inp = torch.cat((x, exc, pred.to(device)), 1) # (bt, 3, n*2400)
            
        exc_dist = model(inp, periods, feat) # (bt, 2, 2400)
        
        # nn.utils.clip_grad_norm_(model.parameters(), 10.)

        loss1 = gaussianloss(exc_dist[:,:,:-1], exc[:,:,1:], size_average=True)
        loss2 = 0

        loss = loss1 + loss2

        epoch_loss += loss
        
        del exc_dist
        
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
    if cfg['orig']:
        # train_dataset = Libri_lpc_data_orig('train', cfg['chunks'], cfg['qtz'])
        train_dataset = Libri_lpc_data_retrain()
        # test_dataset = Libri_lpc_data_orig('val', cfg['chunks'], cfg['qtz'])
    else:
        train_dataset = Libri_lpc_data('train', cfg['chunks'])
        test_dataset = Libri_lpc_data('val', cfg['chunks'])

    train_loader = DataLoader(train_dataset, batch_size=cfg['batch_size'], shuffle=True, num_workers=4, pin_memory=True)
#     test_loader = DataLoader(test_dataset, batch_size=cfg['batch_size'], shuffle=True, num_workers=4, pin_memory=True)
    
    model = build_model()
    model.to(device)
    
    if cfg['transfer_model_s'] is not None:
        
        transfer_model_path = 'saved_models/{}/{}_{}.pth'.format(str(cfg['transfer_model_s']), str(cfg['transfer_model_s']), str(cfg['transfer_epoch_s']))
        print("Load checkpoint from: {}".format(transfer_model_path))
        
        
        model.load_state_dict(torch.load(transfer_model_path))
        
        if cfg['upd_f_only']: # Do not update sample-level network
            for p in model.front_conv.parameters():
                p.requires_grad=False
            for p in model.res_blocks.parameters():
                p.requires_grad=False
            for p in model.final_conv.parameters():
                p.requires_grad=False

    optimizer = optim.Adam(model.parameters(), lr=cfg['learning_rate'])

    global_step = 0
    
    # transform = transforms.MelSpectrogram(sample_rate=16000, n_fft=1024, hop_length=256, n_mels=cfg['n_mels'], f_min=125, f_max=7600).to(device)
    
    min_loss = float('inf')
    for epoch in range(cfg['epochs']):
        
        start = time.time()
        
        train_epoch_loss = train(model, optimizer, train_loader, epoch, model_label, cfg['debugging'], cfg['cin_channels'], cfg['inp_channels'], cfg['stft_loss'])
        
        test_epoch_loss = 0
#         with torch.no_grad():
#             test_epoch_loss = evaluate(model, test_loader, cfg['debugging'], cfg['cin_channels'], cfg['inp_channels'], cfg['stft_loss'])
            
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

        
