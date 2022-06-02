
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
from dataset_orig import Libri_lpc_data_orig
from modules import ExponentialMovingAverage, GaussianLoss

from config import ex
from sacred import Experiment

torch.backends.cudnn.benchmark = True
np.set_printoptions(precision=4)

use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")

criterion = GaussianLoss()
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
    display_step = 500
    display_loss = 0.
    display_start = time.time()
    
    exc_hat = None
    
    for batch_idx, (x, c) in enumerate(train_loader):
        
        # x - torch.Size([3, 1, 2400])
        # y - torch.Size([3, 1, 2400])
        # c - torch.Size([3, 19, 36])
        
        x = x.to(torch.float).to(device)
        # y = y.to(torch.float).to(device)

        if cin_channels == 20:
            feat = torch.transpose(c[:, 2:-2, :-16], 1,2).to(torch.float).to(device) # (bt, 20, 15)
        else:
            feat = torch.transpose(c[:, 2:-2, :], 1,2).to(torch.float).to(device) # (bt, 20, 15)
        lpc = c[:, 2:-2, -16:].to(torch.float).to(device) # (bt, 15, 16) 
        
        periods = (.1 + 50*c[:,2:-2,18:19]+100).to(torch.int32).to(device)
        
        
        pred = utils.lpc_pred(x=x, lpc=lpc) # (bt, 1, 2400)
        exc = x - torch.roll(pred,shifts=1,dims=2) #(bt, 1, L) at i
        
        # x_i, exc_i, pred_i+1
        if inp_channels == 1:
            inp = utils.l2u(x)
        elif inp_channels == 3:
            # inp = torch.cat((utils.l2u(x), utils.l2u(exc), utils.l2u(pred).to(device)), 1) # (bt, 3, n*2400)
            inp = torch.cat((x, exc, pred.to(device)), 1) # (bt, 3, n*2400)
        
        optimizer.zero_grad()
        exc_hat = model(inp, periods, feat) # (bt, 2, 2400) exc_hat_i+1
        # print(exc.shape, exc_hat.shape)
        # print(utils.l2u(exc).shape)
        # fake()
        # y_hat = model(x, feat) # (bt, 2, 2400)
        
        loss1 = criterion(exc_hat[:,:,:-1], exc[:,:,1:], size_average=True)
#         del exc
#         exc_sample = utils.reparam_gaussian(exc_hat)
#         del exc_hat
        
#         x_hat = exc_sample + pred 
        
        loss2 = 0
#         if stft_loss:
#             loss2 = mseloss(
#                 utils.stft(x[:,0,1:]), utils.stft(x_hat[:,0,:-1]))
        
        loss = loss1 + loss2

        # loss = crossentropy(exc_prob[:,:,:-1], utils.l2u(exc)[:,0,1:].to(torch.long))  # (input, target)
        # sparse cross entropy
    
        loss.backward()
        
        nn.utils.clip_grad_norm_(model.parameters(), 10.)
        
        optimizer.step()

        epoch_loss += loss.item()
        display_loss += loss.item()
        
        if batch_idx == 0:

            if not os.path.exists('samples/'+model_label):
                os.mkdir('samples/'+model_label)
            
            # torch.save(x_hat[0,0,:-1], 'samples/{}/x_out_{}.pt'.format(model_label, epoch))
            # torch.save(x[0,0,1:], 'samples/{}/x_{}.pt'.format(model_label, epoch))
            
            # mean_y_hat = exc_hat[0, 0, :].detach().cpu().numpy()
            # exc_out = sample_mu_prob(exc_hat[0,:,:].detach().cpu().numpy(), feat.detach().cpu().numpy())
            # print(exc_out.shape)
            # exc_hat = exc[0,0,:].detach().cpu().numpy()
            
            # Y_hat = 20*np.log10(
            #     np.abs(librosa.feature.melspectrogram(
            #         mean_y_hat, sr=16000, n_fft=1024)))
            
            # plt.imshow(Y_hat, origin='lower', aspect='auto')
            plt.plot(exc_hat[0,0,:].detach().cpu().numpy())
            plt.savefig('samples/{}/exc_out_{}.jpg'.format(model_label, epoch))
            plt.clf()
            
            # Y = 20*np.log10(
            #     np.abs(librosa.feature.melspectrogram(
            #         plot_y, sr=160_00, n_fft=1024)))
            
            # plt.imshow(Y, origin='lower', aspect='auto')
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
    for batch_idx, (x, c) in enumerate(test_loader):
        
        x = x.to(torch.float).to(device)
        
        feat = torch.transpose(c[:, 2:-2, :], 1,2).to(torch.float).to(device) # (bt, 20, 15)
        if cin_channels == 20:
            feat = torch.transpose(c[:, 2:-2, :-16], 1,2).to(torch.float).to(device) # (bt, 20, 15)
        else:
            feat = torch.transpose(c[:, 2:-2, :], 1,2).to(torch.float).to(device) # (bt, 20, 15)
        lpc = c[:, 2:-2, -16:].to(torch.float).to(device) # (bt, 15, 16) 
        
        periods = (.1 + 50*c[:,2:-2,18:19]+100).to(torch.int32).to(device)
        
        pred = utils.lpc_pred(x=x, lpc=lpc) # (bt, 1, 2400)
            
        exc = x - torch.roll(pred,shifts=1,dims=2) #(bt, 1, L) at i
        # exc = x[:,:,1:] - pred[:,:,:-1]
        
        if inp_channels == 1:
            inp = utils.l2u(x)
        elif inp_channels == 3:
            # inp = torch.cat((utils.l2u(x), utils.l2u(exc), utils.l2u(pred).to(device)), 1) # (bt, 3, n*2400)
            inp = torch.cat((x, exc, pred.to(device)), 1) # (bt, 3, n*2400)
            
        exc_hat = model(inp, periods, feat) # (bt, 2, 2400)
        
        nn.utils.clip_grad_norm_(model.parameters(), 10.)
        
        loss1 = criterion(exc_hat[:,:,:-1], exc[:,:,1:], size_average=True)

#         exc_sample = utils.reparam_gaussian(exc_hat)
        
#         x_hat = exc_sample + pred 
        
        loss2 = 0
# #         if stft_loss:
# #             loss2 = mseloss(
# #                 utils.stft(x[:,0,1:]), utils.stft(x_hat[:,0,:-1]))
        
        loss = loss1 + loss2
        
        # loss = crossentropy(exc_hat[:,:,:-1], utils.l2u(exc)[:,0,1:].to(torch.long))
        
        epoch_loss += loss
        
        del exc_hat
        del exc
        
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
        train_dataset = Libri_lpc_data_orig('train', cfg['chunks'])
        test_dataset = Libri_lpc_data_orig('val', cfg['chunks'])
    else:
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
    
    # transform = transforms.MelSpectrogram(sample_rate=16000, n_fft=1024, hop_length=256, n_mels=cfg['n_mels'], f_min=125, f_max=7600).to(device)
    
    min_loss = float('inf')
    for epoch in range(cfg['epochs']):
        
        start = time.time()
        
        train_epoch_loss = train(model, optimizer, train_loader, epoch, model_label, cfg['debugging'], cfg['cin_channels'], cfg['inp_channels'], cfg['stft_loss'])
        
        with torch.no_grad():
            test_epoch_loss = evaluate(model, test_loader, cfg['debugging'], cfg['cin_channels'], cfg['inp_channels'], cfg['stft_loss'])
            
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

        
