import time
import torch
from torchaudio import transforms
from torch.utils.data import Dataset, DataLoader

import os
import gc
import json
import time
import librosa
import numpy as np
from tqdm import tqdm
import soundfile as sf
from matplotlib import pyplot as plt

import torch
import torch.nn as nn
from torch import optim
from torchaudio import transforms
from torch.utils.data import Dataset, DataLoader

import utils
from wavenet import Wavenet
from wavernn import Wavernn
from dataset import Libri_lpc_data
from dataset_orig import Libri_lpc_data_orig
from modules import ExponentialMovingAverage, GaussianLoss

from config import ex
from sacred import Experiment


use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")

MAXI = 24.1

@ex.capture
def build_model(cfg):
    model = Wavenet(out_channels=2,
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


@ex.capture
def saveaudio(cfg, wave, tp, ns):
    
    out_wav = wave.flatten().squeeze().cpu().numpy()
    wav_name = 'samples/{}/{}_{}_{}_predf.wav'.format(cfg['model_label_s'], cfg['model_label_f'], tp, str(ns))
    torch.save(out_wav, 'samples/{}/{}_{}_{}_predf.pt'.format(cfg['model_label_s'], cfg['model_label_f'], tp, str(ns)))
    sf.write(wav_name, out_wav/max(abs(out_wav)), 16000, 'PCM_16')

@ex.automain
def synthesis(cfg):
    # 3s speech
    
    if cfg['model_label'] is None:
        model_label_f = cfg['model_label_f']
        # model_label_s = cfg['model_label_s']
    else: 
        model_label_f = model_label_s = str(cfg['model_label'])
    
    path_f = 'saved_models/'+ model_label_f + '/' + model_label_f + '_'+ str(cfg['epoch_f']) +'.pth'
    # path_s = 'saved_models/'+ model_label_s + '/' + model_label_s + '_'+ str(cfg['epoch_s']) +'.pth'
    
    if not os.path.exists('samples/'+model_label_f):
        os.mkdir('samples/'+model_label_f)
    
    model_f = Wavernn(in_features=20, out_features=cfg['out_features'], num_layers=cfg['num_layers'], fc_units=20).to(device)
    # model_s = build_model().to(device)
    
    # print("Load checkpoint from: {}, {}".format(path_f, path_s))
    model_f.load_state_dict(torch.load(path_f))
    # model_s.load_state_dict(torch.load(path_s))

    model_f.eval()
    # model_s.eval()
    
    length = cfg['total_secs']*cfg['sr']
    tot_chunks = length//cfg['n_sample_seg']//cfg['chunks']*cfg['chunks']
    tot_length = tot_chunks * cfg['n_sample_seg'] - 160
     
    # load test data
    if cfg['orig']:
        test_dataset = Libri_lpc_data_orig('val', tot_chunks) 
    else:
        test_dataset = Libri_lpc_data('val', tot_chunks)
        
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=0, pin_memory=True)
    
    for ns,(sample, c, nm_c) in enumerate(test_loader):
        if ns < cfg['num_samples']:
            
            # ----- Load and prepare data ----
            sample = sample[:,:,160:].to(torch.float).to(device) # (1, 1, tot_chunks*2400)
            # y = y.to(torch.float).to(device) # (1, 1, tot_chunks*2400)
            
            # === Train frame-wise predictive coding ====
            if cfg['cin_channels'] == 20:
                feat = c[:, 2:-2, :-16].to(torch.float).to(device) # (bt, L, C)
                nm_feat = nm_c[:, 2:-2, :-16].to(torch.float).to(device) # (bt, L, C)
            else:
                feat = c[:, 2:-2, :].to(torch.float).to(device) # (bt, L, C)
                nm_feat = nm_c[:, 2:-2, :].to(torch.float).to(device) # (bt, L, C)
            
            nm_feat_out = model_f(nm_feat) # (B, L, C)
            
            for i in 
            
            
            
            
            feat_out = MAXI * nm_feat_out # Scale back
            
            feat = torch.transpose(feat[:, :, :], 1,2) # (bt, C, L-1)
            feat_out = torch.transpose(feat_out[:, :, :], 1,2) # (bt, C, L-1)

            plt.imshow(feat_out[0, :, :-1].detach().cpu().numpy(), origin='lower', aspect='auto')
            plt.colorbar()
            plt.savefig('samples/{}/feat_out_{}_{}.jpg'.format(cfg['model_label_f'], cfg['epoch_f'], ns))
            plt.clf()

            plt.imshow(feat[0, :, 1:].detach().cpu().numpy(), origin='lower', aspect='auto')
            plt.colorbar()
            plt.savefig('samples/{}/feat_{}_{}.jpg'.format(cfg['model_label_f'], cfg['epoch_f'], ns))
            plt.clf()
            
            
            
            plt.imshow((feat[0, :, 1:]-feat[0, :, :-1]).detach().cpu().numpy(), origin='lower', aspect='auto')
            plt.colorbar()
            plt.savefig('samples/{}/adj_residual_tr_{}_{}.jpg'.format(cfg['model_label_f'], cfg['epoch_f'], ns))
            plt.clf()
            
            
            plt.imshow((feat_out[0, :, :-1]-feat[0, :, :-1]).detach().cpu().numpy(), origin='lower', aspect='auto')
            plt.colorbar()
            plt.savefig('samples/{}/adj_residual_{}_{}.jpg'.format(cfg['model_label_f'], cfg['epoch_f'], ns))
            plt.clf()
            
            plt.imshow((feat[0, :, 1:]-feat_out[0, :, :-1]).detach().cpu().numpy(), origin='lower', aspect='auto')
            plt.colorbar()
            plt.savefig('samples/{}/residual_{}_{}.jpg'.format(cfg['model_label_f'], cfg['epoch_f'], ns))
            plt.clf()
            
            
            print('spec', utils.cal_entropy(feat[0, :, 1:].flatten().detach().cpu().numpy()))
            print('spec_out', utils.cal_entropy(feat_out[0, :, :-1].flatten().detach().cpu().numpy()))
            print('adj_residual', utils.cal_entropy((feat[0, :, 1:]-feat[0, :, :-1]).flatten().detach().cpu().numpy()))
            print('residual', utils.cal_entropy((feat[0, :, 1:]-feat_out[0, :, :-1]).flatten().detach().cpu().numpy()))
            
            continue
            
            
            lpc = c[:, 3:-2, -16:].to(torch.float).to(device) # (1, tot*15, 16) 
            periods = (.1 + 50*c[:,3:-2,18:19]+100).to(torch.int32).to(device)
            
            lpc_sample = torch.repeat_interleave(lpc, 160, dim=1) # (bt, tot_chunks*2400, 16)

            # Save ground truth
            saveaudio(wave=sample, tp='truth', ns=ns)
            
            torch.cuda.synchronize()
            
            # ====== synthesis ======

            # ------ Initialize input array ------- 
            
            rf_size = model_s.receptive_field_size()

            x = torch.zeros(1, 1, tot_length + 1).to(torch.device('cuda'))
            pred = torch.zeros(1, 1, tot_length + 1).to(torch.device('cuda'))
            exc = torch.zeros(1, 1, tot_length + 1).to(torch.device('cuda'))   
            x_out = torch.zeros(1, 1, tot_length + 1).to(torch.device('cuda'))   
            
            if not cfg['local']:
                c_upsampled = model_s.upsample(feat_out, periods)
                # c_upsampled = model.upsample(feat)
            else:
                c_upsampled = torch.repeat_interleave(feat_out, 160, dim=-1)

            for i in tqdm(range(tot_length)):

                if i >= rf_size:
                    start_idx = i - rf_size + 1
                else:
                    start_idx = 0

                cond_c = c_upsampled[:, :, start_idx:i + 1]

                i_rf = min(i+1, rf_size)            
                x_in = x[:, :, -i_rf:]

                lpc_coef = lpc_sample[:, start_idx:i + 1, :]
                pred_in = utils.lpc_pred(x=x_in, lpc=lpc_coef, n_repeat=1)

                if cfg['inp_channels'] == 1:
                    x_inp = x_in
                elif cfg['inp_channels'] == 3:
                    x_inp = torch.cat((x_in, exc[:, :, -i_rf:], pred_in.to('cuda')), 1)

                with torch.no_grad():
                    out = model_s.wavenet(x_inp, cond_c)

                exc_out = utils.sample_from_gaussian(out[:, :, -1:])

                x = torch.roll(x, shifts=-1, dims=2)
                x[:, :, -1] = exc_out + pred_in[:,:,-1]
                # print(x[:, :, -1])
                exc = torch.roll(exc, shifts=-1, dims=2)
                exc[:, :, -1] = exc_out
                pred[:, :, i+1] = pred_in[:,:,-1]
                x_out[:, :, i+1] = 0.85 * x[:, :, -2] + x[:, :, -1]
                # print(x[:, :, -1], exc_out, pred_in[:,:,-1])

                torch.cuda.synchronize()

            saveaudio(wave=x[:,:,1:], tp='xin', ns=ns)    
            saveaudio(wave=x_out[:,:,1:], tp='xout', ns=ns)    
            saveaudio(wave=pred[:,:,1:], tp='pred', ns=ns)    
            saveaudio(wave=exc[:,:,1:], tp='exc', ns=ns)    
            


