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

import torch
import torch.nn as nn
from torch import optim
from torchaudio import transforms
from torch.utils.data import Dataset, DataLoader

import utils
from wavenet import Wavenet
from dataset import Libri_lpc_data
from dataset_orig import Libri_lpc_data_orig
from modules import ExponentialMovingAverage, GaussianLoss

from config import ex
from sacred import Experiment


use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")


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
    wav_name = 'samples/{}/{}_{}_{}_{}.wav'.format(cfg['model_label'], cfg['note'], cfg['epoch'], tp, str(ns))
    torch.save(out_wav, 'samples/{}/{}_{}_{}.pt'.format(cfg['model_label'], cfg['note'], tp, str(ns)))
    sf.write(wav_name, out_wav/max(abs(out_wav)), 16000, 'PCM_16')

@ex.automain
def synthesis(cfg):
    # 3s speech
    model_label = str(cfg['model_label'])
    
    path = 'saved_models/'+ model_label + '/' + model_label + '_'+ str(cfg['epoch']) +'.pth'
    
    if not os.path.exists('samples/'+model_label):
        os.mkdir('samples/'+model_label)
    
    model = build_model()
    print("Load checkpoint from: {}".format(path))
    model.load_state_dict(torch.load(path))
    model.to(device)

    model.eval()
    
    length = cfg['total_secs']*cfg['sr']
    tot_chunks = length//cfg['n_sample_seg']  #//cfg['chunks']*cfg['chunks']
    tot_length = tot_chunks * cfg['n_sample_seg']
    
    # load test data
    if cfg['orig']:
        test_dataset = Libri_lpc_data_orig('val', tot_chunks, qtz=cfg['qtz']) 
    else:
        test_dataset = Libri_lpc_data('val', tot_chunks, qtz=cfg['qtz'])
        
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=0, pin_memory=True)
    
    for ns, (_, sample, c, nm_c) in enumerate(test_loader):
        if ns < cfg['num_samples']:
            
            # ----- Load and prepare data ----
            sample = sample.to(torch.float).to(device) # (1, 1, tot_chunks*2400)
            # y = y.to(torch.float).to(device) # (1, 1, tot_chunks*2400)
            
            if cfg['cin_channels'] == 20:
                feat = c[:, 2:-2, :-16].to(torch.float).to(device) # (1, tot*15, 20)
            else:
                feat = c[:, 2:-2, :].to(torch.float).to(device) 
            lpc = c[:, 2:-2, -16:].to(torch.float).to(device) # (1, tot*15, 16) 
            
            periods = (.1 + 50*c[:,2:-2,18:19]+100).to(torch.int32).to(device)
            
            lpc_sample = torch.repeat_interleave(lpc, 160, dim=1) # (bt, tot_chunks*2400, 16)
            
            # Save ground truth
            saveaudio(wave=sample, tp='truth', ns=ns)

            
            torch.cuda.synchronize()
            
            # ------ synthesis -------
            # x = x.reshape(-1, 1, cfg['chunks']*cfg['n_sample_seg']) 
            # feat = torch.transpose(feat.reshape(-1, cfg['chunks']*cfg['n_seg'], feat.shape[2]), 1, 2) 
            feat = torch.transpose(feat, 1, 2) # (1, 36, 7*15*n)
            # lpc = lpc.reshape(-1, cfg['chunks']*15, 16) # (n, chunk*15, 16)
            
            # ------ Initialize input array ------- 
            
            x_out = model.generate_lpc(feat, periods, lpc_sample, tot_length)
            
            
            # saveaudio(wave=x[:,:,1:], tp='xin', ns=ns)    
            saveaudio(wave=x_out[:,:,1:], tp='xout', ns=ns)    
            # saveaudio(wave=pred[:,:,1:], tp='pred', ns=ns)    
            # saveaudio(wave=exc[:,:,1:], tp='exc', ns=ns)    
            


