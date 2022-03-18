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
                    cin_channels=cfg['cin_channels'],
                    cout_channels=cfg['cout_channels'],
                    upsample_scales=[10, 16],
                    local=cfg['local'])
    return model

@ex.automain
def synthesis(cfg):
    # 3s speech
    model_label = str(cfg['model_label'])
    # model_label = str(cfg['model_label'])[:4] + '_' + str(cfg['model_label'])[4:]
    
    path = 'saved_models/'+ model_label + '/' + model_label + '_'+ str(cfg['epoch']) +'.pth'
    
    if not os.path.exists('samples/'+model_label):
        os.mkdir('samples/'+model_label)
    
    length = cfg['total_secs']*cfg['sr']
    
    model = build_model()
    print("Load checkpoint from: {}".format(path))
    model.load_state_dict(torch.load(path))
    model.to(device)

    model.eval()
    
    tot_chunks = length//cfg['n_sample_seg']//cfg['chunks']*cfg['chunks']
    
    # load test data
    if cfg['orig']:
        test_dataset = Libri_lpc_data_orig('val', tot_chunks) 
    else:
        test_dataset = Libri_lpc_data('val', tot_chunks)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=True, num_workers=0, pin_memory=True)
    
    for ns, (x, c) in enumerate(test_loader):
        if ns < cfg['num_samples']:
            
            # ----- Load and prepare data ----
            x = x.to(torch.float).to(device) # (1, 1, tot_chunks*2400)
            # y = y.to(torch.float).to(device) # (1, 1, tot_chunks*2400)
            
            if cfg['cin_channels'] == 20:
                feat = c[:, 2:-2, :-16].to(torch.float).to(device) # (1, tot*15, 20)
            else:
                feat = c[:, 2:-2, :].to(torch.float).to(device) 
            lpc = c[:, 2:-2, -16:].to(torch.float).to(device) # (1, tot*15, 16) 
            
            # ----- Save audio files ------
            pred = utils.lpc_pred(x=x, lpc=lpc, N=tot_chunks*cfg['n_sample_seg']) #(1, 1, tot_chunks*2400)
            # Save lpc_prediction
            
            lpc_wav = torch.reshape(pred, (pred.shape[0]*pred.shape[2],)).squeeze().cpu().numpy()
            wav_lpc_name = 'samples/{}/{}_{}_{}_lpc.wav'.format(model_label, cfg['note'],cfg['epoch'], ns)
            torch.save(lpc_wav, 'samples/{}/{}_lpc_wav_{}.pt'.format(model_label, cfg['note'], str(ns)))
#             sf.write(wav_lpc_name, lpc_wav/max(abs(lpc_wav)), 16000, 'PCM_16')
            
            # Save ground truth
            wav = torch.reshape(x[:,:,1:], (-1,1)).squeeze().cpu().numpy()
            wav_truth_name = 'samples/{}/{}_{}_{}_truth.wav'.format(model_label, cfg['note'], cfg['epoch'], ns)

            sf.write(wav_truth_name, wav/max(abs(wav)), 16000, 'PCM_16')
            
            torch.cuda.synchronize()
            
            # ------ synthesis -------
            # x = x.reshape(-1, 1, cfg['chunks']*cfg['n_sample_seg']) 
            feat = torch.transpose(feat.reshape(-1, cfg['chunks']*cfg['n_seg'], feat.shape[2]), 1, 2) 
            lpc = lpc.reshape(-1, cfg['chunks']*15, 16) # (m, chunk*15, 16)

            y_gen = []
            pred_gen = []
            nc = cfg['chunks']
            for i in range(len(feat)):
                # x_sub = x[i:i+1,:,:]       # [1, 1, chunk*1024]
                feat_sub = feat[i:i+1,:,:] # [1, 36, chunk*15]
                lpc_sub = lpc[i:i+1, :, :] # [1, chunk*15, 16]
                with torch.no_grad():
                    y_sub_gen, pred_sub_gen = model.generate_lpc(cfg['chunks']*cfg['n_sample_seg'], lpc_sub, feat_sub, cfg['inp_channels']) # pick one audio signal # (L,), (L,)
                y_gen.append(y_sub_gen)
                pred_gen.append(pred_sub_gen)

                torch.cuda.synchronize()
                # break
            
            y_wav = torch.reshape(torch.cat(y_gen, 0), (1, -1)).squeeze().cpu().numpy()
            pred_wav = torch.reshape(torch.cat(pred_gen, 0), (1, -1)).squeeze().cpu().numpy()

            wav_name = 'samples/{}/{}_{}_{}.wav'.format(model_label,cfg['note'],  cfg['epoch'], ns)
            pred_name = 'samples/{}/{}_{}_{}_pred.wav'.format(model_label,cfg['note'],  cfg['epoch'], ns)
            
            torch.save(y_wav, 'samples/{}/{}_exc_out_{}.pt'.format(model_label, cfg['note'], str(ns)))
            torch.save(pred_wav, 'samples/{}/{}_pred_out_{}.pt'.format(model_label, cfg['note'], str(ns)))
            
            # print(max(abs(pred_wav)))
            sf.write(wav_name, y_wav/max(abs(y_wav)), 16000, 'PCM_16')
            sf.write(pred_name, pred_wav/max(abs(pred_wav)), 16000, 'PCM_16')
            
            del y_gen
            # fake()

