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
from torch.distributions.normal import Normal
from torch.utils.data import Dataset, DataLoader

import utils
from models.wavenet import Wavenet
from models.wavernn import Wavernn
from datasets.dataset import Libri_lpc_data
from datasets.dataset_orig import Libri_lpc_data_orig
from models.modules import ExponentialMovingAverage, GaussianLoss

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
    wav_name = '../samples/{}/{}_{}_{}_{}_predf.wav'.format(cfg['model_label_s'], cfg['model_label_f'], cfg['note'], tp, str(ns))
    # torch.save(out_wav, 'samples/{}/{}_{}_{}_{}_predf.pt'.format(cfg['model_label_s'], cfg['model_label_f'], tp, str(ns)))
    sf.write(wav_name, out_wav/max(abs(out_wav)), 16000, 'PCM_16')
    
@ex.capture
def draw_cmp(cfg, wav, note, ns):
    
    plt.imshow(wav.detach().cpu().numpy(), origin='lower', aspect='auto')
    plt.colorbar()
    plt.savefig('../samples/{}/{}_{}_{}.jpg'.format(cfg['model_label_f'], note, cfg['epoch_f'], ns))
    plt.clf()

    # etp = utils.cal_entropy((wav.flatten().detach().cpu().numpy()))
    # print(note, etp)
    
    return etp

def take_pred_mean(x):
        
    """
    The function is used in the bidirectional WaveGRU
    x - (bt, L, rnn_out*2)
    """

    units = x.shape[-1]//2
    x_a = x[:,:,:units] # [x_1, x_2 .., x_n,  x_{n+1}]
    x_b = x[:,:,units:] # [x_{-1}, x_0 .., x_{n-1} ]

    y = (x_a[:,:-2,:] + x_b[:,2:,:])/2

    return y
    

@ex.automain
def synthesis(cfg):
    # 3s speech
    
    model_label_f = cfg['model_label_f']
    
    if not os.path.exists('../samples/'+model_label_f):
        os.mkdir('../samples/'+model_label_f)
    
    path_f = '../saved_models/'+ model_label_f + '/' + model_label_f + '_'+ str(cfg['epoch_f']) +'.pth'
    
    model_f = Wavernn(in_features = 20, 
                    gru_units1 = cfg['gru_units1'],
                    gru_units2 = cfg['gru_units2'],
                    attn_units = cfg['attn_units'],
                    bidirectional = cfg['bidirectional'],
                    rnn_layers = cfg['rnn_layers'],
                    fc_units = cfg['fc_units'],).to(device)
    model_f.load_state_dict(torch.load(path_f))
    model_f.eval()
    
    length = cfg['total_secs']*cfg['sr']
    tot_chunks = length//cfg['n_sample_seg']
    tot_length = tot_chunks * cfg['n_sample_seg'] - 160
     
    # load test data
    if cfg['orig']:
        test_dataset = Libri_lpc_data_orig('val', tot_chunks) 
    else:
        test_dataset = Libri_lpc_data('val', tot_chunks)
        
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=0, pin_memory=True)
    # number of samples in test_loader -> 2703
    
    results = []
    
    for sample_name, sample, c, nm_c in tqdm(test_loader):

        # ----- Load and prepare data ----
        # sample = sample[:,:,160:].to(torch.float).to(device) # (1, 1, tot_chunks*2400)

        if cfg['normalize']:
            feat = nm_c[:, 2:-2, :-16].to(torch.float).to(device) # (B, L, C)
        else:
            feat = c[:, 2:-2, :-16].to(torch.float).to(device) # (B, L, C)

        output = model_f(feat) # (B, L, C)

        # feat = torch.transpose(feat[:, :, :cfg['fc_units']], 1,2) # (bt, C, L-1)
        # output = torch.transpose(output, 1,2) # (bt, C, L)

        feat = torch.transpose(feat[:, :, :18], 1,2) # (bt, C, L-1)
        output = torch.transpose(output[:, :, :18], 1,2) # (bt, C, L)

#             # --- When feat_out are sampels ---- 
        frames_out = output[0,:,:-1]

#             # ---- When feat_out are adjacent residuals -----
#             frames_out = output[0,:, :-1] + feat[0,:, :-1]

        frames = feat[0,:,1:]
        adj_res_tr = frames-feat[0, :, :-1] # 
        adj_res_out = frames_out - feat[0,:,:-1]
        res = frames - frames_out

        # Bidirectional 
#             frames = feat[0,:,1:-1]
#             frames_out = output[0,:,:]

#             adj_res_tr = frames-feat[0, :, :-2]
#             adj_res_out = frames_out - feat[0,:,:-2]

#             res = frames - frames_out

        results.append(utils.cal_entropy((frames.flatten().detach().cpu().numpy())))
        results.append(utils.cal_entropy((frames_out.flatten().detach().cpu().numpy())))
        results.append(utils.cal_entropy((adj_res_tr.flatten().detach().cpu().numpy())))
        results.append(utils.cal_entropy((adj_res_out.flatten().detach().cpu().numpy())))
        results.append(utils.cal_entropy((res.flatten().detach().cpu().numpy())))            
        
    results = torch.tensor(np.array(results))
    results = torch.reshape(results, (-1, 5))
    ave_r = torch.mean(results, 0)
    print('spec', np.round(ave_r[0].data.numpy(),4))
    print('spec_out', np.round(ave_r[1].data.numpy(),4))
    print('adj_res_tr', np.round(ave_r[2].data.numpy(),4))
    print('residual', np.round(ave_r[4].data.numpy(),4))
    torch.save(results, '../samples/{}/eval_result_{}.pt'.format(cfg['model_label_f'], cfg['epoch_f']))
            
            