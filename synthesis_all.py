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
    wav_name = 'samples/{}/{}_{}_{}_{}_predf.wav'.format(cfg['model_label_s'], cfg['model_label_f'], cfg['note'], tp, str(ns))
    # torch.save(out_wav, 'samples/{}/{}_{}_{}_{}_predf.pt'.format(cfg['model_label_s'], cfg['model_label_f'], tp, str(ns)))
    sf.write(wav_name, out_wav/max(abs(out_wav)), 16000, 'PCM_16')
    
@ex.capture
def draw_cmp(cfg, wav, note, ns):
    
    # plt.imshow(wav.detach().cpu().numpy(), origin='lower', aspect='auto')
    # plt.colorbar()
    # plt.savefig('samples/{}/{}_{}_{}.jpg'.format(cfg['model_label_f'], note, cfg['epoch_f'], ns))
    # plt.clf()

    etp = utils.cal_entropy((wav.flatten().detach().cpu().numpy()))
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
    
    if cfg['model_label'] is None:
        model_label_f = cfg['model_label_f']
        model_label_s = cfg['model_label_s']
    else: 
        model_label_f = model_label_s = str(cfg['model_label'])
    
    if not os.path.exists('samples/'+model_label_f):
        os.mkdir('samples/'+model_label_f)
    
    path_f = 'saved_models/'+ model_label_f + '/' + model_label_f + '_'+ str(cfg['epoch_f']) +'.pth'
    
    # print(torch.load(path_f).keys())
    # fake()
    model_f = Wavernn(in_features = 20, 
                    gru_units1 = cfg['gru_units1'],
                    gru_units2 = cfg['gru_units2'],
                    attn_units = cfg['attn_units'],
                    bidirectional = cfg['bidirectional'],
                    rnn_layers = cfg['rnn_layers'],
                    fc_units = cfg['fc_units'],).to(device)
    model_f.load_state_dict(torch.load(path_f))
    model_f.eval()
    
    
    # path_s = 'saved_models/'+ model_label_s + '/' + model_label_s + '_'+ str(cfg['epoch_s']) +'.pth'
    # model_s = build_model().to(device)
    # model_s.load_state_dict(torch.load(path_s))
    # model_s.eval()
    
    # print("Load checkpoint from: {}, {}".format(path_f, path_s))
    
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
    
    for ns,(sample, c, nm_c) in enumerate(test_loader):

        if ns < 5:
            
            # ----- Load and prepare data ----
            sample = sample[:,:,160:].to(torch.float).to(device) # (1, 1, tot_chunks*2400)
            # y = y.to(torch.float).to(device) # (1, 1, tot_chunks*2400)
            
            # === Train frame-wise predictive coding ====
            # if cfg['cin_channels'] == 20:
            #     feat = c[:, 2:-2, :-16].to(torch.float).to(device) # (bt, L, C)
            #     nm_feat = nm_c[:, 2:-2, :-16].to(torch.float).to(device) # (bt, L, C)
            # else:
            #     feat = c[:, 2:-2, :].to(torch.float).to(device) # (bt, L, C)
            #     nm_feat = nm_c[:, 2:-2, :].to(torch.float).to(device) # (bt, L, C)
                
            if cfg['normalize']:
                feat = nm_c[:, 2:-2, :-16].to(torch.float).to(device) # (B, L, C)
            else:
                feat = c[:, 2:-2, :-16].to(torch.float).to(device) # (B, L, C)
            
            output = model_f(feat) # (B, L, C)
            # output = take_pred_mean(output)
#             feat_para = model_f(nm_feat) # (B, L, C)
            
#             feat_mu = feat_para[:,:,:cfg['fc_units']//2]
#             feat_logvar = feat_para[:,:,cfg['fc_units']//2:]
            
#             dist = Normal(feat_mu, torch.exp(feat_logvar))
#             nm_feat_out = dist.sample()
            
            # feat_out = MAXI * nm_feat_out # Scale back
            
            
            feat = torch.transpose(feat[:, :, :cfg['fc_units']], 1,2) # (bt, C, L-1)
            output = torch.transpose(output, 1,2) # (bt, C, L-1)

            # plt.imshow(feat_out[0, :, :-1].detach().cpu().numpy(), origin='lower', aspect='auto')
            # plt.colorbar()
            # plt.savefig('samples/{}/feat_out_{}_{}.jpg'.format(cfg['model_label_f'], cfg['epoch_f'], ns))
            # plt.clf()
            
            
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
            
            
            results.append(draw_cmp(cfg, frames, 'feat', ns))
            results.append(draw_cmp(cfg, frames_out, 'feat_out', ns))
            results.append(draw_cmp(cfg, adj_res_tr, 'adj_res_tr', ns))
            results.append(draw_cmp(cfg, adj_res_out, 'adj_res_out', ns))
            results.append(draw_cmp(cfg, res, 'res', ns))
                
#             plt.imshow(frames.detach().cpu().numpy(), origin='lower', aspect='auto')
#             plt.colorbar()
#             plt.savefig('samples/{}/feat_{}_{}.jpg'.format(cfg['model_label_f'], cfg['epoch_f'], ns))
#             plt.clf()
            
#             plt.imshow(frames_out.detach().cpu().numpy(), origin='lower', aspect='auto')
#             plt.colorbar()
#             plt.savefig('samples/{}/feat_out_{}_{}.jpg'.format(cfg['model_label_f'], cfg['epoch_f'], ns))
#             plt.clf()
            
#             plt.imshow(adj_res_tr.detach().cpu().numpy(), origin='lower', aspect='auto')
#             plt.colorbar()
#             plt.savefig('samples/{}/adj_res_tr_{}_{}.jpg'.format(cfg['model_label_f'], cfg['epoch_f'], ns))
#             plt.clf()
            
#             plt.imshow(adj_res_out.detach().cpu().numpy(), origin='lower', aspect='auto')
#             plt.colorbar()
#             plt.savefig('samples/{}/adj_res_{}_{}.jpg'.format(cfg['model_label_f'], cfg['epoch_f'], ns))
#             plt.clf()
            
#             plt.imshow(res.detach().cpu().numpy(), origin='lower', aspect='auto')
#             plt.colorbar()
#             plt.savefig('samples/{}/res_{}_{}.jpg'.format(cfg['model_label_f'], cfg['epoch_f'], ns))
#             plt.clf()

            
#             print('spec', utils.cal_entropy(frames.flatten().detach().cpu().numpy()))
#             print('spec_out', utils.cal_entropy(frames_out.flatten().detach().cpu().numpy()))
#             print('adj_res_tr', utils.cal_entropy((adj_res_tr.flatten().detach().cpu().numpy()))
#             print('residual', utils.cal_entropy((feat[0, :, 1:]-feat_out[0, :, :-1]).flatten().detach().cpu().numpy()))
            
    
            # continue
        
    results = torch.tensor(np.array(results))
    results = torch.reshape(results, (-1, 5))
    ave_r = torch.mean(results, 0)
    print('spec', np.round(ave_r[0].data.numpy(),4))
    print('spec_out', np.round(ave_r[1].data.numpy(),4))
    print('adj_res_tr', np.round(ave_r[2].data.numpy(),4))
    print('residual', np.round(ave_r[4].data.numpy(),4))
    torch.save(results, 'samples/{}/{}_{}.pt'.format(cfg['model_label_f'], cfg['note'], cfg['epoch_f']))
            
            
#             lpc = c[:, 3:-2, -16:].to(torch.float).to(device) # (1, tot*15, 16) 
#             periods = (.1 + 50*c[:,3:-2,18:19]+100).to(torch.int32).to(device)
            
#             lpc_sample = torch.repeat_interleave(lpc, 160, dim=1) # (bt, tot_chunks*2400, 16)

#             # Save ground truth
#             saveaudio(wave=sample, tp='truth', ns=ns)
            
#             torch.cuda.synchronize()
            
#             # ====== synthesis ======

#             # ------ Initialize input array ------- 
            
#             rf_size = model_s.receptive_field_size()

#             x = torch.zeros(1, 1, tot_length + 1).to(torch.device('cuda'))
#             pred = torch.zeros(1, 1, tot_length + 1).to(torch.device('cuda'))
#             exc = torch.zeros(1, 1, tot_length + 1).to(torch.device('cuda'))   
#             x_out = torch.zeros(1, 1, tot_length + 1).to(torch.device('cuda'))   
            

#             if not cfg['local']:
#                 c_upsampled = model_s.upsample(feat_out[:,:,1:], periods)
#                 # c_upsampled = model.upsample(feat)
#             else:
#                 c_upsampled = torch.repeat_interleave(feat_out[:,:,1:], 160, dim=-1)
        
#             # print(c_upsampled.shape)
#             # fake()

#             for i in tqdm(range(tot_length)):

#                 if i >= rf_size:
#                     start_idx = i - rf_size + 1
#                 else:
#                     start_idx = 0

#                 cond_c = c_upsampled[:, :, start_idx:i + 1]

#                 i_rf = min(i+1, rf_size)            
#                 x_in = x[:, :, -i_rf:]

#                 lpc_coef = lpc_sample[:, start_idx:i + 1, :]
#                 pred_in = utils.lpc_pred(x=x_in, lpc=lpc_coef, n_repeat=1)

#                 if cfg['inp_channels'] == 1:
#                     x_inp = x_in
#                 elif cfg['inp_channels'] == 3:
#                     x_inp = torch.cat((x_in, exc[:, :, -i_rf:], pred_in.to('cuda')), 1)

#                 with torch.no_grad():
#                     out = model_s.wavenet(x_inp, cond_c)

#                 exc_out = utils.sample_from_gaussian(out[:, :, -1:])

#                 x = torch.roll(x, shifts=-1, dims=2)
#                 x[:, :, -1] = exc_out + pred_in[:,:,-1]
#                 # print(x[:, :, -1])
#                 exc = torch.roll(exc, shifts=-1, dims=2)
#                 exc[:, :, -1] = exc_out
#                 pred[:, :, i+1] = pred_in[:,:,-1]
#                 x_out[:, :, i+1] = 0.85 * x[:, :, -2] + x[:, :, -1]
#                 # print(x[:, :, -1], exc_out, pred_in[:,:,-1])

#                 torch.cuda.synchronize()

#             # saveaudio(wave=x[:,:,1:], tp='xin', ns=ns)    
#             saveaudio(wave=x_out[:,:,1:], tp='xout', ns=ns)    
#             # saveaudio(wave=pred[:,:,1:], tp='pred', ns=ns)    
#             # saveaudio(wave=exc[:,:,1:], tp='exc', ns=ns)    
            


