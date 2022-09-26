import os
import gc
import json
import time
import math
import librosa
import numpy as np
import soundfile as sf
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
from quantization.vq_func import vq_quantize, scl_quantize
from datasets.dataset import Libri_lpc_data
from ceps2lpc.ceps2lpc_vct import ceps2lpc_v
# from dataset_orig import Libri_lpc_data_orig
from datasets.dataset_syn import Libri_lpc_data_syn
from models.modules import ExponentialMovingAverage, GaussianLoss

from config import ex
from sacred import Experiment

use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")

MAXI = 24.1

@ex.capture
def saveaudio(wave, model_label, sample_name):
    
    out_wav = wave.flatten().squeeze().cpu().numpy()
    wav_name = '../samples/{}/{}_truth.wav'.format(model_label, sample_name)
    # torch.save(out_wav, 'samples/{}/{}_{}_{}.pt'.format(cfg['model_label_s'], cfg['note'], tp, str(ns)))
    
    out_wav /= np.std(out_wav)
    out_wav /= max(abs(out_wav))
    
                        
    sf.write(wav_name, out_wav, 16000, 'PCM_16')
    
    
@ex.capture
def savefig(cfg, data, tp, ns):
    
    # data - (1, n_frames, n_dim)
    
    data = data[0].cpu().data.numpy()
    file_name = '../samples/{}/{}_{}_{}_{}.jpg'.format(cfg['model_label_s'], cfg['note'], cfg['epoch_s'], tp, str(ns))
    # torch.save(out_wav, 'samples/{}/{}_{}_{}.pt'.format(cfg['model_label_s'], cfg['note'], tp, str(ns)))
    plt.imshow(data.T, origin='lower', aspect='auto')
    plt.colorbar()
    plt.savefig(file_name)
    plt.clf()
    
    
@ex.automain
def synthesis(cfg):
    # 3s speech
    model_label_f = cfg['model_label_f']
    # model_label_s = cfg['model_label_s']
    
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
    

    print("Load checkpoint from: {}".format(path_f))
    
    length = cfg['total_secs']*cfg['sr']
    tot_chunks = length//cfg['n_sample_seg']
    tot_length = tot_chunks * cfg['n_sample_seg'] - 160
     
    # load test data
    test_dataset = Libri_lpc_data_syn(tot_chunks) 
        
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=0, pin_memory=True)
    # number of samples in test_loader -> 2703
    
    results = []
    
    l1 = 0.075
    l2 = 0.075
    

    for ns, (sample_name, sample, nm_c, qtz_c) in enumerate(test_loader):
        
        # if ns >= cfg['num_samples']:
        #     break
        
        # Non-mask
        
        saveaudio(sample, model_label_f, sample_name[0])

#         mask = torch.ones(nm_c.shape[1])
        
#         feat = nm_c[:,:,:-16].to('cuda')
#         c_in, r, r_qtz = model_f.encoder(cfg=cfg, feat=feat, n_dim=cfg['code_dim'], mask = mask, l1=l1, l2=l2, vq_quantize=vq_quantize, scl_quantize=scl_quantize)      
#         c_in *= MAXI
        
#         e, lpc_c, rc = ceps2lpc_v(c_in.reshape(-1, c_in.shape[-1]).cpu()) # lpc_c - (N*(L-1), 16)
#         all_features = torch.cat((c_in.cpu(), lpc_c.unsqueeze(0)), -1)
        
#         # all_features = torch.cat((c_in.cpu(), qtz_c[:,:,-16:]), -1)
        
#         np.save('../samples/{}/{}_cin_{}.npy'.format(model_label_f, sample_name[0], cfg['note']), all_features.cpu().data.numpy())
#         np.save('../samples/{}/{}_r_{}.npy'.format(model_label_f, sample_name[0], cfg['note']), r.cpu().data.numpy())
        
        # 1-0 mask
        # mask[:nm_c.shape[1]//2*2] = torch.tensor((1,0)).repeat(nm_c.shape[1]//2)
        mask = None
        
        feat = nm_c[:,:,:-16].to('cuda')
        c_in, r, r_qtz = model_f.encoder(cfg=cfg, feat=feat, n_dim=cfg['code_dim'], mask = mask, l1=l1, l2=l2, vq_quantize=vq_quantize, scl_quantize=scl_quantize)     
        
        c_in *= MAXI
        
        e, lpc_c, rc = ceps2lpc_v(c_in.reshape(-1, c_in.shape[-1]).cpu()) # lpc_c - (N*(L-1), 16)
        all_features = torch.cat((c_in.cpu(), lpc_c.unsqueeze(0)), -1)
        

        np.save('../samples/{}/{}_cin_thrd_{}.npy'.format(model_label_f, sample_name[0], cfg['note']), \
                all_features.cpu().data.numpy())
        
        np.save('../samples/{}/{}_r_thrd_{}.npy'.format(model_label_f, sample_name[0], cfg['note']), \
                r.cpu().data.numpy())
        
        # np.save('../samples/{}/{}_qtz.npy'.format(model_label_f, sample_name[0]), \
                # qtz_c.cpu().data.numpy())
        
        # fake()
        break
        
        
#         if cfg['normalize']:
#             feat_out *= MAXI
        
#         savefig(data=r, tp='r', ns=ns)
#         savefig(data=r_qtz, tp='r_qtz', ns=ns)
#         savefig(data=feat*MAXI, tp='truth', ns=ns)
#         savefig(data=feat_out, tp='out', ns=ns)
        
#         saveaudio(wave=sample, tp='truth', ns=ns)

#         lpc = c[:, :, -16:].to(torch.float).to(device) # (1, tot*15, 16)  # Use original lpc for now
#         periods = (.1 + 50*feat_out[:,:,18:19]+100).to(torch.int32).to(device)
#         lpc_sample = torch.repeat_interleave(lpc, 160, dim=1) # (bt, tot_chunks*2400, 16)
        
#         feat_out = torch.transpose(feat_out, 1, 2) # (1, 20, 7*15*n)
#         x_out = model_s.generate_lpc(feat_out, periods, lpc_sample, tot_length)
        
#         saveaudio(wave=x_out[:,:,1:], tp='xout', ns=ns)

        
# if __name__ == '__main__':

#     r_s = np.random.rand(10, 18)
#     out = torch.tensor(quantize(r_s[:,1:])).to(device)


                    
    
            
            
        
    
        
        
        