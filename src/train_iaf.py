
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
from models.wavenet import Wavenet
from models.wavenet_iaf import Wavenet_IAF
from datasets.dataset import Libri_lpc_data
from datasets.dataset_orig import Libri_lpc_data_orig
from models.modules import ExponentialMovingAverage, GaussianLoss

from config import ex
from sacred import Experiment

torch.backends.cudnn.benchmark = True
np.set_printoptions(precision=4)

use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")

criterion = GaussianLoss()
mseloss = nn.MSELoss()


@ex.capture
def build_wn_model(cfg):
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


@ex.capture
def build_model(cfg):
    model = Wavenet_IAF(
                    num_blocks_iaf=[1, 1, 1, 1, 1, 1],
                    num_layers=cfg['num_layers_iaf'],
                    residual_channels=cfg['residual_channels'],
                    gate_channels=cfg['gate_channels'],
                    skip_channels=cfg['skip_channels'],
                    kernel_size=cfg['kernel_size'],
                    cin_channels=cfg['cin_channels'],
                    cout_channels=cfg['cout_channels'],
                    upsample_scales=[10, 16]
                    )
    return model


def gaussian_ll(mean, logscale, sample):
    
    # mean.shape - (bt, 1, n_in_frames*320)
    # logscale.shape - (bt, 1, n_in_frames*320)
    # sample - (bt, n_out_frames*320)
    
    dist = Normal(mean, torch.exp(logscale))
    logp = dist.log_prob(sample)   
    
    return -torch.mean(torch.flatten(logp))


# def clone_as_averaged_model(model_s, ema):
#     assert ema is not None
#     averaged_model = build_student()
#     averaged_model.to(device)
#     if args.num_gpu > 1:
#         averaged_model = torch.nn.DataParallel(averaged_model)
#     averaged_model.load_state_dict(model_s.state_dict())

#     for name, param in averaged_model.named_parameters():
#         if name in ema.shadow:
#             param.data = ema.shadow[name].clone().data
#     return averaged_model


def train(model, wn_model, optimizer, train_loader, epoch, model_label, debugging, cin_channels):
    
    epoch_loss = 0.
    start_time = time.time()
    display_step = 500
    display_loss = 0.
    display_start = time.time()
    
    if wn_model is not None:
        wn_model.eval()
    model.train()
    
    for batch_idx, (x, c) in enumerate(train_loader):

        x = x.to(torch.float).to(device) # x - torch.Size([3, 1, 2400])
        
        if cin_channels == 20:
            feat = torch.transpose(c[:, 2:-2, :-16], 1,2).to(torch.float).to(device) # (bt, 20, 15)
        else:
            feat = torch.transpose(c[:, 2:-2, :], 1,2).to(torch.float).to(device) # (bt, 20, 15)
        lpc = c[:, 2:-2, -16:].to(torch.float).to(device) # (bt, 15, 16) 
        
        pred = utils.lpc_pred(x=x, lpc=lpc) # (bt, 1, 2400)
          
        q_0 = Normal(x.new_zeros(x.size()), x.new_ones(x.size()))
        z = q_0.sample()
        
        optimizer.zero_grad()
        if wn_model is None:
            feat_up = model.upsample(feat)
        else:
            feat_up = wn_model.upsample(feat)
            
        exc = x - torch.roll(pred,shifts=1,dims=2) #(bt, 1, L) at i
        
        exc_hat, mu_tot, logs_tot = model(z, feat_up)  # q_T ~ N(mu_tot, 

        if batch_idx == 0:
            utils.plot_training_output(exc, exc_hat, model_label, epoch)
        
        spec_out = utils.stft(exc_hat[:, 0, 1:], scale='linear')
        spec_truth = utils.stft(exc[:, 0, 1:], scale='linear') 
#         spec_out = mel_spec(x_student[:, 0, 1:])
#         spec_truth = mel_spec(x[:, 0, 1:])

        loss_f = mseloss(spec_out, spec_truth)
        loss_t = gaussian_ll(mu_tot[:,0,:], logs_tot[:,0,:], exc[:, 0, 1:])
        
        loss = loss_f + loss_t
        
        loss.backward()
        
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
            
        if debugging:
            break
        
        del loss, loss_f, loss_t, x, c, lpc, feat, feat_up, pred, spec_out, spec_truth, q_0, z
        del exc, exc_hat, mu_tot, logs_tot
        
    return epoch_loss / len(train_loader)


def evaluate(model, wn_model, test_loader, debugging, cin_channels):
    
    if wn_model is not None:
        wn_model.eval()
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
        
        pred = utils.lpc_pred(x=x, lpc=lpc) # (bt, 1, 2400)
            
        exc = x - torch.roll(pred,shifts=1,dims=2) #(bt, 1, L) at i
        # exc = x[:,:,1:] - pred[:,:,:-1]
        
        q_0 = Normal(x.new_zeros(x.size()), x.new_ones(x.size()))
        z = q_0.sample()

        if wn_model is None:
            feat_up = model.upsample(feat)
        else:
            feat_up = wn_model.upsample(feat)
        
        # if inp_channels == 1:
        #     inp = x
        # elif inp_channels == 3:
        #     inp = torch.cat((x, exc, pred.to(device)), 1) # (bt, 3, n*2400)
        
        exc_hat, mu_tot, logs_tot = model(z, feat_up)  # q_T ~ N(mu_tot, 
        
        spec_out = utils.stft(exc_hat[:, 0, 1:], scale='linear')
        spec_truth = utils.stft(exc[:, 0, 1:], scale='linear') 
#         spec_out = mel_spec(x_student[:, 0, 1:])
#         spec_truth = mel_spec(x[:, 0, 1:])
        

        loss_f = mseloss(spec_out, spec_truth)
        loss_t = gaussian_ll(mu_tot[:,0,:], logs_tot[:,0,:], exc[:, 0, 1:])
        
        loss = loss_f + loss_t
        
        nn.utils.clip_grad_norm_(model.parameters(), 10.)

        epoch_loss += loss

        del loss, loss_f, loss_t, x, c, lpc, feat, feat_up, pred, spec_out, spec_truth, q_0, z
        del mu_tot, logs_tot, exc, exc_hat
            
        if debugging:
            break

    return epoch_loss / len(test_loader)


def synthesize(model_t, model_s, ema=None):
    global global_step
    if ema is not None:
        model_s_ema = clone_as_averaged_model(model_s, ema)
    model_s_ema.eval()
    for batch_idx, (x, y, c, _) in enumerate(synth_loader):
        if batch_idx == 0:
            x, c = x.to(device), c.to(device)
            
            if args.n_mels is not None:
                transform = transforms.MelSpectrogram(sample_rate=16000, n_fft=1024, hop_length=256, n_mels=args.n_mels, f_min=125, f_max=7600).to(device)
                c = transform(x)[:,0, :,:-1]
            q_0 = Normal(x.new_zeros(x.size()), x.new_ones(x.size()))
            z = q_0.sample()
            wav_truth_name = '{}/{}/{}/generate_{}_{}_truth.wav'.format(args.sample_path, teacher_name,
                            args.model_name, global_step, batch_idx)
#             librosa.output.write_wav(wav_truth_name, y.squeeze().numpy(), sr=22050)
            sf.write(wav_truth_name, y.squeeze().numpy(), 16000, 'PCM_16')
            print('{} Saved!'.format(wav_truth_name))

            torch.cuda.synchronize()
            start_time = time.time()
            if model_t is None:
                c_up = model_s.upsample(c)
            else:
                c_up = model_t.upsample(c)
#             

            with torch.no_grad():
                if args.num_gpu == 1:
                    y_gen = model_s_ema.generate(z, c_up).squeeze()
                else:
                    y_gen = model_s_ema.module.generate(z, c_up).squeeze()
            torch.cuda.synchronize()
            print('{} seconds'.format(time.time() - start_time))
            wav = y_gen.to(torch.device("cpu")).data.numpy()
            wav_name = '{}/{}/{}/generate_{}_{}.wav'.format(args.sample_path, teacher_name,
                                                            args.model_name, global_step, batch_idx)
#             librosa.output.write_wav(wav_name, wav, sr=22050)
            sf.write(wav_name, wav, 16000, 'PCM_16')
            print('{} Saved!'.format(wav_name))
            del y_gen, wav, x,  y, c, c_up, z, q_0
    del model_s_ema



@ex.automain
def run(cfg, model_label): 
    
    # ----- Wirte and print the hyper-parameters -------
    result_path = '../results/'+ model_label +'.txt'
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
        
        transfer_model_path = '../saved_models/{}/{}_{}.pth'.format(str(cfg['transfer_model']), str(cfg['transfer_model']), str(cfg['transfer_epoch']))
        print("Load checkpoint from: {}".format(transfer_model_path))
        model.load_state_dict(torch.load(transfer_model_path))
        
    wn_model = None
    if cfg['wn_model'] is not None:
        
        wn_model = build_wn_model()
        wn_model.to(device)
        
        wn_model_path = '../saved_models/{}/{}_{}.pth'.format(str(cfg['wn_model']), str(cfg['wn_model']), str(cfg['transfer_epoch']))
        print("Load checkpoint from: {}".format(wn_model_path))
        wn_model.load_state_dict(torch.load(wn_model_path))

    optimizer = optim.Adam(model.parameters(), lr=cfg['learning_rate'])

    min_loss = float('inf')
    
    for epoch in range(cfg['epochs']):
        
        start = time.time()
        
        train_epoch_loss = train(model, wn_model, optimizer, train_loader, epoch, model_label, cfg['debugging'], cfg['cin_channels'])
        
        with torch.no_grad():
            test_epoch_loss = evaluate(model, wn_model, test_loader, cfg['debugging'], cfg['cin_channels'])
            
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

        
