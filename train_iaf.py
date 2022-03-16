import os
import gc
import json
import time
import librosa
import numpy as np

import torch
import torch.nn as nn
from torch import optim
from torchaudio import transforms
from torch.distributions.normal import Normal
from torch.utils.data import Dataset, DataLoader

import utils
from wavenet import Wavenet
from dataset import Libri_lpc_data
from modules import ExponentialMovingAverage, GaussianLoss

from config import ex
from sacred import Experiment

torch.backends.cudnn.benchmark = True
np.set_printoptions(precision=4)

use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")

criterion = GaussianLoss()

if args.teacher_name:
    teacher_name = args.teacher_name
else:
    teacher_name = 'iaf'
    

# LOAD DATASETS
train_dataset = Libri_lpc_data('train', cfg['chunks'])
test_dataset = Libri_lpc_data('test', cfg['chunks'])

train_loader = DataLoader(train_dataset, batch_size=cfg['batch_size'], shuffle=True, num_workers=4, pin_memory=True)
test_loader = DataLoader(test_dataset, batch_size=cfg['batch_size'], shuffle=True, num_workers=4, pin_memory=True)


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

def gaussian_ll(mean, logscale, sample):
    
    # mean.shape - (bt, 1, n_in_frames*320)
    # logscale.shape - (bt, 1, n_in_frames*320)
    # sample - (bt, n_out_frames*320)
    
    dist = D.Normal(mean, torch.exp(logscale))
    logp = dist.log_prob(sample)   
    
    return -torch.mean(torch.flatten(logp))

def quantization(c, bins=10):
    
    s = c.shape
    c = c.flatten()
    c *= bins
    for i in range(len(c)):
        ceil = torch.ceil(c[i])
        if ceil - c[i] < 0.5:
            c[i] = ceil
        else:
            c[i] = ceil - 1
    c /= bins
    
    c.to(torch.float32)
    c = c.reshape(s)
    
    return c

def build_student():
    model_s = Wavenet_Student(num_blocks_student=[1, 1, 1, 1, 1, 1],
                              num_layers=args.num_layers_s,
                              in_channels=args.in_channels,
                              residual_channels=args.residual_channels,
                              gate_channels=args.gate_channels,
                              skip_channels=args.skip_channels,
                              kernel_size=args.kernel_size,
                              cin_channels=args.cin_channels,
                              causal=True,
                              upsample_scales=[16, 16])
    return model_s

def clone_as_averaged_model(model_s, ema):
    assert ema is not None
    averaged_model = build_student()
    averaged_model.to(device)
    if args.num_gpu > 1:
        averaged_model = torch.nn.DataParallel(averaged_model)
    averaged_model.load_state_dict(model_s.state_dict())

    for name, param in averaged_model.named_parameters():
        if name in ema.shadow:
            param.data = ema.shadow[name].clone().data
    return averaged_model


def train(epoch, model_t, model_s, optimizer):
    
    epoch_loss = 0.
    start_time = time.time()
    display_step = 500
    display_loss = 0.
    display_start = time.time()
    
    if model_t is not None:
        model_t.eval()
    model_s.train()
    
    for batch_idx, (x, y, c) in enumerate(train_loader):

        x = torch.transpose(x, 1, 2).to(torch.float).to(device)
        y = torch.transpose(y, 1, 2).to(torch.float).to(device)
        
        feat = torch.transpose(c[:, 2:-2, :-16], 1,2).to(torch.float).to(device) # (bt, 20, 15)
        lpc = c[:, 2:-2, -16:].to(torch.float).to(device) # (bt, 15, 16) 
          
        q_0 = Normal(x.new_zeros(x.size()), x.new_ones(x.size()))
        z = q_0.sample()
        
        optimizer.zero_grad()
        if model_t is None:
            feat_up = model_s.upsample(feat)
        else:
            feat_up = model_t.upsample(feat)
            
        x_student, mu_s, logs_s = model_s(z, feat_up)  # q_T ~ N(mu_tot, 

        spec_student = stft(x_student[:, 0, 1:], scale='linear')
        spec_truth = stft(x[:, 0, 1:], scale='linear')
        
#         spec_student = mel_spec(x_student[:, 0, 1:])
#         spec_truth = mel_spec(x[:, 0, 1:])
        
        loss_frame = criterion_frame(spec_student, spec_truth)
#         loss_t = gaussian_ll(mu_s[:,0,:], logs_s[:,0,:], x_student[:, 0, 1:])
#         loss_tot = loss_t + loss_frame
        loss_tot = loss_frame
        loss_tot.backward()

        nn.utils.clip_grad_norm_(model_s.parameters(), 10.)
        optimizer.step()

        running_loss[0] += loss_tot.item() / display_step

        epoch_loss += loss_tot.item()
        if (batch_idx + 1) % display_step == 0:
            end_time = time.time()
            print('Global Step : {}, [{}, {}] [Total Loss, KL Loss, Reg Loss, Frame Loss] : {}'
                  .format(global_step, epoch, batch_idx + 1, np.array(running_loss)))
            print('{} Step Time : {}'.format(display_step, end_time - start_time))
            start_time = time.time()
            running_loss = [0.0, 0.0] # Changed from 4 to 2
#         del loss_tot, loss_frame, loss_KL, loss_reg, loss_t, x, y, c, c_up, stft_student, stft_truth, q_0, z
        del loss_tot, loss_frame, x, y, c, c_up, spec_student, spec_truth, q_0, z
        del x_student, mu_s, logs_s  #, mu_logs_t
        
        # break
    print('{} Epoch Training Loss : {:.4f}'.format(epoch, epoch_loss / (len(train_loader))))
    return epoch_loss / len(train_loader)


def evaluate(model_t, model_s, ema=None):
    if ema is not None:
        model_s_ema = clone_as_averaged_model(model_s, ema)
    if model_t is not None:
        model_t.eval()
    model_s_ema.eval()
    running_loss = [0., 0.]
    epoch_loss = 0.

    display_step = 100
    for batch_idx, (x, y, c, _) in enumerate(test_loader):
        
        x = torch.transpose(x, 1, 2).to(torch.float).to(device)
        y = torch.transpose(y, 1, 2).to(torch.float).to(device)
        
        feat = torch.transpose(c[:, 2:-2, :-16], 1,2).to(torch.float).to(device) # (bt, 20, 15)
        lpc = c[:, 2:-2, -16:].to(torch.float).to(device) # (bt, 15, 16) 
          
        q_0 = Normal(x.new_zeros(x.size()), x.new_ones(x.size()))
        z = q_0.sample()
        
        optimizer.zero_grad()
        if model_t is None:
            feat_up = model_s.upsample(feat)
        else:
            feat_up = model_t.upsample(feat)

        x_student, mu_s, logs_s = model_s_ema(z, feat_up)


        spec_student = stft(x_student[:, 0, 1:], scale='linear')
        spec_truth = stft(x[:, 0, 1:], scale='linear')
        
        loss_frame = criterion_frame(spec_student, spec_truth.detach())

        loss_tot = loss_frame

        running_loss[0] += loss_tot.item() / display_step
#         running_loss[1] += loss_KL.item() / display_step
#         running_loss[2] += loss_reg.item() / display_step
        running_loss[1] += loss_frame.item() / display_step
        epoch_loss += loss_tot.item()

        if (batch_idx + 1) % display_step == 0:
            print('{} [Total, KL, Reg, Frame Loss] : {}'.format(batch_idx + 1, np.array(running_loss)))
            running_loss = [0., 0.] # Changed from 4 to 2
#         del loss_tot, loss_frame, loss_KL, loss_reg, loss_t, x, y, c, c_up, stft_student, stft_truth, q_0, z
        del loss_tot, loss_frame, x, y, c, c_up, spec_student, spec_truth, q_0, z
        del x_student, mu_s, logs_s #, mu_logs_t
        
        # break
    epoch_loss /= len(test_loader)
    print('Evaluation Loss : {:.4f}'.format(epoch_loss))
    del model_s_ema
    return epoch_loss


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


def save_checkpoint(model, optimizer, global_step, global_epoch, ema=None):
    checkpoint_path = os.path.join(args.save, teacher_name, args.model_name, "checkpoint_step{:09d}.pth".format(global_step))
    
    optimizer_state = optimizer.state_dict()
    torch.save({"state_dict": model.state_dict(),
                "optimizer": optimizer_state,
                "global_step": global_step,
                "global_epoch": global_epoch}, checkpoint_path)
    if ema is not None:
        averaged_model = clone_as_averaged_model(model, ema)
        checkpoint_path = os.path.join(args.save, teacher_name, args.model_name, "checkpoint_step{:09d}_ema.pth".format(global_step))
        torch.save({"state_dict": averaged_model.state_dict(),
                    "optimizer": optimizer_state,
                    "global_step": global_step,
                    "global_epoch": global_epoch}, checkpoint_path)


def load_checkpoint(step, model_s, optimizer, ema=None):
    global global_step
    global global_epoch

    checkpoint_path = os.path.join(args.save, teacher_name, args.model_name, "checkpoint_step{:09d}.pth".format(step))
    print("Load checkpoint from: {}".format(checkpoint_path))
    checkpoint = torch.load(checkpoint_path)
    model_s.load_state_dict(checkpoint["state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer"])
    global_step = checkpoint["global_step"]
    global_epoch = checkpoint["global_epoch"]

    if ema is not None:
        checkpoint_path = os.path.join(args.save, teacher_name, args.model_name, "checkpoint_step{:09d}_ema.pth".format(step))
        checkpoint = torch.load(checkpoint_path)
        averaged_model = build_student()
        averaged_model.to(device)
        try:
            averaged_model.load_state_dict(checkpoint["state_dict"])
        except RuntimeError:
            print("INFO: this model is trained with DataParallel. Creating new state_dict without module...")
            state_dict = checkpoint["state_dict"]
            from collections import OrderedDict
            new_state_dict = OrderedDict()
            for k, v in state_dict.items():
                name = k[7:]  # remove `module.`
                new_state_dict[name] = v
            averaged_model.load_state_dict(new_state_dict)
        for name, param in averaged_model.named_parameters():
            if param.requires_grad:
                ema.register(name, param.data)
    return model_s, optimizer, ema


def load_teacher_checkpoint(path, model_t):
    print("Load checkpoint from: {}".format(path))
    checkpoint = torch.load(path, map_location=lambda storage, loc: storage)
    model_t.load_state_dict(checkpoint["state_dict"])
    return model_t

# Loading teacher model
model_t = None
if args.teacher_name:
    teacher_step = args.teacher_load_step
    path = os.path.join(args.load, args.teacher_name, "checkpoint_step{:09d}_ema.pth".format(teacher_step))
    model_t = build_model()
    model_t = load_teacher_checkpoint(path, model_t)
    model_t.to(device)

model_s = build_student()
model_s.to(device)

if args.num_gpu > 1:
    if not model_t is None:
        model_t = torch.nn.DataParallel(model_t)
    model_s = torch.nn.DataParallel(model_s)

optimizer = optim.Adam(model_s.parameters(), lr=args.learning_rate)
criterion_t = KL_Loss()
criterion_frame = nn.MSELoss()

ema = ExponentialMovingAverage(args.ema_decay)
for name, param in model_s.named_parameters():
    if param.requires_grad:
        ema.register(name, param.data)
if not model_t is None:
    for name, param in model_t.named_parameters():
        if param.requires_grad:
            param.requires_grad = False

global_step, global_epoch = 0, 0
load_step = args.load_step

log = open(os.path.join(args.log, '{}.txt'.format(args.model_name)), 'w')
state = {k: v for k, v in args._get_kwargs()}

if load_step == 0:
    list_train_loss, list_loss = [], []
    log.write(json.dumps(state) + '\n')
    test_loss = 100.0
else:
    model_s, optimizer, ema = load_checkpoint(load_step, model_s, optimizer, ema)
    list_train_loss = np.load('{}/{}_train.npy'.format(args.loss, args.model_name)).tolist()
    list_loss = np.load('{}/{}.npy'.format(args.loss, args.model_name)).tolist()
    list_train_loss = list_train_loss[:global_epoch]
    list_loss = list_loss[:global_epoch]
    test_loss = np.min(list_loss)

for epoch in range(global_epoch + 1, args.epochs + 1):
    training_epoch_loss = train(epoch, model_t, model_s, optimizer, ema)
    with torch.no_grad():
        test_epoch_loss = evaluate(model_t, model_s, ema)

    state['training_loss'] = training_epoch_loss
    state['eval_loss'] = test_epoch_loss
    state['epoch'] = epoch
    list_train_loss.append(training_epoch_loss)
    list_loss.append(test_epoch_loss)

    if test_loss > test_epoch_loss:
        test_loss = test_epoch_loss
        save_checkpoint(model_s, optimizer, global_step, epoch, ema)
        print('Epoch {} Model Saved! Loss : {:.4f}'.format(epoch, test_loss))
        synthesize(model_t, model_s, ema)
    np.save('{}/{}_train.npy'.format(args.loss, args.model_name), list_train_loss)
    np.save('{}/{}.npy'.format(args.loss, args.model_name), list_loss)

    log.write('%s\n' % json.dumps(state))
    log.flush()
    print(state)

log.close()
