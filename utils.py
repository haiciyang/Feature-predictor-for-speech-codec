import os
import torch
import librosa
import numpy as np
from config import ex
from sacred import Experiment
from torchaudio import transforms
from matplotlib import pyplot as plt
from torch.distributions.normal import Normal

use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")

# \mu law <-> Linear conversion functions

scale = 255.0/32768.0
scale_1 = 32768.0/255.0

def l2u(x):
    s = torch.sign(x)
    x = torch.abs(x)
    u = s*(128*K.log(1+scale*x)/K.log(256.0))
    u = K.clip(128 + u, 0, 255)
    return u

def tf_u2l(u):
    u = tf.cast(u,"float32")
    u = u - 128.0
    s = K.sign(u)
    u = K.abs(u)
    return s*scale_1*(K.exp(u/128.*K.log(256.0))-1)

def sample_from_gaussian(y_hat):
    assert y_hat.size(1) == 2

    y_hat = y_hat.transpose(1, 2)
    mean = y_hat[:, :, :1]
    log_std = y_hat[:, :, 1:]
    dist = Normal(mean, torch.exp(log_std))
    sample = dist.sample()
    # sample = torch.clamp(torch.clamp(sample, min=-1.), max=1.)
    del dist
    return sample

def reparam_gaussian(y_hat):
    
    assert y_hat.size(1) == 2
    
    y_hat = y_hat.transpose(1, 2)
    mean = y_hat[:, :, :1]
    log_std = y_hat[:, :, 1:]
    eps = torch.normal(0, 1, mean.shape).to(device)
    
    return mean + log_std * eps


def mel_spec(x):

    mels = [16, 80]
    output = []
    for n in mels:
        transform = transforms.MelSpectrogram(sample_rate=16000, n_fft=1024, hop_length=256, n_mels=n, f_min=125, f_max=7600).to('cuda')
        output.append(transform(x))
        
    output = torch.cat(output,2)
    
    return output

def stft(y, scale='linear'):
    D = torch.stft(y, n_fft=1024, hop_length=256, win_length=1024)#, window=torch.hann_window(1024).cuda())
    D = torch.sqrt(D.pow(2).sum(-1) + 1e-10)
    # D = torch.sqrt(torch.clamp(D.pow(2).sum(-1), min=1e-10))
    if scale == 'linear':
        return D
    elif scale == 'log':
        S = 2 * torch.log(torch.clamp(D, 1e-10, float("inf")))
        return S
    else:
        pass
    


def plot_spec(x):
    
    X = 20*np.log10(np.abs(librosa.feature.melspectrogram(x, sr=16000, n_fft=1024)))
    plt.imshow(X, origin='lower', aspect='auto')
    
    return plt

@ex.capture
def lpc_pred(cfg, x, lpc, N=None, n_repeat = None):
    
    # in_x - (bt, 1, 2400)
    # lpc - (bt, 15, 16)
    # N - number of 2400 segments
    
    N = cfg['chunks']*2400 if N is None else N
    n_repeat = cfg['frame_size'] if n_repeat is None else n_repeat
    
    lpc_N = cfg['lpcoeffs_N']
    lpc = torch.repeat_interleave(lpc, n_repeat, dim=1) # (bt, 2400, 16)
    
    x = torch.transpose(x, 1, 2) # (bt, 2400, 1)
    xs = x.shape
    pad_x = torch.cat((torch.zeros(xs[0], lpc_N, xs[2]).cuda(), x), 1) # (bt, 2416, 1))
    stack_x = torch.cat(
        [pad_x[:,(lpc_N - i):(lpc_N - i + N),:] for i in range(lpc_N)],
        2) # (bt, 2400, 16)
    
    pred = lpc * stack_x
    pred = - torch.sum(pred, axis=2)[:,None, :] # (bt, 1, 2400)
    
    return pred


def checkpoint(debugging, epoch, batch_id, duration, model_label, state_dict, train_loss, valid_loss, min_loss):

    result_path = 'results/'+ model_label +'.txt'
    if not os.path.exists('saved_models/'+model_label):
        os.mkdir('saved_models/'+model_label)
        
    model_path = 'saved_models/'+ model_label + '/' + model_label + '_' +str(epoch) +'.pth'

    if state_dict is not None: # when an epoch is finished
        # print(epoch, duration, train_loss, valid_loss)
        records = 'Epoch: {} | time: {:.2f} | train_loss: {:.4f} | valid_loss: {:.4f} \n'.format(epoch, duration, train_loss, valid_loss)
        if not debugging:
            if valid_loss < min_loss:
                min_loss = valid_loss
            torch.save(state_dict, model_path)
    else:
        records = 'Epoch: {} | step: {} | time: {:.2f} | train_loss: {:.4f} \n'.format(epoch, batch_id, duration, train_loss)
    
    print(records)
    if not debugging:
        with open(result_path, 'a+') as file:
            file.write(records)
            file.flush()

    return min_loss


def plot_training_output(y, y_hat, model_label, epoch):
    
    mean_y_hat = y_hat[0, 0, :].detach().cpu().numpy()
    plot_y = y[0,0,:].detach().cpu().numpy()

    if not os.path.exists('samples/'+model_label):
        os.mkdir('samples/'+model_label)

    Y_hat = 20*np.log10(
        np.abs(librosa.feature.melspectrogram(
            mean_y_hat, sr=16000, n_fft=1024)))
    plt.imshow(Y_hat, origin='lower', aspect='auto')
    plt.savefig('samples/{}/exc_out_{}.jpg'.format(model_label, epoch))
    plt.clf()
    Y = 20*np.log10(
        np.abs(librosa.feature.melspectrogram(
            plot_y, sr=16000, n_fft=1024)))
    plt.imshow(Y, origin='lower', aspect='auto')
    plt.savefig('samples/{}/exc_{}.jpg'.format(model_label, epoch))
    plt.clf()
