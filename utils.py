import os
import torch
from config import ex
from sacred import Experiment
from torch.distributions.normal import Normal


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

@ex.capture
def lpc_pred(cfg, x, lpc, N=None, n_repeat = None):
    
    # in_x - (bt, 1, 2400)
    # lpc - (bt, 15, 16)
    
    # N - number of 2400 segments
    
    N = cfg['chunks']*1024 if N is None else N
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
