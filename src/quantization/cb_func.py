'''
Learn codebooks for the cepstrum vector quantization

'''
import os
import gc
import json
import time
import math
import librosa
import numpy as np
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


def vq_train(data, codebook, nb_entries):
    
    # data - (nb_vectors, ndims)

    ndims = data.shape[1]
    
    codebook[0] = np.mean(data, 0)

    e = 1
    while e < nb_entries:
        
        # split
        codebook[e, :] = codebook[0, :]
        delta = .001 * (np.random.rand(e, ndims) / 2)
        codebook[:e, :] += delta
        e += 1

        # update
        for _ in range(4):  
            codebook[:e, :] = update(data, codebook[:e, :], e)

    test = np.sum(codebook, -1) # (nb_entries, )
    
    for _ in range(20):
        codebook = update(data, codebook, nb_entries)
        
    return codebook

def find_nearest(data, codebook):
    
    # data - (nb_vectors, ndims)
    # codebook - (nb_entries, ndims)
    
    data = np.expand_dims(data, axis=0) # (1, nb_vectors, ndims)
    codebook = np.expand_dims(codebook, axis=1) # (nb_entries, 1, ndims)
    
    dist = np.sum((data - codebook) ** 2, -1) # (nb_entries, nb_vectors)
    
    min_index = np.argmin(dist, 0) # (nb_vectors,)
    
    return min_index
        
    
def update(data, codebook, nb_entries_tmp):
    
    nb_vectors = data.shape[0]
    ndims = data.shape[1]
    
    count = np.zeros((nb_entries_tmp,1))
    
    min_index = find_nearest(data, codebook)
    
    codebook = np.zeros((nb_entries_tmp, ndims))
    
    for i in range(nb_vectors):
        
        n = min_index[i]
        count[n] += 1
        codebook[n] += data[i]
    
    codebook /= count + 1e-20
    
    # for i in range(nb_entries_tmp):
    #     if sum(codebook_upd[i]) != 0:
    #         codebook[i] = codebook_upd[i]

    w2 = np.sum((count/nb_vectors)**2)

    print('{} - min: {}, max: {}, small: {}, error: {}'\
          .format(nb_entries_tmp, min(count), max(count), sum(count == 0), w2))
    # print(np.mean(codebook, 0))

    return codebook
    
        
def quantize(codebook, data):
    
    # codebook - (nb_entries, ndims)
    # data - (nb_vectors, ndims)
    
    # find nearest
    min_index = find_nearest(data, codebook)
    qdata = np.array([codebook[i] for i in min_index])
    
    return qdata
    
