import os
import glob
import torch
import numpy as np
from tqdm import tqdm

task = 'train'
print(task)

if task == 'train':
    path = '/media/sdb1/Data/libri_lpc/train/*.s16'
elif task == 'val':
    path = '/media/sdb1/Data/libri_lpc/val/*.s16'
    
files = glob.glob(path)

# The numbers are hard_coded
frame_size = 160
nb_features = 20 + 16
nb_used_features = 20
feature_chunk_size = 15
pcm_chunk_size = frame_size*feature_chunk_size
lookahead=2
lpcoeffs_N = 16

for data_f in tqdm(files):
    

    feature_f = data_f[:-8] + 'features.f32'
    name = data_f[:-8].split('/')[-1]
    
    if os.path.exists('/media/sdb1/Data/libri_lpc_pt/train/'+name+'_in_data.pt') and os.path.exists('/media/sdb1/Data/libri_lpc_pt/train/'+name+'_out_data.pt') and os.path.exists('/media/sdb1/Data/libri_lpc_pt/train/'+name+'_features.pt'):
        
        os.system('rm ' + data_f)
        os.system('rm ' + feature_f)
        continue
    else:
        data = np.memmap(data_f, dtype='int16', mode='r')
        features = np.memmap(feature_f, dtype='float32', mode='r')
        
        os.system('rm ' + data_f)
        os.system('rm ' + feature_f)

        nb_frames = (len(data)//(2*pcm_chunk_size)-1)
        data = data[(4-lookahead)*2*frame_size:]
        data = data[:nb_frames*2*pcm_chunk_size]

        data = np.reshape(data, (nb_frames, pcm_chunk_size, 2))

        in_data = torch.tensor(data[:,:,:1])
        out_data = torch.tensor(data[:,:,1:])

        sizeof = features.strides[-1]
        features = np.lib.stride_tricks.as_strided(
            features, 
            shape=(nb_frames, feature_chunk_size+4, nb_features),
            strides=(feature_chunk_size*nb_features*sizeof, nb_features*sizeof, sizeof)
        )

        features = torch.tensor(features)


        torch.save(in_data, '/media/sdb1/Data/libri_lpc_pt/'+task+'/'+name+'_in_data.pt')
        torch.save(out_data, '/media/sdb1/Data/libri_lpc_pt/'+task+'/'+name+'_out_data.pt')
        torch.save(features, '/media/sdb1/Data/libri_lpc_pt/'+task+'/'+name+'_features.pt')

    del data, features
    



