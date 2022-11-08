import os
from sacred import Ingredient
from sacred import Experiment
import numpy as np

import time

ex = Experiment('config')
os.environ['TZ'] = 'EST+05EDT,M4.1.0,M10.5.0'
time.tzset()

@ex.config
def my_config():
    cfg = {
            # Data  
            'frame_size': 160, 
            'lpcoeffs_N': 16,
            'chunks': 7, # One second - 6.666 chunks
            'sr': 16000,
            'n_sample_seg': 2400, 
            'n_seg': 15, 
            'orig': True,
            'normalize': True, 
            'qtz': True,
            'scl_cb_path': '../codebook/scalar_center_256.npy',
            'cb_path': '../codebook/ceps_vq_codebook_2_1024_large_17.npy',
            'bl_scl_cb_path': '',
            'bl_cb_path': '',
            'n_entries': [2048],
            'code_dim': 17,
            'l1': 0, 
            'l2': 0,
            
            # Training
            'epochs': 1000,
            'batch_size': 10,
            'learning_rate': 0.001,
            'ema_decay':0.9999, 
            'upd_f_only': True,
    
            'transfer_model_f': None,
            'transfer_epoch_f': None,
            'transfer_model_s': None,
            'transfer_epoch_s': None,

            # Model
            'n_mels': None, 
            'num_blocks': 2,
            'num_layers': 10,
            'inp_channels': 1,
            'out_channels': 2, 
            'residual_channels': 128,
            'gate_channels': 256,
            'skip_channels': 128, 
            'kernel_size': 2,
            'cin_channels': 80,
            'cout_channels': 128,
            'num_workers': 2,
            'local': False,
            'fat_upsampler': True,
            'stft_loss': False,
        
            # Model hyper-parameters of WaveRNN
            'out_features': 20, 
            'gru_units1': 384, 
            'gru_units2': 16,
            'rnn_layers':2,
            'attn_units': 20,
            'fc_units': 20, 
            'packing': False,
            'bidirectional': False,
            

            'debugging': False,    
        
            # Synthesis
            'total_secs': 3, 
            'num_samples': 2,
            'model_label_s': None,
            'model_label_f': None,
            'epoch_s': None,
            'epoch_f': None,
            'note': '',
            
    }
    
    model_label = time.strftime("%m%d_%H%M%S")

