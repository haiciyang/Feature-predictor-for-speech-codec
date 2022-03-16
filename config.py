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
            'chunks': 7,
            'sr': 16000,
            'n_sample_seg': 2400, 
            'n_seg': 15, 
            
            # Training
            'epochs': 1000,
            'batch_size': 10,
            'learning_rate': 0.001,
            'ema_decay':0.9999, 
            
            'transfer_model': None,
            'transfer_epoch': None,

            # Model
            'n_mels': None, 
            'num_blocks': 2,
            'num_layers': 5,
            'residual_channels': 128,
            'gate_channels': 256,
            'skip_channels': 128, 
            'kernel_size': 2,
            'cin_channels': 80,
            'num_workers': 2,
            'local': False,

            'debugging': False,    
        
            # Synthesis
            'total_secs': 3, 
            'num_samples': 2,
            'model_label': None,
            'epoch': None,
            'note': '',
            'orig': True,
    }
    
    model_label = time.strftime("%m%d_%H%M%S")

