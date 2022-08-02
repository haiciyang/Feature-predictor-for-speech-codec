import time
import torch
from torch import nn

from torch.nn.utils.rnn import pad_packed_sequence

import utils
# from modules import Conv, ResBlock


class Wavernn(nn.Module):
    
    def __init__(self, in_features=20, out_features=20, num_layers=2, fc_unit=20):
        
        super(Wavernn, self). __init__()