import time
import torch
from torch import nn

from torch.nn.utils.rnn import pad_packed_sequence

import utils
# from modules import Conv, ResBlock


class Wavernn(nn.Module):
    
    def __init__(self, in_features=20, out_features=20, num_layers=2, fc_unit=20):
        
        super(Wavernn, self). __init__()
        
        self.rnn = nn.GRU(in_features, out_features,2, batch_first=True)
        
        self.dual_fc = nn.Sequential(
            nn.Linear(out_features, fc_unit),
            nn.Tanh()
        )

    def forward(self, x):
        
        # x - (bt, C, L)
        x, _ = self.rnn(x) # x - (bt, L, rnn_out)
        
        x, out_lens = pad_packed_sequence(x, batch_first=True)
        x = torch.cat((x.unsqueeze(1),x.unsqueeze(1)),1) # (bt, 2, L, C_in)
        x = self.dual_fc(x) # x - (bt, 2, L, fc_out)
        
        # x = nn.functional.softmax(torch.sum(x, dim=1), dim = -1)
        x = torch.sum(x, dim=1)
        
        return x#, out_lens
        
        
        
        
        