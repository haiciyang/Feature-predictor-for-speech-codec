import time
import torch
import numpy as np
from torch import nn
from torch import Tensor
from typing import Optional, Tuple

from torch.nn.utils.rnn import pad_packed_sequence

import utils
# from modules import Conv, ResBlock


class Wavernn(nn.Module):
    
    def __init__(self, in_features=20, gru_units1=384, gru_units2=16, fc_units=20, attn_units=20, rnn_layers=2, bidirectional = False, packing=False):
        
        super(Wavernn, self). __init__()
        
        self.relu = nn.ReLU()
        
        self.packing = packing
        self.bidirectional = bidirectional
        
        self.rnn1 = nn.GRU(in_features, gru_units1, 1, bidirectional=bidirectional, batch_first=True)
        self.rnn2 = nn.GRU(gru_units1, gru_units2, 1, bidirectional=bidirectional, batch_first=True)
        
        # self.rnn = nn.GRU(in_features, gru_units2, rnn_layers, bidirectional=bidirectional, batch_first=True)
        
        # self.dual_fc = nn.Sequential(
        #     nn.Linear(out_features, fc_units),
        #     nn.Tanh()
        # )
        
        self.dual_fc = nn.Sequential(
            nn.Linear(gru_units2, fc_units),
            # nn.ReLU()
            nn.Tanh()
        )
        
        # self.loc_attn = LocationAwareAttention(hidden_dim = attn_units, smoothing=True)


    def forward(self, x: Tensor) -> Tensor :
        
        """
        Input: x - (bt, L, C)
        """
        
        # x = self.loop_attention(x)
        
        x, _ = self.rnn1(x) 
#         if self.bidirectional == True:
#             x = slef.take_mean(x)   # x - (bt, L, rnn_out)

        x, _ = self.rnn2(x) 
#         if self.bidirectional == True:
#             x = slef.take_mean(x)   # x - (bt, L, rnn_out)
        
        # x, _ = self.rnn(x) # x - (bt, L, rnn_out)
        
        if self.packing:
            x, out_lens = pad_packed_sequence(x, batch_first=True)
        
        if self.bidirectional == True:        
            x = self.take_pred_mean(x)
        x = self.relu(x) 
        
        x = torch.cat((x.unsqueeze(1),x.unsqueeze(1)),1) # (bt, 2, L, C_in)
        x = self.dual_fc(x) # x - (bt, 2, L, fc_out)
        # x = nn.functional.softmax(torch.sum(x, dim=1), dim = -1)
        x = torch.sum(x, dim=1)        
        # x = torch.tanh(x)
        
        # x = self.loop_attention(x)

        return x#, out_lens

    def loop_attention(self, x: Tensor, attn_range: int = 10) -> Tensor :
        
        """
        The attention layer process sample by sample in an autoregressive manner. 
        Input: x - (bt, L, fc_out(18))
               att_range - the attention range for one sample
        Output: 
        
        """
        L = x.shape[1]
        
        last_attn = None
        x_out = []
        
        for i in range(L):
            
            """
            For every step of the loop, localizaiton attention takes h_{i-l}, .., h{i}, and output y{i} 
            """
            start_id = max(0, i-attn_range+1)
            
            query = x[:, i:i+1, :] # (bt, 1, hidden_dim)
            value = x[:, start_id:i+1, :] # (bt, l, hidden_dim)
            
            out, last_attn = self.loc_attn(query, value, last_attn)
            
            x_out.append(out)
        
        x_out = torch.cat(x_out, 1)
        
        return x_out         
        
          
    def take_mean(self, x):
        
        """
        The function is used in the bidirectional WaveGRU
        x - (bt, L, rnn_out*2)
        """
        units = x.shape[-1]//2
        x = torch.mean(x[:,:,:unites], x[:,:,unites:]) 
        
        return x
    
    def take_pred_mean(self, x):
        
        """
        The function is used in the bidirectional WaveGRU
        x - (bt, L, rnn_out*2)
        """
        
        units = x.shape[-1]//2
        x_a = x[:,:,:units] # [x_1, x_2 .., x_n,  x_{n+1}]
        x_b = x[:,:,units:] # [x_{-1}, x_0 .., x_{n-1} ]
        
        y = (x_a[:,:-2,:] + x_b[:,2:,:])/2
        
        return y
            
          
        
class LocationAwareAttention(nn.Module):
    
    """
    Applies a location-aware attention mechanism on the output features from the decoder.
    Location-aware attention proposed in "Attention-Based Models for Speech Recognition" paper.
    The location-aware attention mechanism is performing well in speech recognition tasks.
    We refer to implementation of ClovaCall Attention style.
    Args:
        hidden_dim (int): dimesion of hidden state vector
        smoothing (bool): flag indication whether to use smoothing or not.
    Inputs: query, value, last_attn, smoothing
        - **query** (batch, q_len, hidden_dim): tensor containing the output features from the decoder.
        - **value** (batch, v_len, hidden_dim): tensor containing features of the encoded input sequence.
        - **last_attn** (batch_size * num_heads, v_len): tensor containing previous timestep`s attention (alignment)
    Returns: output, attn
        - **output** (batch, output_len, dimensions): tensor containing the feature from encoder outputs
        - **attn** (batch * num_heads, v_len): tensor containing the attention (alignment) from the encoder outputs.
    Reference:
        - **Attention-Based Models for Speech Recognition**: https://arxiv.org/abs/1506.07503
        - **ClovaCall**: https://github.com/clovaai/ClovaCall/blob/master/las.pytorch/models/attention.py
    """
    
    def __init__(self, hidden_dim: int, smoothing: bool = True) -> None:
        
        super(LocationAwareAttention, self).__init__()
        self.hidden_dim = hidden_dim
        self.conv1d = nn.Conv1d(in_channels=1, out_channels=hidden_dim, kernel_size=3, padding=1)
        self.query_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.value_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.score_proj = nn.Linear(hidden_dim, 1, bias=True)
        self.bias = nn.Parameter(torch.rand(hidden_dim).uniform_(-0.1, 0.1))
        self.smoothing = smoothing

    def forward(self, query: Tensor, value: Tensor, last_attn: Tensor) -> Tuple[Tensor, Tensor]:
        batch_size, hidden_dim, seq_len = query.size(0), query.size(2), value.size(1)

        # Initialize previous attention (alignment) to zeros
        if last_attn is None:
            last_attn = value.new_zeros(batch_size, seq_len)
        elif last_attn.shape[-1] != seq_len:
            last_attn = torch.cat((last_attn, torch.zeros(batch_size, 1).to('cuda')), 1)
            
        conv_attn = torch.transpose(self.conv1d(last_attn.unsqueeze(1)), 1, 2)
        score = self.score_proj(torch.tanh(
                self.query_proj(query.reshape(-1, hidden_dim)).view(batch_size, -1, hidden_dim)
                + self.value_proj(value.reshape(-1, hidden_dim)).view(batch_size, -1, hidden_dim)
                + conv_attn
                + self.bias
        )).squeeze(dim=-1)

        if self.smoothing:
            score = torch.sigmoid(score)
            attn = torch.div(score, score.sum(dim=-1).unsqueeze(dim=-1))
        else:
            attn = F.softmax(score, dim=-1)

        context = torch.bmm(attn.unsqueeze(dim=1), value) # Bx1xT X BxTxD => Bx1xD

        return context, attn
        
        
        
        
        