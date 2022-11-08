import time
import torch
import numpy as np
from tqdm import tqdm

from torch import nn
from torch import Tensor
# from quantization.vq_func import quantize, scl_quantize
from typing import Optional, Tuple

from torch.nn.utils.rnn import pad_packed_sequence

from config import ex
from sacred import Experiment

import utils
# from modules import Conv, ResBlock

device = 'cuda'

class Wavernn_para(nn.Module):
    
    def __init__(self, in_features=20, gru_units1=384, gru_units2=16, fc_units=20, attn_units=20, rnn_layers=2, bidirectional = False, packing=False):
        
        super(Wavernn_para, self). __init__()
        
        self.relu = nn.ReLU()
        
        self.packing = packing
        self.bidirectional = bidirectional
        
        self.rnn1 = nn.GRU(in_features, gru_units1, 1, bidirectional=bidirectional, batch_first=True)
        self.rnn2 = nn.GRU(gru_units1, gru_units2, 1, bidirectional=bidirectional, batch_first=True)
        
        self.rnn3 = nn.GRU(fc_units, fc_units, 1, bidirectional=bidirectional, batch_first=True)
        
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


    def forward(self, x: Tensor, h1=None, h2=None, h3=None) -> Tensor :
        
        """
        Input: x - (bt, L, C)
        """
        
        x, h1 = self.rnn1(x, h1) # 
        x, h2 = self.rnn2(x, h2) 

        x = self.relu(x) 
        
        x = torch.cat((x.unsqueeze(1),x.unsqueeze(1)),1) # (bt, 2, L, C_in)
        x = self.dual_fc(x) # x - (bt, 2, L, fc_out)
        x_mid = torch.sum(x, dim=1)  
        
        x_out, h3 = self.rnn3(torch.flip(x_mid, [1]), h3) # (bt, L, 18)
        x_out = torch.tanh(x_out)
        
        

        return x_mid, x_out, h1, h2, h3 #, out_lens
      
        
    
    # @ex.capture
    def encoder(self, cfg, feat, mask, l1, l2, vq_quantize = None, scl_quantize = None, qtz=1):

        '''
        Input: feat, n_dim, mask
            - **feat**: (batch_size, seq_length, n_dims)
            - **n_dim**:
            - **mask**: (seq_length)
        Output: c_in, r
            - **c_in** (batch_size, seq_length, dim): prediced frames plus quantized residual
            - **r** (batch_size, seq_length, dim): quantized residual
        '''
        B, L, C = feat.shape
        c_in = torch.zeros(B, L+1, C).to(device) # (B, L, C)
        c_in[:,1:,-2:] = feat[:,:,-2:]
        r = torch.zeros(B, L, 18).to(device)
        r_qtz = torch.zeros(B, L, 18).to(device)
        r_under = torch.zeros(B, L, 18).to(device)
        h1 = h2 = None

        num_ind1 = 0
        num_ind2 = 0

        for i in range(c_in.shape[1]-1):

            f_out, h1, h2 = self.forward(x=c_in[:, i:i+1, :], h1=h1, h2=h2)# inputs previous frames; predicts i+1th frame
            f_out = f_out[:,-1,:] 
            r_s = feat[:,i,:-2] - f_out.clone()
            # r[:,i,:] = r_s
            
            
            # --- Define indicator for scalar quantization and VQ --- 
            if mask is None: # Use threshold
                ind1 = (abs(r_s[:,0]) > l1).to(int).unsqueeze(1) # (bt, 1)
                num_ind1 += sum(ind1)
                
                ind2 = (torch.sum(abs(r_s[:,1:]), -1) > l2).to(int).unsqueeze(1) # (bt, 1)
                num_ind2 += sum(ind2)
                
            if mask is not None: # Use input mask
                ind1 = ind2 = mask[i]
                
            r_under[:,i,1:] = r_s[:,1:] * (1-ind2)
            
            r_s[:,0:1] = r_s[:,0:1] * ind1
            r_s[:,1:] = r_s[:,1:] * ind2
            
            r[:,i,:] = r_s
            
            if qtz: # Dont qtz when synthesize residuals for training codebook
                
                # --- Scalar quantization for c0 ---
                for k in range(len(ind1)):
                    if ind1[k,0]:
                        r_qtz[k:k+1,i,0:1] = torch.tensor(scl_quantize((r_s[k:k+1,0:1]).cpu().data.numpy(), cfg['scl_cb_path'])).to(device)
                    elif cfg['bl_scl_cb_path']:
                        r_qtz[k:k+1,i,0:1] = torch.tensor(scl_quantize((r_s[k:k+1,0:1]).cpu().data.numpy(), cfg['bl_scl_cb_path'])).to(device)

                # --- VQ for C1-C17 ---
                for k in range(len(ind2)):
                    if ind2[k,0]:
                        r_qtz[k:k+1,i,1:] = torch.tensor(vq_quantize((r_s[k:k+1,1:]).cpu().data.numpy(), cfg['cb_path'])).to(device)
                    elif cfg['bl_cb_path']:
                        r_qtz[k:k+1,i,1:] = torch.tensor(vq_quantize((r_s[k:k+1,1:]).cpu().data.numpy(), cfg['bl_cb_path'])).to(device)
            
                c_in[:,i+1,:-2] = f_out + r_qtz[:,i,:]
                
            else:
                c_in[:,i+1,:-2] = f_out + r[:,i,:]  # 0.01 * (torch.rand(f_out.shape)-0.5).to(device)/2
        
        # print(num_ind1 / L, num_ind2 / L)
        return c_in[:,1:,:], r, r_qtz, r_under, (num_ind1/B/L).data, (num_ind2/B/L).data
    
    @ex.capture
    def decoder(self, cfg, feat, r):


        c = torch.zeros(feat.shape).to(device) # (B, L, C)
        c[:,:,-2:] = feat[:,:,-2:]

        for i in range(c.shape[1]-1):

            f_out, h1, h2 = self.forward(x=c[:, i:i+1, :], h1=h1, h2=h2)[:,-1,:] # inputs previous frames; predicts i+1th frame
            c[:,i+1,:-2] = f_out + r[:,i+1, :]
            
        
        return c
            
          
        
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
        
        
        
        
        