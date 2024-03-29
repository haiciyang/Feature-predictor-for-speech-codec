import time
import torch
import numpy as np
from tqdm import tqdm

from torch import nn
from torch import Tensor
# from quantization.vq_func import quantize, scl_quantize
from typing import Optional, Tuple

import torch.nn.functional as F
from torch.nn.utils.rnn import pad_packed_sequence

from config import ex
from sacred import Experiment

import utils
# from modules import Conv, ResBlock

device = 'cuda'

class Wavernn(nn.Module):
    
    def __init__(self, in_features=20, gru_units1=384, gru_units2=16, fc_units=20, attn_units=20, rnn_layers=2, bidirectional = False, packing=False):
        
        # scale: how much portion of the residuals left 
        
        super(Wavernn, self). __init__()
        
        self.scale = 1
        
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
        
        # self.mask_rnn = nn.GRU(in_features, fc_units, 1, bidirectional=True, batch_first=True) # (Bt, L, n_dim)
        # self.mask_fc = nn.Sequential(
        #     nn.Linear(fc_units*2, 2),
        #     nn.Tanh()
        # )
        
        # self.loc_attn = LocationAwareAttention(hidden_dim = attn_units, smoothing=True)


    def forward(self, x: Tensor, h1=None, h2=None) -> Tensor :
        
        """
        Input: x - (bt, L, C)
        """
        
        # x = self.loop_attention(x)
        
        x, h1 = self.rnn1(x, h1) 
        # if self.bidirectional == True:
        #     x = slef.take_mean(x)   # x - (bt, L, rnn_out)
        # print(x.shape)
        # x = self.loop_attention(x)
        x, h2 = self.rnn2(x, h2) 
        # if self.bidirectional == True:
        #     x = slef.take_mean(x)   # x - (bt, L, rnn_out)
        
        # x, _ = self.rnn(x) # x - (bt, L, rnn_out)
        
        if self.packing:
            x, out_lens = pad_packed_sequence(x, batch_first=True)
        
#         if self.bidirectional == True:        
#             x = self.take_pred_mean(x)
        x = self.relu(x) 
        
        x = torch.cat((x.unsqueeze(1),x.unsqueeze(1)),1) # (bt, 2, L, C_in)
        x = self.dual_fc(x) # x - (bt, 2, L, fc_out)
        # x = nn.functional.softmax(torch.sum(x, dim=1), dim = -1)
        x = torch.sum(x, dim=1)        
        # x = torch.tanh(x)
        
        # mask = self.mask(x)
        # mask = F.tanh((mask+scale) * 100) # (bt, L, 2)
        
        
        
        # x = self.loop_attention(x)

        return x, h1, h2#, out_lens

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
    
    
    # @ex.capture
    def encoder(self, cfg, feat, mask, l1, l2, vq_quantize = None, scl_quantize = None, qtz=True):

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
        ind1_mask = torch.zeros(B, L, 1).to(device)
        ind2_mask = torch.zeros(B, L, 1).to(device)
        
        cb_tot = [0,0,0,0,0]
        
        
        for i in range(c_in.shape[1]-1):

            f_out, h1, h2 = self.forward(x=c_in[:, i:i+1, :], h1=h1, h2=h2)# inputs previous frames; predicts i+1th frame
            f_out = f_out[:,-1,:] 
            r_s = feat[:,i,:-2] - f_out.clone()
            r[:,i,:] = r_s
            
            
            # --- Define indicator for scalar quantization and VQ --- 
            if mask is None: # Use threshold
                ind1 = (abs(r_s[:,0]) > l1).to(int).unsqueeze(1) # (bt, 1)
                num_ind1 += sum(ind1)
                ind1_mask[:, i, :] = ind1
                
                ind2 = (torch.sum(abs(r_s[:,1:]), -1) > l2).to(int).unsqueeze(1) # (bt, 1)
                num_ind2 += sum(ind2)
                ind2_mask[:, i, :] = ind2
                
            if mask is not None: # Use input mask
                ind1 = mask[:,i,0]
                ind2 = mask[:,i,1]
    
            if qtz: # Dont qtz when synthesize residuals for training codebook
                
                # --- Scalar quantization for c0 ---
                for k in range(len(ind1)):
                    if ind1[k,0]:
                        rq, cb_t = scl_quantize((r_s[k:k+1,0:1]).cpu().data.numpy(), cfg['scl_cb_path'])
                        r_qtz[k:k+1,i,0:1] = torch.tensor(rq).to(device)
                        cb_tot[0] += cb_t
                    elif cfg['bl_scl_cb_path']:
                        rq, cb_t = scl_quantize((r_s[k:k+1,0:1]).cpu().data.numpy(), cfg['bl_scl_cb_path'])
                        r_qtz[k:k+1,i,0:1] = torch.tensor(rq).to(device)
                        cb_tot[1] += cb_t
                        
                # --- VQ for C1-C17 ---
                for k in range(len(ind2)):
                    if ind2[k,0]:
                        rq, cb_t = vq_quantize((r_s[k:k+1,1:]).cpu().data.numpy(), \
                                               cfg['cb_path'])
                        r_qtz[k:k+1,i,1:] = torch.tensor(rq).to(device)
                        cb_tot[2] += cb_t[0]
                        cb_tot[3] += cb_t[1]
                    elif cfg['bl_cb_path']:
                        rq, cb_t = vq_quantize((r_s[k:k+1, 1:]).cpu().data.numpy(), \
                                               cfg['bl_cb_path'])
                        r_qtz[k:k+1,i,1:] = torch.tensor(rq).to(device)
                        # cb_tot1 += cb_t[0]
                        cb_tot[4] += cb_t[-1]

                c_in[:,i+1,:-2] = f_out + r_qtz[:,i,:]
                
            else:
                r_under[:,i,0:1] = r_s[:,0:1] * (1-ind1)
                r_under[:,i,1:] = r_s[:,1:] * (1-ind2)

                r[:,i, 0:1] = r_s[:,0:1] * ind1
                r[:,i, 1:] = r_s[:,1:] * ind2

                
                c_in[:,i+1,:-2] = f_out + r[:,i,:]  # 0.01 * (torch.rand(f_out.shape)-0.5).to(device)/2
        
        # print(num_ind1 / L, num_ind2 / L)
        # return c_in[:,1:,:], r, r_qtz, r_under, (num_ind1/B/L).data, (num_ind2/B/L).data, cb_tot
        return c_in[:,1:,:], r, r_qtz, r_under, ind1_mask, ind2_mask, cb_tot
    
    
    def mask_enc(self, feat, cfg=None, vq_quantize=None, scl_quantize=None, qtz=False):

        B, L, C = feat.shape

        mask, _ = self.mask_rnn(feat) # (B, L, 2) (-1,1)
        mask = self.mask_fc(mask) # (B, L, 2) (-1,1)
        
        mask = torch.sigmoid(mask * self.scale) #  (B, L, 1)
        
        
        scl_mask = mask[:,:,0:1]
        vct_mask = mask[:,:,1:2]

        
        c_in = torch.zeros(B, L, C-2).to(device) # (B, L, C)
        r_orig = torch.zeros(B, L, C-2).to(device)
        r = torch.zeros(B, L, C-2).to(device)
        r_bl = torch.zeros(B, L, C-2).to(device)
        
        # print(scl_mask.shape)
        
        h1 = h2 = None
        
        c_inp = torch.zeros(B, 1, C-2).to(device)
        
        cb_tot = [0,0,0,0,0]
        
        for i in range(L):

            c_inp = torch.cat((c_inp, feat[:,i:i+1, -2:]), -1)
            f_out, h1, h2 = self.forward(x=c_inp, h1=h1, h2=h2)# inputs previous frames; predicts i+1th frame
            r_s = feat[:,i:i+1,:-2].clone() - f_out.clone() # (B, 1, 2)
            r_orig[:,i:i+1,:] = r_s
            
            r_mask, r_mask_bl, cb_tot = self.apply_mask(cfg, r_s, scl_mask[:,i:i+1, :], vct_mask[:,i:i+1,:], vq_quantize, scl_quantize, cb_tot, qtz)

#             r_scl = r_s[:,:,0:1] * scl_mask[:,i:i+1, :].clone()
#             r_vct = r_s[:,:,1:] * vct_mask[:,i:i+1,:].clone()
 
#             r_all = torch.cat((r_scl, r_vct), -1)

            c_inp = f_out.clone() + r_mask.clone()
            
            c_in[:,i:i+1,:] = c_inp

            r[:,i:i+1,:] = r_mask.clone()
            if r_mask_bl is not None:
                r_bl[:,i:i+1,:] = r_mask_bl.clone()
            
        c_in = torch.cat((c_in, feat[:,:,-2:]), -1)
        
        return c_in, r_orig, r, r_bl, scl_mask, vct_mask, cb_tot
    
    
    def apply_mask(self, cfg, r_s, scl_mask, vct_mask, vq_quantize, scl_quantize, cb_tot, qtz=False):
        
        # mask - [B, 1, 1]
        # r_s - [B, 1, 18]
        
        B = r_s.shape[0]
        C = r_s.shape[-1]
        
        r_mask_bl = None

        if not qtz:
            r_scl = r_s[:,:,0:1] * scl_mask
            r_vct = r_s[:,:,1:] * vct_mask
            
            r_scl_bl = r_s[:,:,0:1] * (1-scl_mask)
            r_vct_bl = r_s[:,:,1:] * (1-vct_mask)
        
            r_mask_bl = torch.cat((r_scl_bl, r_vct_bl), -1)


        if qtz: # Dont qtz when synthesize residuals for training codebook
            
            r_scl = torch.zeros(B, 1, 1).to(device)
            r_vct = torch.zeros(B, 1, C-1).to(device)
            
            # --- Scalar quantization for c0 ---
            for k in range(len(scl_mask)):
                if scl_mask[k,0,0]:
                    rq, cb_t = scl_quantize((r_s[k:k+1,0,0:1]).cpu().data.numpy(), cfg['scl_cb_path'])
                    r_scl[k:k+1,0,:] = torch.tensor(rq).to(device)
                    cb_tot[0] += cb_t
                elif cfg['bl_scl_cb_path']:
                    rq, cb_t = scl_quantize((r_s[k:k+1,0:1]).cpu().data.numpy(), cfg['bl_scl_cb_path'])
                    r_scl[k:k+1,0,:] = torch.tensor(rq).to(device)
                    cb_tot[1] += cb_t
                        
            # --- VQ for C1-C17 ---
            for k in range(len(scl_mask)):
                if vct_mask[k,0,0]:
                    rq, cb_t = vq_quantize((r_s[k:k+1,0, 1:]).cpu().data.numpy(), cfg['cb_path'])
                    r_vct[k:k+1,0,:] = torch.tensor(rq).to(device)
                    cb_tot[2] += cb_t[0]
                    cb_tot[3] += cb_t[1]
                elif cfg['bl_cb_path']:
                    rq, cb_t = vq_quantize((r_s[k:k+1, 0, 1:]).cpu().data.numpy(), cfg['bl_cb_path'])
                    r_vct[k:k+1,0,:] = torch.tensor(rq).to(device)
                    cb_tot[3] += cb_t
        
        r_mask = torch.cat((r_scl, r_vct), -1)
        
        return r_mask, r_mask_bl, cb_tot


    
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
        
        
        
        
        