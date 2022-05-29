import time
import torch
from torch import nn

import utils
from modules import Conv, ResBlock


class Wavenet(nn.Module):
    def __init__(self, out_channels=1, num_blocks=3, num_layers=10, inp_channels=1,
                 residual_channels=512, gate_channels=512, skip_channels=512,
                 kernel_size=2, cin_channels=36, cout_channels=36,
                 upsample_scales=None, causal=True, local=True, fat_upsampler=True):
        super(Wavenet, self). __init__()

        self.causal = causal
        self.num_blocks = num_blocks
        self.num_layers = num_layers
        self.inp_channels = inp_channels
        self.out_channels = out_channels
        self.gate_channels = gate_channels
        self.residual_channels = residual_channels
        self.skip_channels = skip_channels
        self.cin_channels = cin_channels
        self.cout_channels = cout_channels 
        self.kernel_size = kernel_size
        self.fat_upsampler = fat_upsampler
        
        if not self.fat_upsampler: 
            assert self.cout_channels == self.cin_channels 
        
        self.local = local

        self.front_channels = 32
        self.front_conv = nn.Sequential(
            Conv(self.inp_channels, self.residual_channels, self.front_channels, causal=self.causal),
            nn.ReLU()
        )

        self.res_blocks = nn.ModuleList()
        for b in range(self.num_blocks):
            for n in range(self.num_layers):
                self.res_blocks.append(ResBlock(self.residual_channels, self.gate_channels, self.skip_channels,
                                                self.kernel_size, dilation=self.kernel_size**n,
                                                cout_channels=self.cout_channels, local_conditioning=True,
                                                causal=self.causal, mode='SAME'))

        self.final_conv = nn.Sequential(
            nn.ReLU(),
            Conv(self.skip_channels, self.skip_channels, 1, causal=self.causal),
            nn.ReLU(),
            Conv(self.skip_channels, self.out_channels, 1, causal=self.causal), 
            nn.Softmax(dim=1) # For using mu-law
            
        )
        
        if self.fat_upsampler:
            self.c_conv = nn.Sequential(
                nn.Conv1d(self.cin_channels, self.cout_channels, 3, padding=1),
                nn.Tanh(),
                nn.Conv1d(self.cout_channels, self.cout_channels, 3, padding=1),
                nn.Tanh()
            )

            self.c_fc = nn.Sequential(
                nn.Linear(self.cout_channels, self.cout_channels),
                nn.Tanh(),
                nn.Linear(self.cout_channels, self.cout_channels),
                nn.Tanh(),
            )
        
        self.upsample_conv = nn.ModuleList()
        for s in upsample_scales:
            convt = nn.ConvTranspose2d(1, 1, (3, 2 * s), padding=(1, s // 2), stride=(1, s))
            convt = nn.utils.weight_norm(convt)
            nn.init.kaiming_normal_(convt.weight)
            self.upsample_conv.append(convt)
            self.upsample_conv.append(nn.LeakyReLU(0.4))
            
        self.embedding = nn.Embedding(256, 64)

    def forward(self, x, periods, c):
        
        if not self.local:
            cfeat = self.upsample(c, periods)
        else:
            cfeat = torch.repeat_interleave(c, 160, dim=-1)

        out = self.wavenet(x, cfeat)
        return out

#     def generate_lpc(self, x, exc, c, lpc, inp_channels,):
        
#         # lpc -  (1, chunks*15, 16)
#         # c - (1, 36, chunks*15)
        
#         rf_size = self.receptive_field_size()
#         x_rf = torch.zeros(1, 1, rf_size).to(torch.device('cuda'))

#         # x_rf = torch.zeros(1, 1, rf_size).to(torch.device('cuda'))
#         # x = torch.zeros(1, 1, num_samples + 1).to(torch.device('cuda'))
#         # pred = torch.zeros(1, 1, num_samples + 1).to(torch.device('cuda'))
#         # exc = torch.zeros(1, 1, num_samples + 1).to(torch.device('cuda'))
#         # x_out = torch.zeros(1, 1, num_samples + 1).to(torch.device('cuda'))
            
#         if not self.local:
#             c_upsampled = self.upsample(c)
#         else:
#             c_upsampled = torch.repeat_interleave(c, 160, dim=-1)
            
#         local_cond = c_upsampled

#         timer = time.perf_counter()
#         torch.cuda.synchronize()
        
#         # for i in range(num_samples):
            
#             if (i+1) % 1000 == 0:
#                 torch.cuda.synchronize()
#                 timer_end = time.perf_counter()
#                 print("generating {}-th sample: {:.4f} samples per second..".format(i+1, 1000/(timer_end - timer)))
#                 timer = time.perf_counter()

#             if i >= rf_size:
#                 start_idx = i - rf_size + 1
#             else:
#                 start_idx = 0

#             if local_cond is not None:
#                 cond_c = local_cond[:, :, start_idx:i + 1]
#             else:
#                 cond_c = None

#             i_rf = min(i+1, rf_size)            
#             x_in = x[:, :, -i_rf:]

#             lpc_coef = lpc[:,i // 160,:].unsqueeze(1) #(bt, 1, 16)

#             pred_in = utils.lpc_pred(x=x_in, lpc=lpc_coef, N=i_rf, n_repeat=i_rf)

#             if inp_channels == 1:
#                 x_inp = x_in
#             elif inp_channels == 3:
#                 x_inp = torch.cat((x_in, exc[:, :, start_idx:i + 1], pred_in.to('cuda')), 1)

#             out = self.wavenet(x_inp, cond_c)

#             exc_out = utils.sample_from_gaussian(out[:, :, -1:])


#         return exc_out, pred_in

            
            # # Add de-emphasis filter
            # coef = 0.85
            # x_out[:, :, i+1] = coef * x[:, :, -2] + x[:, :, -1]

        # return x_out[:, :, 1:].cpu(), pred[:, :, 1:].cpu(), exc[:,:,1:].cpu()
        # return x[:,:,1:].cpu(), pred[:, :, 1:].cpu(), exc[:,:,1:].cpu()

    
    def generate(self, num_samples, c=None):
        # lpc -  (1, tot*15, 16)
        
        rf_size = self.receptive_field_size()

        # x_rf = torch.zeros(1, 1, rf_size).to(torch.device('cuda'))
        x = torch.zeros(1, 1, num_samples + 1).to(torch.device('cuda'))

        if not self.local:
            c_upsampled = self.upsample(c)
        else:
            c_upsampled = torch.repeat_interleave(c, 160, dim=-1)
            
        local_cond = c_upsampled

        timer = time.perf_counter()
        torch.cuda.synchronize()
        for i in range(num_samples):
            if (i+1) % 1000 == 0:
                torch.cuda.synchronize()
                timer_end = time.perf_counter()
                print("generating {}-th sample: {:.4f} samples per second..".format(i+1, 1000/(timer_end - timer)))
                timer = time.perf_counter()
            if i >= rf_size:
                start_idx = i - rf_size + 1
            else:
                start_idx = 0

            cond_c = local_cond[:, :, start_idx:i + 1]
            
            i_rf = min(i, rf_size)
            x_in = x[:, :, -i_rf:]

            out = self.wavenet(x_in, cond_c)
            x = torch.roll(x, shifts=-1, dims=2)
            x[:, :, -1] = utils.sample_from_gaussian(out[:, :, -1:])

        return x[:, :, 1:].cpu()


    def roll_dim2_and_zerofill_last(self, x, n):
        x_left, x_right = x[:, :, :-n], x[:, :, -n:]
        x_left.zero_()
        return torch.cat((x_right, x_left), dim=2)

    def upsample(self, c, periods):
        
        emb_p = self.embedding(periods) # (bt, L, 1, 64)
        emb_p = torch.transpose(emb_p[:,:,0,:], 1, 2) # (bt, 64, L) 
        
        cfeat = torch.cat((c, emb_p), 1) # (bt, 64+C, L)
        
        if self.fat_upsampler:
            cfeat = torch.transpose(self.c_conv(cfeat), 1, 2) # (bt, L, C)
            cfeat = torch.transpose(self.c_fc(cfeat), 1, 2) # (br, C, L) 
        
        if self.upsample_conv is not None:
            # B x 1 x C x T'
            cfeat = cfeat.unsqueeze(1)
            for f in self.upsample_conv:
                cfeat = f(cfeat)
            # B x C x T
            cfeat = cfeat.squeeze(1)
        return cfeat

    def wavenet(self, tensor, c=None):
        
        h = self.front_conv(tensor)
        skip = 0
        for i, f in enumerate(self.res_blocks):
            h, s = f(h, c)
            skip += s
        out = self.final_conv(skip)
        return out

    def receptive_field_size(self):
        num_dir = 1 if self.causal else 2
        dilations = [2 ** (i % self.num_layers) for i in range(self.num_layers * self.num_blocks)]
        return num_dir * (self.kernel_size - 1) * sum(dilations) + self.front_channels
