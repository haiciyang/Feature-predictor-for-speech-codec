import os
import glob
import torch
import random
import librosa
import numpy as np



FRAME_SIZE_5MS = 2
OVERLAP_SIZE_5MS = 2
WINDOW_SIZE_5MS = FRAME_SIZE_5MS + OVERLAP_SIZE_5MS
FRAME_SIZE = 80 * FRAME_SIZE_5MS
OVERLAP_SIZE = 80 * OVERLAP_SIZE_5MS

FRAME_SIZE = 80*FRAME_SIZE_5MS
OVERLAP_SIZE = 80*OVERLAP_SIZE_5MS
WINDOW_SIZE = FRAME_SIZE + OVERLAP_SIZE
FREQ_SIZE = WINDOW_SIZE//2 + 1
NB_BANDS = 18
LPC_ORDER = 16

COMPENSATION = torch.tensor([
    0.8, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.666667, 0.5, 0.5, 0.5, 0.333333, 0.25, 0.25, 0.2, 0.166667, 0.173913
])

DCT_TABLE = torch.zeros(NB_BANDS, NB_BANDS)
for i in range(NB_BANDS):
    for j in range(NB_BANDS):
        DCT_TABLE[i, j] = torch.cos(torch.tensor((i+.5)*j*np.pi/NB_BANDS))
        if j==0:
            DCT_TABLE[i, j] *= torch.sqrt(torch.tensor(.5))
            
            
def idct(in_):
    
    out = torch.zeros(in_.shape)
    for i in range(NB_BANDS):
        sm = 0
        for j in range(NB_BANDS):
            sm += in_[:, j] * DCT_TABLE[i, j]
        out[:, i] = sm * torch.sqrt(torch.tensor(2./NB_BANDS))
    return out

def interp_band_gain(bandE):
    
    eband5ms = [
# 0  200 400 600 800  1k 1.2 1.4 1.6  2k 2.4 2.8 3.2  4k 4.8 5.6 6.8  8k*/
  0,  1,  2,  3,  4,  5,  6,  7,  8, 10, 12, 14, 16, 20, 24, 28, 34, 40
]
    g = torch.zeros(bandE.shape[0], FREQ_SIZE)
    for i in range(NB_BANDS - 1):
        band_size = (eband5ms[i+1]-eband5ms[i])*WINDOW_SIZE_5MS
        for j in range(band_size):
            frac = float(j) / band_size
            g[:, (eband5ms[i]*WINDOW_SIZE_5MS) + j] = (1-frac) * bandE[:, i] + frac * bandE[:, i+1]
    return g


def _celt_lpc_s(ac, p):
    
    error = ac[0]
    lpc = torch.zeros(p)#.to('cuda')
    rc = torch.zeros(p).to(torch.float64)#.to('cuda')
    if ac[0] != 0:
        for i in range(p):
            rr = torch.tensor(0.)#.to('cuda')
            for j in range(i):
                rr = rr + lpc[j].clone() * ac[i - j]
            rr = rr + (ac[i + 1])
            r = -(rr) / error
            rc[i] = r
            lpc[i] = r
            
            for j in range(int((i+1) / 2)):
                tmp1 = lpc[j].clone()
                tmp2 = lpc[i-1-j].clone()
                lpc[j]     = tmp1 + (r * tmp2)
                lpc[i-1-j] = tmp2 + (r * tmp1)

            error = error - ((r * r) * error)
            if error < (ac[0]  / (2**10)):
                break
            if (error < 0.001 * ac[0]):
                break
    # print(f'ac[0]={ac[0]} error={error}')

    return error, lpc, rc

# def _celt_lpc(ac, p):
    
    # error = ac[:, 0]
    # lpc = torch.zeros(ac.shape[0], p).requires_grad_ #.to('cuda')
    # rc = torch.zeros(ac.shape[0], p).to(torch.float64)#.to('cuda')

#     if ac[:, 0].any() != 0:
#         for i in range(p):
#             rr = torch.zeros(ac.shape[0]).to(torch.float)
#             for j in range(i):
#                 rr = rr + lpc[:, j] * ac[:, i - j]
                
#             rr = rr + ac[:, i + 1]
#             r = - rr / error
#             rc[:,i] = r
#             lpc[:,i] = r
            
#             # for j in range(int((i+1) / 2)):
#             #     tmp1 = lpc[:,j]
#             #     tmp2 = lpc[:,i-1-j]
#             #     lpc[:,j]     = tmp1 + (r * tmp2)
#             #     lpc[:,i-1-j] = tmp2 + (r * tmp1)

#             error = error - ((r * r) * error)
#             # if error < (ac[0]  / (2**10)):
#             #     break
#             # Bail out once we get 30 dB gain 
#             if (error < 0.001 * ac[:, 0]).any():
#                 break
#     print(f'ac[0]={ac[0]} error={error}')
    # return error, lpc, rc

def ceps2lpc_v(cepstrum):
    
    # cepstrum - torch.size (N, C)
    
    tmp = cepstrum[:, :NB_BANDS]
    
    scale = torch.zeros(NB_BANDS)#.to('cuda')
    scale[0] += 4
    
    # tmp[:,0] += 4
    # Ex = idct(tmp) # inverse cepstrums to Bank-scale spectrogram
    Ex = idct(tmp + scale[None,:]) # inverse cepstrums to Bank-scale spectrogram
    Ex = (10.0 ** Ex) * COMPENSATION

    Xr = interp_band_gain(Ex) #  interpolate linear spectrogram 
    # Xr[: FREQ_SIZE-1] = 0

    
    acr = torch.fft.irfft(Xr) #  calculate autocorrelation
    # acr = torch.real(torch.fft.ifft(Xr))

    acr = acr[:, :LPC_ORDER+1]  # [:, p] autocorrelation values
    

    # ---- -40 dB noise floor --- 
    acr[:,0] += acr[:, 0] * 0.0001 + 320/12/38.
    
    # ---- Lag windowing ---- 
    for i in range(1, LPC_ORDER+1):
        acr[:,i] *= (1 - 0.00006 * i * i)
    
    
    lpc = []
    for l in range(len(acr)):
        e, lpc_, rc = _celt_lpc_s(acr[l], LPC_ORDER)
        lpc.append(lpc_.unsqueeze(0))

    # e, lpc_, rc = _celt_lpc(acr, LPC_ORDER) #  calculate lpc
    lpc = torch.cat(lpc, 0)
    
    return e, lpc, rc