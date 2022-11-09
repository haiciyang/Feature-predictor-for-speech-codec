# NEURAL FEATURE PREDICTOR AND DISCRIMINATIVE RESIDUAL CODING FOR LOW-BITRATE SPEECH CODING

Open-source code for  H. Yang, W. Lim, M. Kim, Neural Feature Predictor and Discriminative Residual Coding for Low-bitrate speech coding",https://arxiv.org/pdf/2211.02506.pdf

## Prerequisites
#### Environment
- Python 3.6.0 <br>
- torch 1.8.0+cu111 <br>
- torchaudio 0.8.0 <br>
#### Data
- Librispeech - [https://sigsep.github.io/datasets/musdb.html#musdb18-compressed-stems](https://sigsep.github.io/datasets/musdb.html#musdb18-compressed-stems)
- This paper'e experiments uses 
#### LPCNet
We reuse part of the LPCNet's data processing code, and its model as our sample-level vocoder.
You chttps://github.com/asteroid-team/asteroid](https://github.com/asteroid-team/asteroid)

## Experiments
### Training Steps
#### Data Loaders

#### Feature predictor training

#### Codebook training

#### Vocoder (LPCNet) training
Note: We update LPCNet's original data retrieval , and apply our own dataset
1. Train LPCNet with clean features
1. Generate coded features
2. Fineturn LPCNet with the coded features

### Synthesis
#### Generate coded features 

#### Vocoder (LPCNet) Synthesis

### Model parameters
### Other numbers


