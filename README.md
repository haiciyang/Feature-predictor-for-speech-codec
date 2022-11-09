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
### Model parameters
Model parameters are written in <code>src/config.py</code>.
Some important parameters


### Training Steps
#### 1. Feature predictor training
<code>python train_frame.py</code> Assumingly saved model in [NEW_Model]
#### 2. Codebook training

Learn codebook for the above-threshold residuals
<code> python3 train_cb.py with cfg.transfer_model_f=0722_001326 cfg.transfer_epoch_f=4000 cfg.gru_units1=384 cfg.gru_units2=128 cfg.fc_units=18 cfg.l1=0.09 cfg.l2=0.28 cfg.n_entries=1024,1024 cfg.train_bl=False cfg.scl_clusters=256 cfg.scl_clusters_bl=16 cfg.note=[CB_Name]</code>

Learn codebook for the below-threshold residuals
<code> python3 train_cb.py with cfg.transfer_model_f=0722_001326 cfg.transfer_epoch_f=4000 cfg.gru_units1=384 cfg.gru_units2=128 cfg.fc_units=18 cfg.l1=0.09 cfg.l2=0.28 cfg.n_entries=512 cfg.train_bl=True cfg.scl_clusters=256 cfg.scl_clusters_bl=16 cfg.note=[CB_Name]</code>
#### 3. Vocoder (LPCNet) training
Note: We update LPCNet's original data retrieval , and apply our own dataset

- Train LPCNet with clean features

- Generate coded features

<code> python3 generate_qtz_features.py with cfg.transfer_model_f=[NEW_Model] cfg.transfer_epoch_f=[New_Model_Epoch] cfg.gru_units1=384 cfg.gru_units2=128 cfg.fc_units=18 cfg.orig=True cfg.scl_cb_path='../codebooks/scalar_center_256_[CB_Name].npy' cfg.cb_path='../codebooks/ceps_vq_codebook_[CB_Name].npy' cfg.l1=0.09 cfg.l2=0.28 cfg.qtz=True cfg.bl_scl_cb_path='../codebooks/scalar_center_16_[CB_Name]_bl.npy' cfg.bl_cb_path='../codebooks/ceps_vq_codebook_[CB_Name]_bl.npy'</code>

- Fineturn LPCNet with the coded features

Go to LPCNet


### Synthesis
#### 1. Generate coded features 
<code> python synthesis_qtz.py with cfg.model_label_f=0722_001326 cfg.epoch_f=4000 cfg.gru_units1=384 cfg.gru_units2=128 cfg.fc_units=18 cfg.orig=True cfg.num_samples=3 cfg.total_secs=3 cfg.l1=0.09 cfg.l2=0.28 cfg.qtz=True cfg.note='syn20_bl' cfg.scl_cb_path='../codebooks/scalar_center_256_syn20_tr.npy' cfg.cb_path='../codebooks/ceps_vq_codebook_2_1024_syn20.npy' cfg.bl_scl_cb_path='../codebooks/scalar_center_16_syn20_bl_tr.npy' cfg.bl_cb_path='../codebooks/ceps_vq_codebook_1_512_syn20_bl.npy' </code>
#### 2. Vocoder (LPCNet) Synthesis


