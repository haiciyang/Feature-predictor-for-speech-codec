import numpy as np

SURVIVORS = 5
NB_BANDS = 18


# CEPS_CODEBOOK = np.random.rand(3, 1024, 17)
        

def vq_quantize_mbest(codebook, nb_entries, x, ndim, mbest):
    
    # x - vector (ndim,)
    # codebook (nb_entries, ndim)
    
    x = np.expand_dims(x, axis=0) # (1, ndims)
    codebook = np.expand_dims(codebook, axis=1) # (nb_entries, ndims)
    
    dist = np.sum((x - codebook) ** 2, -1) # (nb_entries, )
    
    idx_mat = np.array(sorted(np.arange(nb_entries, dtype=int), key = lambda x: dist[x])[:mbest])

    curr_dist = np.array(sorted(dist)[:mbest])
        
    return idx_mat, curr_dist

def quantize_1stage(x, n_entries, CEPS_CODEBOOK): 
    
    # This function only takes in one vector. No batch operation
    # CODEBOOK - (nb_entries, ndims)
    
    curr_idx, curr_dist = vq_quantize_mbest(CEPS_CODEBOOK[0], n_entries[0], x, NB_BANDS-1, SURVIVORS) # (mbest, ), (mbest, )
    
    qx = CEPS_CODEBOOK[0][curr_idx[0]]

    return qx


def quantize_2stage_mbest(x, n_entries, CEPS_CODEBOOK): 
    
    # This function only takes in one vector. No batch operation
    # CODEBOOK - (nb_entries, ndims)
    
    glob_dist = np.zeros(SURVIVORS)
    index1 = np.zeros((2, SURVIVORS), dtype=int)
    index2 = np.zeros((2, SURVIVORS), dtype=int)
    
    curr_idx, curr_dist = vq_quantize_mbest(CEPS_CODEBOOK[0], n_entries[0], x, NB_BANDS-1, SURVIVORS) # (mbest, ), (mbest, )
    index1[0] = curr_idx
    
    for k in range(SURVIVORS):

        diff = x - CEPS_CODEBOOK[0][index1[0, k]]

        curr_idx, curr_dist = vq_quantize_mbest(CEPS_CODEBOOK[1], n_entries[1], diff, NB_BANDS-1, SURVIVORS)

        if k == 0:
            index2[0] = index1[0, k]
            index2[1] = curr_idx
            glob_dist = curr_dist


        elif curr_dist[0] < glob_dist[-1]:
            m = 0
            for p in range(SURVIVORS):
                if curr_dist[m] < glob_dist[p]:
                    for j in range(SURVIVORS-1, p, -1):
                        glob_dist[j] = glob_dist[j-1]
                        index2[0, j] = index2[0, j-1]
                        index2[1, j] = index2[1, j-1]
                    glob_dist[p] = curr_dist[m]
                    index2[0, p] = index1[0, k]
                    index2[1, p] = curr_idx[m]

                    m += 1

    qx = CEPS_CODEBOOK[0][index2[0, 0]] + CEPS_CODEBOOK[1][index2[1, 0]]
    
    return qx

        
        
def quantize_mstage(x, n_entries, CEPS_CODEBOOK): 
    
    # This function only takes in one vector. No batch operation
    # CODEBOOK - (nb_entries, ndims)
    
    n_stages = len(n_entries)

    glob_dist = np.zeros(SURVIVORS)
    
    index = np.zeros((n_stages, SURVIVORS), dtype=int)
    
    curr_idx, curr_dist = vq_quantize_mbest(CEPS_CODEBOOK[0], n_entries[0], x, NB_BANDS-1, SURVIVORS)
    
    index[0] = curr_idx
    
    for st in range(1, n_stages):
        
        last_idx = np.copy(index)
        
        for k in range(SURVIVORS):
            
            csum = 0
            for i in range(st):
                csum += CEPS_CODEBOOK[i][last_idx[i, k]]
            diff = x - csum

            curr_idx, curr_dist = vq_quantize_mbest(CEPS_CODEBOOK[st], n_entries[st], diff, NB_BANDS-1, SURVIVORS)

            if k == 0:
                index[:st] = last_idx[:st,k]
                index[st] = curr_idx
                glob_dist = curr_dist
                
            elif curr_dist[0] < glob_dist[-1]:
                m = 0
                for p in range(SURVIVORS):
                    if curr_dist[m] < glob_dist[p]:
                        for j in range(SURVIVORS-1, p, -1):
                            glob_dist[j] = glob_dist[j-1]
                            index[:st+1, j] = index[:st+1, j-1]
                        glob_dist[p] = curr_dist[m]
                        index[:st, p] = last_idx[:st, k]
                        index[st, p] = curr_idx[m]
                        m += 1

    csum = 0
    for i in range(n_stages):
        csum += CEPS_CODEBOOK[i][index[i, 0]]
    
    return csum


def quantize(r, n_entries, cb_path):
    
    '''
    Input - features (batch_size, ndims)
    Output - quantized featurs (batch_size, ndims)
    '''
    
    CEPS_CODEBOOK = np.load(cb_path, allow_pickle=True)
    
    if len(CEPS_CODEBOOK.shape) == 2:
        CEPS_CODEBOOK = np.expand_dims(CEPS_CODEBOOK, 0)
    
    batch_size, ndims = r.shape
    
    qr = []
    for vec in r:
        qvec = quantize_mstage(vec, n_entries, CEPS_CODEBOOK)
        qr.append(qvec)
    
    qr = np.reshape(qr, (batch_size, ndims))

    
    return qr

