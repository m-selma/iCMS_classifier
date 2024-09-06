# iCMS_classifier
# Date: 2024-09-06

import pandas as pd
import numpy as np
from scipy.stats import rankdata
from scipy.spatial.distance import cosine
import warnings
from sklearn.preprocessing import StandardScaler
import multiprocessing as mp
import random

# Loading table of up/downregulated genes for iCMS2/3
markers = pd.read_csv("~/Desktop/CRC/Selma/supp_table.csv", sep=',', header=0)

# Loading counts matrix
cnts = pd.read_csv("~/Desktop/CRC/CRC_TPM_1063_counts.csv", sep=',', header=0, index_col=0)

# Checking that we have 716 unique markers
marker_list = pd.unique(markers[['iCMS2_Up', 'iCMS2_Down', 'iCMS3_Up', 'iCMS3_Down']].values.ravel('K'))
marker_list_iCMS2 = pd.unique(markers[['iCMS2_Up', 'iCMS2_Down']].values.ravel('K'))
marker_list_iCMS3 = pd.unique(markers[['iCMS3_Up', 'iCMS3_Down']].values.ravel('K'))

# Splitting markers by iCMS type
iCMS2 = pd.unique(markers[['iCMS2_Up', 'iCMS2_Down']].values.ravel('K'))
iCMS2_df = pd.DataFrame({'markers': iCMS2, 'status': 'iCMS2'})

iCMS3 = pd.unique(markers[['iCMS3_Up', 'iCMS3_Down']].values.ravel('K'))
iCMS3_df = pd.DataFrame({'markers': iCMS3, 'status': 'iCMS3'})

i2_i3 = pd.concat([iCMS2_df, iCMS3_df], ignore_index=True)
i2_i3['status'] = pd.Categorical(i2_i3['status'])


# Define the pred_iCMS function
def pred_iCMS(cnts_mat, i2_i3, nPerm=2000, nCores=0, setSeed=False):
    
    # Step 1: Clean cnts_mat (handle missing values)
    cnts_mat = cnts_mat.dropna()
    
    # Step 2: Clean i2_i3 markers
    i2_i3 = i2_i3[i2_i3['markers'].isin(cnts_mat.index)]
    
    # Step 3: Prepare inputs
    N = cnts_mat.shape[1]  # Number of columns
    K = i2_i3['status'].nunique()  # Number of unique status categories
    S = i2_i3.shape[0]  # Number of markers
    P = cnts_mat.shape[0]  # Number of rows
    class_names = i2_i3['status'].cat.categories
    i2_i3['status'] = i2_i3['status'].cat.codes + 1  # Convert to numeric
    
    # Step 4: Check for normalization warning
    cnts_mat_mean = cnts_mat.mean().round(2)
    if abs(cnts_mat_mean).max() > 1:
        cnts_mat_sd = cnts_mat.std().round(2)
        warnings.warn(f"emat mean={cnts_mat_mean.mean()}, sd={cnts_mat_sd.mean()} <- check feature centering!", stacklevel=2)

    # Step 5: Matching markers and cnts_mat
    mm = i2_i3['markers'].apply(lambda x: cnts_mat.index.get_loc(x) if x in cnts_mat.index else None).dropna()
    if not np.all(cnts_mat.index[mm] == i2_i3['markers']):
        raise ValueError("Error matching probes, check rownames(cnts_mat) and i2_i3['markers']")
    
    # Step 6: Prepare templates
    tmat = np.zeros((S, K))
    for k in range(K):
        tmat[:, k] = (i2_i3['status'] == (k + 1)).astype(int)
    
    if K == 2:
        tmat[tmat == 0] = -1
    
    # Step 7: Define similarity and distance functions
    def sim_fn(x, y):
        return 1 - cosine(x, y)
    
    def simToDist(cos_sim):
        return np.sqrt(1 / 2 * (1 - cos_sim))

    # Step 8: Define ntp_fn (nearest template prediction function)
    def ntp_fn(n):
        n_sim = np.array([sim_fn(cnts_mat.iloc[mm, n], tmat[:, k]) for k in range(K)])
        n_sim_perm_max = np.max([sim_fn(np.random.permutation(cnts_mat.iloc[:, n])[:S], tmat[:, k]) for _ in range(nPerm) for k in range(K)])
        n_ntp = np.argmax(n_sim)
        n_sim_ranks = rankdata([-n_sim[n_ntp]] + [-n_sim_perm_max])
        n_pval = n_sim_ranks[0] / len(n_sim_ranks)
        return n_ntp, simToDist(n_sim[n_ntp]), n_pval

    # Step 9: Parallel or serial processing based on nCores and setSeed
    if setSeed:
        random.seed(7)
        nCores = 1
        res = [ntp_fn(n) for n in range(N)]
    else:
        nCores = nCores or mp.cpu_count()
        with mp.Pool(nCores) as pool:
            res = pool.map(ntp_fn, range(N))

    # Step 10: Prepare output
    res = np.array(res)
    predictions = pd.DataFrame(res, columns=["prediction", *[f'd.{name}' for name in class_names], "p.value"])
    predictions['prediction'] = predictions['prediction'].apply(lambda x: class_names[int(x)])
    predictions.index = cnts_mat.columns
    predictions['p.value'] = predictions['p.value'].apply(lambda p: max(p, 1 / nPerm))
    predictions['FDR'] = pd.Series(np.minimum.accumulate(sorted(predictions['p.value']))).values

    return predictions

# Example usage:
# result = pred_iCMS(cnts, i2_i3, nPerm=2000, nCores=0, setSeed=False)
