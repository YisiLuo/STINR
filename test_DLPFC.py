import pandas as pd
import numpy as np
import scanpy as sc
import anndata as ad
import scipy.io
import matplotlib.pyplot as plt
import os
import sys
from STINR import model
import warnings
warnings.filterwarnings("ignore")

import torch
import random
seed=1
torch.manual_seed(seed) 
torch.cuda.manual_seed(seed) 
torch.cuda.manual_seed_all(seed)  
np.random.seed(seed)  
random.seed(seed)  
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

slice_idx = [151673, 151674, 151675, 151676]

adata_st_list_raw0 = ad.read_h5ad('adata_st_list_raw0.h5ad')
adata_st_list_raw1 = ad.read_h5ad('adata_st_list_raw1.h5ad')
adata_st_list_raw2 = ad.read_h5ad('adata_st_list_raw2.h5ad')
adata_st_list_raw3 = ad.read_h5ad('adata_st_list_raw3.h5ad')
adata_st_list_raw = []
adata_st_list_raw.append(adata_st_list_raw0)
adata_st_list_raw.append(adata_st_list_raw1)
adata_st_list_raw.append(adata_st_list_raw2)
adata_st_list_raw.append(adata_st_list_raw3)

celltype_list_use = ['Astros_1', 'Astros_2', 'Astros_3', 
                     'Endo', 'Micro/Macro',
                     'Oligos_1', 'Oligos_2', 'Oligos_3',
                     'Ex_1_L5_6', 'Ex_2_L5', 'Ex_3_L4_5', 
                     'Ex_4_L_6', 'Ex_5_L5',
                     'Ex_6_L4_6', 'Ex_7_L4_6', 'Ex_8_L5_6', 
                     'Ex_9_L5_6', 'Ex_10_L2_4']

adata_st = ad.read_h5ad('adata_st_DLPFC.h5ad')
adata_basis = ad.read_h5ad('adata_basis_DLPFC.h5ad')

model = model.Model(adata_st_list_raw, adata_st, adata_basis)

model.train()

save_path = ""
result = model.eval(adata_st_list_raw, save=True, output_path=save_path)

from sklearn.mixture import GaussianMixture

np.random.seed(1234)
gm = GaussianMixture(n_components=7, covariance_type='tied', 
                     reg_covar = 10e-4, init_params='kmeans')
y = gm.fit_predict(model.adata_st.obsm['latent'], y=None)
model.adata_st.obs["GM"] = y
model.adata_st.obs["GM"].to_csv(os.path.join(save_path, "clustering_result.csv"))

order = [0,1,2,3,4,5,6] # reordering cluster labels

model.adata_st.obs["Cluster"] = [order[label] for label in model.adata_st.obs["GM"].values]

for i in range(len(result)):
    result[i].obs["GM"] = model.adata_st.obs.loc[result[i].obs_names, ]["GM"]
    result[i].obs["Cluster"] = model.adata_st.obs.loc[result[i].obs_names, ]["Cluster"]
    
for i, adata_st_i in enumerate(result):
    print("Slice %d cell-type deconvolution result:" % slice_idx[i])
    
    print(list(adata_basis.obs.index))
    sc.pl.spatial(adata_st_i, img_key="lowres", 
                  color=list(adata_basis.obs.index), size=1.)