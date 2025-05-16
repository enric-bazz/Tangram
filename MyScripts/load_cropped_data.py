import os

import anndata as ad

adata_sc = ad.read(os.getcwd() + "/myDataCropped/test_sc_crop.h5ad")
adata_st = ad.read(os.getcwd() + "/myDataCropped/test_sp_crop.h5ad")
adata_sc.file.close()
adata_st.file.close()

from scipy.sparse import vstack
adata_tot = vstack([adata_sc.X, adata_st.X]) #csr matrix

gene_counts = adata_tot.sum(axis=0)

import numpy as np

zero_count_genes = np.sum(gene_counts==0)
print('number of genes with total null expresion on data: ', zero_count_genes)

# run in console with
# %run "Myscripts/load_cropped_data"