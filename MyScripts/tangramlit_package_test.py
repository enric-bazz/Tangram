# Load cropped datasets
import anndata as ad
import numpy as np
path = "C:/Users/enric/tangram/myDataCropped"
adata_sc = ad.read_h5ad(path + "/test_sc_crop.h5ad")
adata_st = ad.read_h5ad(path + "/slice200_norm_reduced.h5ad")
# considerably reduce the number of spots as refinements require to compute and store several versions of the neighbors graph

import tangramlit as tg

# Set parameters for mapping
mode = "vanilla"
target_count = None
cluster_label = "cluster_labels"

# Set seed for reproducibility
random_state = 123

ad_map_lt, mapper, datamodule = tg.map_cells_to_space(
    adata_sc,
    adata_st,
    mode=mode,
    target_count=target_count,
    num_epochs=30,
    lambda_d=1,
    lambda_g1=1,
    lambda_g2=1,
    lambda_r=0.001,
    lambda_count=1,
    lambda_f_reg=1,
    lambda_l2=1e-5,
    lambda_l1=1e-5,
    random_state=random_state,
    lambda_sparsity_g1=1,
    lambda_neighborhood_g1=1,
    lambda_ct_islands=1,
    cluster_label=cluster_label,
    lambda_getis_ord=1,
    lambda_moran=1,
    lambda_geary=1 ,
    )

# Plot loss terms
tg.plot_loss_terms(adata_map=ad_map_lt, log_scale=False)

# Plot final filter values distribution
if mode == "filter":
    tg.plot_filter_weights_light(ad_map_lt, plot_spaghetti=True, plot_envelope=True)
    tg.plot_filter_count(ad_map_lt, target_count=target_count)

# Validation
# Get shared genes (case-insensitive)
sc_genes = {gene.lower(): gene for gene in adata_sc.var_names}
st_genes = {gene.lower(): gene for gene in adata_st.var_names}

# Find intersection of lowercase gene names
shared_lower = set(sc_genes.keys()) & set(st_genes.keys())

# Use original case from sc_genes for consistency
shared_genes = [sc_genes[gene_lower] for gene_lower in shared_lower]

# Random split
if random_state is not None:
    np.random.seed(random_state)

# Shuffle the shared genes
shared_genes = np.array(shared_genes)
np.random.shuffle(shared_genes)

# Split into train and validation
train_ratio = 0.8  # Adjust the ratio as needed
n_train = int(len(shared_genes) * train_ratio)
train_genes = shared_genes[:n_train]
val_genes = shared_genes[n_train:]

#Train
ad_map_lt, mapper, datamodule = tg.map_cells_to_space(
    adata_sc,
    adata_st,
    mode=mode,
    train_genes_names=train_genes,
    val_genes_names=val_genes,
    target_count=target_count,
    num_epochs=30,
    lambda_d=1,
    lambda_g1=1,
    lambda_g2=1,
    lambda_r=0.001,
    lambda_count=1,
    lambda_f_reg=1,
    lambda_l2=1e-5,
    lambda_l1=1e-5,
    random_state=random_state,
    lambda_sparsity_g1=1,
    lambda_neighborhood_g1=1,
    lambda_ct_islands=1,
    cluster_label=cluster_label,
    lambda_getis_ord=1,
    lambda_moran=1,
    lambda_geary=1 ,
    )

#Validate
results = tg.validate_mapping_experiment(mapper, datamodule)

# Test cv
cv_results = tg.cross_validate_mapping(adata_sc,
    adata_st,
    mode=mode,
   input_genes=shared_genes,
    num_epochs=30,
    lambda_d=1,
    lambda_g1=1,
    lambda_g2=1,
    lambda_r=0.001,
    lambda_count=1,
    lambda_f_reg=1,
    lambda_l2=1e-5,
    lambda_l1=1e-5,
    random_state=random_state,
    lambda_sparsity_g1=1,
    lambda_neighborhood_g1=1,
    lambda_ct_islands=1,
    cluster_label=cluster_label,
    lambda_getis_ord=1,
    lambda_moran=1,
    lambda_geary=1 ,)