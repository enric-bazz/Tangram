# Load cropped datasets
import anndata as ad
path = "C:/Users/enric/tangram/myDataCropped"
adata_sc = ad.read_h5ad(path + "/test_sc_crop.h5ad")
adata_st = ad.read_h5ad(path + "/slice200_norm_reduced.h5ad")
# considerably reduce the number of spots as refinements require to compute and store several versions of the neighbors graph


# Standardize gene names to lowercase
"""adata_sc.var_names = adata_sc.var_names.str.lower()
adata_st.var_names = adata_st.var_names.str.lower()

# Make gene names unique if they aren't already
adata_sc.var_names_make_unique()
adata_st.var_names_make_unique()"""


import tangramlit as tg

# Set parameters for mapping
mode = "filter"
target_count = None
cluster_label = "cluster_labels"

# Set seed for reproducibility
random_state = 123




ad_map_lt = tg.map_cells_to_space(
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
## Create train/val split to test
import numpy as np

# Plot loss terms
tg.plot_loss_terms(adata_map=ad_map_lt, log_scale=False)

# Plot final filter values distribution
if mode == "filter":
    tg.plot_filter_weights_light(ad_map_lt, plot_spaghetti=True, plot_envelope=True)
    tg.plot_filter_count(ad_map_lt, target_count=target_count)

def create_test_splits(adata_sc, adata_st):
    # Get the intersection of genes between datasets
    shared_genes = list(set(adata_sc.var_names) & set(adata_st.var_names))

    # Get indices of shared genes in original adata_sc
    shared_genes_idx = [i for i, gene in enumerate(adata_sc.var_names)
                        if gene in shared_genes]
    n_shared = len(shared_genes_idx)

    # Case 1: Valid non-overlapping train/val split (80/20)
    np.random.seed(42)
    train_size = int(0.8 * n_shared)
    train_indices = np.random.choice(shared_genes_idx, size=train_size, replace=False)
    val_indices = np.array([idx for idx in shared_genes_idx if idx not in train_indices])

    # Case 2: Overlapping train/val split
    val_indices_overlap = np.concatenate([val_indices, train_indices[:10]])  # Add 10 training genes to validation

    # Case 3: Invalid genes (indices that aren't in the shared set)
    non_shared_idx = [i for i, gene in enumerate(adata_sc.var_names)
                      if gene not in shared_genes][:10]  # Get first 10 non-shared genes
    train_indices_invalid = np.concatenate([train_indices, non_shared_idx])

    return {
        'valid_split': (train_indices, val_indices),
        'overlapping_split': (train_indices, val_indices_overlap),
        'invalid_split': (train_indices_invalid, val_indices)
    }


# Example usage:
splits = create_test_splits(adata_sc, adata_st)
print(f"Valid split - Train size: {len(splits['valid_split'][0])}, Val size: {len(splits['valid_split'][1])}")
print(
    f"Overlapping split - Train size: {len(splits['overlapping_split'][0])}, Val size: {len(splits['overlapping_split'][1])}")
print(f"Invalid split - Train size: {len(splits['invalid_split'][0])}, Val size: {len(splits['invalid_split'][1])}")

"""ad_map_lt = tg.map_cells_to_space(
    adata_sc,
    adata_st,
    mode=mode,
    train_genes_idx=splits['invalid_split'][0],
    val_genes_idx=splits['invalid_split'][1],
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
    )"""

ad_map_lt = tg.map_cells_to_space(
    adata_sc,
    adata_st,
    mode=mode,
    train_genes_idx=splits['valid_split'][0],
    val_genes_idx=splits['valid_split'][1],
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



"""# Cross-validation test
cv_results = tg.cross_validate_lightning(
    adata_sc,
    adata_st,   
    mode=mode,
    lambda_d=1,
    lambda_g1=1,
    lambda_g2=1,
    lambda_r=0.001,
    lambda_count=1,
    lambda_f_reg=1,
    target_count=target_count,
    num_epochs=30,
    learning_rate=0.1,
    cv_mode="kfold",
    cv_k=3,
    density_prior='rna_count_based',
    verbose=False,
    metrics=["SSIM", "PCC", "RMSE", "JS"]
)

print(cv_results)"""