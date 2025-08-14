# Load cropped datasets
import anndata as ad
path = "C:/Users/enric/tangram/myDataCropped"
adata_sc = ad.read_h5ad(path + "/test_sc_crop.h5ad")
adata_st = ad.read_h5ad(path + "/slice200_norm_with_spatial.h5ad")

# Standardize gene names to lowercase
"""adata_sc.var_names = adata_sc.var_names.str.lower()
adata_st.var_names = adata_st.var_names.str.lower()

# Make gene names unique if they aren't already
adata_sc.var_names_make_unique()
adata_st.var_names_make_unique()"""


import tangramlit as tg

# Set parameters for mapping
mode = "vanilla"
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
    lambda_l2=1,
    lambda_l1=1,
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