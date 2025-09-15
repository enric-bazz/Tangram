# Load cropped datasets
import anndata as ad

path = "C:/Users/enric/tangram/myDataCropped"
adata_sc = ad.read_h5ad(path + "/test_sc_crop.h5ad")
adata_st = ad.read_h5ad(path + "/slice200_norm_reduced.h5ad")
# considerably reduce the number of spots as refinements require to compute and store several versions of the neighbors graph

import tangramlit as tg

# Set parameters for mapping
mode = "filter"
target_count = None
cluster_label = "cluster_labels"

# Set seed for reproducibility
random_state = 123

input_genes=None
train_genes_names=None
val_genes_names=None

hyperparams = tg.validate_mapping_inputs(
    adata_sc,
    adata_st,
    input_genes=input_genes,
    train_genes_names=train_genes_names,
    val_genes_names=val_genes_names,
    mode="filter",
    learning_rate=0.1,
    num_epochs=101,
    random_state=None,
    lambda_d=1,
    lambda_g1=1,
    lambda_g2=0,
    lambda_r=0,
    lambda_l1=0,
    lambda_l2=0,
    lambda_count=1,
    lambda_f_reg=1,
    target_count=None,
    lambda_sparsity_g1=0,
    lambda_neighborhood_g1=0,
    lambda_getis_ord=0,
    lambda_moran=0,
    lambda_geary=0,
    lambda_ct_islands=0,
    cluster_label=None,
    )

filter_mat = tg.run_multiple_mappings(adata_sc, adata_st, config=hyperparams,
                                                   n_runs=10,
                                                    compute_mapping_cube=False,
                                                    compute_filtered_cube=False,
                                                    compute_filter_square=True,)
# always have to unwrap the output

filter_result = tg.filter_cell_choice_consistency(filter_mat[0])

ad_map_lt, mapper, datamodule = tg.map_cells_to_space(
    adata_sc,
    adata_st,
    mode=mode,
    input_genes=None,
    train_genes_names=None,
    val_genes_names=None,
    target_count=target_count,
    num_epochs=1000,
    lambda_d=1,
    lambda_g1=1000,
    lambda_g2=0,
    lambda_r=0,
    lambda_count=1,
    lambda_f_reg=1e-5,
    lambda_l2=0,
    lambda_l1=1e-5,
    random_state=random_state,
    lambda_sparsity_g1=1,
    lambda_neighborhood_g1=1,
    lambda_ct_islands=0,
    #cluster_label=cluster_label,
    lambda_getis_ord=1,
    lambda_moran=1,
    lambda_geary=1 ,
    )
