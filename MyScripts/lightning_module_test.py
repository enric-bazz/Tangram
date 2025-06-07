## Load cropped datasets
import anndata as ad
path = "C:/Users/enric/tangram/myDataCropped"
adata_sc = ad.read_h5ad(path + "/test_sc_crop.h5ad")
adata_st = ad.read_h5ad(path + "/test_sp_crop.h5ad")

# remove zero count genes
import numpy as np

# Overwrite adata without explicitly using .copy()
#adata_sc = adata_sc[:, np.array(adata_sc.X.sum(axis=0)).flatten() > 0]
#adata_st = adata_st[:, np.array(adata_st.X.sum(axis=0)).flatten() > 0]

# get genes intersection
import mytangram as tg
tg.pp_adatas(adata_sc, adata_st)

mode = "constrained"
target_count = np.round(len(adata_st.var.index)/3)

# set seed for comparison
random_state = 123



ad_map_lt = tg.map_cells_to_space_lightning(adata_sc, adata_st, mode=mode, target_count=target_count,
                               density_prior='rna_count_based', num_epochs=30,
                               lambda_d=1,
                               lambda_g1=1,
                               lambda_g2=1,
                               lambda_r=0.001,
                               lambda_count=0,
                               lambda_f_reg=1,
                                random_state=random_state,
                                )




print(ad_map_lt.uns['training_history'].keys())

tg.plot_loss_terms(adata_map=ad_map_lt, log_scale=False)


    # Plot final filter values distribution


if mode == "constrained":
    tg.plot_filter_weights_light(ad_map_lt)
    tg.plot_filter_count(ad_map_lt, target_count=target_count)
