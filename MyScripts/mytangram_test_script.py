## Load cropped datasets
import anndata as ad
path = "C:/Users/enric/tangram/myDataCropped"
adata_sc = ad.read_h5ad(path + "/test_sc_crop.h5ad")
adata_st = ad.read_h5ad(path + "/test_sp_crop.h5ad")

# remove zero count genes
import numpy as np

# Overwrite adata without explicitly using .copy()
adata_sc = adata_sc[:, np.array(adata_sc.X.sum(axis=0)).flatten() > 0]
adata_st = adata_st[:, np.array(adata_st.X.sum(axis=0)).flatten() > 0]

# get genes intersection
import mytangram as tg
tg.pp_adatas(adata_sc, adata_st)


# run mapping
mode = "cells"
target_count = np.round(len(adata_st.var.index)/3)
ad_map = tg.map_cells_to_space(adata_sc, adata_st, mode=mode, target_count=target_count,
                               density_prior='rna_count_based', num_epochs=100,device='cpu',
                               lambda_d=1,
                               lambda_g1=1,
                               lambda_g2=1,
                               lambda_r=1,
                               lambda_count=1,
                               lambda_f_reg=1
                               )

# plot loss
tg.plot_loss_terms(adata_map=ad_map, log_scale=False)

# plot filter colormap
if mode == "constrained":
    tg.plot_filter_weights(ad_map)

ad_ge = tg.project_genes(ad_map, adata_sc)

cv_results = tg.cross_validate(
        adata_sc,
        adata_st,
        cluster_label=None,
        mode="cells",
        scale=False,
        lambda_d=0,
        lambda_g1=1,
        lambda_g2=0,
        lambda_r=0,
        lambda_count=1,
        lambda_f_reg=1,
        target_count=None,
        num_epochs=1000,
        device="cpu",
        learning_rate=0.1,
        cv_mode="10fold",
        density_prior=None,
        random_state=None,
        verbose=False,
        metrics=["SSIM","PCC","RMSE","JS"]
)

cv_results2 = tg.cross_validate(
        adata_sc,
        adata_st,
        cluster_label=None,
        mode="cells",
        scale=False,
        lambda_d=0.5,
        lambda_g1=1,
        lambda_g2=0.5,
        lambda_r=0.5,
        lambda_count=0.5,
        lambda_f_reg=1,
        target_count=None,
        num_epochs=1000,
        device="cpu",
        learning_rate=0.1,
        cv_mode="10fold",
        density_prior=None,
        random_state=None,
        verbose=False,
        metrics=["SSIM", "PCC", "JS"]
)





