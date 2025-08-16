import logging

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from tqdm import tqdm

from tangram import utils as ut
from . import lightning_mapping_utils as lmu
from . import validation_metrics as vm


def get_cv_data(adata_st, k=10):
    """
    Generates pair of training/test gene indexes for cross-validation datasets

    Args:
        adata_st (AnnData): gene spatial data
        k (int): Number of folds for k-folds cross-validation. Default is 10.

    Yields:
        tuple: list of train_genes, list of test_genes
    """

    genes_array = np.array(adata_st.uns["training_genes"])

    cv = KFold(n_splits = k)

    for train_idx, test_idx in cv.split(genes_array):
        train_genes = list(genes_array[train_idx])
        test_genes = list(genes_array[test_idx])
        yield train_genes, test_genes



def cross_validate_mapping(
        adata_sc,
        adata_st,
        k=10,
        cv_train_genes=None,
        train_genes_idx=None,
        val_genes_idx=None,
        cluster_label=None,
        mode="vanilla",
        learning_rate=0.1,
        num_epochs=1000,
        lambda_d=0,
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
        lambda_ct_islands=0,
        lambda_getis_ord=0,
        lambda_moran=0,
        lambda_geary=0,
        random_state=None,
        verbose=True,
        metrics=["SSIM", "PCC", "RMSE", "JS"]
):
    """
    Executes genes set cross-validation using Lightning mapper

    Args:
        adata_sc (AnnData): single cell data
        adata_sp (AnnData): gene spatial data
        mode (str): Optional. Tangram mapping mode. Currently supported: 'vanilla', 'filter', 'refined'. Default is 'vanilla'.
        lambda_g1 (float): Optional. Strength of Tangram loss function. Default is 1.
        lambda_d (float): Optional. Strength of density regularizer. Default is 0.
        lambda_g2 (float): Optional. Strength of voxel-gene regularizer. Default is 0.
        lambda_r (float): Optional. Strength of entropy regularizer. Default is 0.
        lambda_count (float): Optional. Regularizer for the count term. Default is 1. Only valid when mode == 'constrained'
        lambda_f_reg (float): Optional. Regularizer for the filter, which promotes Boolean values (0s and 1s) in the filter. Only valid when mode == 'constrained'. Default is 1.
        target_count (int): Optional. The number of cells to be filtered. Default is None.
        num_epochs (int): Optional. Number of epochs. Default is 1000.
        learning_rate (float): Optional. Learning rate for the optimizer. Default is 0.1.
        k (int): Number of cross-validation folds. Default is 10.
        density_prior (ndarray or str): Spatial density of spots, when is a string, value can be 'rna_count_based' or 'uniform', when is a ndarray, shape = (number_spots,). This array should satisfy the constraints sum() == 1. If not provided, the density term is ignored.
        random_state (int): Optional. pass an int to reproduce training. Default is None.
        verbose (bool): Optional. If print k-fold details. Default is True.
        metrics (list): Optional. List of metrics to compute. Default is ["SSIM","PCC","RMSE","JS"].

    Returns:
        cv_metrics (dict): Dictionary containing average metric scores across all folds
    """

    logger_root = logging.getLogger()
    logger_root.disabled = True
    logger_ann = logging.getLogger("anndata")
    logger_ann.disabled = True

    curr_cv_set = 1

    # Init fold metrics dictionary
    fold_metrics = {metric: [] for metric in metrics}

    # Check n folds
    if k > 0:
        length = k
    else:
        raise ValueError("Invalid number of folds. Please enter a positive integer greater than 0.")

    for train_genes, test_genes in tqdm(get_cv_data(adata_st, k), total=length):

        # Train mapper
        adata_map = lmu.map_cells_to_space(
            adata_sc=adata_sc,
            adata_st=adata_st,
            cv_train_genes=cv_train_genes,
            train_genes_idx=train_genes_idx,
            val_genes_idx=val_genes_idx,
            cluster_label=cluster_label,
            mode=mode,
            learning_rate=learning_rate,
            num_epochs=num_epochs,
            lambda_d=lambda_d,
            lambda_g1=lambda_g1,
            lambda_g2=lambda_g2,
            lambda_r=lambda_r,
            lambda_l1=lambda_l1,
            lambda_l2=lambda_l2,
            lambda_count=lambda_count,
            lambda_f_reg=lambda_f_reg,
            target_count=target_count,
            lambda_sparsity_g1=lambda_sparsity_g1,
            lambda_neighborhood_g1=lambda_neighborhood_g1,
            lambda_ct_islands=lambda_ct_islands,
            lambda_getis_ord=lambda_getis_ord,
            lambda_moran=lambda_moran,
            lambda_geary=lambda_geary,
        )

        # Project test genes on space
        adata_ge = ut.project_genes(
            adata_map, adata_sc[:, test_genes], scale=False,
        )

        # Echo
        if verbose:
            print(f"cv set: {curr_cv_set}")
        curr_cv_set += 1

        # Define imputed and raw dataframes for metrics evaluation
        impute_data = adata_ge[:, test_genes].X.toarray()  # projected test genes on space
        raw_data = adata_sc[:, test_genes].X.toarray()  # ground truth test genes
        impute = pd.DataFrame(data=impute_data, index=adata_ge.obs.index, columns=adata_ge.var.index)
        raw = pd.DataFrame(data=raw_data, index=adata_sc[:, test_genes].obs.index,
                           columns=adata_sc[:, test_genes].var.index)

        # Metrics evaluation on current fold
        for metric in metrics:
            if metric == "SSIM":
                fold_metrics[metric].append(vm.ssim(raw, impute))
            elif metric == "PCC":
                fold_metrics[metric].append(vm.pearsonr(raw, impute))
            elif metric == "RMSE":
                fold_metrics[metric].append(vm.RMSE(raw, impute))
            elif metric == "JS":
                fold_metrics[metric].append(vm.JS(raw, impute))

    # Calculate average metrics across folds
    cv_metrics = {}
    for metric in metrics:
        temp_arr = np.zeros(len(fold_metrics[metric]))  # shape = (k,)
        for fold in range(len(fold_metrics[metric])):
            temp_arr[fold] = np.mean(fold_metrics[metric][fold])  # scalar
        cv_metrics[metric] = np.array(temp_arr, dtype='float32').mean().item()  # assing metric mean over folds

    return cv_metrics