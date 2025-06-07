import logging

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.model_selection import LeaveOneOut
from tqdm import tqdm

from mytangram import validation_metrics as vm
from tangram import utils as ut
from . import mapping_utils as mu


def get_cv_data(adata_sc, adata_sp, cv_mode="kfold", k=10 ):
    """ Generates pair of training/test gene indexes cross validation datasets

    Args:
        adata_sc (AnnData): single cell data
        adata_sp (AnnData): gene spatial data
        cv_mode (str or int): mode, support 'loo' and 'kfold' cross validaton
        k (int): Optional. Number of folds for k-folds cross validation

    Yields:
        tuple: list of train_genes, list of test_genes
    """

    # Check if training_genes key exist/is valid in adatas.uns
    if "training_genes" not in adata_sc.uns.keys():
        raise ValueError("Missing tangram parameters. Run `pp_adatas()`.")

    if "training_genes" not in adata_sp.uns.keys():
        raise ValueError("Missing tangram parameters. Run `pp_adatas()`.")

    if not list(adata_sp.uns["training_genes"]) == list(adata_sc.uns["training_genes"]):
        raise ValueError(
            "Unmatched training_genes field in two Anndatas. Run `pp_adatas()`."
        )

    genes_array = np.array(adata_sp.uns["training_genes"])

    if cv_mode == "loo":
        cv = LeaveOneOut()
    elif cv_mode == "kfold":
        cv = KFold(n_splits=k)

    for train_idx, test_idx in cv.split(genes_array):
        train_genes = list(genes_array[train_idx])
        test_genes = list(genes_array[test_idx])
        yield train_genes, test_genes


def cross_validate(
        adata_sc,
        adata_sp,
        cluster_label=None,
        mode="cells",
        scale=True,
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
        cv_mode="kfold",
        cv_k=10,
        density_prior=None,
        random_state=None,
        verbose=False,
        metrics=["SSIM","PCC","RMSE","JS"]
):
    """
    Executes cross validation

    Args:
        adata_sc (AnnData): single cell data
        adata_sp (AnnData): gene spatial data
        cluster_label (str): the level that the single cell data will be aggregate at, this is only valid for clusters mode mapping
        mode (str): Optional. Tangram mapping mode. Currently supported: 'cell', 'clusters', 'constrained'. Default is 'clusters'.
        scale (bool): Optional. Whether weight input single cell by # of cells in cluster, only valid when cluster_label is not None. Default is True.
        lambda_g1 (float): Optional. Strength of Tangram loss function. Default is 1.
        lambda_d (float): Optional. Strength of density regularizer. Default is 0.
        lambda_g2 (float): Optional. Strength of voxel-gene regularizer. Default is 0.
        lambda_r (float): Optional. Strength of entropy regularizer. Default is 0.
        lambda_count (float): Optional. Regularizer for the count term. Default is 1. Only valid when mode == 'constrained'
        lambda_f_reg (float): Optional. Regularizer for the filter, which promotes Boolean values (0s and 1s) in the filter. Only valid when mode == 'constrained'. Default is 1.
        target_count (int): Optional. The number of cells to be filtered. Default is None.
        num_epochs (int): Optional. Number of epochs. Default is 1000.
        learning_rate (float): Optional. Learning rate for the optimizer. Default is 0.1.
        device (str or torch.device): Optional. Default is 'cuda:0'.
        cv_mode (str): Cross validation mode, 'loo' ('leave-one-out') and 'kfold' supported. Default is 'kfold'.
        cv_k (int): Number of cross validation folds. Default is 10.
        return_gene_pred (bool): Optional. if return prediction and true spatial expression data for test gene, only applicable when 'loo' mode is on, default is False.
        density_prior (ndarray or str): Spatial density of spots, when is a string, value can be 'rna_count_based' or 'uniform', when is a ndarray, shape = (number_spots,). This array should satisfy the constraints sum() == 1. If not provided, the density term is ignored.
        random_state (int): Optional. pass an int to reproduce training. Default is None.
        verbose (bool): Optional. If print training details. Default is False.

    Returns:
        cv_dict (dict): a dictionary contains information of cross validation (hyperparameters, average test score and train score, etc.)
        adata_ge_cv (AnnData): predicted spatial data by LOOCV. Only returns when `return_gene_pred` is True and in 'loo' mode.
        test_gene_df (Pandas dataframe): dataframe with columns: 'score', 'is_training', 'sparsity_sp'(spatial data sparsity)
    """

    logger_root = logging.getLogger()
    logger_root.disabled = True
    logger_ann = logging.getLogger("anndata")
    logger_ann.disabled = True

    curr_cv_set = 1

    # Init fold metrics dictionary
    fold_metrics = {}

    if cv_mode == "loo":
        length = len(list(adata_sc.uns["training_genes"]))
        for metric in metrics:
            fold_metrics[metric] = []  # np.zeros(len(adata_sc.uns["training_genes"]))
    elif cv_mode == "kfold":
        length = cv_k
        for metric in metrics:
            fold_metrics[metric] = []  # np.zeros(10)

    if mode == "clusters":
        adata_sc_agg = mu.adata_to_cluster_expression(adata_sc, cluster_label, scale)



    for train_genes, test_genes in tqdm(
            get_cv_data(adata_sc, adata_sp, cv_mode, cv_k), total=length
    ):
        # train
        adata_map = mu.map_cells_to_space(
            adata_sc=adata_sc,
            adata_sp=adata_sp,
            cv_train_genes=train_genes,
            mode=mode,
            device=device,
            learning_rate=learning_rate,
            num_epochs=num_epochs,
            cluster_label=cluster_label,
            scale=scale,
            lambda_d=lambda_d,
            lambda_g1=lambda_g1,
            lambda_g2=lambda_g2,
            lambda_r=lambda_r,
            lambda_count=lambda_count,
            lambda_f_reg=lambda_f_reg,
            target_count=target_count,
            random_state=random_state,
            verbose=False,
            density_prior=density_prior,
        )


        # project on space
        adata_ge = ut.project_genes(
            adata_map, adata_sc[:, test_genes], cluster_label=cluster_label, scale=scale,
        )

        if verbose == True:
            msg = "cv set: {}----train score: {:.3f}----test score: {:.3f}".format(
                curr_cv_set)
            print(msg)

        curr_cv_set += 1

        impute_data = adata_ge[:, test_genes].X.toarray()
        raw_data = adata_sc[:, test_genes].X.toarray()
        impute = pd.DataFrame(data=impute_data, index=adata_ge.obs.index, columns=adata_ge.var.index)
        raw = pd.DataFrame(data=raw_data, index=adata_sc[:, test_genes].obs.index, columns=adata_sc[:, test_genes].var.index)

        # Metrics evaluation on fold

        if "SSIM" in metrics:
            fold_metrics["SSIM"].append(vm.ssim(raw, impute, scale=None))
        if "PCC" in metrics:
            fold_metrics["PCC"].append(vm.pearsonr(raw, impute, scale=None))
        if "RMSE" in metrics:
            fold_metrics["RMSE"].append(vm.RMSE(raw, impute, scale=None))
        if "JS" in metrics:
            fold_metrics["JS"].append(vm.JS(raw, impute, scale=None))

        # results storing
        #fold_metrics is a dictionrary, each item is a list of n_fold elements, each list contains a pd.Series
        #of n_genes_fold values of the metric for each gene in the fold across all spots
        # I need to averge across genes in the fold before averaging across folds
        # average metrics in folds
    cv_metrics = {}
    for metric in metrics:
        temp_arr = np.zeros(len(fold_metrics[metric]))
        for fold in range(len(fold_metrics[metric])):
            temp_arr[fold] = np.mean((fold_metrics[metric][fold]))
        cv_metrics[metric] = np.array(temp_arr, dtype='float32').mean()

    # Add cross-validated metrics dictionary to object as attribute
    #self.cv_metrics = cv_metrics

    return cv_metrics