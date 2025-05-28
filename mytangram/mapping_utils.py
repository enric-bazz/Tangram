import logging

import numpy as np
import pandas as pd
import scanpy as sc
import torch
from anndata import *
from scipy.sparse.csc import csc_matrix
from scipy.sparse.csr import csr_matrix

from tangram import utils as ut
from . import mapping_optimizer as mo


def pp_adatas(adata_sc, adata_sp, genes=None, gene_to_lowercase=True):
    """
    Pre-process AnnDatas so that they can be mapped. Specifically:
    - Remove genes that all entries are zero
    - Find the intersection between adata_sc, adata_sp and given marker gene list, save the intersected markers in two adatas
    - Calculate density priors and save it with adata_sp

    Args:
        adata_sc (AnnData): single cell data
        adata_sp (AnnData): spatial expression data
        genes (List): Optional. List of genes to use. If `None`, all genes are used.

    Returns:
        update adata_sc by creating `uns` `training_genes` `overlap_genes` fields
        update adata_sp by creating `uns` `training_genes` `overlap_genes` fields and creating `obs` `rna_count_based_density` & `uniform_density` field
    """

    # remove all-zero-valued genes
    sc.pp.filter_genes(adata_sc, min_cells=1)
    sc.pp.filter_genes(adata_sp, min_cells=1)

    if genes is None:
        # Use all genes
        genes = adata_sc.var.index

    # put all var index to lower case to align
    if gene_to_lowercase:
        adata_sc.var.index = [g.lower() for g in adata_sc.var.index]
        adata_sp.var.index = [g.lower() for g in adata_sp.var.index]
        genes = list(g.lower() for g in genes)

    adata_sc.var_names_make_unique()
    adata_sp.var_names_make_unique()

    # Refine `marker_genes` so that they are shared by both adatas
    genes = list(set(genes) & set(adata_sc.var.index) & set(adata_sp.var.index))
    # logging.info(f"{len(genes)} shared marker genes.")

    adata_sc.uns["training_genes"] = genes
    adata_sp.uns["training_genes"] = genes
    logging.info(
        "{} training genes are saved in `uns``training_genes` of both single cell and spatial Anndatas.".format(
            len(genes)
        )
    )

    # Find overlap genes between two AnnDatas
    overlap_genes = list(set(adata_sc.var.index) & set(adata_sp.var.index))
    # logging.info(f"{len(overlap_genes)} shared genes.")

    adata_sc.uns["overlap_genes"] = overlap_genes
    adata_sp.uns["overlap_genes"] = overlap_genes
    logging.info(
        "{} overlapped genes are saved in `uns``overlap_genes` of both single cell and spatial Anndatas.".format(
            len(overlap_genes)
        )
    )

    # Calculate uniform density prior as 1/number_of_spots
    adata_sp.obs["uniform_density"] = np.ones(adata_sp.X.shape[0]) / adata_sp.X.shape[0]
    logging.info(
        f"uniform based density prior is calculated and saved in `obs``uniform_density` of the spatial Anndata."
    )

    # Calculate rna_count_based density prior as % of rna molecule count
    rna_count_per_spot = np.array(adata_sp.X.sum(axis=1)).squeeze()
    adata_sp.obs["rna_count_based_density"] = rna_count_per_spot / np.sum(rna_count_per_spot)
    logging.info(
        f"rna count based density prior is calculated and saved in `obs``rna_count_based_density` of the spatial Anndata."
    )


def adata_to_cluster_expression(adata, cluster_label, scale=True, add_density=True):
    """
    Convert an AnnData to a new AnnData with cluster expressions. Clusters are based on `cluster_label` in `adata.obs`.  The returned AnnData has an observation for each cluster, with the cluster-level expression equals to the average expression for that cluster.
    All annotations in `adata.obs` except `cluster_label` are discarded in the returned AnnData.

    Args:
        adata (AnnData): single cell data
        cluster_label (String): field in `adata.obs` used for aggregating values
        scale (bool): Optional. Whether weight input single cell by # of cells in cluster. Default is True.
        add_density (bool): Optional. If True, the normalized number of cells in each cluster is added to the returned AnnData as obs.cluster_density. Default is True.

    Returns:
        AnnData: aggregated single cell data

    """
    ## Retrieve cell counts-per-cluster Series object (df column) with .value_count()
    try:
        value_counts = adata.obs[cluster_label].value_counts(normalize=True)
    except KeyError as e:
        raise ValueError("Provided label must belong to adata.obs.")
    # Retrieve unique cell type/clusters labels from Series
    unique_labels = value_counts.index
    # Build dataframe (one column) of clusters labels (as entries of the df, indexes are int-range)
    new_obs = pd.DataFrame({cluster_label: unique_labels})
    # Define AnnData object with inherited var (genes) and uns
    adata_ret = sc.AnnData(obs=new_obs, var=adata.var, uns=adata.uns)

    ### Build clusters count matrix as ndarray (no sparsity management at cluster level)
    # Init array of shape: (number of cluster labels, number of genes)
    X_new = np.empty((len(unique_labels), adata.shape[1]))

    for index, l in enumerate(unique_labels):
        # Assign row entries
        if not scale:
            # Mean gene expression values over adata.obs selected by cluster label (shape: (1, n_genes))
            X_new[index] = adata[adata.obs[cluster_label] == l].X.mean(axis=0)
        else:
            # Summation of gene expression values over adata.obs selected by cluster label (shape: (1, n_genes))
            X_new[index] = adata[adata.obs[cluster_label] == l].X.sum(axis=0)
    # Assign filled expression values ndarray
    adata_ret.X = X_new

    # Add clusters density values to adata object:
    # densities are computed above as relative frequencies of clusters with .value_count()
    if add_density:
        adata_ret.obs["cluster_density"] = adata_ret.obs[cluster_label].map(
            lambda i: value_counts[i]
        )
    # The syntax exploits a lambda inline function to map density values to the corresponding row in the anndata object

    return adata_ret


def validate_mapping_inputs(
        adata_sc,
        adata_sp,
        cv_train_genes=None,
        cluster_label=None,
        mode="cells",
        device="cpu",
        learning_rate=0.1,
        num_epochs=1000,
        lambda_d=0,
        lambda_g1=1,
        lambda_g2=0,
        lambda_r=0,
        lambda_count=1,
        lambda_f_reg=1,
        target_count=None,
        density_prior='rna_count_based',):
    """
    Validates inputs for cell-to-space mapping functions in the Tangram framework.

    Args:
        adata_sc (AnnData): Single-cell RNA-seq data object for source cells
        adata_sp (AnnData): Spatial transcriptomics data object for target space
        genes (list): Optional. List of genes to use for mapping. If None, all genes in both datasets will be used.
        mode (str): Optional. Mapping mode, either 'cells' or 'clusters'. Default is 'clusters'.
        cluster_label (str): Optional. If mode=='clusters', name of the column in adata_sc.obs containing cluster labels.
        count_normalize (bool): Optional. If True, data will be count-normalized. Default is True.
        num_epochs (int): Optional. Number of epochs for training. Default is 1000.
        device (str): Optional. Device for computation ('cpu' or 'cuda'). Default is 'cpu'.
        learning_rate (float): Optional. Learning rate for training. Default is 0.1.

    Returns:
        tuple: A tuple containing:
            - adata_sc: Validated/preprocessed single-cell data
            - adata_sp: Validated/preprocessed spatial data
            - genes: List of validated genes
            - hyperparameters: Dict of validated hyperparameters

    Raises:
        ValueError: If inputs are invalid or incompatible
    """

    ### INPUT Control
    # check invalid values for arguments
    if lambda_g1 == 0:
        raise ValueError("lambda_g1 cannot be 0.")

    if (type(density_prior) is str) and (
            density_prior not in ["rna_count_based", "uniform", None]
    ):
        raise ValueError("Invalid input for density_prior.")

    if density_prior is not None and (lambda_d == 0 or lambda_d is None):
        lambda_d = 1

    if lambda_d > 0 and density_prior is None:
        raise ValueError("When lambda_d is set, please define the density_prior.")


    # Validate data objects
    if not isinstance(adata_sc, AnnData) or not isinstance(adata_sp, AnnData):
        raise ValueError("Both adata_sc and adata_sp must be AnnData objects")

    # Validate mapping mode
    if mode not in ["cells", "constrained", "cluster"]:
        raise ValueError('Argument "mode" must be "cells" or "constrained')

    # Validate cluster information if needed
    if mode == 'clusters':
        if cluster_label is None:
            raise ValueError("A cluster_label must be specified if mode is 'clusters'.")
        if cluster_label not in adata_sc.obs.columns:
            raise ValueError(f"cluster_label '{cluster_label}' not found in adata_sc.obs")

    # Validate and process genes
    ### Training Genes
    # Check if training_genes key exist/is valid in adatas.uns
    if not set(["training_genes", "overlap_genes"]).issubset(set(adata_sc.uns.keys())):
        raise ValueError("Missing tangram parameters. Run `pp_adatas()`.")

    if not set(["training_genes", "overlap_genes"]).issubset(set(adata_sp.uns.keys())):
        raise ValueError("Missing tangram parameters. Run `pp_adatas()`.")

    assert list(adata_sp.uns["training_genes"]) == list(adata_sc.uns["training_genes"])

    # get training_genes
    if cv_train_genes is None:
        training_genes = adata_sc.uns["training_genes"]
    elif cv_train_genes is not None:
        if set(cv_train_genes).issubset(set(adata_sc.uns["training_genes"])):
            training_genes = cv_train_genes
        else:
            raise ValueError(
                "Given training genes list should be subset of two AnnDatas."
            )

    # Validate numerical parameters
    if num_epochs <= 0:
        raise ValueError("num_epochs must be positive")
    if learning_rate <= 0:
        raise ValueError("learning_rate must be positive")

    # Validate device
    if device not in ['cpu', 'cuda']:
        raise ValueError("device must be either 'cpu' or 'cuda'")
    if device == 'cuda' and not torch.cuda.is_available():
        logging.warning("CUDA device requested but not available. Falling back to CPU.")
        device = 'cpu'

        ### Density prior
        # define density_prior if 'rna_count_based' is passed to the density_prior argument:
    d_str = density_prior
    if type(density_prior) is np.ndarray:
        d_str = "customized"
    else:
        if density_prior == "rna_count_based":
            density_prior = adata_sp.obs["rna_count_based_density"]

        # define density_prior if 'uniform' is passed to the density_prior argument:
        elif density_prior == "uniform":
            density_prior = adata_sp.obs["uniform_density"]

    if mode == "cells":
        d = density_prior

    # Filter-mode density prior: uniform or input
    if mode == "constrained":
        if density_prior is None:
            d = adata_sp.obs["uniform_density"]
            d_str = "uniform"
        else:
            d = density_prior
        if lambda_d is None or lambda_d == 0:
            lambda_d = 1

    # Cluster-mode density: density annotated with adata_to_cluster_expression()
    if mode == "clusters":
        d_source = np.array(adata_sc.obs["cluster_density"])
    else:  # cell mode
        d_source = None

    ## Create hyperparameters dictionary fo all next calls
    if mode == "cells":
        hyperparameters = {
            "lambda_d": lambda_d,  # KL (ie density) term
            "lambda_g1": lambda_g1,  # gene-voxel cos sim
            "lambda_g2": lambda_g2,  # voxel-gene cos sim
            "lambda_r": lambda_r,  # regularizer: penalize entropy
            "d_source": d_source,
        }
    elif mode == "constrained":
        hyperparameters = {
            "lambda_d": lambda_d,  # KL (ie density) term
            "lambda_g1": lambda_g1,  # gene-voxel cos sim
            "lambda_g2": lambda_g2,  # voxel-gene cos sim
            "lambda_r": lambda_r,  # regularizer: penalize entropy
            "lambda_count": lambda_count,  # regularizer: enforce target number of cells
            "lambda_f_reg": lambda_f_reg,  # regularizer: push sigmoid values to 0,1
            "target_count": target_count,  # target number of cells
        }




    return training_genes, hyperparameters, d, d_str


def map_cells_to_space(
        adata_sc,
        adata_sp,
        cv_train_genes=None,
        cluster_label=None,
        mode="cells",
        device="cpu",
        learning_rate=0.1,
        num_epochs=1000,
        scale=True,
        lambda_d=0,
        lambda_g1=1,
        lambda_g2=0,
        lambda_r=0,
        lambda_count=1,
        lambda_f_reg=1,
        target_count=None,
        random_state=None,
        verbose=True,
        density_prior='rna_count_based',
):
    """
    Map single cell data (`adata_sc`) on spatial data (`adata_sp`).

    Args:
        adata_sc (AnnData): single cell data
        adata_sp (AnnData): gene spatial data
        cv_train_genes (list): Optional. Training gene list. Default is None.
        cluster_label (str): Optional. Field in `adata_sc.obs` used for aggregating single cell data. Only valid for `mode=clusters`.
        mode (str): Optional. Tangram mapping mode. Currently supported: 'cell', 'constrained', 'cluster'. Default is 'cell'.
        device (string or torch.device): Optional. Default is 'cpu'.
        learning_rate (float): Optional. Learning rate for the optimizer. Default is 0.1.
        num_epochs (int): Optional. Number of epochs. Default is 1000.
        scale (bool): Optional. Whether weight input single cell data by the number of cells in each cluster, only valid when cluster_label is not None. Default is True.
        lambda_d (float): Optional. Hyperparameter for the density term of the optimizer. Default is 0.
        lambda_g1 (float): Optional. Hyperparameter for the gene-voxel similarity term of the optimizer. Default is 1.
        lambda_g2 (float): Optional. Hyperparameter for the voxel-gene similarity term of the optimizer. Default is 0.
        lambda_r (float): Optional. Strength of entropy regularizer. A higher entropy promotes probabilities of each cell peaked over a narrow portion of space. lambda_r = 0 corresponds to no entropy regularizer. Default is 0.
        lambda_count (float): Optional. Regularizer for the count term. Default is 1. Only valid when mode == 'constrained'
        lambda_f_reg (float): Optional. Regularizer for the filter, which promotes Boolean values (0s and 1s) in the filter. Only valid when mode == 'constrained'. Default is 1.
        target_count (int): Optional. The number of cells to be filtered. Default is None.
        random_state (int): Optional. pass an int to reproduce training. Default is None.
        verbose (bool): Optional. If print training details. Default is True.
        density_prior (str, ndarray or None): Spatial density of spots, when is a string, value can be 'rna_count_based' or 'uniform', when is a ndarray, shape = (number_spots,). This array should satisfy the constraints sum() == 1. If None, the density term is ignored. Default value is 'rna_count_based'.

    Returns:
        a cell-by-spot AnnData containing the probability of mapping cell i on spot j.
        The `uns` field of the returned AnnData contains the training genes and the history over epochs of loss terms.
        if mode = 'constrained', then the history of filter values is also returned.
    """

    # Invoke input control function
    training_genes, hyperparameters, d, d_str = validate_mapping_inputs(
        adata_sc=adata_sc,
        adata_sp=adata_sp,
        cv_train_genes=cv_train_genes,
        cluster_label=cluster_label,
        mode=mode,
        device=device,
        learning_rate=learning_rate,
        num_epochs=num_epochs,
        lambda_d=lambda_d,
        lambda_g1=lambda_g1,
        lambda_g2=lambda_g2,
        lambda_r=lambda_r,
        lambda_count=lambda_count,
        lambda_f_reg=lambda_f_reg,
        target_count=target_count,
        density_prior=density_prior
    )
    # Compute cluster-level anndata object
    if mode == "clusters":
        adata_sc = adata_to_cluster_expression(
            adata_sc, cluster_label, scale, add_density=True
        )
    #### perhaps there is a better way of managing cluster mode-specific args like scale, cluster_label

    ### INPUT Tensors: allocate them as arrays even if sparse matrices
    logging.info("Allocate tensors for mapping.")

    ## S matrix
    if isinstance(adata_sc.X, csc_matrix) or isinstance(adata_sc.X, csr_matrix):
        S = np.array(adata_sc[:, training_genes].X.toarray(), dtype="float32",)
    elif isinstance(adata_sc.X, np.ndarray):
        S = np.array(adata_sc[:, training_genes].X.toarray(), dtype="float32",)
    else:
        X_type = type(adata_sc.X)
        logging.error("AnnData X has unrecognized type: {}".format(X_type))
        raise NotImplementedError

    # G matrix
    if isinstance(adata_sp.X, csc_matrix) or isinstance(adata_sp.X, csr_matrix):
        G = np.array(adata_sp[:, training_genes].X.toarray(), dtype="float32")
    elif isinstance(adata_sp.X, np.ndarray):
        G = np.array(adata_sp[:, training_genes].X, dtype="float32")
    else:
        X_type = type(adata_sp.X)
        logging.error("AnnData X has unrecognized type: {}".format(X_type))
        raise NotImplementedError


    # Choose device
    device = torch.device(device)  # for gpu

    # Set echo
    if verbose:
        print_each = 100
    else:
        print_each = None

    ### MAPPING no constraint
    if mode == "cells":
        logging.info(
            "Begin training with {} genes and {} density_prior in {} mode...".format(
                len(training_genes), d_str, mode
            )
        )
        mapper = mo.Mapper(
            S=S, G=G, d=d, device=device, random_state=random_state, constraint=False, **hyperparameters,
        )

        # Print out coefficients values
        msg = []
        for k in hyperparameters.keys() - "d_source":  # keys are the iterable of a dictionary
            m = "{}: {}".format(k, hyperparameters[k])  # "{}: {:.3f}" does not work on float (only str)
            msg.append(m)
        print("Regularizer coefficients:" + "\n" + str(msg).replace("[", "").replace("]", "").replace("'", ""))

        mapping_matrix, training_history = mapper.train(
            learning_rate=learning_rate, num_epochs=num_epochs, print_each=print_each,
        )

        ### MAPPING with coonstraint
    elif mode == "constrained":
        logging.info(
            "Begin training with {} genes and {} density_prior in {} mode...".format(
                len(training_genes), d_str, mode
            )
        )

        # Print out coefficients values
        msg = []
        for k in hyperparameters:
            m = "{}: {}".format(k, hyperparameters[k])
            msg.append(m)
        print("Regularizer coefficients:" + "\n" + str(msg).replace("[", "").replace("]", "").replace("'", ""))

        mapper = mo.Mapper(
            S=S, G=G, d=d, device=device, random_state=random_state, constraint=True, **hyperparameters,
        )

        mapping_matrix, F_out, training_history, filter_history = mapper.train(
            learning_rate=learning_rate, num_epochs=num_epochs, print_each=print_each,
        )

    logging.info("Saving results..")
    adata_map = sc.AnnData(
        X=mapping_matrix,
        obs=adata_sc[:, training_genes].obs.copy(),
        var=adata_sp[:, training_genes].obs.copy(),
    )

    if mode == "constrained":
        adata_map.obs["F_out"] = F_out

    # Annotate cosine similarity of each training gene
    G_predicted = adata_map.X.T @ S
    cos_sims = []
    for v1, v2 in zip(G.T, G_predicted.T):
        norm_sq = np.linalg.norm(v1) * np.linalg.norm(v2)
        cos_sims.append((v1 @ v2) / norm_sq)

    df_cs = pd.DataFrame(cos_sims, training_genes, columns=["train_score"])
    df_cs = df_cs.sort_values(by="train_score", ascending=False)
    adata_map.uns["train_genes_df"] = df_cs

    # Annotate sparsity of each training genes
    ut.annotate_gene_sparsity(adata_sc)
    ut.annotate_gene_sparsity(adata_sp)
    adata_map.uns["train_genes_df"]["sparsity_sc"] = adata_sc[
                                                     :, training_genes
                                                     ].var.sparsity
    adata_map.uns["train_genes_df"]["sparsity_sp"] = adata_sp[
                                                     :, training_genes
                                                     ].var.sparsity
    adata_map.uns["train_genes_df"]["sparsity_diff"] = (
            adata_sp[:, training_genes].var.sparsity
            - adata_sc[:, training_genes].var.sparsity
    )

    adata_map.uns["training_history"] = training_history

    if mode == "constrained":
        adata_map.uns["filter_history"] = filter_history

    return adata_map

