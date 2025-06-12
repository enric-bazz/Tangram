"""
Mapping functions for Tangram using the Lightning framework.
"""

import logging

import pandas as pd
import scanpy as sc
from anndata import *

from .lightning_mapping_optim import *


def validate_mapping_inputs(
        adata_sc,
        adata_sp,
        mode="cells",
        learning_rate=0.1,
        num_epochs=1000,
        lambda_d=0,
        lambda_g1=1,
        lambda_g2=0,
        lambda_r=0,
        lambda_count=1,
        lambda_f_reg=1,
        target_count=None,
        density_prior='rna_count_based',
):
    """
    Validates inputs for cell-to-space mapping functions in the Tangram framework.

    Args:
        adata_sc (AnnData): Single-cell RNA-seq data object for source cells
        adata_sp (AnnData): Spatial transcriptomics data object for target space
        mode (str): Mapping mode, either 'cells' or 'constrained'. Default is 'cells'.
        learning_rate (float): Optional. Learning rate for training. Default is 0.1.
        num_epochs (int): Optional. Number of epochs for training. Default is 1000.
        lambda_d (float): Optional. Weight for KL divergence term. Default is 0.
        lambda_g1 (float): Optional. Weight for gene-voxel cosine similarity term. Default is 1.
        lambda_g2 (float): Optional. Weight for voxel-gene cosine similarity term. Default is 0.
        lambda_r (float): Optional. Weight for regularizer term. Default is 0.
        lambda_count (float): Optional. Weight for target cell count regularizer. Default is 1.
        lambda_f_reg (float): Optional. Weight for sigmoid regularizer. Default is 1.
        target_count (int): Optional. Target number of cells. Default is None (computed automatically).
        density_prior (str or np.ndarray): Optional. Density prior for spatial locations. Default is 'rna_count_based'.


    Returns:
        hyperparameters (dict): Dictionary of hyperparameters for training
        d (np.ndarray): Density prior for spatial locations
        d_str (str): String representation of density prior for logging purposes

    Raises:
        ValueError: If inputs are invalid or incompatible
    """

    ### INPUT Control
    # checks invalid values for arguments
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
    if mode not in ["cells", "constrained"]:
        raise ValueError('Argument "mode" must be "cells" or "constrained')

    # Validate and process genes
    ### Training Genes
    # Check if training_genes key exists/is valid in adata.uns
    if not set(["training_genes", "overlap_genes"]).issubset(set(adata_sc.uns.keys())):
        raise ValueError("Missing tangram parameters. Run `pp_adatas()`.")

    if not set(["training_genes", "overlap_genes"]).issubset(set(adata_sp.uns.keys())):
        raise ValueError("Missing tangram parameters. Run `pp_adatas()`.")

    assert list(adata_sp.uns["training_genes"]) == list(adata_sc.uns["training_genes"])

    # Validate numerical parameters
    if num_epochs <= 0:
        raise ValueError("num_epochs must be positive")
    if learning_rate <= 0:
        raise ValueError("learning_rate must be positive")

    ### Density prior
    # defines density_prior if 'rna_count_based' is passed to the density_prior argument:
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

    ## Create hyperparameters dictionary for all next calls
    if mode == "cells":
        hyperparameters = {
            "lambda_d": lambda_d,  # KL (ie density) term
            "lambda_g1": lambda_g1,  # gene-voxel cos sim
            "lambda_g2": lambda_g2,  # voxel-gene cos sim
            "lambda_r": lambda_r,  # regularizer: penalize entropy
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

    return hyperparameters, d, d_str


def map_cells_to_space_lightning(
        adata_sc,
        adata_sp,
        mode="cells",
        learning_rate=0.1,
        num_epochs=1000,
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
        cv_train_genes=None,
 ):
    """
    Maps single cells to spatial locations using the Lightning-based Tangram implementation.

    Args:
        (same as validate_mapping_inputs)
        random_state (int): Optional. Random seed for reproducibility. Default is None.
        verbose (bool): Optional. Sets echo for training. Default is True.

    Returns:
        AnnData: Updated single-cell object with mapping results
    """

    # Invoke input control function
    hyperparameters, d, d_str = validate_mapping_inputs(adata_sc=adata_sc,
                                                        adata_sp=adata_sp,
                                                        mode=mode,
                                                        learning_rate=learning_rate,
                                                        num_epochs=num_epochs,
                                                        lambda_d=lambda_d,
                                                        lambda_g1=lambda_g1,
                                                        lambda_g2=lambda_g2,
                                                        lambda_r=lambda_r,
                                                        lambda_count=lambda_count,
                                                        lambda_f_reg=lambda_f_reg,
                                                        target_count=target_count,
                                                        density_prior=density_prior,
                                                        )

    # Set echo
    if verbose:
        print_each = 100
    else:
        print_each = None

    # Call the data module to retrieve batch size
    data = MyDataModule(adata_sc, adata_sp, train_genes=cv_train_genes)

    # Initialize the model
    model = MapperLightning(
        d=d,
        lambda_g1=lambda_g1,
        lambda_d=lambda_d,
        lambda_g2=lambda_g2,
        lambda_r=lambda_r,
        lambda_count=lambda_count,
        lambda_f_reg=lambda_f_reg,
        target_count=target_count,
        learning_rate=learning_rate,
        constraint=mode == "constrained",
        random_state=random_state,
    )

    # Initialize trainer
    trainer = pl.Trainer(
        max_epochs=num_epochs,
        log_every_n_steps=print_each,
        enable_progress_bar=True
    )

    # Train the model
    trainer.fit(model, datamodule=data)

    # Get the final mapping matrix
    with torch.no_grad():
        if model.hparams.constraint:
            mapping, _, filter_probs = model()  # Unpack values (skip filtered M matrix)
            final_mapping = mapping.cpu().numpy()
            final_filter = filter_probs.cpu().numpy()
        else:
            final_mapping = model().cpu().numpy()

    logging.info("Saving results..")
    if cv_train_genes:  # if it is cross-validating use fold training genes
        training_genes = cv_train_genes
    else:  # else automatically retrieve training genes from adata_sc and adata_sp
        training_genes = adata_sc.uns['training_genes']

    adata_map = sc.AnnData(
        X=final_mapping,
        obs=adata_sc[:, training_genes].obs.copy(),
        var=adata_sp[:, training_genes].obs.copy(),
    )

    if mode == "constrained":
        adata_map.obs["F_out"] = final_filter

    # Store history in adata_map
    adata_map.uns['training_history'] = {
        'total_loss' : model.history['loss'],
        'main_loss': model.history['main_loss'],
        'vg_reg': model.history['vg_reg'],
        'kl_reg': model.history['kl_reg'],
        'entropy_reg': model.history['entropy_reg'],
        'count_reg': model.history['count_reg'],
        'lambda_f_reg': model.history['lambda_f_reg']
    }

    # Store filter-related information if using constrained mode
    if model.hparams.constraint:
        adata_map.uns['filter_history'] = {
            'filter_values': model.filter_history['filter_values'],
            'n_cells': model.filter_history['n_cells']
        }
        # Store final filter values
        adata_map.uns['filter'] = model.get_filter()

    # Annotate cosine similarity of each training gene (needed to use tangram.utils.project_genes)
    G_predicted = adata_map.X.T @ data.train_dataset[0]["S"]  # access S matrix through model attributes
    cos_sims = []
    for v1, v2 in zip(data.train_dataset[0]["G"].T, G_predicted.T):
        norm_sq = np.linalg.norm(v1) * np.linalg.norm(v2)
        cos_sims.append((v1 @ v2) / norm_sq)

    df_cs = pd.DataFrame(cos_sims, training_genes, columns=["train_score"])
    df_cs = df_cs.sort_values(by="train_score", ascending=False)
    adata_map.uns["train_genes_df"] = df_cs

    return adata_map