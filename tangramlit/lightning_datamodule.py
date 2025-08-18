import logging

import numpy as np
import pytorch_lightning as pl
import scanpy as sc
import squidpy as sq
import torch
from scipy.sparse import csc_matrix, csr_matrix
from torch.utils.data import Dataset, DataLoader


class MyDataModule(pl.LightningDataModule):
    """
        Lightning DataModule for Tangram mapping.
    """

    def __init__(self,
                 adata_sc=None,
                 adata_st=None,
                 input_genes=None,
                 refined_mode=False,
                 train_genes_names=None,
                 val_genes_names=None
                 ):
        """
        Lightly preprocessed single-cell and spatial anndata objects.

        Args:
            adata_sc (AnnData): Single-cell AnnData object.
            adata_st (AnnData): Spatial AnnData object.
            input_genes (list): List of input genes to use for training. If None, use all genes shared between adata_sc and adata_st.
            refined_mode (bool): Whether to use refined mode for training. If True, use refined mode. If False, use unrefined mode. Default is False.
            train_genes_names (list): List of names of genes to use for training. If None, use all genes shared between adata_sc and adata_st.
            val_genes_names (list): List of names of genes to use for validation.
        """
        super().__init__()
        self.adata_sc = adata_sc
        self.adata_st = adata_st
        self.input_genes = input_genes  # Allow passing specific genes for training
        self.refined_mode = refined_mode  # Flag to require spatial coordinates for refined mode
        self.train_genes_names = train_genes_names
        self.val_genes_names = val_genes_names

        # Turn all gene names to lowercase
        if self.input_genes is not None:
            self.input_genes = [g.lower() for g in self.input_genes]
        if self.train_genes_names is not None:
            self.train_genes_names = [g.lower() for g in self.train_genes_names]
        if self.val_genes_names is not None:
            self.val_genes_names = [g.lower() for g in self.val_genes_names]

        # Compute spatial neighbors needed for the neighborhood extension of Tangram
        if self.refined_mode:
            sq.gr.spatial_neighbors(self.adata_st, set_diag=False, key_added="spatial")
        # If not in refined mode, spatial coordinates are not required in the input anndata

    def prepare_data(self):
        """
        Takes anndata objects and prepares them for mapping.
        Executed before setup() is called.
        """

        # Preprocess data - define adata.uns['training_genes'] - originally implemented in tg.mapping_utils.pp_adatas()
        logging.info("Preprocessing data...")
        # 1. Remove genes with zero counts
        self.adata_sc = self.adata_sc[:, np.array(self.adata_sc.X.sum(axis=0)).flatten() > 0]
        self.adata_st = self.adata_st[:, np.array(self.adata_st.X.sum(axis=0)).flatten() > 0]

        # 2. Remove all-zero-valued genes with scanpy utility
        sc.pp.filter_genes(self.adata_sc, min_cells=1)
        sc.pp.filter_genes(self.adata_st, min_cells=1)

        # 3. Put all var indexes to lower case to align
        self.adata_sc.var.index = [g.lower() for g in self.adata_sc.var.index]
        self.adata_st.var.index = [g.lower() for g in self.adata_st.var.index]

        # 4. Make genes unique
        self.adata_sc.var_names_make_unique()
        self.adata_st.var_names_make_unique()

        # 5. Define training genes as intersection of input training genes and anndata var indexes
        if self.input_genes is not None:
            genes = list(set(self.input_genes) & set(self.adata_sc.var.index) & set(self.adata_st.var.index))
            logging.info(f"Using {len(genes)} training genes provided by user.")
        else:
            genes = list(set(self.adata_sc.var.index) & set(self.adata_st.var.index))
        logging.info(f"Using {len(genes)} shared marker genes.")

        # 6. Store genes in adata.uns['training_genes']
        self.adata_sc.uns["training_genes"] = genes
        self.adata_st.uns["training_genes"] = genes



    def setup(self, stage=None):
        """
        Setup datasets for use in dataloaders.
        This method is called on every GPU separately.
        Execute after prepare_data() and before train/val_dataloader().
        Defines dataset based on the current training mode (stage variable).
        """
        if stage == 'fit' or stage is None:
            self.train_dataset = AdataPairDataset(self.adata_sc,
                                                  self.adata_st,
                                                  mode='train',
                                                  train_genes_names=self.train_genes_names,
                                                  )
        if stage == 'validate' or stage is None:
            self.val_dataset = AdataPairDataset(self.adata_sc,
                                                self.adata_st,
                                                mode='val',
                                                val_genes_names=self.val_genes_names,
                                                )

    def train_dataloader(self):
        """
        Return a DataLoader for training.
        For Tangram, we use a single batch containing all data.
        """
        return DataLoader(
            self.train_dataset,
            batch_size=1,  # Always use batch_size=1 as each item contains all data
            shuffle=False,  # No need to shuffle as we have just one batch
            num_workers=0,  # Process in the main thread
            pin_memory=True,  # Speed up data transfer to GPU if using CUDA
            collate_fn=lambda x: x[0]  # Prevent adding batch dimension [1, n_cells/spots, n_genes] => [n_cells/spots, n_genes]
        )

    def val_dataloader(self):
        """
        Return a DataLoader for validation.
        """
        return DataLoader(
            self.val_dataset,
            batch_size=1,  # Always use batch_size=1 as each item contains all data
            shuffle=False,  # No need to shuffle as we have just one batch
            num_workers=0,  # Process in the main thread
            pin_memory=True,  # Speed up data transfer to GPU if using CUDA
            collate_fn=lambda x: x[0]  # Prevent adding batch dimension [1, n_cells/spots, n_genes] => [n_cells/spots, n_genes]
        )


class AdataPairDataset(Dataset):
    """
    Dataset class for single-cell and spatial anndata objects.
    Returns a single batch containing all data, sliced according to the provided names and based on the
    current mode.

    Args:
        adata_sc (AnnData): Single-cell AnnData object.
        adata_st (AnnData): Spatial AnnData object.
        train_genes_names (list): List of names of genes to use for training. If None, use all genes shared between adata_sc and adata_st.
        val_genes_names (list): List of names of genes to use for validation.
        mode (str): Training mode. Can be 'train' or 'val'. Default is 'train'.
    """
    def __init__(self,
                 adata_sc,
                 adata_st,
                 mode='train',
                 train_genes_names=None,
                 val_genes_names=None,
                 ):

        # Get training genes from adata.uns['training_genes'] - defined in prepare_data()
        assert adata_sc.uns['training_genes'] == adata_st.uns['training_genes'], "Training genes must be the same for single-cell and spatial data."
        training_genes = adata_sc.uns['training_genes']

        ## S matrix (single-cell)
        if isinstance(adata_sc.X, csc_matrix) or isinstance(adata_sc.X, csr_matrix):
            self.S = torch.tensor(adata_sc[:, training_genes].X.toarray(), dtype=torch.float32)
        elif isinstance(adata_sc.X, np.ndarray):
            self.S = torch.tensor(adata_sc[:, training_genes].X, dtype=torch.float32)
        else:
            X_type = type(adata_sc.X)
            logging.error(f"Single-cell AnnData X has unrecognized type: {X_type}")
            raise NotImplementedError

        # G matrix (spatial)
        if isinstance(adata_st.X, csc_matrix) or isinstance(adata_st.X, csr_matrix):
            self.G = torch.tensor(adata_st[:, training_genes].X.toarray(), dtype=torch.float32)
        elif isinstance(adata_st.X, np.ndarray):
            self.G = torch.tensor(adata_st[:, training_genes].X, dtype=torch.float32)
        else:
            X_type = type(adata_st.X)
            logging.error(f"Spatial AnnData X has unrecognized type: {X_type}")
            raise NotImplementedError

        # Store mode and train/val genes indexes retrieved from names
        self.mode = mode
        self.train_genes_idx = gene_names_to_indices(gene_names=train_genes_names, adata=adata_st) if train_genes_names is not None else slice(None)
        self.val_genes_idx = gene_names_to_indices(gene_names=val_genes_names, adata=adata_st) if val_genes_names is not None else slice(None)
        # NOTE: When both indices are `None`, it defaults to using all genes for both training and validation

        # Store metadata
        self.training_genes = training_genes
        self.n_cells = self.S.shape[0]
        self.n_spots = self.G.shape[0]
        self.n_genes = len(training_genes)

    def __len__(self):
        return 1

    def __getitem__(self, idx):
        """
        Returns sliced data according to the current mode.
        """
        # Return only the relevant slice based on mode
        if self.mode == 'train':
            return {
                'S_train': self.S[:, self.train_genes_idx],
                'G_train': self.G[:, self.train_genes_idx],
                'training_genes_number': len(self.train_genes_idx),
            }
        else:  # validation mode
            return {
                'S_val': self.S[:, self.val_genes_idx],
                'G_val': self.G[:, self.val_genes_idx],
                'validation_genes_number': len(self.val_genes_idx),
            }

def gene_names_to_indices(gene_names, adata):
    """
    Get indices of genes in AnnData object's var, handling case sensitivity.
    Only includes genes that are present in adata.uns['training_genes'].

    Args:
        gene_names (list): List of gene names to find indices for
        adata (AnnData): AnnData object to search in

    Returns:
        list: List of indices corresponding to the input gene names

    Raises:
        ValueError: If any gene name is not found in the AnnData object
        KeyError: If 'training_genes' is not present in adata.uns
    """

    # Find indices for each gene name
    indices = []
    missing_genes = []

    for gene in gene_names:
        gene_lower = gene.lower()
        # Check if gene is in training_genes
        if gene_lower in adata.uns['training_genes']:
            indices.append(adata.uns['training_genes'].index(gene_lower))
        else:
            missing_genes.append(gene)  # use .item() if array

    if missing_genes:
        logging.warning(f"The following train/val input genes were removed with preprocessing: {missing_genes}.")

    return indices