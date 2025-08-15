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
                 train_genes=None,
                 refined_mode=False,
                 train_genes_idx=None,
                 val_genes_idx=None
                 ):
        """
        Lightly preprocessed single-cell and spatial anndata objects.

        Args:
            adata_sc (AnnData): Single-cell AnnData object.
            adata_st (AnnData): Spatial AnnData object.
            train_genes (list): List of genes to use for training. If None, use all genes shared between adata_sc and adata_st.
            refined_mode (bool): Whether to use refined mode for training. If True, use refined mode. If False, use unrefined mode. Default is False.
            train_genes_idx (list): List of indices of genes to use for training. If None, use all genes shared between adata_sc and adata_st.
            val_genes_idx (list): List of indices of genes to use for validation.
        """
        super().__init__()
        self.adata_sc = adata_sc
        self.adata_st = adata_st
        self.train_genes = train_genes  # Allow passing specific genes for CV and training
        self.refined_mode = refined_mode
        self.train_genes_idx = train_genes_idx
        self.val_genes_idx = val_genes_idx

        # 2. Compute spatial neighbors needed for the neighborhood extension of Tangram
        if self.refined_mode:
            sq.gr.spatial_neighbors(self.adata_st, set_diag=False, key_added="spatial")
        # If not in refined mode, spatial coordinates are not required in the input anndata

    def prepare_data(self):
        """
        Takes anndata objects and prepares them for mapping.
        Executed before setup() is called.
        """

        # Preprocess data

        # 1. Training genes
        # Remove genes with zero counts
        self.adata_sc = self.adata_sc[:, np.array(self.adata_sc.X.sum(axis=0)).flatten() > 0]
        self.adata_st = self.adata_st[:, np.array(self.adata_st.X.sum(axis=0)).flatten() > 0]

        # Execute Tangram preprocessing steps (originally implemented in tg.mapping_utils.pp_adatas()
        logging.info("Preprocessing data...")

        # remove all-zero-valued genes with scanpy utility
        sc.pp.filter_genes(self.adata_sc, min_cells=1)
        sc.pp.filter_genes(self.adata_st, min_cells=1)

        # put all var indexes to lower case to align
        self.adata_sc.var.index = [g.lower() for g in self.adata_sc.var.index]
        self.adata_st.var.index = [g.lower() for g in self.adata_st.var.index]

        # make genes unique
        self.adata_sc.var_names_make_unique()
        self.adata_st.var_names_make_unique()

        # Define training genes as intersection of input training genes and anndata var indexes
        if self.train_genes is not None:
            genes = list(set(self.train_genes) & set(self.adata_sc.var.index) & set(self.adata_st.var.index))
        else:
            genes = list(set(self.adata_sc.var.index) & set(self.adata_st.var.index))
        logging.info(f"{len(genes)} shared marker genes.")

        self.adata_sc.uns["training_genes"] = genes
        self.adata_st.uns["training_genes"] = genes



    def setup(self, stage=None):
        """
        Setup datasets for use in dataloaders.
        This method is called on every GPU separately.
        Execute after prepare_data() and before train_dataloader().
        """
        if stage == 'fit' or stage is None:
            self.train_dataset = AdataPairDataset(self.adata_sc,
                                                  self.adata_st,
                                                  train_genes=self.train_genes,
                                                  train_genes_idx=self.train_genes_idx,
                                                  val_genes_idx=self.val_genes_idx
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


class AdataPairDataset(Dataset):
    def __init__(self,
                 adata_sc,
                 adata_st,
                 train_genes=None,
                 train_genes_idx=None,
                 val_genes_idx=None,
                 ):

        if train_genes is not None:
            # If specific genes are provided for CV/training, intersect with the preprocessed genes
            training_genes = list(set(train_genes) & set(adata_sc.uns['training_genes']))
            logging.info(f"Using {len(training_genes)} training genes from user input (after intersection)")
        else:
            # Use all preprocessed genes from prepare_data
            training_genes = adata_sc.uns['training_genes']
            logging.info(f"Using {len(training_genes)} training genes from preprocessing (intersection)")

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

        # Store train/val genes indexes
        self.train_genes_idx = train_genes_idx if train_genes_idx is not None else slice(None)
        self.val_genes_idx = val_genes_idx if val_genes_idx is not None else slice(None)
        # NOTE: When both indices are `None`, it defaults to using all genes for both training and validation

        # Store metadata
        self.training_genes = training_genes
        self.n_cells = self.S.shape[0]
        self.n_spots = self.G.shape[0]
        self.n_genes = len(training_genes)

    def __len__(self):
        return 1

    def __getitem__(self, idx):
        return {
            'S': self.S,
            'G': self.G,
            'training_genes': self.training_genes,
            'train_genes_idx': self.train_genes_idx,
            'val_genes_idx': self.val_genes_idx

        }