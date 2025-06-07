'''
Lightning module for Tangram
'''
import random

import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn as nn
from torch.nn.functional import softmax, cosine_similarity
from torch.optim.lr_scheduler import ReduceLROnPlateau


class MapperLightning(pl.LightningModule):
    def __init__(
            self,
            d=None,
            lambda_g1=1.0,
            lambda_d=0,
            lambda_g2=0,
            lambda_r=0,
            lambda_count=1,
            lambda_f_reg=1,
            target_count=None,
            constraint=False,
            learning_rate=0.1,
            random_state=None
    ):
        """
        Lightning Module for Tangram mapping.

        Args:
            d (ndarray, optional): Target density vector (1D array)
            lambda_g1 (float): Weight for gene-voxel similarity term
            lambda_d (float): Weight for density term
            lambda_g2 (float): Weight for voxel-gene similarity term
            lambda_r (float): Weight for entropy regularizer
            lambda_count (float): Weight for count regularizer
            lambda_f_reg (float): Weight for filter regularizer
            target_count (int, optional): Target number of cells for filtering
            constraint (bool): Whether to use cell filtering
            learning_rate (float): Learning rate for optimizer
            random_state (int, optional): Random seed for initialization
        """

        super().__init__()
        self.save_hyperparameters()


        # Turn target density vector (ndarray) into torch.tensor
        self.target_density_enabled = d is not None
        if self.target_density_enabled:
            self.d = torch.tensor(d, dtype=torch.float32)


        # Pass all regularization coefficients (input) as class attributes
        self.lambda_d = lambda_d
        self.lambda_g1 = lambda_g1
        self.lambda_g2 = lambda_g2
        self.lambda_r = lambda_r
        self.lambda_count = lambda_count
        self.lambda_f_reg = lambda_f_reg
        self.learning_rate = learning_rate

        # Set filter mode as attribute
        self.constraint = constraint

        # Initialize density criterion
        self._density_criterion = nn.KLDivLoss(reduction="sum")

        # Set target number (if filter is True) ---> see setup method
        #if self.constraint:
            # if the target is not provided, set to number of voxels
         #   if target_count is None:
          #     self.target_count = self.G.shape[0]
            #else:
           #     self.target_count = target_count



        # Parameters M and F will be initialized in setup()
        self.M = None
        self.F = None

        #self.M = nn.Parameter(torch.randn(S.shape[0], G.shape[0]))
        # Set initial conditions of F to Standard distributed values (if filter is True)
        #if self.constraint:
         #   self.F = nn.Parameter(torch.randn(S.shape[0],))

        # Create training history dictionary
        self.history = {
            'loss': [],
            'main_loss': [],
            'vg_reg': [],
            'kl_reg': [],
            'entropy_reg': [],
            'count_reg': [],
            'lambda_f_reg': []
        }

        # Add filter history tracking
        self.filter_history = {
            'filter_values': [],  # Store filter values per epoch
            'n_cells': []  # Store number of cells that pass the filter per epoch
        }

    def setup(self, stage=None):
        """
        Initialize mapping matrices using data dimensions from datamodule.
        Called after datamodule is set but before training starts.
        """
        if stage == 'fit' or stage is None:
            # Get a batch from the datamodule to determine dimensions
            dataloader = self.trainer.datamodule.train_dataloader()
            batch = next(iter(dataloader))
            S, G = batch['S'], batch['G']

            n_cells, n_genes_sc = S.squeeze().shape
            n_spots, n_genes_st = G.squeeze().shape

            assert n_genes_sc == n_genes_st, "Number of genes must match between datasets"

            # Set random seed if specified
            if self.hparams.random_state is not None:
                torch.manual_seed(self.hparams.random_state)
                random.seed(self.hparams.random_state)
                np.random.seed(self.hparams.random_state)

            # Initialize mapping matrix M
            self.M = nn.Parameter(torch.randn(n_cells, n_spots))

            # Initialize filter F if using constraints
            if self.constraint:
                self.F = nn.Parameter(torch.randn(n_cells))

                # Set target count if not provided
                if self.hparams.target_count is None:
                    self.hparams.target_count = n_spots

    def forward(self):
        """
        Compute the mapping probabilities.
        """
        if self.constraint:
            F_probs = torch.sigmoid(self.F)
            M_probs = softmax(self.M, dim=1)
            return M_probs * F_probs[:, None], F_probs
        else:
            return softmax(self.M, dim=1)

    def training_step(self, batch, batch_idx):
        """
        Training step using data from the datamodule.
        """
        S = batch['S']  # single-cell data
        G = batch['G']  # spatial data

        # Forward step to get mapping probabilities
        if self.constraint:
            M_probs, F_probs = self()  # Get softmax probabilities and filter probabilities
        else:
            M_probs = self()  # Get softmax probabilities


        ## Loss computation
        # Calculate density term
        density_term = None
        self.target_density_enabled = self.d is not None
        if self.target_density_enabled:
            d_pred = torch.log(M_probs.sum(axis=0) / self.M.shape[0])
            density_term = self.lambda_d * self._density_criterion(d_pred, self.d)

        # Calculate expression terms
        G_pred = torch.matmul(M_probs.t(), S)
        gv_term = self.hparams.lambda_g1 * cosine_similarity(G_pred, G, dim=0).mean()
        vg_term = self.hparams.lambda_g2 * cosine_similarity(G_pred, G, dim=1).mean()
        expression_term = gv_term + vg_term

        # Calculate regularizer term
        regularizer_term = self.hparams.lambda_r * (torch.log(M_probs) * M_probs).sum()

        # Calculate total loss
        total_loss = -expression_term - regularizer_term

        self.target_density_enabled = self.d is not None
        if density_term is not None:
            total_loss = total_loss + density_term

        # Define count term and filter regularizers (if filter mode)
        if self.constraint:
            # Count term: abs( sum(f_i, over cells i) - n_target_cells)
            count_term = self.lambda_count * torch.abs(F_probs.sum() - self.hparams.target_count)
            # Filter regularizer: sum(f_i - f_i^2, over cells i)
            f_reg = self.lambda_f_reg * (F_probs - F_probs * F_probs).sum()
            total_loss = total_loss + count_term + f_reg


        # Log metrics
        self.log('train_loss', total_loss, prog_bar=True)
        self.log('gv_loss', -gv_term)
        self.log('vg_loss', -vg_term)
        self.log('reg_term', regularizer_term)
        if density_term is not None:
            self.log('density_loss', density_term)


        ## Compute history terms
        step_output = {
            "loss": total_loss,
            "main_loss": gv_term,
            "vg_reg": vg_term,
            "kl_reg": density_term if density_term is not None else torch.tensor(float('nan')),
            "entropy_reg": regularizer_term,
            "count_reg": count_term if self.constraint else torch.tensor(float('nan')),
            "lambda_f_reg": f_reg if self.constraint else torch.tensor(float('nan'))
        }

        # Store values for the epoch
        self.last_step_values = {k: v.detach() for k, v in step_output.items()}

        # Store filter values if in constrained mode
        if self.constraint:
            self.last_filter_values = {
                'filter_values': F_probs.detach(),
                'n_cells': torch.sum(F_probs > 0.5).detach()  # Count cells with filter prob > 0.5
            }

        return step_output

    def on_train_epoch_end(self):
        # Store values at the end of each epoch
        for key in self.history.keys():
            if key in self.last_step_values:
                value = self.last_step_values[key].cpu().numpy()
                self.history[key].append(float(value))

        # Store filter history if in constrained mode
        if self.constraint:
            self.filter_history['filter_values'].append(
                self.last_filter_values['filter_values'].cpu().numpy()
            )
            self.filter_history['n_cells'].append(
                self.last_filter_values['n_cells'].cpu().numpy()
            )

    def configure_optimizers(self):
        # optimizer and learning rate
        params = [self.M]

        if self.constraint:
            optimizer = torch.optim.Adam([self.M, self.F], lr=self.learning_rate)
        else:
            optimizer = torch.optim.Adam([self.M], lr=self.learning_rate)
        scheduler = {
            'scheduler': ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=10),
            'monitor': 'train_loss',
            'interval': 'epoch',
            'frequency': 1
        }
        return [optimizer], [scheduler]

    # Add a method to get the final filter values
    def get_filter(self):
        """
        Returns the final filter values after training.
        """
        if not self.constraint:
            return None

        with torch.no_grad():
            _, F_probs = self()
            return F_probs.cpu().numpy()


from torch.utils.data import DataLoader

class MyDataModule(pl.LightningDataModule):
    def __init__(self, adata_sc=None, adata_st=None, data_dir=None, batch_size=None):
        super().__init__()
        self.adata_sc = adata_sc
        self.adata_st = adata_st
        self.data_dir = data_dir
        self.batch_size = batch_size


    def prepare_data(self):
        """
        Load h5ad files from the specified directory if not provided directly.
        This method is called only once and on the main process only.
        """
        import anndata as ad
        import os
        import logging

        if self.adata_sc is None or self.adata_st is None:
            if self.data_dir is None:
                raise ValueError("Either provide adata_sc and adata_st directly or specify data_dir")

            logging.info(f"Loading data from {self.data_dir}")

            # Find h5ad files in the directory
            sc_files = [f for f in os.listdir(self.data_dir) if f.endswith('_sc.h5ad') or 'sc_' in f]
            st_files = [f for f in os.listdir(self.data_dir) if f.endswith('_st.h5ad') or 'st_' in f or 'sp_' in f]

            if not sc_files:
                raise FileNotFoundError(f"No single-cell h5ad files found in {self.data_dir}")
            if not st_files:
                raise FileNotFoundError(f"No spatial h5ad files found in {self.data_dir}")

            # Use the first file found for each type
            sc_path = os.path.join(self.data_dir, sc_files[0])
            st_path = os.path.join(self.data_dir, st_files[0])

            logging.info(f"Loading single-cell data from {sc_path}")
            logging.info(f"Loading spatial data from {st_path}")

            # Load the data
            self.adata_sc = ad.read_h5ad(sc_path)
            self.adata_st = ad.read_h5ad(st_path)

            # Preprocess data --- equivalent to pp_adatas but internal
            import numpy as np
            # Remove genes with zero counts
            self.adata_sc = self.adata_sc[:, np.array(self.adata_sc.X.sum(axis=0)).flatten() > 0]
            self.adata_st = self.adata_st[:, np.array(self.adata_st.X.sum(axis=0)).flatten() > 0]

            # Import mytangram for preprocessing
            try:
                import mytangram as tg
                tg.pp_adatas(self.adata_sc, self.adata_st)
                logging.info("Preprocessed data with mytangram.pp_adatas")
            except ImportError:
                logging.warning("mytangram not found, skipping pp_adatas step")

    def setup(self, stage=None):
        """
        Setup datasets for use in dataloaders.
        This method is called on every GPU separately.
        """
        if stage == 'fit' or stage is None:
            self.train_dataset = AdataPairDataset(self.adata_sc, self.adata_st)

            # If batch_size wasn't specified in __init__, use the full dataset
            if self.batch_size is None:
                self.batch_size = len(self.train_dataset)

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
            pin_memory=True  # Speed up data transfer to GPU if using CUDA
        )


from torch.utils.data import Dataset


class AdataPairDataset(Dataset):
    def __init__(self, adata_sc, adata_sp):
        import logging
        from scipy.sparse import csc_matrix, csr_matrix

        # Get training genes from adata_sc.uns if available
        if 'training_genes' in adata_sc.uns and 'training_genes' in adata_sp.uns:
            training_genes = adata_sc.uns['training_genes']
            logging.info(f"Using {len(training_genes)} training genes from adata.uns")
        else:
            # Use all genes shared between datasets
            training_genes = list(set(adata_sc.var_names).intersection(set(adata_sp.var_names)))
            logging.info(f"Using {len(training_genes)} shared genes between datasets")

        ## S matrix (single-cell)
        if isinstance(adata_sc.X, csc_matrix) or isinstance(adata_sc.X, csr_matrix):
            self.S = np.array(adata_sc[:, training_genes].X.toarray(), dtype="float32")
        elif isinstance(adata_sc.X, np.ndarray):
            self.S = np.array(adata_sc[:, training_genes].X, dtype="float32")
        else:
            X_type = type(adata_sc.X)
            logging.error(f"AnnData X has unrecognized type: {X_type}")
            raise NotImplementedError

        # G matrix (spatial)
        if isinstance(adata_sp.X, csc_matrix) or isinstance(adata_sp.X, csr_matrix):
            self.G = np.array(adata_sp[:, training_genes].X.toarray(), dtype="float32")
        elif isinstance(adata_sp.X, np.ndarray):
            self.G = np.array(adata_sp[:, training_genes].X, dtype="float32")
        else:
            X_type = type(adata_sp.X)
            logging.error(f"AnnData X has unrecognized type: {X_type}")
            raise NotImplementedError

        # Store metadata
        self.training_genes = training_genes
        self.n_cells = self.S.shape[0]
        self.n_spots = self.G.shape[0]
        self.n_genes = len(training_genes)

        logging.info(f"Created dataset with {self.n_cells} cells, {self.n_spots} spots, and {self.n_genes} genes")


    def __len__(self):
        # Return 1 as we want to process the entire dataset in a single batch
        # This is appropriate for Tangram which maps all cells at once
        return 1

    def __getitem__(self, i):
        # Return all data in one batch (Tangram maps entire datasets at once)
        return {
          "S": self.S,  # Single-cell data (cells x genes)
          "G": self.G,  # Spatial data (spots x genes)
          "training_genes": self.training_genes  # List of gene names used
        }