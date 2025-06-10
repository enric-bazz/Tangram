import numpy as np
import torch
from torch.nn.functional import softmax, cosine_similarity

class Mapper:

    def __init__(
        self,
        S,
        G,
        d=None,
        d_source=None,
        lambda_g1=1.0,
        lambda_d=0,
        lambda_g2=0,
        lambda_r=0,
        lambda_count=1,
        lambda_f_reg=1,
        target_count=None,
        constraint=False,
        device="cpu",
        random_state=None,
    ):
        """
            Instantiate the Tangram optimizer (with possible filtering).

            Args:
                S (ndarray): Single nuclei matrix, shape = (number_cell, number_genes).
                G (ndarray): Spatial transcriptomics matrix, shape = (number_spots, number_genes).
                    Spots can be single cells or they can contain multiple cells.
                d (ndarray): Spatial density of cells, shape = (number_spots,).
                             This array should satisfy the constraints d.sum() == 1.
                d_source (ndarray): Density of single cells in single cell clusters. To be used when S corresponds to cluster-level expression.
                              This array should satisfy the constraint d_source.sum() == 1.
                lambda_d (float): Optional. Hyperparameter for the density term of the optimizer. Default is 1.
                lambda_g1 (float): Optional. Hyperparameter for the gene-voxel similarity term of the optimizer. Default is 1.
                lambda_g2 (float): Optional. Hyperparameter for the voxel-gene similarity term of the optimizer. Default is 1.
                lambda_r (float): Optional. Entropy regularizer for the learned mapping matrix. An higher entropy promotes
                                  probabilities of each cell peaked over a narrow portion of space.
                                  lambda_r = 0 corresponds to no entropy regularizer. Default is 0.
                lambda_count (float): Optional. Regularizer for the count term. Default is 1.
                lambda_f_reg (float): Optional. Regularizer for the filter, which promotes Boolean values (0s and 1s) in the filter. Default is 1.
                target_count (int): Optional. The number of cells to be filtered. If None, this number defaults to the number of
                                    voxels inferred by the matrix 'G'. Default is None.
                constraint (bool): Cell filtering for spot-level mapping. Default is False.
                device (str or torch.device): Optional. Device is 'cpu'.
                adata_map (scanpy.AnnData): Optional. Mapping initial condition (for resuming previous mappings). Default is None.
                random_state (int): Optional. pass an int to reproduce training. Default is None.
        """

        # Turn input np.ndarray count matrices into torch.tensor
        self.S = torch.tensor(S, device=device, dtype=torch.float32)
        self.G = torch.tensor(G, device=device, dtype=torch.float32)

        # Turn target density vector (ndarray) into torch.tensor
        self.target_density_enabled = d is not None
        if self.target_density_enabled:
            self.d = torch.tensor(d, device=device, dtype=torch.float32)

        # Turn cluster density vector into torch.tensor (if mode == 'cluster')
        self.source_density_enabled = d_source is not None
        if self.source_density_enabled:
            self.d_source = torch.tensor(d_source, device=device, dtype=torch.float32)


        # Pass all regularization coefficients (input) as class attributes
        self.lambda_d = lambda_d
        self.lambda_g1 = lambda_g1
        self.lambda_g2 = lambda_g2
        self.lambda_r = lambda_r
        self.lambda_count = lambda_count
        self.lambda_f_reg = lambda_f_reg

        # Set filter mode as attribute
        self.constraint = constraint

        # Define loss function for density distribution as attribute: KL Divergence
        self._density_criterion = torch.nn.KLDivLoss(reduction="sum")

        # Set target number (if filter is True)
        if self.constraint:
            # if the target is not provided, set to number of voxels
            if target_count is None:
                self.target_count = self.G.shape[0]
            else:
                self.target_count = target_count

        # Initialize learned matrices

        # Set RNG seed
        self.random_state = random_state
        if self.random_state:
            np.random.seed(seed=self.random_state)

        # Set initial conditions of M to Standard distributed values
        self.M = torch.randn((S.shape[0], G.shape[0]), device=device, requires_grad=True, dtype=torch.float32)
        # Set initial conditions of F to Standard distributed values (if filter is True)
        if self.constraint:
            self.F = torch.randn((S.shape[0],), device=device, requires_grad=True, dtype=torch.float32)

    def _loss_fn(self, verbose=True):

        # Softmax matrices along columns
        M_probs = softmax(self.M, dim=1)
        # In filter mode, filter the M matrix
        if self.constraint:
            F_probs = torch.sigmoid(self.F)
            M_probs_filtered = M_probs * F_probs[:, np.newaxis]
        ## Recall that the loss function utilizes both the filtered and unfiltered M matrices
        ## To operate easily, M_probs_filtered is used in place of M_porbs every time it is required
        ## In non-filter mode it is simply assigned to M_probs

        # Compute estimated density vector according to definition m_j = sum(M_ij, over i)/n_cells
        # Cluster-level mode
        if self.target_density_enabled and self.source_density_enabled:
            d_pred = torch.log(self.d_source @ M_probs)
            # np.matmul() works because the anndata object, and thus the M matrix, is ensured to have n_cells = n_clusters = len(d_source)
            density_term = self.lambda_d * self._density_criterion(d_pred, self.d)
        # Cell-level mode
        elif self.target_density_enabled and not self.source_density_enabled:
            # Filter mode
            if self.constraint:
                d_pred = torch.log(M_probs_filtered.sum(axis=0) / (F_probs.sum()))  # sum of filtered cells
            # No filter
            else:
                d_pred = torch.log(M_probs.sum(axis=0) / self.M.shape[0]) # number of cells
            density_term = self.lambda_d * self._density_criterion(d_pred, self.d)
        else:
            density_term = None

        # Compute predicted G matrix as M^T S
        if self.constraint:
            S_filtered = self.S * F_probs[:, np.newaxis]
            G_pred = torch.matmul(M_probs.t(), S_filtered)
        else:
            G_pred = torch.matmul(M_probs.t(), self.S)

        # Compute similarity terms
        # The paper computes sum, original code computes mean: the difference is the respective scaling factors n_genes, n_voxels
        # Each of the terms is scaled by a lambda_g factor which can absorb the scaling resulting from sum and adjust scale
        # Empirical results show that the sum() is out of scaled compared to other terms
        gv_term = self.lambda_g1 * cosine_similarity(G_pred, self.G, dim=0).mean()
        vg_term = self.lambda_g2 * cosine_similarity(G_pred, self.G, dim=1).mean()
        expression_term = gv_term + vg_term
        #gv_term_sum = self.lambda_g1 * cosine_similarity(G_pred, self.G, dim=0).sum()
        #vg_term_sum = self.lambda_g2 * cosine_similarity(G_pred, self.G, dim=1).sum()


        # Define entropy(M) regularizer term (sum all entries - return scalar)
        regularizer_term = self.lambda_r * (torch.log(M_probs) * M_probs).sum()

        # Define count term and filter regularizers (if filter mode)
        if self.constraint:
            # Count term: abs( sum(f_i, over cells i) - n_target_cells)
            count_term = self.lambda_count * torch.abs(F_probs.sum() - self.target_count)
            # Filter regularizer: sum(f_i - f_i^2, over cells i)
            f_reg = self.lambda_f_reg * (F_probs - F_probs * F_probs).sum()

        # Compute final loss
        total_loss = -expression_term - regularizer_term
        # same but with sum
        if density_term is not None:
            total_loss = total_loss + density_term
        if self.constraint:
            total_loss = total_loss + count_term + f_reg



        # Define outputs: each variable corresponds to a term of the loss, re-scaled of the regularizer and turned to list
        main_loss = (gv_term).tolist()  # not .tolist() to exclude string metadata
        kl_reg = (
            (density_term).tolist()
            if density_term is not None
            else np.nan
        )  # conditional assignment must return something
        vg_reg = (vg_term).tolist()
        if self.constraint:
            count_reg = (count_term).tolist()
            lambda_f_reg = (f_reg).tolist()
        else:
            count_reg = None
            lambda_f_reg = None

        entropy_reg = (regularizer_term).tolist()

        # verbose print message
        if verbose:

            term_numbers = [main_loss, vg_reg, kl_reg, entropy_reg]
            term_names = ["Score", "VG reg", "KL reg", "Entropy reg"]

            if self.constraint:
                temp_list = [[count_reg, "Count reg"],
                             [lambda_f_reg, "Lambda f reg"]]
                for sub_list in temp_list:
                    term_numbers.append(sub_list[0])
                    term_names.append(sub_list[1])

            d = dict(zip(term_names, term_numbers))
            clean_dict = {k: d[k] for k in d if not np.isnan(d[k])}
            msg = []
            for k in clean_dict:
                m = "{}: {:.3f}".format(k, clean_dict[k])
                msg.append(m)
            print(str(msg).replace("[", "").replace("]", "").replace("'", ""))


        return (
            total_loss,
            main_loss,
            vg_reg,
            kl_reg,
            entropy_reg,
            count_reg,
            lambda_f_reg,
        )

    def train(self, num_epochs, learning_rate=0.1, print_each=100):

        """
        Run the optimizer and returns the mapping outcome.

        Args:
            num_epochs (int): Number of epochs.
            learning_rate (float): Optional. Learning rate for the optimizer. Default is 0.1.
            print_each (int): Optional. Prints the loss each print_each epochs. If None, the loss is never printed. Default is 100.

        Returns:
            A tuple (output, F_out, training_history), with:
                M (ndarray): is the optimized mapping matrix, shape = (number_cells, number_spots).
                f (ndarray): is the optimized filter, shape = (number_cells,).
                training_history (dict): loss for each epoch
        """

        # Set seed (if given)
        if self.random_state:
            torch.manual_seed(seed=self.random_state)

        # optimizer and learning rate
        if self.constraint:
            optimizer = torch.optim.Adam([self.M, self.F], lr=learning_rate)
        else:
            optimizer = torch.optim.Adam([self.M], lr=learning_rate)

        # Define keys of stored values (loss terms)
        keys = [
            "total_loss",
            "main_loss",
            "vg_reg",
            "kl_reg",
            "entropy_reg",
        ]
        if self.constraint:
            for name in ["count_reg", "lambda_f_reg"]:
                keys.append(name)

        # Init values and training history dictionary
        values = [[] for i in range(len(keys))]
        training_history = {key: value for key, value in zip(keys, values)}

        # Init filter history: it will output with shape (n_cells, n_epochs)
        if self.constraint:
            filter_history = []

        # Training loop
        for t in range(num_epochs):
            # Run _loss_fn() method as verbose only for epochs that are multiples of _print_each_
            if print_each is None or t % print_each != 0:
                run_loss = self._loss_fn(verbose=False)
            else:
                run_loss = self._loss_fn(verbose=True)

            # Retrieve total_loss from _loss_fn() method
            loss = run_loss[0]

            # Append current values to dictionary lists
            for i in range(len(keys)):
                if i == 0:
                    training_history[keys[i]].append(loss.item())  # detach tensor from grad
                else:
                    training_history[keys[i]].append(run_loss[i])  # remove str(run_loss[i]) to maintain float values

            # Append current filter weights
            if self.constraint:
                filter_history.append(torch.sigmoid(self.F).detach().numpy())

            # Learning steps
            # set optimizer gradient entries to zero
            optimizer.zero_grad()
            # compute gradient of loss
            loss.backward()
            # compute parameters update/gradient step
            optimizer.step()

        # Take final softmax w/o computing gradients
        with torch.no_grad():
            output = softmax(self.M, dim=1).cpu().numpy()
            if self.constraint:
                F_out = torch.sigmoid(self.F).cpu().numpy()
                return output, F_out, training_history, filter_history
            else:
                return output, training_history
