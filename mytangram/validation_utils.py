"""
    Tools for validating a single mapping and store results.
    Tools for cross-validating multiple mappings run on different hyperparameters sets.

"""
from operator import itemgetter

import numpy as np
from scipy.sparse.csc import csc_matrix
from scipy.sparse.csr import csr_matrix

import tangram as tgo
from mytangram import mapping_optimizer as mo
from mytangram import mapping_utils as mu
from mytangram.mapping_optimizer import Mapper


# Make class for model(datasets, hyperparameters set, dataset metadata to keep and identify model)
# define cross-validation method with input: n_folds, list of metrics to compute
# define stand-alone functions to compute each metric
# in the cv method iterate through metrics list
# store cv results as attributes of the model

class CVModel(Mapper):
    def __init__(
        self,
        adata_sc,
        adata_sp,
        hyperparameters,
        k,
        cluster_label=None,
        mode="cells",
        device="cpu",
        learning_rate=0.1,
        num_epochs=1000,
        scale=True
    ):
        super().__init__()
        self.adata_sc = adata_sc
        self.adata_sp = adata_sp
        self.k = k
        self.hyperparameters = {
            "lambda_d": self.lambda_d,  # KL (ie density) term
            "lambda_g1": self.model.lambda_g1,  # gene-voxel cos sim
            "lambda_g2": self.model.lambda_g2,  # voxel-gene cos sim
            "lambda_r": self.model.lambda_r,  # regularizer: penalize entropy
            "d_source": self.model.d_source,
        }
        if self.model.constraint:
            self.hyperparameters.pop("d_source")
            self.hyperparameters.update({
                "lambda_count": self.model.lambda_count,  # regularizer: enforce target number of cells
                "lambda_f_reg": self.model.lambda_f_reg,  # regularizer: push sigmoid values to 0,1
                "target_count": self.model.target_count,  # target number of cells
            })
        if self.model.random_state:
            np.random.seed(seed=self.model.random_state)

        self.scale = scale
        self.mode = mode
        self.device = device
        self.learning_rate = learning_rate
        self.num_epochs = num_epochs
        self.cluster_label = cluster_label



        # Section where datasets metadata is retrieved and stored in the object

    def get_genes_folds(
        self,
    ):

        """
        Returns sc and spatial anndata wtih added uns filed containing gene names fold to use as cv_training_genes arg in maP_CELLS_TO_SPAce
        Both need to contain them for mao_cells_to_space to work
        """

        if self.model.random_state:
            np.random.seed(seed=self.model.random_state)
        n_genes = len(self.adata_sc.uns["training_genes"])
        indices = np.random.permutation(n_genes)
        fold_size = n_genes // self.k
        self.adata_sc.uns["folds"] = []
        self.adata_sp.uns["folds"] = []
        for i in range(self.k):
            idx_fold = indices[i * fold_size: (i + 1) * fold_size]
            getter = itemgetter(*idx_fold)
            self.adata_sc.uns["folds"].append(getter(self.adata_sc.uns["training_genes"]))
            self.adata_sp.uns["folds"].append(getter(self.adata_sp.uns["training_genes"]))





    # Builds the 10 folds of training genes from the subset provided as input (scRNA DE or spatially DE genes,
    # total genes)

def cross_validate(
    k,
    metrics,
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
    verbose=False,
    density_prior='rna_count_based'
):
    """
    choose metrics you want to use. ['SSIM','PCC','RMSE','JS']
    """



    # Choose device
    device = torch.device(device)  # for gpu



    if mode == "cells":
        hyperparameters = {
            "lambda_d": lambda_d,  # KL (ie density) term
            "lambda_g1": lambda_g1,  # gene-voxel cos sim
            "lambda_g2": lambda_g2,  # voxel-gene cos sim
            "lambda_r": lambda_r,  # regularizer: penalize entropy
            "d_source": d_source,
        }


        mapper = CVModel(
            device=device, random_state=random_state, constraint=False, **hyperparameters,
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
        hyperparameters = {
            "lambda_d": lambda_d,  # KL (ie density) term
            "lambda_g1": lambda_g1,  # gene-voxel cos sim
            "lambda_g2": lambda_g2,  # voxel-gene cos sim
            "lambda_r": lambda_r,  # regularizer: penalize entropy
            "lambda_count": lambda_count,  # regularizer: enforce target number of cells
            "lambda_f_reg": lambda_f_reg,  # regularizer: push sigmoid values to 0,1
            "target_count": target_count,  # target number of cells
        }

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
    cv_model = CVModel()
    cv_model.get_genes_folds(self)
    assert self.adata_sc.uns["folds"] == self.adata_sp.uns["folds"]

    # Init fold metrics dictionary
    fold_metrics = {}
    for metric in metrics:
        fold_metrics[metric] = np.zeros(k)

    for fold in range(k):
        # train test split of current fold

        training_genes = self.adata_sc.uns["folds"][:fold] + self.adata_sc.uns["folds"][fold + 1:]
        test_genes = self.adata_sc.uns["folds"][fold]

        # run pp_adatas on train genes only, ensure tangram will use only the intersection subset of those
        # tg.pp_adatas(adata_sc, adata_st, genes = trn_genesns)
        ### IMPLEMENT BEFORE CREATING FOLDS
        ### INPUT Tensors: allocate them as arrays even if sparse matrices

        ## S matrix
        if isinstance(adata_sc.X, csc_matrix) or isinstance(adata_sc.X, csr_matrix):
            S = np.array(adata_sc[:, training_genes].X.toarray(), dtype="float32", )
        elif isinstance(adata_sc.X, np.ndarray):
            S = np.array(adata_sc[:, training_genes].X.toarray(), dtype="float32", )
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
        # Mapping
        adata_map = mu.map_cells_to_space(
            adata_sc=self.adata_sc,
            adata_sp=self.adata_sp,
            cv_train_genes=training_genes,
            mode=self.mode,
            device=self.model.device,
            learning_rate=self.learning_rate,
            num_epochs=self.num_epochs,
            cluster_label=self.cluster_label,
            scale=self.scale,
            **self.hyperparameters,
            random_state=self.model.random_state,
            verbose=False,
            density_prior="rna_count_based",
        )

        # Gene projection
        adata_ge = tgo.project_genes(
            adata_map, self.adata_sc[:, test_genes], cluster_label=self.cluster_label, scale=self.scale,
        )
        impute = np.array(adata_ge[:, test_genes].X.toarray(), dtype="float32",)
        raw = np.array(self.adata_sc[:, test_genes].X.toarray(), dtype="float32",)

        # Metrics evaluation on fold

        if "SSIM" in metrics:
            fold_metrics["SSIM"][k] = self.ssim(self, raw, impute, scale='scale_max')
        if "PCC" in metrics:
            fold_metrics["PCC"][k] = self.pearsonr(self, raw, impute, scale=None)
        if "RMSE" in metrics:
            fold_metrics["RMSE"][k] = self.RMSE(self, raw, impute, scale=None)
        if "JS" in metrics:
            fold_metrics["JS"][k] = self.JS(self, raw, impute, scale=None)

        # results storing

    # average metrics in folds
    cv_metrics = {}
    for metric in metrics:
        cv_metrics[metric] = fold_metrics[metric].mean()

    # Add cross-validated metrics dictionary to object as attribute
    self.cv_metrics = cv_metrics

    return cv_metrics


