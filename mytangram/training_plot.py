import matplotlib.pyplot as plt
import numpy as np
import torch


def plot_loss_terms(adata_map, log_scale=True):
    """
        Plots a panel for each loss term curve in the training step

        Args:
            adata_map (anndata object): input containing .uns["training_history"] returned by map_cells_to_space()
            log_scale (bool): Whether the y axis plots should be in log-scale  

        Returns:

        """
    # Check if training history is present
    if not "training_history" in adata_map.uns.keys():
        raise ValueError("Missing training history in mapped input data.")

    # Retrieve loss terms labels
    loss_terms_labels = adata_map.uns['training_history'].keys()

    # Initiate empty dict containing numpy arrays
    loss_dict = {key: None for key in loss_terms_labels}

    # Some terms are returned as a list of torch tensors (scalars) others as lists of float: turn all into ndarray
    for k in loss_terms_labels:
        if type(adata_map.uns["training_history"][k][0]) == torch.Tensor and not torch.isnan(adata_map.uns["training_history"][k][0]):
            loss_term_values = []
            for entry in adata_map.uns["training_history"][k]:
                loss_term_values.append(entry.detach())
            loss_term_values = np.asarray(loss_term_values)
        elif type(adata_map.uns["training_history"][k][0]) == float and not np.isnan(adata_map.uns["training_history"][k][0]):
            loss_term_values = np.asarray(adata_map.uns["training_history"][k])
            # does not implement .copy()
        loss_dict[k] = loss_term_values

    # Retrieve number of epochs
    n_epochs = len(adata_map.uns['training_history']['total_loss'])

    # Add filter to remove nan vectors
    # Create plot
    plt.figure(figsize=(10,20))


    for curve in loss_dict:
        if loss_dict[curve].any():  # truthy keys only
            if log_scale:
                plt.semilogy(loss_dict[curve], label=curve)
            else:
                plt.plot(loss_dict[curve], label=curve)
            plt.legend()
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.show()

def plot_filter_weights(adata_map):
    """
        Plots a heatmap of filter weights values through training epochs.
        This should help evaluate how the filtering works and settle to the final values.

        Args:
            adata_map (anndata object): input containing .uns["filter_history"] returned by map_cells_to_space()

        Returns:
        """

    # Check if filter history is present
    if not "filter_history" in adata_map.uns.keys():
        raise ValueError("Missing filter history in mapped input data (constrained mode only).")

    # Set colormap
    #c_map = mpl.cm.get_cmap()

    # Build matrix
    matrix = np.column_stack(adata_map.uns['filter_history'])

    plt.figure(figsize=(12, 12))
    plt.imshow(matrix)
    plt.colorbar(fraction=0.03, pad=0.05)
    plt.xlabel('Epoch')
    plt.ylabel('Cell')
    plt.title('Sigmoid weights over epochs')
    plt.show()
    plt.gca().set_aspect(0.1)
