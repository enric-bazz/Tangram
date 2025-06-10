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
    #plt.close()

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
    #plt.close()


def plot_filter_weights_light(adata_map, plot_spaghetti=False, plot_envelope=False):
    """
    Plots the filter weights evolution over epochs with optional additional visualizations.

    Args:
        adata_map (anndata object): input containing .uns["filter_history"] returned by map_cells_to_space()
        plot_spaghetti (bool): If True, plots individual cell trajectories over epochs
        plot_envelope (bool): If True, plots the mean signal with ±1 std deviation envelope
    """
    matrix = np.column_stack(adata_map.uns['filter_history']['filter_values'])
    
    # Calculate appropriate figure size and aspect ratio
    n_cells, n_epochs = matrix.shape
    aspect_ratio = n_epochs / n_cells  # This gives us the data aspect ratio
    
    # Set base width and adjust height accordingly
    base_width = 12
    fig_height = base_width / aspect_ratio
    
    # Limit maximum height to keep plot reasonable
    fig_height = min(fig_height, 16)
    
    # Main heatmap plot
    plt.figure(figsize=(base_width, fig_height))
    im = plt.imshow(matrix, aspect='auto')  # 'auto' ensures the plot fills the figure
    plt.colorbar(im, fraction=0.03, pad=0.05)
    plt.xlabel('Epoch')
    plt.ylabel('Cell')
    plt.title('Sigmoid weights over epochs')
    plt.show()

    # Additional plots if requested
    if plot_spaghetti or plot_envelope:
        epochs = np.arange(n_epochs)
        
        if plot_spaghetti and plot_envelope:
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(base_width, 10))
            
            # Spaghetti plot
            for cell_idx in range(matrix.shape[0]):
                ax1.plot(epochs, matrix[cell_idx, :], alpha=0.1, color='blue')
            ax1.set_xlabel('Epoch')
            ax1.set_ylabel('Filter Weight')
            ax1.set_title('Individual Cell Filter Weight Trajectories')
            ax1.set_ylim(0, 1)
            
            # Envelope plot
            mean_signal = np.mean(matrix, axis=0)
            std_signal = np.std(matrix, axis=0)
            
            ax2.plot(epochs, mean_signal, 'b-', label='Mean')
            ax2.fill_between(epochs, 
                           mean_signal - std_signal,
                           mean_signal + std_signal,
                           alpha=0.3, color='blue', label='±1 std dev')
            ax2.set_xlabel('Epoch')
            ax2.set_ylabel('Filter Weight')
            ax2.set_title('Mean Filter Weight with Standard Deviation Envelope')
            ax2.set_ylim(0, 1)
            ax2.legend()
            
        elif plot_spaghetti:
            plt.figure(figsize=(base_width, 5))
            for cell_idx in range(matrix.shape[0]):
                plt.plot(epochs, matrix[cell_idx, :], alpha=0.1, color='blue')
            plt.xlabel('Epoch')
            plt.ylabel('Filter Weight')
            plt.title('Individual Cell Filter Weight Trajectories')
            plt.ylim(0, 1)
            
        elif plot_envelope:
            plt.figure(figsize=(base_width, 5))
            mean_signal = np.mean(matrix, axis=0)
            std_signal = np.std(matrix, axis=0)
            
            plt.plot(epochs, mean_signal, 'b-', label='Mean')
            plt.fill_between(epochs,
                           mean_signal - std_signal,
                           mean_signal + std_signal,
                           alpha=0.3, color='blue', label='±1 std dev')
            plt.xlabel('Epoch')
            plt.ylabel('Filter Weight')
            plt.title('Mean Filter Weight with Standard Deviation Envelope')
            plt.ylim(0, 1)
            plt.legend()
        
        plt.show()


def plot_filter_count(adata_map, target_count=None, figsize=(10, 5)):
    """
    Plot the number of cells passing the filter threshold over epochs.

    Args:
        adata_map: anndata object returned my the mapping containing the filter history
        and target count equalt to the one used for the mapping (if missing it is internally computed
        as in the optimizer class)

        This is a useful diagnostic plot as it shows how far the final number of cells is from the target.
        It should be related to the corresponding term in the loss function.
    """
    n_spots = adata_map.X.squeeze().shape[0]

    # Set target count if not provided
    if target_count is None:
        target_count = n_spots

    fig, ax = plt.subplots(figsize=figsize)
    n_cells = adata_map.uns['filter_history']['n_cells']
    epochs = range(1, len(n_cells) + 1)

    # Plot number of cells
    ax.plot(epochs, n_cells, '-o', label='Filtered cells')

    # Add horizontal line for target count
    ax.axhline(y=target_count, color='r', linestyle='--', label='Target count')

    ax.set_xlabel('Epoch')
    ax.set_ylabel('Number of cells')
    ax.set_title('Number of cells passing filter threshold per epoch')
    ax.grid(True)
    ax.legend()
    plt.show()

    #return fig