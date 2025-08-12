import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.patches import Patch


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

    title = 'Loss terms over epochs'
    if log_scale:
        title = title + ' (logscale)'

    for curve in loss_dict:
        if loss_dict[curve].any():  # truthy keys only
            if log_scale:
                plt.semilogy(abs(loss_dict[curve]), label=curve)
            else:
                plt.plot(loss_dict[curve], label=curve)
            plt.legend()
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title(title)
    plt.show()
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


def vis_gene_intersection(genes_list, condition_mask_sc=None, condition_mask_sp=None, figsize=(10, 10)):
    """
    DEPRECATED: Use traffic_light_plot() instead.

    Creates a traffic light plot to visualize the intersection of two gene lists and their conditions.
    Each gene that satisfies condition1 creates a red row (1,0,0).
    Each gene that satisfies condition2 creates a green column (0,1,0).
    Where red and green overlap, they add to create yellow (1,1,0).
    
    Args:
        genes_list (list): List of intersecting gene names from scRNA-seq and spatial data
        condition_mask_sc (numpy.ndarray, optional): Boolean mask indicating which genes from scRNA-seq data satisfy the condition
        condition_mask_sp (numpy.ndarray, optional): Boolean mask indicating which genes from spatial data satisfy the condition
        figsize (tuple): Optional. Figure size in inches (width, height)
    
    Returns:
        None (displays the plot)
    """
    # If condition masks are not provided, assume all genes satisfy the condition
    if condition_mask_sc is None:
        condition_mask_sc = np.ones(len(genes_list), dtype=bool)
    if condition_mask_sp is None:
        condition_mask_sp = np.ones(len(genes_list), dtype=bool)
        
    # Create the matrix
    matrix = np.zeros((len(genes_list), len(genes_list), 3))  # RGB matrix
    
    # Add red for rows (genes_list1)
    for i, mask_sc in enumerate(condition_mask_sc):
        if mask_sc:
            matrix[i, :, 0] = 1  # Set red channel to 1 for entire row
            
    # Add green for columns (genes_list2)
    for j, mask_sp in enumerate(condition_mask_sp):
        if mask_sp:
            matrix[:, j, 1] = 1  # Set green channel to 1 for entire column

    # The logic is the following:
    #     Convert masks to 2D boolean arrays
    #     row_mask = condition_mask_sc[:, np.newaxis]  # Shape: (n_genes_sc, 1)
    #     col_mask = condition_mask_sp[np.newaxis, :]  # Shape: (1, n_genes_sp)
    #
    #     # Create the RGB channels separately
    #     red_channel = row_mask * np.ones((len(genes_list_sc), len(genes_list_sp)))  # Red for rows
    #     green_channel = col_mask * np.ones((len(genes_list_sc), len(genes_list_sp)))  # Green for columns
    #     blue_channel = np.zeros((len(genes_list_sc), len(genes_list_sp)))  # Blue channel stays zero
    #
    #     # Stack the channels to create the RGB matrix
    #     matrix = np.stack([red_channel, green_channel, blue_channel], axis=2)

    # Create the plot
    plt.figure(figsize=figsize)
    plt.imshow(matrix)
    
    # Set ticks and labels
    plt.xticks(range(len(genes_list)), genes_list, rotation=90)
    plt.yticks(range(len(genes_list)), genes_list)
    plt.ylabel('Gene List single cell')
    plt.xlabel('Gene List spatial')
    
    # Add colorbar legend
    legend_elements = [
        Patch(facecolor='yellow', label='Overlapping conditions (Red + Green)'),
        Patch(facecolor='red', label='Gene from List 1 satisfies condition'),
        Patch(facecolor='green', label='Gene from List 2 satisfies condition')
    ]
    plt.legend(handles=legend_elements, bbox_to_anchor=(1.05, 1), loc='upper left')
    
    plt.xlabel('Gene List 2')
    plt.ylabel('Gene List 1')
    plt.title('Gene Intersection Traffic Light Plot')
    
    # Adjust layout to prevent label cutoff
    plt.tight_layout()
    plt.show()


def traffic_light_plot(genes_list, values_sc=None, values_sp=None, figsize=(10, 10)):
    """
    Creates a traffic light visualization where genes are represented as RGB elements
    arranged in a square/rectangular matrix. The first set of values controls the red channel,
    the second set controls the green channel. Values are automatically normalized to [0,1] range.
    
    Args:
        genes_list (list): List of gene names
        values_sc (numpy.ndarray, optional): Values for single cell data (controls red channel).
            Can be continuous or boolean. If None, defaults to ones.
        values_sp (numpy.ndarray, optional): Values for spatial data (controls green channel).
            Can be continuous or boolean. If None, defaults to ones.
        figsize (tuple): Figure size in inches (width, height)
    
    Returns:
        None (displays the plot)
    """
    n_genes = len(genes_list)
    
    # If values are not provided, raise error
    if values_sc is None:
        raise ValueError("single-cell values must be provided.")
    if values_sp is None:
        raise ValueError("spatial values must be provided.")
    if not n_genes == len(values_sc) == len(values_sp):
        raise ValueError("values must be of the same length as genes_list.")

    # Convert to numpy arrays if they aren't already
    values_sc = np.asarray(values_sc)
    values_sp = np.asarray(values_sp)
    
    # Normalize values to [0,1] if they aren't boolean
    if not values_sc.dtype == bool:
        if values_sc.max() != values_sc.min():
            values_sc = (values_sc - values_sc.min()) / (values_sc.max() - values_sc.min())
    if not values_sp.dtype == bool:
        if values_sp.max() != values_sp.min():
            values_sp = (values_sp - values_sp.min()) / (values_sp.max() - values_sp.min())
    
    # Convert boolean arrays to float
    values_sc = values_sc.astype(float)
    values_sp = values_sp.astype(float)
    
    # Create the RGB array (n_genes x 3)
    rgb_array = np.zeros((n_genes, 3))
    rgb_array[:, 0] = values_sc  # Red channel
    rgb_array[:, 1] = values_sp  # Green channel
    # Blue channel remains 0
    
    # Calculate dimensions for the square/rectangular matrix
    width = int(np.ceil(np.sqrt(n_genes)))
    height = int(np.ceil(n_genes / width))
    
    # Create the padded matrix
    total_cells = width * height
    padding_needed = total_cells - n_genes
    
    # Add padding (black cells) if needed
    if padding_needed > 0:
        padding = np.zeros((padding_needed, 3))
        rgb_array = np.vstack([rgb_array, padding])
    
    # Reshape into 2D matrix
    matrix = rgb_array.reshape(height, width, 3)
    
    # Create the plot
    plt.figure(figsize=figsize)
    plt.imshow(matrix)
    
    # Remove all axes, labels, and ticks
    plt.axis('off')
    
    # Add legend
    legend_elements = [
        Patch(facecolor='red', label='Single cell signal (Red)'),
        Patch(facecolor='green', label='Spatial signal (Green)'),
        Patch(facecolor='yellow', label='High in both (Red + Green)'),
        Patch(facecolor='black', label='Padding')
    ]
    plt.legend(handles=legend_elements, bbox_to_anchor=(1.05, 0.5), loc='center left')

    plt.title(f'Gene Traffic Light Matrix ({height}×{width})')
    
    plt.tight_layout()
    plt.show()