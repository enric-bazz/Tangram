import matplotlib.pyplot as plt
import numpy as np

"""
Utilities for models comparison.
"""

# Plot losses side by side for comparison
def compare_loss_trajectories(hist1, hist2, key='total_loss'):
    plt.figure(figsize=(12, 6))
    plt.plot(hist1[key], label='Original', alpha=0.7)
    plt.plot(hist2[key], label='Lightning', alpha=0.7)
    plt.title(f'Comparison of {key}')
    plt.xlabel('Epoch')
    plt.ylabel(key)
    plt.legend()
    plt.grid(True)
    plt.show()


# Compare intermediate states
def analyze_mapping_evolution(matrix1, matrix2):
    # Compute correlation
    correlation = np.corrcoef(matrix1.flatten(), matrix2.flatten())[0, 1]

    # Compute cosine similarity
    cos_sim = np.dot(matrix1.flatten(), matrix2.flatten()) / \
              (np.linalg.norm(matrix1.flatten()) * np.linalg.norm(matrix2.flatten()))

    # Find largest differences
    diff = np.abs(matrix1 - matrix2)
    max_diff_pos = np.unravel_index(np.argmax(diff), diff.shape)

    return {
        'correlation': correlation,
        'cosine_similarity': cos_sim,
        'max_diff': np.max(diff),
        'max_diff_position': max_diff_pos,
        'max_diff_values': (matrix1[max_diff_pos], matrix2[max_diff_pos])
    }


# Analyze distribution of values
def plot_mapping_distributions(matrix1, matrix2, title="Distribution of Mapping Values"):
    plt.figure(figsize=(12, 6))

    plt.subplot(1, 2, 1)
    plt.hist(matrix1.flatten(), bins=50, alpha=0.7, label='Original')
    plt.hist(matrix2.flatten(), bins=50, alpha=0.7, label='Lightning')
    plt.title(title)
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.scatter(matrix1.flatten(), matrix2.flatten(), alpha=0.1)
    plt.plot([0, 1], [0, 1], 'r--')  # diagonal line for reference
    plt.xlabel('Original Values')
    plt.ylabel('Lightning Values')
    plt.title('Value Correlation')

    plt.tight_layout()
    plt.show()


# Analyze sparsity patterns
def compare_sparsity(matrix1, matrix2, threshold=1e-5):
    sparse1 = (matrix1 > threshold).astype(float)
    sparse2 = (matrix2 > threshold).astype(float)

    agreement = np.sum(sparse1 == sparse2) / sparse1.size
    diff_positions = np.where(sparse1 != sparse2)

    return {
        'sparsity_agreement': agreement,
        'different_positions': list(zip(diff_positions[0], diff_positions[1]))[:5]  # first 5 differences
    }


def compare_cell_choices(ad_map, ad_map_lt):
    """
    Compare how similarly the two models choose cells by analyzing F_out values.

    Parameters:
    -----------
    ad_map : AnnData
        Result from the original mapping
    ad_map_lt : AnnData
        Result from the lightning mapping

    Returns:
    --------
    dict
        Dictionary containing comparison metrics
    """
    # Get F_out from both models
    f_probs_original = ad_map.obs['F_out'].to_numpy()
    f_probs_lightning = ad_map_lt.obs['F_out'].to_numpy()

    if f_probs_original is None or f_probs_lightning is None:
        return {"error": "F_out not found in one or both models"}

    # Calculate various similarity metrics
    correlation = np.corrcoef(f_probs_original, f_probs_lightning)[0, 1]

    # Calculate cosine similarity
    cos_sim = np.dot(f_probs_original, f_probs_lightning) / \
              (np.linalg.norm(f_probs_original) * np.linalg.norm(f_probs_lightning))

    # Calculate absolute differences
    diff = np.abs(f_probs_original - f_probs_lightning)
    max_diff = np.max(diff)
    mean_diff = np.mean(diff)


    results = {
        'correlation': correlation,
        'cosine_similarity': cos_sim,
        'max_difference': max_diff,
        'mean_difference': mean_diff,
    }

    # Visualize the comparison
    plt.figure(figsize=(15, 5))

    # Plot 1: Scatter plot of F_out values
    plt.subplot(131)
    plt.scatter(f_probs_original, f_probs_lightning, alpha=0.1)
    plt.plot([0, 1], [0, 1], 'r--')  # diagonal line for reference
    plt.xlabel('Original F_out')
    plt.ylabel('Lightning F_out')
    plt.title('F_out Correlation')

    # Plot 2: Distribution comparison
    plt.subplot(132)
    plt.hist(f_probs_original, bins=50, alpha=0.5, label='Original')
    plt.hist(f_probs_lightning, bins=50, alpha=0.5, label='Lightning')
    plt.xlabel('F_out values')
    plt.ylabel('Frequency')
    plt.title('Distribution of F_out')
    plt.legend()

    # Plot 3: Difference plot
    plt.subplot(133)
    plt.plot(diff, alpha=0.7)
    plt.xlabel('Cell index')
    plt.ylabel('Absolute difference')
    plt.title('F_out Differences')

    plt.tight_layout()
    plt.show()

    return results