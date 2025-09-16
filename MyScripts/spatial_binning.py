import matplotlib.pyplot as plt
import scanpy as sc
import squidpy as sq

adata_st = sc.read("C:/Users/enric/tangram/myDataCropped/slice200_norm_reduced.h5ad")
print(adata_st)

sq.gr.spatial_neighbors(adata_st, set_diag=False, key_added="spatial")
print(adata_st.obsp.keys())

import numpy as np
from collections import deque
from sklearn.metrics import pairwise_distances


def fps_seeds(coords, num_seeds, random_state=None):
    """
    Farthest-Point Sampling (FPS) to select well-spread seeds.
    coords: (n_cells,2) array of spatial positions
    num_seeds: number of seeds to select
    returns: list of seed indices
    """
    n = coords.shape[0]
    if random_state is not None:
        np.random.seed(random_state)
    seeds = [np.random.randint(n)]
    min_dists = pairwise_distances(coords.values, coords.iloc[seeds].values).flatten()

    for _ in range(1, num_seeds):
        idx = np.argmax(min_dists)
        seeds.append(idx)
        dist_to_new = np.linalg.norm(coords.values - coords.iloc[idx].values, axis=1)
        min_dists = np.minimum(min_dists, dist_to_new)

    return seeds



coords = adata_st.obsm['spatial']
adj_matrix = adata_st.obsp['spatial_connectivities']  # Squidpy 6-NN graph
n_cells = coords.shape[0]
target_size = 25
num_bins = int(np.ceil(n_cells / target_size))

# 1. Select well-spread seeds
seeds = fps_seeds(coords, num_seeds=num_bins, random_state=0)


def bfs_bin_growth(coords_df, seeds, adj_matrix, target_size=50):
    """
        Grow bins from seeds using BFS on a sparse adjacency matrix.
        coords: (n_cells,2)
        seeds: list of seed indices
        adj_matrix: sparse CSR (n_cells x n_cells), e.g., Squidpy 6-NN connectivities
        target_size: approximate number of cells per bin
        returns: labels array (n_cells,) bin assignment
    """

    n = coords_df.shape[0]
    labels = -np.ones(n, dtype=int)
    bin_id = 0
    adj = adj_matrix.tocsr()

    for s in seeds:
        if labels[s] != -1:
            continue
        q = deque([s])
        labels[s] = bin_id
        count = 1
        while q and count < target_size:
            v = q.popleft()
            neighbors = adj.indices[adj.indptr[v]:adj.indptr[v + 1]]
            for nb in neighbors:
                if labels[nb] == -1:
                    labels[nb] = bin_id
                    q.append(nb)
                    count += 1
                    if count >= target_size:
                        break
        bin_id += 1

    # Assign leftover cells
    leftover = np.where(labels == -1)[0]
    if leftover.size > 0:
        bin_ids = np.unique(labels[labels >= 0])
        centroids = np.vstack([coords_df.iloc[labels == b].values.mean(axis=0) for b in bin_ids])
        from sklearn.neighbors import NearestNeighbors
        tree = NearestNeighbors(n_neighbors=1).fit(centroids)
        _, nearest = tree.kneighbors(coords_df.iloc[leftover].values)
        labels[leftover] = bin_ids[nearest[:, 0]]

    return labels

# 2. Grow bins
labels = bfs_bin_growth(coords, seeds, adj_matrix, target_size=target_size)

# 3. Save to AnnData
adata_st.obs['bin'] = labels
adata_st.obs['bin_cell_count'] = adata_st.obs.groupby('bin')['bin'].transform('count')

# 4. Plot
plt.figure(figsize=(6,6))
scatter = plt.scatter(coords.values[:,0], coords.values[:,1], c=labels, cmap='tab20', s=20)  # retrieve df values
plt.gca().invert_yaxis()  # optional: match typical tissue orientation
plt.axis('equal')
plt.xlabel('X (µm)')
plt.ylabel('Y (µm)')
plt.title('Spatial bins')
plt.colorbar(scatter, label='Bin ID')
plt.show()


## Conditional prior
cluster_label = 'class_label'  # or 'sublclass'

import pandas as pd


def compute_bin_celltype_fractions(adata, cluster_label='cluster', bin_label='bin'):
    """
    Compute relative frequencies of cell types in each spatial bin.

    Parameters
    ----------
    adata : AnnData
        AnnData object with obs[cluster_label] and obs[bin_label].
    cluster_label : str
        Column in adata.obs indicating cell type or cluster.
    bin_label : str
        Column in adata.obs indicating bin assignment.

    Returns
    -------
    pd.DataFrame
        Rows: cell types, Columns: bins, values: fraction of cells of that type in the bin.
    """
    # Crosstab counts: cell types vs bins
    counts = pd.crosstab(adata.obs[cluster_label], adata.obs[bin_label])

    # Convert counts to fractions per bin
    fractions = counts.divide(counts.sum(axis=0), axis=1)

    return fractions

fractions_df = compute_bin_celltype_fractions(adata_st, cluster_label=cluster_label, bin_label='bin')
fractions_df.head()


# visualize
coords = adata_st.obsm['spatial'].values
bins = adata_st.obs['bin'].values
clusters = adata_st.obs[cluster_label].values

# Map clusters to a small set of markers
markers = ['o', 's', '^', 'v', 'D', 'P', '*', 'X']
cluster_to_marker = {c: markers[i % len(markers)] for i, c in enumerate(np.unique(clusters))}

plt.figure(figsize=(6,6))
for c in np.unique(clusters):
    idx = clusters == c
    plt.scatter(coords[idx,0], coords[idx,1], c=bins[idx], cmap='tab20', s=20,
                marker=cluster_to_marker[c], label=str(c))
plt.gca().invert_yaxis()
plt.axis('equal')
plt.legend(markerscale=1, fontsize=8, title='Cluster')
plt.show()