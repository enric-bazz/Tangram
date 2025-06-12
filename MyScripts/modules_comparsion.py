import anndata as ad

import mytangram as tg
from mytangram.comparison_utils import *

## Load cropped datasets
path = "C:/Users/enric/tangram/myDataCropped"
adata_sc = ad.read_h5ad(path + "/test_sc_crop.h5ad")
adata_st = ad.read_h5ad(path + "/test_sp_crop.h5ad")

# get genes intersection
tg.pp_adatas(adata_sc, adata_st)

mode = "constrained"
target_count = np.round(len(adata_st.var.index)/3)

# set seed for comparison
random_state = 123

ad_map = tg.map_cells_to_space(adata_sc, adata_st, mode=mode, target_count=target_count,
                               density_prior='rna_count_based', num_epochs=30,device='cpu',
                               lambda_d=1,
                               lambda_g1=1,
                               lambda_g2=1,
                               lambda_r=0.0003,
                               lambda_count=0.005,
                               lambda_f_reg=0.01,
                               random_state=random_state
                               )


ad_map_lt = tg.map_cells_to_space_lightning(adata_sc, adata_st, mode=mode, target_count=target_count,
                                density_prior='rna_count_based', num_epochs=30,
                                lambda_d=1,
                                lambda_g1=1,
                                lambda_g2=1,
                                lambda_r=0.0003,
                                lambda_count=0.005,
                                lambda_f_reg=0.01,
                                random_state=random_state,
                                )

# Get the mapping matrices from both results
mapping_matrix = ad_map.X
mapping_matrix_lt = ad_map_lt.X

# Check if the matrices are equal within a small tolerance
are_equal = np.allclose(mapping_matrix, mapping_matrix_lt, rtol=1e-5, atol=1e-8)
print(f"Mapping matrices are equal: {are_equal}")

# Compare loss histories
original_history = ad_map.uns['training_history']
lightning_history = ad_map_lt.uns['training_history']

# Compare all loss terms
for key in ad_map.uns['training_history']:
    if key in original_history and key in lightning_history:
        compare_loss_trajectories(original_history, lightning_history, key=key)

# Run the loss analyses
final_comparison = analyze_mapping_evolution(ad_map.X, ad_map_lt.X)
sparsity_analysis = compare_sparsity(ad_map.X, ad_map_lt.X)
plot_mapping_distributions(ad_map.X, ad_map_lt.X)

# Print results
print("\nFinal Mapping Comparison:")
for k, v in final_comparison.items():
    print(f"{k}: {v}")

print("\nSparsity Analysis:")
print(f"Agreement in sparsity patterns: {sparsity_analysis['sparsity_agreement']:.2%}")

# Run filters comparison
comparison_results = compare_cell_choices(ad_map, ad_map_lt)
print("\nCell Choice Comparison Results:")
for metric, value in comparison_results.items():
    print(f"{metric}: {value:.6f}")
