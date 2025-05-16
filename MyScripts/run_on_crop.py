## Load cropped datasets
# either running the script
#%run "Myscripts/load_cropped_data"
# or loading the datasests
import anndata as ad
path = "C:/Users/enric/tangram/myDataCropped"
adata_sc = ad.read_h5ad(path + "/test_sc_crop.h5ad")
adata_st = ad.read_h5ad(path + "/test_sp_crop.h5ad")

# remove zero count genes
import numpy as np

# Overwrite adata without explicitly using .copy()
adata_sc = adata_sc[:, np.array(adata_sc.X.sum(axis=0)).flatten() > 0]
adata_st = adata_st[:, np.array(adata_st.X.sum(axis=0)).flatten() > 0]

# get genes intersection
import squidpy as sq
import mytangram as tg
tg.pp_adatas(adata_sc, adata_st)

# run mapping
ad_map = tg.map_cells_to_space(adata_sc, adata_st, mode="cells", density_prior='rna_count_based', num_epochs=500,device='cpu')


# plot scores
#scores_fig = tg.plot_training_scores(ad_map, bins=20, alpha=.5)
#print('figure saved correctly as {}'.format(type(scores_fig)))
#print(scores_fig)
#tg.plot_training_scores(ad_map, bins=20, alpha=.5)
#plt.show()
#plt.close()

import pandas as pd
tg.project_cell_annotations(ad_map, adata_st, annotation="class_label")
annotation_list = list(pd.unique(adata_sc.obs['class_label']))
#tg.plot_cell_annotation_sc(adata_st, annotation_list,perc=0.02,spot_size=1,scale_factor=1)

# project new genes
ad_ge = tg.project_genes(adata_map=ad_map, adata_sc=adata_sc)
df_all_genes = tg.compare_spatial_geneexp(ad_ge, adata_st, adata_sc)

## FILTERED MAPPPING
#load image and segment
img = sq.datasets.visium_fluo_image_crop()
sq.im.process(img=img, layer="image", method="smooth")
sq.im.segment(
    img=img,
    layer="image_smooth",
    method="watershed",
    channel=0,
)
features_kwargs = {
    "segmentation": {
        "label_layer": "segmented_watershed",
        "props": ["label", "centroid"],
        "channels": [1, 2],
    }
}
 #load full spatial to retrieve coordinates
adata_st_full = sq.datasets.visium_fluo_adata_crop()
adata_st.obsm["spatial"] = adata_st_full.obsm["spatial"][0:400]
adata_st.uns["spatial"] = adata_st_full.uns["spatial"]
del adata_st_full

#compute features
if __name__ == "__main__":
    sq.im.calculate_image_features(
        adata_st,
        img,
        layer="image",
        key_added="image_features",
        features_kwargs=features_kwargs,
        features="segmentation",
        mask_circle=True,
    )

adata_st.obs["cell_count"] = adata_st.obsm["image_features"]["segmentation_label"]

# map with filter
ad_map_filt = tg.map_cells_to_space(
    adata_sc,
    adata_st,
    mode="constrained",
    target_count=adata_st.obs.cell_count.sum(),
    density_prior=np.array(adata_st.obs.cell_count) / adata_st.obs.cell_count.sum(),
    num_epochs=100,
    device='cpu'
)