import anndata as ad

## Load datasets already preprocessed and on the intersection genes only (training sets)
# scRNA-seq data
adata_sc = ad.read("C:/Users/enric/tangram/data/test_ad_sc.h5ad")
print('original scRNA dataset size=', adata_sc.X)
# crop to 400 observations
adata_sc_sub = adata_sc[:400].copy()
# remove raw coints if present
#del adata_sc_sub.raw
# save file
path = "C:/Users/enric/tangram/myDataCropped"
adata_sc_sub.write(path + "/test_sc_crop.h5ad")

# spatial dataset
adata_st = ad.read("C:/Users/enric/tangram/data/test_ad_sp.h5ad")
print('original spatial dataset size=', adata_st.X)
# crop to 400 observations
adata_st_sub = adata_st[:400].copy()
# remove raw coints if present
#del adata_st_sub.raw
# save file
path = "C:/Users/enric/tangram/myDataCropped"
adata_st_sub.write(path + "/test_sp_crop.h5ad")


## load in console with
#adata_sc = ad.read(os.getcwd() + "/myDataCropped/test_sc_crop.h5ad")
#adata_st = ad.read(os.getcwd() + "/myDataCropped/test_sp_crop.h5ad")
#adata_sc.file.close()
#adata_st.file.close()

from scipy.sparse import vstack
adata_tot = vstack([adata_sc.X, adata_st.X]) #csr matrix



