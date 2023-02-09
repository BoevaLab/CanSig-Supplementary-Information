import anndata
import matplotlib.pyplot as plt
import pandas as pd
import scanpy as sc
import pathlib as pl

# %%

counts = pd.read_table("/home/barkmann/BoevaLab/Code/CanSig-Supplementary-Information/spatial/data/GSM4284316_P2_ST_rep1_stdata.tsv",
                       index_col=0)

# %%

spatial_cords = pd.read_table("/home/barkmann/BoevaLab/Code/CanSig-Supplementary-Information/spatial/data/GSM4284316_spot_data-selection-P2_ST_rep1.tsv")
idx = (spatial_cords["x"].astype(str) + "x" + spatial_cords["y"].astype(str)).values
# %%


adata = anndata.AnnData(counts)
adata = adata[idx].copy()
adata.obsm["spatial"] = spatial_cords[["pixel_x", "pixel_y"]].values

# %%
image = mpimg.imread("/home/barkmann/BoevaLab/Code/CanSig-Supplementary-Information/spatial/data/smaller.jpg")

# %%


adata.obs["spot"] = "blue"

# %%

sc.pp.normalize_total(adata, target_sum=1e4)
sc.pp.log1p(adata)


# %%
import squidpy as sq

sq.pl.spatial_scatter(adata, color=[f"metasig{i}" for i in range(1, 6)], size=15, ncols=2)
plt.show()


# %%
import pathlib as pl

sig_basepath = pl.Path("/home/barkmann/BoevaLab/Code/CanSig-Supplementary-Information/spatial/sigs")
for sig_path in sig_basepath.iterdir():
    sig = pd.read_csv(sig_path, index_col=0)
    sig = sig.iloc[:, 0].values[:50]
    sc.tl.score_genes(adata, gene_list=sig, score_name=f"{sig_path.stem}")


# %%
