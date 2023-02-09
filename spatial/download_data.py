import gzip
import json
import pathlib as pl

import GEOparse
import anndata
import numpy as np
import pandas as pd
import scanpy as sc
import squidpy as sq
import wget
from PIL import Image
import scipy.io as io

def load_img(path, downsize_factor=0.1):
    Image.MAX_IMAGE_PIXELS = 933120000
    image = Image.open(gzip.open(path))
    size = (downsize_factor * np.array(image.size)).astype(int)
    image = image.resize(size)
    image = np.asarray(image)
    return image


# %%
gse = GEOparse.get_GEO(geo="GSE144239", destdir="./")
base_dir = pl.Path(
    "/home/barkmann/BoevaLab/Code/CanSig-Supplementary-Information/spatial/data")
for sample, info in gse.gsms.items():
    print(sample)
    data_dir = base_dir.joinpath(sample)
    data_dir.mkdir(exist_ok=True, parents=True)
    for key, val in info.metadata.items():
        if key.startswith("supplementary_file"):
            wget.download(val[0], out=str(data_dir))

# %%
DOWNSIZE_FACTOR = 0.25

sig_basepath = pl.Path(
    "/home/barkmann/BoevaLab/Code/CanSig-Supplementary-Information/spatial/sigs")
results_dir = pl.Path(
    "/home/barkmann/BoevaLab/Code/CanSig-Supplementary-Information/spatial/adata")
results_dir.mkdir(exist_ok=True, parents=True)

# %%
for sample_path in pl.Path(
        "/home/barkmann/BoevaLab/Code/CanSig-Supplementary-Information/spatial/data/").iterdir():

    print(sample_path.stem)

    if len(list(sample_path.iterdir())) == 3:

        continue
        size = 15
        image_path = list(sample_path.glob("*.jpg.gz"))[0]
        image = load_img(image_path, DOWNSIZE_FACTOR)
        sample_name = image_path.stem.rsplit(".", 1)[0]
        counts_path = list(sample_path.glob("*stdata.tsv.gz"))[0]
        counts = pd.read_table(counts_path, index_col=0)
        spatial_path = list(sample_path.glob("*spot_data-selection*"))[0]
        spatial_cords = pd.read_table(spatial_path)
        idx = (spatial_cords["x"].astype(str) + "x" + spatial_cords["y"].astype(str)).values
        spatial_cords.index = idx

        adata = anndata.AnnData(counts)
        adata = adata[adata.obs_names.intersection(idx)].copy()
        spatial_cords = spatial_cords.loc[adata.obs_names.intersection(idx), :]
        adata.obsm["spatial"] = spatial_cords[["pixel_x", "pixel_y"]].values

        sc.pp.normalize_total(adata, target_sum=1e4)
        sc.pp.log1p(adata)

        spatial_key = "spatial"
        library_id = "tissue42"
        adata.uns[spatial_key] = {library_id: {}}
        adata.uns[spatial_key][library_id]["images"] = {}
        adata.uns[spatial_key][library_id]["images"] = {"hires": image}
        adata.uns[spatial_key][library_id]["scalefactors"] = {
            "tissue_hires_scalef": DOWNSIZE_FACTOR,
            "spot_diameter_fullres": 10}

    elif len(list(sample_path.iterdir())) == 6:
        size = None
        image_path = list(sample_path.glob("*.png.gz"))[0]
        image = load_img(image_path, 1.)
        sample_name = image_path.stem.rsplit(".", 1)[0]
        counts_path = list(sample_path.glob("*_matrix.mtx.gz"))[0]
        counts = io.mmread(counts_path)
        barcodes_path = list(sample_path.glob("*_barcodes.tsv.gz"))[0]
        barcodes = pd.read_table(barcodes_path, index_col=0, header=None)
        features_path = list(sample_path.glob("*_features.tsv.gz"))[0]
        features = pd.read_table(features_path, index_col=0, header=None)
        spatial_path = list(sample_path.glob("*_tissue_positions_list.csv.gz"))[0]
        spatial_cords = pd.read_csv(spatial_path, header=None, index_col=0)
        config_path = list(sample_path.glob("*json.gz"))[0]
        config = json.load(gzip.open(config_path))

        adata = anndata.AnnData(counts.transpose().tocsc(), obs=barcodes, var=features)
        adata.var_names = adata.var.iloc[:, 0]

        idx = spatial_cords.index.intersection(adata.obs_names)

        adata = adata[idx].copy()
        adata.var_names_make_unique()
        spatial_cords = spatial_cords.loc[idx].copy()

        adata.obsm["spatial"] = np.flip(spatial_cords.iloc[:, 3:5].values, axis=1)

        sc.pp.normalize_total(adata, target_sum=1e4)
        sc.pp.log1p(adata)

        spatial_key = "spatial"
        library_id = "tissue42"
        adata.uns[spatial_key] = {library_id: {}}
        adata.uns[spatial_key][library_id]["images"] = {}
        adata.uns[spatial_key][library_id]["images"] = {"hires": image}
        adata.uns[spatial_key][library_id]["scalefactors"] = config

    else:
        raise ValueError("Unknown sample type")



    for sig_path in sig_basepath.iterdir():
        sig = pd.read_csv(sig_path, index_col=0)
        sig = sig.iloc[:, 0].values[:50]
        sc.tl.score_genes(adata, gene_list=sig, score_name=f"{sig_path.stem}")

    for i in range(1, 6):
        sq.pl.spatial_scatter(adata, color=f"metasig{i}", size=size,
                              save=sample_name+f"metasig{i}.png")
    #adata.write_h5ad(results_dir.joinpath(sample_name + ".h5ad"))
