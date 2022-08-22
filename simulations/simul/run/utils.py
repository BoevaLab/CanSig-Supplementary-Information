from lib2to3.pgen2.token import OP
import pandas as pd
import anndata as ad
import scanpy as sc
import scvi
import scipy
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import itertools

from typing import Optional, List, Dict, Tuple
from matplotlib.colors import TwoSlopeNorm

from ..patients.dataset import Dataset

####### Plotting ###########
def plot_subclone_profile(dataset: Dataset, filename: Optional[str] = None) -> None:
    """Function to plot the true CNV profile as a heatmap

    Args:

        dataset: an instantiated dataset object
        filename: if not None, will save the figure in the provided path

    """
    subclone_df = dataset.get_subclone_profiles()
    subclone_plot_df = dataset.order_subclone_profile(subclone_df=subclone_df)
    fig, ax = plt.subplots(1, 1, figsize=(15, 10))
    sns.heatmap(subclone_plot_df, center=0, cmap="vlag", ax=ax)

    if filename is not None:
        fig.savefig(filename, bbox_inches="tight")


def plot_cnv_heatmap(
    dataset: Dataset,
    patient: str,
    adata: ad.AnnData,
    figsize: Tuple[int, int] = (15, 10),
    filename: Optional[str] = None,
) -> None:

    subclone_df = dataset.get_subclone_profiles()
    subclone_plot_df = dataset.order_subclone_profile(subclone_df=subclone_df)

    pat_subclones = subclone_plot_df.loc[subclone_plot_df.index.str.endswith(patient)]

    cnv_adata = ad.AnnData(
        pat_subclones.reset_index(drop=True), obs=pat_subclones.reset_index()[["index"]]
    )

    cnv_adata.obs.columns = ["subclone"]

    chromosomes = [f"chr{i}" for i in np.append(np.arange(1, 23), ["X", "Y", "M"])]
    chr_pos_dict = {}
    for chrom in chromosomes:
        chr_pos_dict[chrom] = np.where(
            adata.var.loc[subclone_plot_df.columns].chromosome == chrom
        )[0][0]

    chr_pos = list(chr_pos_dict.values())

    # center color map at 0
    tmp_data = cnv_adata.X.data if scipy.sparse.issparse(cnv_adata.X) else cnv_adata.X
    norm = TwoSlopeNorm(0, vmin=np.nanmin(tmp_data), vmax=np.nanmax(tmp_data))

    # add chromosome annotations
    var_group_positions = list(zip(chr_pos, chr_pos[1:] + [cnv_adata.shape[1]]))

    return_ax_dic = sc.pl.heatmap(
        cnv_adata,
        var_names=cnv_adata.var.index.values,
        groupby="subclone",
        figsize=figsize,
        cmap="vlag",
        show_gene_labels=False,
        var_group_positions=var_group_positions,
        var_group_labels=list(chr_pos_dict.keys()),
        norm=norm,
        show=False,
    )

    return_ax_dic["heatmap_ax"].vlines(chr_pos[1:], ymin=-1, ymax=cnv_adata.shape[0])

    if filename is not None:
        return_ax_dic["heatmap_ax"].figure.savefig(filename, bbox_inches="tight")


######### Prob distributions ############
def generate_anchor_alphas(
    anchors: List[str],
    start_alpha: List[int] = [5, 5, 5],
    alpha_add: int = 10,
) -> Dict[Tuple, List[int]]:
    """Function to generate the alphas for the dirichlet distribution associated with each anchor
    combination (2^n_anchors)

    Args:

        anchors: the list of anchors
        start_alpha: optionally, an initial alpha distribution to start from. Eg, if you want to create a "rare"
            program, you can start off with (10, 10, 5). Defaults to (5, 5, 5)
        alpha_add: optionally, the alpha to add when the anchor is gained. Defaults to 10.

    Returns:

        a dictionary with the anchor combination as key
        (eg (True, False, True) if anchor 1 and 3 are gained)
        and the associated alphas as value

    """
    l = [False, True]
    anchor_profiles = list(itertools.product(l, repeat=len(anchors)))
    alphas = {}
    for profile in anchor_profiles:
        alphas[profile] = [
            x + alpha_add if profile[i] else x for i, x in enumerate(start_alpha)
        ]
    return alphas


########### Distribution parameters ############
### WARNING: This section is specific to the dataset we use
# TODO(Josephine): make this usable whatever the dataset


def get_param_patient(
    adata: ad.AnnData, patient: str, model: scvi.model._scvi.SCVI
) -> Dict[str, Dict[str, np.ndarray]]:
    """Function to retrieve the parameters associated with a specific patient

    Args:

        adata: the original adata object
        patient: the name of the patient from which to use the cells
        model: the scVI model pretrained on the original adata object

    Returns:

        a dictionary with the name of the simulated patient as key and a
            dictionary as value containing the name of the program as key
            and the associated mean, dispersion, dropout and lib size as value
    """
    mapping_params = {
        "Macro": "Macro",
        "Plasma": "Plasma",
        "TCD4": "program1",
        "TCD8": "program2",
        "Tgd": "program2",
        "TZBTB16": "program3",
    }

    params = {}
    for ct in ["Macro", "Plasma", "TCD4", "TCD8", "Tgd", "TZBTB16"]:
        ind = np.where((adata.obs.sample_id == patient) & (adata.obs.celltype == ct))[0]
        params[mapping_params[ct]] = model.get_likelihood_parameters(
            n_samples=1, indices=ind
        )
        params[mapping_params[ct]]["libsize"] = adata.obs.iloc[ind][["n_counts"]].values
    return params


def add_chemistry_obs(
    all_malignant_obs: Dict[str, pd.DataFrame],
    sample_patients: Dict[str, str],
    adata: ad.AnnData,
) -> Dict[str, pd.DataFrame]:

    for patient in sample_patients:
        chem = list(
            adata.obs.loc[adata.obs.sample_id == sample_patients[patient]]
            .SINGLECELL_TYPE.unique()
            .astype(str)
        )
        all_malignant_obs[patient]["chemistry"] = (
            chem * all_malignant_obs[patient].shape[0]
        )
    return all_malignant_obs
