import pandas as pd
import anndata as ad
import scvi
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import itertools

from typing import Optional, List, Dict, Tuple

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


######### Prob distributions ############
def generate_anchor_alphas(anchors: List[str]) -> Dict[Tuple, List[int]]:
    """Function to generate the alphas for the dirichlet distribution associated with each anchor
    combination (2^n_anchors)

    Args:

        anchors: the list of anchors

    Returns:

        a dictionary with the anchor combination as key
        (eg (True, False, True) if anchor 1 and 3 are gained)
        and the associated alphas as value

    Note:

        right now we hardcode 20 as alpha is the anchor is gained and 1 if not
    """
    l = [False, True]
    anchor_profiles = list(itertools.product(l, repeat=len(anchors)))
    alphas = {}
    for profile in anchor_profiles:
        alphas[profile] = [20 if profile[i] else 1 for i in range(len(profile))]
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
        "Macro": "healthy",
        "TCD4": "program1",
        "TCD8": "program2",
        "Tgd": "program3",
    }

    params = {}
    for ct in ["Macro", "TCD4", "TCD8", "Tgd"]:
        ind = np.where((adata.obs.sample_id == patient) & (adata.obs.celltype == ct))[0]
        params[mapping_params[ct]] = model.get_likelihood_parameters(
            n_samples=1, indices=ind
        )
        params[mapping_params[ct]]["libsize"] = adata.obs.iloc[ind][["n_counts"]].values
    return params
