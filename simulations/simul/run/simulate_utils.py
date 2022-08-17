import pandas as pd
import numpy as np
import anndata as ad
import scvi

from typing import Dict, List

from ..patients.dataset import Dataset
from ..cna.api import ProgramDistribution
import simul.cna.gene_expression as gex

from .utils import get_param_patient

####### Prob dist ##################


def get_zinb_gex(
    mean_array: np.ndarray,
    dispersion_array: np.ndarray,
    log_dropout_array: np.ndarray,
    libsize_array: np.ndarray,
) -> np.ndarray:
    """Returns a sampled counts matrix using a ZINB and multiplying by observed library size

    Args:

        mean_array: array of shape (n_samples, n_genes) with the mu
        dispersion array: array of shape (n_samples, n_genes) with the associated theta
        log_dropout_array: array of shape (n_sample, n_genes) with the pi in logit scale
        libsize_array: array of shape (n_sample, ) with the observed library size of the cell

    Returns:

        x the sampled count matrix
    """

    # first draw from a gamma distribution
    w = np.random.gamma(mean_array, dispersion_array)
    # then sample from a poisson using the gamma to obtain a negative binomial
    x = np.random.poisson(w)
    # to turn into a ZINB, get a mask using the dropout probability
    ps = np.exp(log_dropout_array) / (1 + np.exp(log_dropout_array))
    mask = 1 - np.random.binomial(np.ones((ps.shape)).astype(int), ps)
    # finally get the sampled expression
    x = mask * x
    # now we transform the sampled counts so they fit the original size
    # TODO(Josephine): think about a way of multiplying the mu by the libsize and sampling from
    # that distribution as it is more correct (issue being making sure the dispersion and
    # dropout correspond to the new mu)
    x = np.round((x.T / x.sum(axis=1)).T * libsize_array)
    return x


########## Simulation ##################
def simulate_malignant_comp_batches(
    dataset: Dataset, prob_dist: ProgramDistribution
) -> Dict[str, pd.DataFrame]:
    """Simulates the composition of mallignant cells for a dataset

    Args:

        dataset: an instantiated Dataset object
        prob_dist: an instantiated ProgramDistribution object with the conditional probabilities
            of P (program | anchors, patient)

    Returns:

        a dictionary with patient name as key and an observation dataframe as value

    """
    all_malignant_obs = {}
    for patient in dataset.patients:

        patient_subclones = [
            patient.subclones[i].name for i in range(len(patient.subclones))
        ]
        patient_subclone_profiles = {
            patient.subclones[i].name: tuple(patient.subclones[i].anchor_profile)
            for i in range(len(patient.subclones))
        }

        # pick the subclones the cells belong to
        batch_clones = np.random.choice(
            patient_subclones,
            size=(patient.n_malignant_cells,),
            p=patient.subclone_proportions,
        )
        cell_programs = []
        for c in batch_clones:
            cell_programs.append(
                prob_dist.sample(
                    anchors=patient_subclone_profiles[c],
                    batch=patient.batch,
                    n_samples=1,
                )[0]
            )
        malignant = ["malignant"] * patient.n_malignant_cells
        df_obs = pd.DataFrame(
            np.array([batch_clones, cell_programs, malignant]),
            index=["subclone", "program", "malignant_key"],
            columns=[f"cell{i+1}" for i in range(batch_clones.shape[0])],
        ).T
        all_malignant_obs[patient.batch] = df_obs
    return all_malignant_obs


def simulate_healthy_comp_batches(dataset: Dataset) -> Dict[str, pd.DataFrame]:
    """Simulates the composition of healthy cells for a dataset

    Args:

        dataset: an instantiated Dataset object

    Returns:

        a dictionary with patient name as key and an observation dataframe as value

    """
    all_healthy_obs = {}
    for patient in dataset.patients:
        clones = ["NA"] * patient.n_healthy_cells
        cell_programs = ["NA"] * patient.n_healthy_cells
        malignant = ["non_malignant"] * patient.n_healthy_cells

        df_obs = pd.DataFrame(
            np.array([clones, cell_programs, malignant]),
            index=["subclone", "program", "malignant_key"],
            columns=[
                f"cell{i+1+patient.n_malignant_cells}" for i in range(len(clones))
            ],
        ).T
        all_healthy_obs[patient.batch] = df_obs

    return all_healthy_obs


def sample_patient_original(
    dataset: Dataset, selected_patients: List[str]
) -> Dict[str, str]:
    """Function to sample the patients in the original set that we will generate the parameters from
    for the simulated set

    Args:

        dataset: an instantiated Dataset object
        selected_patients: a list of patients in the original adata that can be used for generation

    Returns:

        a dictionary with the name of the simulated patient as key and the name of the original
            patient to sample from as value
    """
    list_patients = [dataset.patients[i].batch for i in range(len(dataset.patients))]
    if len(list_patients) > len(selected_patients):
        raise ValueError(
            f"The number of patients to generate is too big. Max number of patients is {len(selected_patients)}"
        )
    sample_patients = np.random.choice(
        selected_patients,
        size=(len(list_patients),),
        replace=False,
    )
    return {list_patients[i]: sample_patients[i] for i in range(len(list_patients))}


def sample_patient_original_replacement(
    dataset: Dataset, selected_patients: List[str]
) -> Dict[str, str]:
    """Function to sample with replacement the patients in the original set that we will generate the parameters from
    for the simulated set

    Args:

        dataset: an instantiated Dataset object
        selected_patients: a list of patients in the original adata that can be used for generation

    Returns:

        a dictionary with the name of the simulated patient as key and the name of the original
            patient to sample from as value
    """
    list_patients = [dataset.patients[i].batch for i in range(len(dataset.patients))]

    sample_patients = np.random.choice(
        selected_patients,
        size=(len(list_patients),),
    )
    return {list_patients[i]: sample_patients[i] for i in range(len(list_patients))}


def simulate_gex_malignant(
    adata: ad.AnnData,
    model: scvi.model._scvi.SCVI,
    dataset: Dataset,
    all_malignant_obs: pd.DataFrame,
    sample_patients: Dict[str, str],
) -> Dict[str, np.ndarray]:
    """Function to simulate the malignant components of all patients in the dataset using the
    original adata and a pretrained scVI model. the model should be trained on the same adata as provided
    here.

    Args:

        adata: the original adata object
        model: the scVI model pretrained on the original adata object
        dataset: an instantiated Dataset object
        all_malignant_obs: a dictionary with sim patients as keys and an observation df as value
            as generated by simulate_malignant_comp_batches
        sample_patients: a mapping between simulated and original patients as generated by
            sample_patient_original(_replacement)

    Returns:

        a dictionary with the name of the simulated patient as key and the simulated counts as value
    """
    all_malignant_gex = {}
    for patient in all_malignant_obs:
        df_obs = all_malignant_obs[patient]

        cell_programs = df_obs.program.ravel()
        cell_subclones = df_obs.subclone.ravel()

        print("Getting patient parameters")
        # get the parameters associated with the patient
        zinb_params = get_param_patient(
            adata=adata, patient=sample_patients[patient], model=model
        )

        n_vars = zinb_params[list(zinb_params.keys())[0]]["mean"].shape[1]

        # simulate the effect of a gain/loss for a specific gene separately for each patient
        gain_expr = gex.sample_gain_vector(n_vars)
        loss_expr = gex.sample_loss_vector(n_vars)

        # retrieve the subclone profiles
        mapping_patients = dataset.name_to_patient()
        patient_subclone_profiles = {
            mapping_patients[patient]
            .subclones[i]
            .name: mapping_patients[patient]
            .subclones[i]
            .profile
            for i in range(len(mapping_patients[patient].subclones))
        }
        print("Getting cell specific parameters")
        mean_array, dispersion_array, log_dropout_array, libsize_array = [], [], [], []

        for i, program in enumerate(cell_programs):
            # here we do not draw without replacement because the number of cells we generate might
            # be very different from the original number of cells in the patient
            cell_index = np.random.randint(zinb_params[program]["mean"].shape[0])
            subclone_profile = patient_subclone_profiles[cell_subclones[i]].ravel()

            mean_gex = zinb_params[program]["mean"][cell_index]
            mean_gex = gex.perturb(mean_gex, sigma=0.01)
            # this will modify the expression using the subclone profile of the cell
            mean_gex = gex.change_expression(
                mean_gex,
                changes=subclone_profile,
                gain_change=gain_expr,
                loss_change=loss_expr,
            )
            mean_array.append(mean_gex)
            dispersion_array.append(zinb_params[program]["dispersions"][cell_index])
            log_dropout_array.append(zinb_params[program]["dropout"][cell_index])
            libsize_array.append(zinb_params[program]["libsize"][cell_index])

        print("Starting ZINB sampling")
        batch_gex = get_zinb_gex(
            mean_array=mean_array,
            dispersion_array=dispersion_array,
            log_dropout_array=log_dropout_array,
            libsize_array=libsize_array,
        )
        all_malignant_gex[patient] = batch_gex

    return all_malignant_gex


def simulate_gex_healthy(
    adata: ad.AnnData,
    model: scvi.model._scvi.SCVI,
    all_healthy_obs: Dict[str, pd.DataFrame],
    sample_patients: Dict[str, str],
) -> Dict[str, np.ndarray]:
    """Function to simulate the healthy components of all patients in the dataset using the
    original adata and a pretrained scVI model. the model should be trained on the same adata as provided
    here.

    Args:

        adata: the original adata object
        model: the scVI model pretrained on the original adata object
        all_healthy_obs: a dictionary with sim patients as keys and an observation df as value
            as generated by simulate_healthy_comp_batches
        sample_patients: a mapping between simulated and original patients as generated by
            sample_patient_original(_replacement)

    Returns:

        a dictionary with the name of the simulated patient as key and the simulated counts as value
    """
    all_healthy_gex = {}
    program = "healthy"

    for patient in all_healthy_obs:
        df_obs = all_healthy_obs[patient]

        # get the parameters associated with the patient
        print("Getting patient parameters")
        zinb_params = get_param_patient(
            adata=adata, patient=sample_patients[patient], model=model
        )

        print("Getting cell specific parameters")
        mean_array, dispersion_array, log_dropout_array, libsize_array = [], [], [], []

        for _ in range(df_obs.shape[0]):

            cell_index = np.random.randint(zinb_params[program]["mean"].shape[0])

            mean_array.append(zinb_params[program]["mean"][cell_index])
            dispersion_array.append(zinb_params[program]["dispersions"][cell_index])
            log_dropout_array.append(zinb_params[program]["dropout"][cell_index])
            libsize_array.append(zinb_params[program]["libsize"][cell_index])

        print("Starting ZINB sampling")
        batch_gex = get_zinb_gex(
            mean_array=mean_array,
            dispersion_array=dispersion_array,
            log_dropout_array=log_dropout_array,
            libsize_array=libsize_array,
        )
        all_healthy_gex[patient] = batch_gex

    return all_healthy_gex
