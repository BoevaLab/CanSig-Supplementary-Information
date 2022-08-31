import pandas as pd
import numpy as np
import anndata as ad
import scvi

from typing import Dict, List, Tuple

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
    """Returns a sampled counts matrix using a ZINB corrected by lib size

    Args:

        mean_array: array of shape (n_samples, n_genes) with the mu
        dispersion array: array of shape (n_samples, n_genes) with the associated theta
        log_dropout_array: array of shape (n_sample, n_genes) with the pi in logit scale
        libsize_array: array of shape (n_sample, ) with the observed library size of the cell

    Returns:

        x the sampled count matrix
    """
    # draw from a gamma distribution
    w = np.random.gamma(dispersion_array, mean_array / dispersion_array)
    # then sample from a poisson using the gamma to obtain a negative binomial
    x = np.random.poisson(w)
    # to turn into a ZINB, get a mask using the dropout probability
    ps = np.exp(log_dropout_array) / (1 + np.exp(log_dropout_array))
    mask = 1 - np.random.binomial(np.ones((ps.shape)).astype(int), ps)
    # finally get the sampled expression
    x = mask * x
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


def drop_rarest_program(
    all_malignant_obs: Dict[str, pd.Series],
    dataset: Dataset,
    p_1: float = 0.3,
    p_2: float = 0.5,
) -> Tuple[Dict[str, pd.Series], Dataset]:

    mapping_patients = dataset.name_to_patient()
    new_obs = {}

    for patient in all_malignant_obs:
        new_obs[patient] = all_malignant_obs[patient].copy()
        sort_programs = new_obs[patient].program.value_counts().sort_values()
        rarest_program = sort_programs.index[0]
        second_rarest_program = sort_programs.index[1]
        if np.random.binomial(p=p_1, n=1, size=1):
            # drop the rarest progam
            new_obs[patient] = new_obs[patient][
                ~(new_obs[patient].program == rarest_program)
            ]
            # update the number of malignant cells
            mapping_patients[patient].n_malignant_cells = new_obs[patient].shape[0]

            # only if the rarest program was dropped is there the possibility for
            # the second rarest to be dropped
            if np.random.binomial(p=p_2, n=1, size=1):
                # drop the second rarest progam
                new_obs[patient] = new_obs[patient][
                    ~(new_obs[patient].program == second_rarest_program)
                ]
                # update the number of malignant cells
                mapping_patients[patient].n_malignant_cells = new_obs[patient].shape[0]
    return new_obs, dataset


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
        cell_programs = [
            np.random.choice(["Macro", "Plasma"])
            for _ in range(patient.n_healthy_cells)
        ]
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

        # we first select which genes belong to the highly/lowly expressed, as the effect of
        # gains/losses on gene expression depends on the original expression of the gene
        mask_high = gex.get_mask_high(adata=adata, quantile=0.3)
        # simulate the effect of a gain/loss for a specific gene separately for each patient
        gain_expr = gex.sample_gain_vector(mask_high=mask_high)
        loss_expr = gex.sample_loss_vector(mask_high=mask_high)
        pd.DataFrame(gain_expr).to_csv(f"cnvvectors/{patient}_gain.csv")
        pd.DataFrame(loss_expr).to_csv(f"cnvvectors/{patient}_loss.csv")

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

            # this will modify the expression using the subclone profile of the cell
            mean_gex = gex.change_expression(
                mean_gex,
                changes=subclone_profile,
                gain_change=gain_expr,
                loss_change=loss_expr,
            )
            # we clip the values so that 0 entries become 0.0001. This is because we
            # sample from a gamma distribution at the beginning
            # the % of 0 in the data is small enough that the approximation should be ok
            mean_gex = np.clip(mean_gex, a_min=0.0001, a_max=None)
            mean_array.append(mean_gex)
            dispersion_array.append(zinb_params[program]["dispersions"][cell_index])
            log_dropout_array.append(zinb_params[program]["dropout"][cell_index])
            libsize_array.append(zinb_params[program]["libsize"][cell_index])

        print("Starting ZINB sampling")

        batch_gex = get_zinb_gex(
            mean_array=np.array(mean_array),
            dispersion_array=np.array(dispersion_array),
            log_dropout_array=np.array(log_dropout_array),
            libsize_array=np.array(libsize_array),
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

    for patient in all_healthy_obs:
        df_obs = all_healthy_obs[patient]

        cell_programs = df_obs.program.ravel()

        # get the parameters associated with the patient
        print("Getting patient parameters")
        zinb_params = get_param_patient(
            adata=adata, patient=sample_patients[patient], model=model
        )

        print("Getting cell specific parameters")
        mean_array, dispersion_array, log_dropout_array, libsize_array = [], [], [], []

        for program in cell_programs:
            # here we do not draw without replacement because the number of cells we generate might
            # be very different from the original number of cells in the patient
            cell_index = np.random.randint(zinb_params[program]["mean"].shape[0])

            mean_array.append(zinb_params[program]["mean"][cell_index])
            dispersion_array.append(zinb_params[program]["dispersions"][cell_index])
            log_dropout_array.append(zinb_params[program]["dropout"][cell_index])
            libsize_array.append(zinb_params[program]["libsize"][cell_index])

        print("Starting ZINB sampling")
        batch_gex = get_zinb_gex(
            mean_array=np.array(mean_array),
            dispersion_array=np.array(dispersion_array),
            log_dropout_array=np.array(log_dropout_array),
            libsize_array=np.array(libsize_array),
        )
        all_healthy_gex[patient] = batch_gex

    return all_healthy_gex
