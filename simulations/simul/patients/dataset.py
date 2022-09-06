import numpy as np
import pandas as pd

from typing import Optional, List, Dict

from ..cna.clone import Subclone, CNVPerBatchGenerator
from .patient import Patient
from ..rand import Seed


class Dataset:
    """A class to generate datasets. Will hold all information about patients in the dataset and
    their subclones. Automatically generates patients with associated subclones"""

    def __init__(
        self,
        n_batches: int,
        n_programs: int,
        CNVGenerator: CNVPerBatchGenerator,
        n_subclones_min: int = 2,
        n_subclones_max: int = 5,
        n_healthy_min: int = 200,
        n_healthy_max: int = 1000,
        n_malignant_min: int = 200,
        n_malignant_max: int = 1000,
        subclone_alpha: int = 1,
        seed: Seed = 123,
    ) -> None:
        """
        Args:
            n_batches: the number of batches/patients in the dataset
            n_programs: the number of distinct programs in the set
            CNVGenerator: an initialized CNVPerBatchGenerator that will be able to generate CNVs
            n_subclones_min: the minimal number of subclones per patient
            n_subclones_max: the maximal number of subclones per patient
            n_healthy_min: the minimal number of healthy cells per patient
            n_healthy_max: the maximal number of healthy cells per patient
            n_malignant_min: the minimal number of malignant cells per patient
            n_malignant_max: the maximal number of malignant cells per patient
            subclone_alpha: the alpha used for the dirichlet dist to sample subclone proportions
            seed: the random seed for the generator
        """

        self.n_batches = n_batches
        # get the list corresponding to all the batches
        self.batches = self._generate_batches()
        self.n_programs = n_programs
        # get the list corresponding to all the programs
        self.programs = self._generate_programs()
        self._CNVGenerator = CNVGenerator
        self.anchors = CNVGenerator.anchors
        self._n_subclones_min = n_subclones_min
        self._n_subclones_max = n_subclones_max
        self._n_healthy_min = n_healthy_min
        self._n_healthy_max = n_healthy_max
        self._n_malignant_min = n_malignant_min
        self._n_malignant_max = n_malignant_max
        self._subclone_alpha = subclone_alpha
        self._rng = np.random.default_rng(seed)
        # get the list corresponding to all the patients
        self.patients = self._generate_patients()

    def _generate_batches(self) -> List[str]:
        """Returns a list of strings that represent the batches"""
        return [f"patient{i+1}" for i in range(self.n_batches)]

    def _generate_programs(self) -> List[str]:
        """Returns a list of strings that represent programs"""
        return [f"program{i+1}" for i in range(self.n_programs)]

    def _generate_patients(self) -> List[Patient]:
        """Generates patients in the dataset"""
        dataset = []
        for batch in self.batches:
            # get the number of subclones
            n_subclones = self._rng.integers(
                self._n_subclones_min, self._n_subclones_max, endpoint=True
            )
            # get the proportion of subclones
            alphas = np.array([self._subclone_alpha] * n_subclones)
            subclone_proportions = self._rng.dirichlet(alphas, size=1)[0]

            # sample the number of malignant and healthy cells
            n_malignant_cells = self._rng.integers(
                self._n_malignant_min, self._n_malignant_max, endpoint=True
            )
            n_healthy_cells = self._rng.integers(
                self._n_healthy_min, self._n_healthy_max, endpoint=True
            )

            # create the list of subclones with one ancestral subclone and n_subclones - 1 children
            subclones = [
                Subclone(
                    ancestral=True,
                    name="ancestral",
                    batch=batch,
                    CNVGenerator=self._CNVGenerator,
                    anchors=self.anchors,
                )
            ]
            if n_subclones > 1:
                for i in range(n_subclones - 1):
                    subclones.append(
                        Subclone(
                            ancestral=False,
                            name=f"child{i+1}",
                            batch=batch,
                            CNVGenerator=self._CNVGenerator,
                            anchors=self.anchors,
                            ancestral_changes=subclones[0].profile,
                        )
                    )
            # add the patient to the dataset
            dataset.append(
                Patient(
                    batch=batch,
                    subclones=subclones,
                    n_malignant_cells=n_malignant_cells,
                    n_healthy_cells=n_healthy_cells,
                    subclone_proportions=subclone_proportions,
                )
            )
        return dataset

    def get_subclone_profiles(self) -> pd.DataFrame:
        """Returns the profiles of all the subclones in the set as a dataframe
        Rows are subclones, columns are genes"""
        all_subclones = []
        for pat in self.patients:
            for sub in pat.subclones:
                all_subclones.append(sub.profile)
                all_subclones[-1].name = f"{sub.name}_{pat.batch}"
        return pd.concat(all_subclones, axis=1).T

    def order_subclone_profile(self, subclone_df: pd.DataFrame) -> pd.DataFrame:
        """Returns the subclone profile ordered by chromosome and within a chromosome"""
        sorted_index = []
        for chrom in [f"chr{i}" for i in range(1, 23)] + ["chrX", "chrY", "chrM"]:
            sorted_index += list(self._CNVGenerator._genome.chromosome_index(chrom))
        return subclone_df.loc[:, sorted_index].copy()

    def name_to_patient(self) -> Dict[str, Patient]:
        """Returns a mapping for the name of a patient to the patient object"""
        mapping = {}
        for patient in self.patients:
            mapping[patient.batch] = patient
        return mapping
