import numpy as np
import pandas as pd

from typing import Optional, List

from ..cna.clone import Subclone


class Patient:
    """A class that stores information about a specific patient"""

    def __init__(
        self,
        batch: str,
        subclones: List[Subclone],
        n_malignant_cells: int,
        n_healthy_cells: int,
        subclone_proportions: np.ndarray,
    ) -> None:
        """
        Args:
            batch: the batch associated with this patient.
            subclones: the list of subclone instances associated with the patient
            n_malignant_cells: the number of malignant cells associated with the patient
            n_healthy_cells: the number of healthy cells associated with the patient
            subclones_proportions: the (n_subclones,) array containing in what proportion the sublones
                are represented
        """

        self.batch = batch
        self.subclones = subclones
        self.n_malignant_cells = n_malignant_cells
        self.n_healthy_cells = n_healthy_cells
        self.subclone_proportions = subclone_proportions

    def n_total_cells(self) -> int:
        """The total number of cells for this patient."""
        return self.n_malignant_cells + self.n_healthy_cells
