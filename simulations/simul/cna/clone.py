import numpy as np
import pandas as pd

from typing import Optional, List

from .profiles import CNVPerBatchGenerator


class Subclone:
    """A class that stores information about a subclone"""

    def __init__(
        self,
        ancestral: bool,
        name: str,
        batch: str,
        CNVGenerator: CNVPerBatchGenerator,
        anchors: List[str],
        ancestral_changes: Optional[np.ndarray] = None,
    ) -> None:
        """
        Args:
            ancestral: a boolean that flags if the subclone is an ancestral subclone of a batch or not
            name: name of the subclone
            batch: the batch the subclone belongs to
            CNVGenerator: a CNV generator that will serve to create the subclone
            anchors: list of genes that function as anchors
            ancestral_changes: the CNV profile of the ancestral clone if the subclone is a child
        """

        self._ancestral = ancestral
        self.name = name
        self.batch = batch
        self._CNVGenerator = CNVGenerator
        self._ancestral_changes = ancestral_changes
        self.anchors = anchors
        # generate the CNV profile associated with this subclone
        self.profile = self._generate()
        # generate the boolean list associated with the anchor gains for this subclone
        # eg if anchor 1 and 2 are gained in the subclone but not anchor 3
        # then the anchor profile will be [True, True, False]
        self.anchor_profile = self._anchor_gain_profile()

    def is_ancestral(self) -> bool:
        """whether the subclone is an ancestral subclone"""
        return self._ancestral

    def _generate(self) -> np.ndarray:
        """
        returns the array of changes, either created from the ancestral clone or generated de novo
            if the clone is ancestral
            changes are shape (n_genes,)
        """
        if self._ancestral:
            return self._CNVGenerator.generate_ancestral_subclone()
        else:
            return self._CNVGenerator.generate_child_subclone(self._ancestral_changes)

    def _anchor_gain_profile(self) -> List[bool]:
        """Returns a boolean array corresponding to if the anchor is gained or not"""
        return [True if self.profile[anchor] == 1 else False for anchor in self.anchors]
