"""The code used to simulate CNA profiles across genes
as well as utilities for finding anchors."""
from typing import Sequence, Tuple, TypeVar

import numpy as np
import pandas as pd

from ..rand import Seed

ChromosomeName = TypeVar("ChromosomeName")


class Genome:
    """A convenient wrapper around pandas dataframe to
    access genes in the right order.
    """
    def __init__(
            self,
            genes_df: pd.DataFrame,
            chromosome_column: str = "chromosome",
            start_column: str = "start",
    ) -> None:
        """
        Args:
            genes_df: dataframe with genes. The index should consist of gene names.
            chromosome_column: column name in `genes_df` keeping chromosome name
            start_column: column name in `genes_df` keeping the position of the gene
                on the chromosome
        """
        if start_column not in genes_df.columns:
            raise ValueError(f"Start column {start_column} missing in the dataframe.")
        if chromosome_column not in genes_df.columns:
            raise ValueError(f"Chromosome column {start_column} missing in the dataframe.")

        self._genes_dataframe = genes_df
        self._start_column = start_column
        self._chromosome_column = chromosome_column

    def chromosome_length(self, chromosome: ChromosomeName) -> int:
        """The length of a given chromosome."""
        return len(self.chromosome_index(chromosome))

    def _chromosome_dataframe(self, chromosome: ChromosomeName) -> pd.DataFrame:
        mask = self._genes_dataframe[self._chromosome_column] == chromosome
        return self._genes_dataframe[mask].sort_values(self._start_column)

    def chromosome_index(self, chromosome: ChromosomeName) -> pd.Index:
        """Genes in the chromosome, ordered by the position in the chromosome
        (which doesn't need to correspond to the order in the original index)."""
        return self._chromosome_dataframe(chromosome).index

    def __len__(self) -> int:
        return len(self._genes_dataframe)

    @property
    def original_index(self) -> pd.Index:
        """The index of `genes_df`."""
        return self._genes_dataframe.index


GAIN_CHROMOSOMES: Tuple[ChromosomeName, ...] = tuple(
    f"chr{nr}" for nr in [1, 4, 6, 7, 10, 12, 17, 20]
)
LOSS_CHROMOSOMES: Tuple[ChromosomeName, ...] = tuple(
    f"chr{nr}" for nr in [2, 3, 8, 14, 15, 18]
)


class CNVProfileGenerator:
    def __init__(
        self,
        genome: Genome,
        chromosomes_gain: Sequence[ChromosomeName] = GAIN_CHROMOSOMES,
        chromosomes_loss: Sequence[ChromosomeName] = LOSS_CHROMOSOMES,
        min_region_length: int = 25,
        max_region_length: int = 150,
        seed: Seed = 111,
    ) -> None:
        """

        Args:
            genome: object storing the information about genes and chromosomes
            chromosomes_gain: which chromosomes can have gain
            chromosomes_loss: which chromosomes can have loss
            min_region_length: minimal length of the region to be changed
            max_region_length: maximal length of the region to be changed
            seed: random seed
        """
        self._genome = genome
        self._rng = np.random.default_rng(seed)  # Random number generator

        if intersection := set(chromosomes_gain).intersection(chromosomes_loss):
            raise ValueError(f"We assume that each chromosome can only have one region with CNVs. "
                             f"Decide whether it should be loss or gain. Currently there is a non-empty "
                             f"intersection: {intersection}.")
        self._chromosomes_gain = list(chromosomes_gain)
        self._chromosomes_loss = list(chromosomes_loss)

        self.min_region_length = min_region_length
        self.max_region_length = max_region_length

    def _index_with_changes(self, chromosome: ChromosomeName) -> pd.Index:
        """Returns the index of the genes in a chromosomes with a mutation."""

        length = self._rng.integers(self.min_region_length, self.min_region_length, endpoint=True)
        chromosome_length = self._genome.chromosome_length(chromosome)

        # TODO(Pawel): Beware of a one-off error.
        start_position = self._rng.integers(0, chromosome_length - length - 1)
        end_position = start_position + length

        assert end_position <= chromosome_length, "End position must be at most chromosome length."
        return self._genome.chromosome_index(chromosome)[start_position:end_position]

    def generate_subclone(self) -> np.ndarray:
        """

        Returns:
            numpy array with values {-1, 0, 1} for each gene
                (the order of the genes is the same as in `genes_dataframe`)
                -1: there is a copy lost
                0: no change
                1: there is a copy gain
        """
        changes = pd.Series(data=0, index=self._genome.original_index)

        for chromosome in self._chromosomes_gain:
            index = self._index_with_changes(chromosome)
            changes[index] = 1

        for chromosome in self._chromosomes_loss:
            index = self._index_with_changes(chromosome)
            changes[index] = -1

        return changes.values
