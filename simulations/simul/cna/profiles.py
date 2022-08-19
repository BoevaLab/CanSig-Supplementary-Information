"""The code used to simulate CNA profiles across genes
as well as utilities for finding anchors."""
from typing import List, Sequence, Tuple, TypeVar, Union, Optional

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
            raise ValueError(
                f"Chromosome column {start_column} missing in the dataframe."
            )

        self._genes_dataframe = genes_df
        self._start_column = start_column
        self._chromosome_column = chromosome_column

    def chromosome_length(self, chromosome: ChromosomeName) -> int:
        """The length of a given chromosome."""
        return len(self.chromosome_index(chromosome))

    def get_gene_chromosome(self, gene: str) -> ChromosomeName:
        return str(self._genes_dataframe.loc[gene][self._chromosome_column])

    def get_gene_pos(self, gene: str) -> int:
        chrom = self.get_gene_chromosome(gene)
        return int(np.where(self.chromosome_index(chrom) == gene)[0])

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


class CNVPerBatchGenerator:
    def __init__(
        self,
        genome: Genome,
        anchors: List[str],
        p_anchor: float = 0.5,
        chromosomes_gain: Sequence[ChromosomeName] = GAIN_CHROMOSOMES,
        chromosomes_loss: Sequence[ChromosomeName] = LOSS_CHROMOSOMES,
        dropout: float = 0.1,
        dropout_child: float = 0.1,
        min_region_length: int = 25,
        max_region_length: int = 150,
        seed: Seed = 111,
    ) -> None:
        """
        Args:
            genome: object storing the information about genes and chromosomes
            anchors: list of names of genes that function as anchors
            p_anchor: the probability of getting a gain on an anchor gene
            chromosomes_gain: which chromosomes can have gain
            chromosomes_loss: which chromosomes can have loss
            dropout: probability of a chromosome being dropped out for gain or loss in ancestral clone
            dropout_child: probability of a chromosome being dropped out for gain or loss in child clones
            min_region_length: minimal length of the region to be changed
            max_region_length: maximal length of the region to be changed
            seed: random seed
        """
        self._genome = genome
        self.anchors = anchors
        self._p_anchor = p_anchor
        self._rng = np.random.default_rng(seed)  # Random number generator

        if intersection := set(chromosomes_gain).intersection(chromosomes_loss):
            raise ValueError(
                f"We assume that each chromosome can only have one region with CNVs. "
                f"Decide whether it should be loss or gain. Currently there is a non-empty "
                f"intersection: {intersection}."
            )

        self.p = 1 - dropout
        self.p_child = 1 - dropout_child
        self._chromosomes_gain = list(chromosomes_gain)
        self._chromosomes_loss = list(chromosomes_loss)

        self.min_region_length = min_region_length
        self.max_region_length = max_region_length

    def _index_with_changes(self, chromosome: ChromosomeName) -> pd.Index:
        """Returns the index of the genes in a chromosomes with a CNV."""

        length = self._rng.integers(
            self.min_region_length, self.max_region_length, endpoint=True
        )
        chromosome_length = self._genome.chromosome_length(chromosome)
        if (chromosome_length - length) <= 0:
            start_position = 0
        else:
            start_position = self._rng.integers(0, chromosome_length - length)
        end_position = min(start_position + length, chromosome_length)

        assert (
            end_position <= chromosome_length
        ), "End position must be at most chromosome length."
        return self._genome.chromosome_index(chromosome)[start_position:end_position]

    def _index_with_changes_anchor(self, anchor: str) -> pd.Index:
        """Returns the index of the genes in a chromosomes with a CNV around the anchor."""

        length = self._rng.integers(
            self.min_region_length, self.max_region_length, endpoint=True
        )
        # get the chromosome associated with the anchor
        anchor_chrom = self._genome.get_gene_chromosome(anchor)
        chromosome_length = self._genome.chromosome_length(anchor_chrom)

        # ideally we want the anchor to be in the middle of the gain
        start_position = max(0, self._genome.get_gene_pos(anchor) - (length // 2))
        end_position = min(start_position + length, chromosome_length)

        assert (
            end_position <= chromosome_length
        ), "End position must be at most chromosome length."
        return self._genome.chromosome_index(anchor_chrom)[start_position:end_position]

    def _index_with_changes_child(
        self, chromosome: ChromosomeName, changes: np.ndarray
    ) -> Optional[pd.Index]:
        """Returns the index of the genes in a chromosomes with a CNV for child subclones."""

        df_change = pd.Series(data=changes, index=self._genome.original_index)
        chrom_change = df_change[self._genome.chromosome_index(chromosome)].values

        length = self._rng.integers(
            self.min_region_length, self.max_region_length, endpoint=True
        )
        chromosome_length = self._genome.chromosome_length(chromosome)

        # the start position cannot be where there is already a change
        # the start+length position cannot be where there is already a change
        end_array = np.zeros((chromosome_length))
        for i in range(1, length):
            end_array += np.roll(np.pad(chrom_change, i), -i)[i:-i]
        end_array = (end_array != 0).astype(int)
        array = chrom_change + end_array
        # set the probability of choosing the unauthorized positions to 0, set uniform prob for the rest
        prob_mask = 1 - (array != 0).astype(int)
        # the change can't occur at the end if there isn't length genes left
        prob_mask[-length:] = 0

        # maybe none of the position can create non-overlapping gains/losses, in which case
        # we skip this chromosome
        if all(prob_mask == 0):
            return None

        prob_mask = prob_mask / prob_mask.sum()

        start_position = self._rng.choice(np.arange(chromosome_length), p=prob_mask)
        end_position = min(start_position + length, chromosome_length)

        assert (
            end_position <= chromosome_length
        ), "End position must be at most chromosome length."
        return self._genome.chromosome_index(chromosome)[start_position:end_position]

    def _generate_gain_anchors(self, changes: pd.Series) -> pd.Series:
        """Generates a gain around anchors with prob p_anchor"""
        anchor_changes = changes.copy()
        for anchor in self.anchors:
            if bool(np.random.binomial(p=self._p_anchor, n=1)):
                index = self._index_with_changes_anchor(anchor)
                anchor_changes[index] = 1
        return anchor_changes

    def generate_ancestral_subclone(self) -> pd.Series:
        """
        This function is used to generate the ancestral subclone from which all children
            subclones will branch for a specific patient
        Returns:
            numpy array with values {-1, 0, 1} for each gene
                (the order of the genes is the same as in `genes_dataframe`)
                -1: there is a copy lost
                0: no change
                1: there is a copy gain
        """
        changes = pd.Series(data=0, index=self._genome.original_index)

        for chromosome in self._chromosomes_gain:
            # randomly drop a chromosome gain with prob p
            if bool(np.random.binomial(p=self.p, n=1)):
                index = self._index_with_changes(chromosome)
                changes[index] = 1

        for chromosome in self._chromosomes_loss:
            # randomly drop a chromosome loss with prob p
            if bool(np.random.binomial(p=self.p, n=1)):
                index = self._index_with_changes(chromosome)
                changes[index] = -1
        # we treat anchors separately, so that they follow three characteristics:
        # (a) the anchors are differentially gained across patients
        # (b) the anchors are differentially gained across subclones
        # (c) the anchors are more often gained than not
        changes = self._generate_gain_anchors(changes)
        return changes

    def generate_child_subclone(self, changes) -> pd.Series:
        """
        This function is used to generate child subclones using the ancestral subclone of a
            specific patient
        Args:
             changes: numpy array of changes of the ancestral subclone
        Returns:
            numpy array with values {-1, 0, 1} for each gene
                (the order of the genes is the same as in `genes_dataframe`)
                -1: there is a copy lost
                0: no change
                1: there is a copy gain
        """

        # the gains/losses are added to the ancestral clone
        # we do not want the gains or losses to be overlapping with the existing ones, so we
        # select the start position so that the generated gain/loss does not overlap

        child_change = changes.copy()
        ancestral_gains, ancestral_losses = self.return_chromosomes_ancestral(
            child_change
        )

        for chromosome in np.setdiff1d(self._chromosomes_gain, ancestral_gains):
            # randomly drop a chromosome gain with prob p_child
            if bool(np.random.binomial(p=self.p_child, n=1)):
                index = self._index_with_changes_child(chromosome, changes)
                # if no available spots for non-overlapping gains exist, we do not change anything
                if not (index is None):
                    child_change[index] = 1

        for chromosome in np.setdiff1d(self._chromosomes_loss, ancestral_losses):
            # randomly drop a chromosome loss with prob p_child
            if bool(np.random.binomial(p=self.p_child, n=1)):
                index = self._index_with_changes_child(chromosome, changes)
                # if no available spots for non-overlapping losses exist, we do not change anything
                if not (index is None):
                    child_change[index] = -1

        # we treat anchors separately, so that they follow three characteristics:
        # (a) the anchors are differentially gained across patients
        # (b) the anchors are differentially gained across subclones
        # (c) the anchors are more often gained than not
        child_change = self._generate_gain_anchors(child_change)
        if (child_change == changes).all():
            print("Child similar to ancestral, regenerating...")
            return self.generate_child_subclone(changes)

        return child_change

    def return_chromosomes_ancestral(self, changes):

        df_change = pd.Series(data=changes, index=self._genome.original_index)

        chromosomes_gain, chromosomes_losses = [], []
        for chromosome in self._chromosomes_gain:
            chrom_change = df_change[self._genome.chromosome_index(chromosome)].values
            if (chrom_change > 0).sum() != 0:
                chromosomes_gain.append(chromosome)
        for chromosome in self._chromosomes_loss:
            chrom_change = df_change[self._genome.chromosome_index(chromosome)].values
            if (chrom_change < 0).sum() != 0:
                chromosomes_losses.append(chromosome)
        return chromosomes_gain, chromosomes_losses
