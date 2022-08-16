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

        start_position = self._rng.integers(0, chromosome_length)
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
        start_array = (chrom_change != 0).astype(int)
        # the start+length position cannot be where there is already a change
        end_array = (
            np.roll(np.pad(chrom_change, length), -length)[length:-length] != 0
        ).astype(int)
        array = start_array + end_array
        # set the probability of choosing the unauthorized positions to 0, set uniform prob for the rest
        prob_mask = 1 - (array != 0).astype(int)

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

        for chromosome in self._chromosomes_gain:
            # randomly drop a chromosome gain with prob p_child
            if bool(np.random.binomial(p=self.p_child, n=1)):
                index = self._index_with_changes_child(chromosome, changes)
                # if no available spots for non-overlapping gains exist, we do not change anything
                if not (index is None):
                    child_change[index] = 1

        for chromosome in self._chromosomes_loss:
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

        return child_change


GainLossAnchor = Tuple[bool, bool]
GeneName = str


class MostFrequentGainLossAnchorsEstimator:
    """This class takes a family of subclone profiles,
    calculates the anchors and offers a way
    of calculating the anchors from the profiles.

    The API is based on Sci-Kit Learn.

    In the fitting procedure, we feed the CNV profiles and select the "gain gene"
    and the "loss gene".

    The "gain gene" is the gene with most frequent gains. Ties are

    The anchor is a boolean tuple representing:
    (is a gain in the "gain gene", is a loss on the "loss gene")

    Note that if there is a loss on the "gain gene", the value in the anchor
    is going to be False (similarly for the gain on the loss gene).
    """

    def __init__(self, gene_names: Union[Genome, Sequence[GeneName]]) -> None:
        """
        Args:
            gene_names: the gene order, matching the profile vectors. Can be a Genome object or a sequence of names
        """
        self._gene_names: List[GeneName]
        if isinstance(gene_names, Genome):
            self._gene_names = gene_names.original_index.tolist()
        else:
            self._gene_names = list(gene_names)

        # Number of genes
        self._n_genes = len(self._gene_names)

        # Flag to check whether the model is fitted before predictions
        self._is_fitted: bool = False
        # The gain gene name and its index. Will be set during the fitting procedure.
        self._gene_gain_name: GeneName = None
        self._gene_gain_index: int = None

        # The loss gene name and its index. Will be set during the fitting procedure.
        self._gene_loss_name: GeneName = None
        self._gene_loss_index: int = None

    @property
    def gene_gain(self) -> GeneName:
        assert self._is_fitted, "The model must be fitted first."
        return self._gene_gain_name

    @property
    def gene_loss(self) -> GeneName:
        assert self._is_fitted, "The model must be fitted first."
        return self._gene_loss_name

    def _set_gain_gene(self, profiles: np.ndarray) -> None:
        gain_occurences = np.sum(profiles > 0, axis=0)  # Shape (n_genes,)

        self._gene_gain_index = gain_occurences.argmax()
        self._gene_gain_name = self._gene_names[self._gene_gain_index]

    def _set_loss_gene(self, profiles: np.ndarray) -> None:
        loss_occurences = np.sum(profiles < 0, axis=0)  # Shape (n_genes,)

        self._gene_loss_index = loss_occurences.argmax()
        self._gene_loss_name = self._gene_names[self._gene_loss_index]

    def fit(self, profiles: np.ndarray) -> None:
        """Fits the model to subclones, finding the anchor (gain and loss) genes.

        Args:
            profiles: CNV profiles, shape (n_subclones, n_genes)

        Raises:
            ValueError if the "gain gene" and the "loss gene" are the same.
        """
        profiles = np.asarray(profiles)
        assert profiles.shape == (profiles.shape[0], self._n_genes), "Shape mismatch."

        # Find the gain gene
        self._set_gain_gene(profiles)
        assert self._gene_gain_name is not None
        assert self._gene_gain_index is not None

        # Find the loss gene
        self._set_loss_gene(profiles)
        assert self._gene_loss_name is not None
        assert self._gene_loss_index is not None

        # Check that the gain gene and the loss gene are different
        if self._gene_gain_name == self._gene_loss_name:
            raise ValueError(
                f"Gene {self._gene_gain_name} was selected to be both gain gene and the loss gene."
            )

        # Toggle the flag
        self._is_fitted = True

    def _predict_single(self, profile: np.ndarray) -> GainLossAnchor:
        """Like `predict`  but for a single CNV profile."""
        return (
            profile[self._gene_gain_index] > 0,
            profile[self._gene_loss_index] < 0,
        )

    def predict(self, profiles: np.ndarray) -> List[GainLossAnchor]:
        """Generates anchors for given CNV profiles.

        Args:
            profiles: CNV profiles, shape (n_cells, n_genes)

        Returns:
            anchors for each cell, length n_cells
        """
        profiles = np.asarray(profiles)

        assert self._is_fitted, "The model needs to be fit before predictions."
        return [self._predict_single(profile) for profile in profiles]
