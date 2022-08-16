"""Sampling different programs for different anchors and batches."""
from typing import Dict, Sequence, Tuple, TypeVar

import numpy as np

from .types import Anchors
from ..rand import Seed

# Type for a batch/sample ID, e.g., it can be a string or int
Batch = TypeVar("Batch")
# Programs
Program = TypeVar("Program")


class ProgramDistribution:
    """An object used to keep the probability distribution
        P(programs | anchors, batch)
    """
    def __init__(self, distribution: Dict[Tuple[Anchors, Batch], Sequence[float]], programs: Sequence[Program] = None, seed: Seed = 256) -> None:
        """

        Args:
            distribution: dictionary representing the conditional probability distribution P(program | anchors, batch)
                The keys are the tuples (anchors, batch) and the values are lists specifying conditional probabilities
                of programs
            programs: programs to be used. If not specified, they are 0-index integers
            seed: random seed, used to initialize the random state inside

        Example:
            An item in the `distribution`
              (ANCHOR, BATCH): [0.1, 0.9, 0.0]
            means that if anchor is ANCHOR and batch is BATCH, there is 10% chance for program 0,
            90% for program 1, and 0% chance for program 2
        """
        self._conditional_probability = {
            key: np.asarray(val) for key, val in distribution.items()
        }
        self._rng = np.random.default_rng(seed)

        some_key = list(distribution.keys())[0]
        self.n_programs = len(distribution[some_key])

        # Validate whether all the vectors are of the same length
        for key, val in self._conditional_probability.items():
            if len(val) != self.n_programs:
                raise ValueError(f"At key {key} the length of the distribution is {len(val)} "
                                 f"instead of {self.n_programs}")

        if programs is None:
            self._programs = [i for i in range(self.n_programs)]
        else:
            self._programs = list(programs)

        if len(self._programs) != self.n_programs:
            raise ValueError("Program lenth mismatch")

    def probabilities(self, anchors: Anchors, batch: Batch) -> np.ndarray:
        """Returns the conditional probability vector

            P(programs | anchors, batch)

        Returns:
            array, shape (n_programs,)

        """
        return self._conditional_probability[(anchors, batch)]

    def sample(self, anchors: Anchors, batch: Batch, n_samples: int = 1) -> np.ndarray:
        """Samples from the distribution P(programs | batch, anchors).

        Args:
            anchors: anchors
            batch: batch
            n_samples: how many samples to take

        Returns:
            array of shape (n_samples,), with entries in the set {0, 1, ..., n_programs-1}
        """
        probs = self.probabilities(anchors=anchors, batch=batch)
        return self._rng.choice(
            self._programs,
            p=probs,
            size=n_samples
        )

    def todict(self) -> Dict:
        return {
            "distribution": {key: val.tolist() for key, val in self._conditional_probability},
            "programs": self._programs,
        }

    @classmethod
    def fromdict(cls, dct: Dict) -> "ProgramDistribution":
        return cls(
            distribution=dct["distribution"],
            programs=dct["programs"],
        )


def get_mask(n_programs: int, prob_dropout: float, min_programs: int, seed: Seed = 554) -> np.ndarray:
    """Calculates mask for dropping out some programs.

    Args:
        n_programs: number of programs
        prob_dropout: probability that each program will be dropped out
        min_programs: after dropout is applied, there may not be enough programs.
            The mask is resampled in that case.

    Returns:
        binary mask, True at `i`th position means that program `i` is available,
            False means that it was dropped out
    """
    rng = np.random.default_rng(seed)

    if min_programs < 0:
        raise ValueError("min_programs must be nonnegative")
    if n_programs < min_programs:
        raise ValueError("min_programs must be at most n_programs")
    if not (0 <= prob_dropout < 1):
        raise ValueError("prob_dropout must be in the interval [0, 1)")

    mask = rng.binomial(1, 1 - prob_dropout, n_programs)
    if sum(mask) >= min_programs:
        return mask
    else:
        return get_mask(n_programs=n_programs, prob_dropout=prob_dropout, min_programs=min_programs, seed=rng)


def probabilities_after_dropout(probabilities: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Drops out programs specified by `mask` and rescales the rest of probabilities.

    Args:
        probabilities: base probabilities, will be rescaled after some programs have been dropped.
            Shape (n_programs,)
        mask: binary mask (values in {0, 1}), programs with 0 will be dropped.
            Shape (n_programs,)

    Returns:
        probabilities vector, shape (n_programs,)
    """
    unnormalized = mask * probabilities
    return unnormalized / np.sum(unnormalized)


def generate_probabilities(anchors_to_alphas: Dict[Anchors, Sequence[float]],
                           batches: Sequence[Batch],
                           prob_dropout: float,
                           min_programs: int,
                           program_names: Sequence[Program] = None,
                           seed: Seed = 3457) -> ProgramDistribution:
    """A factory method for `ProgramDistribution`, implementing a procedure we discussed
    during a whiteboard session:

    For each anchor (calculated from the CNA profile) we have a vector of alphas, parameters of the Dirichlet
    distribution.
    Then, for each batch we sample the program proportions from the Dirichlet distribution parametrized by alphas.

    Hence, we have the conditional probabilities

        P_initial( programs | anchors, batch )

    Alphas control how much these vectors may vary between different batches (e.g., if alphas are very large,
    then the variation will be very small).

    However, to further increase inter-patient heterogeneity we assume that some programs are not present at all
    in some patients.
    Hence, for every batch we generate a binary mask (see `get_mask`), which controls which programs will not be
    present in a given patient.

    Then, we set the "dropped out" programs to 0
        P_initial( programs | anchors, batch)
    for all anchors and rescale the probability vector to obtain the final probability vector

        P_final( programs | anchors, batch)

    Args:
        anchors_to_alphas: for each anchor
        batches: a sequence of considered batches
        prob_dropout: controls the probability of dropping programs, see `get_mask`
        min_programs: controls the minimal number of programs that need to be present in each batch,
            see `get_mask`
        program_names: program names, passed to `ProgramDistribution`
        seed: random seed

    Note:
        For a given batch we generate *one* "drop out" mask, which is shared among all the anchors.
    """
    rng = np.random.default_rng(seed)

    dct = {}
    n_programs = len(list(anchors_to_alphas.values())[0])

    for batch in batches:
        mask = get_mask(n_programs=n_programs, prob_dropout=prob_dropout, min_programs=min_programs, seed=rng)

        for anchor, alphas in anchors_to_alphas.items():
            base_probs = rng.dirichlet(alphas)
            new_probs = probabilities_after_dropout(probabilities=base_probs, mask=mask)

            key = (anchor, batch)
            dct[key] = new_probs

    return ProgramDistribution(dct, programs=program_names, seed=rng)
