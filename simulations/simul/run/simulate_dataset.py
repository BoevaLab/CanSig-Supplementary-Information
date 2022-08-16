import argparse
import pandas as pd
import numpy as np
import pathlib

# these patients fulfill the following criterion: they have at least 50 cells for each
# program and at least 50 healthy cells
SELECTED_PATIENTS = ["C123", "C130", "C137", "C143", "C144"]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "data",
        type=pathlib.Path,
        help="the path to the adata object used to generate the model",
    )
    parser.add_argument("n_batches", type=int, help="number of batches in the dataset")
    parser.add_argument(
        "n_programs", type=int, help="number of programs in the dataset"
    )
    parser.add_argument(
        "n_subclones_min",
        type=int,
        help="the min  number of subclones per patient",
        default=2,
    )
    parser.add_argument(
        "n_subclones_max",
        type=int,
        help="the max number of subclones per patient",
        default=5,
    )
    parser.add_argument(
        "n_healthy_min",
        type=int,
        help="the min  number of healthy cells per patient",
        default=200,
    )
    parser.add_argument(
        "n_healthy_max",
        type=int,
        help="the max number of healty cells per patient",
        default=1000,
    )
    parser.add_argument(
        "n_malignant_min",
        type=int,
        help="the min  number of malignant cells per patient",
        default=200,
    )
    parser.add_argument(
        "n_malignant_max",
        type=int,
        help="the max number of malignant cells per patient",
        default=1000,
    )
    parser.add_argument(
        "subclone_alpha",
        type=int,
        help="the alpha used in the dirichlet dist to generate the proportion of subclones per patient",
        default=1,
    )
    parser.add_argument(
        "anchors",
        nargs="+",
        help="the list of anchors used for the programs - must be the same length as the programs",
    )
    parser.add_argument(
        "chrom_gains",
        nargs="+",
        help="the list of chromosomes to gain on",
    )
    parser.add_argument(
        "chrom_losses",
        nargs="+",
        help="the list of chromosomes to lose on",
    )
    parser.add_argument(
        "chrom_dropout",
        type=float,
        help="the probability of dropping out a chromosome in the CNV generation process for ancestral subclones",
        default=0.1,
    )
    parser.add_argument(
        "chrom_dropout_child",
        type=float,
        help="the probability of dropping out a chromosome in the CNV generation process for child subclones",
        default=0.5,
    )
    parser.add_argument(
        "p_anchor",
        type=float,
        help="the probability of adding a gain on an anchor",
        default=0.5,
    )
    parser.add_argument(
        "min_region_length",
        type=int,
        help="the minimal length of a gain region (without counting reduction because of end of chrom)",
        default=200,
    )
    parser.add_argument(
        "max_region_length",
        type=int,
        help="the maximal length of a gain region (without counting reduction because of end of chrom)",
        default=500,
    )

    args = parser.parse_args()
    return args


# if __name__=="__main__":
