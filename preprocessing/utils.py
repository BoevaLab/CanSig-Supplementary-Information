import errno
import logging
import os

import anndata
import pandas as pd
from scipy import sparse


def get_samples(cfg):
    """
    Returns a list of dicts. Each dict contains the path to raw_counts and
    nexus_annotation, as well as the sample_id.
    Sample ids that are in excluded_samples are excluded.
    :param cancer_type: Cancer type for which to run the preprocessing.
    :param df_path: Path to a dataframe containing sample information. The columns are
    expected to be (index, sample_id, path_counts, path_annotation).
    :return: List of dicts containing sample path information.
    """

    df = pd.read_csv(cfg.meta_data_path, index_col=0, sep=cfg.sep)
    samples = df.to_dict("records")

    if cfg.excluded_samples:
        samples = filter(lambda x: x["sample_id"] not in cfg.excluded_samples, samples)

    if cfg.included_samples:
        samples = filter(lambda x: x["sample_id"] in cfg.included_samples, samples)

    samples = list(samples)

    if len(samples) == 0:
        raise ValueError("No samples found.")

    return samples


def load_adata(sample_info):
    """Creates an anndata object containing the raw counts at .X and the annotations
    provided in the annotations path.
    :param sample_info: Dict containing
            "data_path": path to raw count data as a .h5 file,
            "annotation_path": path to annotations provided as a dataframe
            "sample_id": sample_id
    :return: Anndata object
    """
    data_path, sample_id = sample_info["data_path"], sample_info["sample_id"]

    adata = anndata.read_h5ad(data_path)
    if not isinstance(adata.X, sparse.csr_matrix):
        adata.X = sparse.csr_matrix(adata.X)

    adata.obs["sample_id"] = str(sample_id)

    return adata


def get_scoring_dict(cfg):
    scoring_dict = {}
    for score in cfg.scores:
        if score["type"] == "gene_scoring":
            scoring_gene = pd.read_csv(score.annotation).iloc[:, 0].to_list()
            scoring_dict[score.name] = scoring_gene

    return scoring_dict


def get_reference_groups(groups):
    reference_groups = []
    for group in groups:
        if type(group) == str:
            reference_groups.append((group,))
        else:
            reference_groups.append(tuple(group))

    return reference_groups


def mkdirs(cfg) -> None:
    """
    Creates all folders specified in cfg.dirs
    :param cfg: Dict Config containing dirs.
    """
    for path in cfg.dirs.values():
        if path.endswith("_LAST"):
            continue
        logging.info(f"Making dir: {path}")
        os.makedirs(path, exist_ok=True)


def symlink_force(target, link_name):
    try:
        os.symlink(target, link_name)
    except OSError as e:
        if e.errno == errno.EEXIST:
            os.remove(link_name)
            os.symlink(target, link_name)
        else:
            raise e