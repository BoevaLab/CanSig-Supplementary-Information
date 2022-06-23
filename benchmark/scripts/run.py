import argparse
import json
import pathlib
import random
import string
from datetime import datetime
from typing import Callable, Optional

import anndata as ad
import pydantic
import scib
from sklearn import metrics

import cansig.models.scvi as scvi
import cansig.models.cansig as cs
import cansig.integration.model as intmodel


class Scores(pydantic.BaseModel):
    silhouette: Optional[float] = pydantic.Field(default=None)
    calinski_harabasz: Optional[float] = pydantic.Field(default=None)
    davies_bouldin: Optional[float] = pydantic.Field(default=None)


class Results(pydantic.BaseModel):
    method: str
    scores: Scores
    params: dict


MALIGNANT = "maligant"
BATCH = "Batch"
GROUP = "Group"


def generate_filename(len_random_suffix: int = 5) -> str:
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    suffix = "".join(
        random.choice(string.ascii_lowercase) for _ in range(len_random_suffix)
    )
    return f"{timestamp}-{suffix}"


def get_malignant_cells(data_path: str) -> ad.AnnData:
    data = ad.read_h5ad(data_path)
    return data[data.obs["malignant"] == MALIGNANT].copy()


def run_matrix_method(
    method: Callable,
    method_name: str,
    data_path: str,
    batch: str = BATCH,
) -> Results:
    malignant_data = get_malignant_cells(data_path)
    x = method(malignant_data, batch=batch)

    silhouette = metrics.silhouette_score(
        x.obsp["distances"].toarray(),
        labels=x.obs[GROUP].values,
        metric="precomputed",
        random_state=0,
    )

    return Results(method=method_name, params={}, scores=Scores(silhouette=silhouette))


def run_scvi(
    data_path: str,
    args: argparse.Namespace,
    batch: str = BATCH,
) -> Results:
    malignant_data = get_malignant_cells(data_path)

    config = scvi.SCVIConfig(
        batch=batch,
        n_latent=args.scvi_latent,
        preprocessing=scvi.PreprocessingConfig(n_top_genes=args.n_top_genes),
        train=scvi.TrainConfig(max_epochs=args.max_epochs),
    )

    model = scvi.SCVI(config=config, data=malignant_data)
    codes = model.get_latent_codes()
    assert (codes.index == malignant_data.obs.index).all(), "Index mismatch."

    codes_array = codes.values
    labels = malignant_data.obs[GROUP].values

    res = Results(
        method="scvi",
        params=config.dict(),
        scores=Scores(
            silhouette=metrics.silhouette_score(codes_array, labels=labels, random_state=0),
            calinski_harabasz=metrics.calinski_harabasz_score(codes_array, labels),
            davies_bouldin=metrics.davies_bouldin_score(codes_array, labels),
        ),
    )

    print(res)
    return res


def run_cansig(
    data_path: str,
    batch: str = BATCH,
    non_malignant: str = "non-malignant"
) -> Results:

    data = ad.read_h5ad(data_path)
    data.obs['subclonal'] = data.obs[BATCH]

    adata = intmodel.CanSig.preprocessing(data)
    intmodel.CanSig.setup_anndata(
        adata,
        cnv_key="X_cnv",  # column present in data.obs
        celltype_key=GROUP,
        malignant_key="malignant",  # column present in data.obs
        malignant_cat=MALIGNANT,
        non_malignant_cat=non_malignant,
    )

    malignant_index = None
    raise NotImplementedError

    idx = cansig.get_index(malignant_cells=True)
    bdata = adata[idx, :].copy()
    bdata.obsm['latent'] = cansig.get_latent_representation()

    raise NotImplementedError


def create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("data", type=str, help="Path to the H5AD file.")
    parser.add_argument(
        "--method",
        choices=["combat", "bbknn", "scvi"],
        help="Method to be applied.",
        default="bbknn",
    )
    parser.add_argument(
        "--output_dir",
        type=pathlib.Path,
        default=pathlib.Path("results"),
        help="Directory where a JSON file with results will be created.",
    )
    parser.add_argument(
        "--scvi-latent",
        type=int,
        default=5,
        help="Latent space dimension."
    )
    parser.add_argument(
        "--n-top-genes",
        type=int,
        default=1000,
        help="Number of highly-variable genes.",
    )
    parser.add_argument(
        "--max-epochs",
        type=int,
        default=200,
        help="Maximal number of epochs.",
    )

    return parser


def get_results(args) -> Results:
    if args.method == "combat":
        return run_matrix_method(
            scib.ig.combat,
            method_name=args.method,
            data_path=args.data,
        )
    elif args.method == "bbknn":
        return run_matrix_method(
            scib.ig.bbknn,
            method_name=args.method,
            data_path=args.data,
        )
    elif args.method == "scvi":
        return run_scvi(data_path=args.data, args=args)
    else:
        raise ValueError(f"Method {args.method} not recognized.")


def main() -> None:
    args = create_parser().parse_args()

    # Get the result
    results = get_results(args)

    # Generate output path
    output_dir: pathlib.Path = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{generate_filename()}.json"

    with open(output_path, "w") as fp:
        fp.write(results.json())


if __name__ == "__main__":
    main()
