import argparse
import dataclasses
import json
import pathlib
import random
import string
from datetime import datetime
from typing import Callable, Optional

import anndata as ad
import scib
from sklearn import metrics


@dataclasses.dataclass
class Scores:
    silhouette: Optional[float] = None
    calinski_harabasz: Optional[float] = None
    davies_bouldin: Optional[float] = None


@dataclasses.dataclass
class Results:
    method: str
    scores: Scores
    params: dict


MALIGNANT = "maligant"
BATCH = "Batch"


def generate_filename(len_random_suffix: int = 5) -> str:
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    suffix = ''.join(random.choice(string.ascii_lowercase) for _ in range(len_random_suffix))
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
        labels=x.obs["Group"].values,
        metric="precomputed",
        random_state=0,
    )

    return Results(
        method=method_name,
        params={},
        scores=Scores(silhouette=silhouette)
    )


def run_scvi(
    data_path: str,
    batch: str = BATCH,
) -> Results:
    raise NotImplementedError


def run_cansig(
    data_path: str,
    batch: str = BATCH,
) -> Results:
    raise NotImplementedError


def create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("data", type=str, help="Path to the H5AD file.")
    parser.add_argument("--method", choices=["combat", "bbknn", "scvi"], help="Method to be applied.", default="bbknn")
    parser.add_argument("--output_dir", type=pathlib.Path, default=pathlib.Path("results"), help="Directory where a JSON file with results will be created.")

    return parser


def get_results(args) -> Results:
    if args.method == "combat":
        return run_matrix_method(
            scib.ig.combat, method_name=args.method, data_path=args.data,
        )
    elif args.method == "bbknn":
        return run_matrix_method(
            scib.ig.bbknn, method_name=args.method, data_path=args.data,
        )
    elif args.method == "scvi":
        return run_scvi(data_path=args.data)
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
        json.dump(
            fp=fp,
            obj=dataclasses.asdict(results),
        )


if __name__ == "__main__":
    main()
