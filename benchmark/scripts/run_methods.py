import argparse
import json
import pathlib
import random
import string
from datetime import datetime
from typing import Callable

import anndata as ad
import scib
from sklearn import metrics


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
    data_path: str,
    batch: str = BATCH,
):
    malignant_data = get_malignant_cells(data_path)
    x = method(malignant_data, batch=batch)

    return metrics.silhouette_score(
        x.obsp["distances"].toarray(),
        labels=x.obs["Group"].values,
        metric="precomputed",
        random_state=0,
    )


def create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("data", type=str, help="Path to the H5AD file.")
    parser.add_argument("--method", choices=["combat", "bbknn"], help="Method to be applied.", default="bbknn")
    parser.add_argument("--output_dir", type=pathlib.Path, default=pathlib.Path("results"), help="Directory where a JSON file with results will be created.")

    return parser


def main() -> None:
    args = create_parser().parse_args()

    if args.method == "combat":
        score = run_matrix_method(
            scib.ig.combat, data_path=args.data,
        )
    elif args.method == "bbknn":
        score = run_matrix_method(
            scib.ig.bbknn, data_path=args.data,
        )
    else:
        raise ValueError(f"Method {args.method} not recognized.")

    output_dir: pathlib.Path = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    
    output_path = output_dir / f"{generate_filename()}.json"

    with open(output_path, "w") as fp:
        json.dump(
            fp=fp,
            obj={
                "method": args.method,
                "silhouette_score": score,
            }
        )


if __name__ == "__main__":
    main()
