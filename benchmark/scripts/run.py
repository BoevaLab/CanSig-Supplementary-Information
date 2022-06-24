import argparse
import pathlib
import random
import string
from datetime import datetime
from typing import Optional

import anndata as ad
import pydantic
import scanpy as sc
import scanpy.external as sce
import scib
from sklearn import metrics

import cansig.models.scvi as scvi
import cansig.models.cansig as cs


class Scores(pydantic.BaseModel):
    silhouette: Optional[float] = pydantic.Field(default=None)
    calinski_harabasz: Optional[float] = pydantic.Field(default=None)
    davies_bouldin: Optional[float] = pydantic.Field(default=None)
    kbet: Optional[float] = pydantic.Field(default=None)


class Results(pydantic.BaseModel):
    method: str
    scores: Scores
    params: dict


MALIGNANT = "maligant"
BATCH = "Batch"
GROUP = "Group"


def log(message: str) -> None:
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"=== [{timestamp}] {message} ===")


def generate_filename(len_random_suffix: int = 5) -> str:
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    suffix = "".join(
        random.choice(string.ascii_lowercase) for _ in range(len_random_suffix)
    )
    return f"{timestamp}-{suffix}"


def get_malignant_cells(data_path: str) -> ad.AnnData:
    data = ad.read_h5ad(data_path)
    return data[data.obs["malignant"] == MALIGNANT].copy()


def run_bbknn(
    data_path: str,
    args: argparse.Namespace,
    batch: str = BATCH,
) -> Results:
    adata = get_malignant_cells(data_path)

    # TODO(Pawel): Which preprocessing should I use?
    #  Seems that the silhouette score is actually the best
    #  without any preprocessing, although this may be spurious...
    adata = scib.preprocessing.scale_batch(adata, batch)

    x = scib.ig.bbknn(adata, batch)
    silhouette = metrics.silhouette_score(
        x.obsp["distances"].toarray(),
        labels=x.obs[GROUP].values,
        metric="precomputed",
        random_state=0,
    )
    return Results(method="bbknn", params={}, scores=Scores(silhouette=silhouette))


def run_scanorama(
    data_path: str,
    args: argparse.Namespace,
    batch: str = BATCH,
) -> Results:
    adata = get_malignant_cells(data_path)

    sc.pp.recipe_zheng17(adata)
    sc.pp.pca(adata, n_comps=args.latent)
    sce.pp.scanorama_integrate(adata, batch)

    codes_array = adata.obsm["X_scanorama"]
    labels = adata.obs[GROUP].values

    return Results(
        method="scanorama",
        params={
            "pca_dim": args.latent,
        },
        scores=Scores(
            silhouette=metrics.silhouette_score(
                codes_array, labels=labels, random_state=0
            ),
            calinski_harabasz=metrics.calinski_harabasz_score(codes_array, labels),
            davies_bouldin=metrics.davies_bouldin_score(codes_array, labels),
            kbet=scib.metrics.kBET(
                adata,
                batch_key=batch,
                label_key=GROUP,
                embed="X_scanorama",
            )
        ),
    )


def run_scvi(
    data_path: str,
    args: argparse.Namespace,
    batch: str = BATCH,
) -> Results:
    malignant_data = get_malignant_cells(data_path)

    config = scvi.SCVIConfig(
        batch=batch,
        n_latent=args.latent,
        preprocessing=scvi.PreprocessingConfig(n_top_genes=args.n_top_genes),
        train=scvi.TrainConfig(max_epochs=args.max_epochs),
        model=scvi.ModelConfig(n_hidden=args.scvi_hidden, n_layers=args.scvi_layers)
    )

    model = scvi.SCVI(config=config, data=malignant_data)
    codes = model.get_latent_codes()
    assert (codes.index == malignant_data.obs.index).all(), "Index mismatch."

    codes_array = codes.values
    labels = malignant_data.obs[GROUP].values

    malignant_data.obsm["X_scvi"] = codes.values

    return Results(
        method="scvi",
        params=config.dict(),
        scores=Scores(
            silhouette=metrics.silhouette_score(
                codes_array, labels=labels, random_state=0
            ),
            calinski_harabasz=metrics.calinski_harabasz_score(codes_array, labels),
            davies_bouldin=metrics.davies_bouldin_score(codes_array, labels),
            kbet=scib.metrics.kBET(
                malignant_data,
                batch_key=batch,
                label_key=GROUP,
                embed="X_scvi",
            )
        ),
    )


def run_cansig(
    data_path: str,
    args: argparse.Namespace,
    batch: str = BATCH,
) -> Results:
    adata = ad.read_h5ad(data_path)
    adata.obs["subclonal"] = adata.obs[BATCH]
    adata.obs['malignant_key'] = adata.obs['Group'].apply(
        lambda celltype: 'non-malignant' if celltype == 'Group3' else 'malignant')
    adata.obs["celltype"] = adata.obs["Group"].values
    adata.obs["sample_id"] = adata.obs[BATCH]

    adata = cs.CanSig.preprocessing(adata)
    cs.CanSig.setup_anndata(
        adata,
        cnv_key="X_cnv",  # column present in data.obs
        celltype_key=GROUP,
    )

    config = cs.CanSigConfig(
        batch=batch,
        preprocessing=cs.PreprocessingConfig(n_top_genes=args.n_top_genes),
        train=cs.TrainConfig(max_epochs=args.max_epochs),
        n_latent=args.latent,
        n_latent_batch_effect=args.latent,
        n_latent_cnv=args.latent,
        model=cs.ModelConfig(n_hidden=args.scvi_hidden, n_layers=args.scvi_layers),
    )
    model = cs.CanSigWrapper(config=config, data=adata)

    codes = model.get_latent_codes()
    malignant_data = get_malignant_cells(data_path)

    codes_array = codes.values

    assert (codes.index == malignant_data.obs.index).all(), "Index mismatch."

    malignant_data.obsm["X_cansig"] = codes
    labels = malignant_data.obs[GROUP].values

    return Results(
        method="cansig",
        params=config.dict(),
        scores=Scores(
            silhouette=metrics.silhouette_score(
                codes_array, labels=labels, random_state=0
            ),
            calinski_harabasz=metrics.calinski_harabasz_score(codes_array, labels),
            davies_bouldin=metrics.davies_bouldin_score(codes_array, labels),
            kbet=scib.metrics.kBET(
                malignant_data,
                batch_key=batch,
                label_key=GROUP,
                embed="X_cansig",
            )
        ),
    )


def create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("data", type=str, help="Path to the H5AD file.")
    parser.add_argument(
        "--method",
        choices=["bbknn", "scvi", "scanorama", "cansig"],
        help="Method to be applied.",
        default="bbknn",
    )
    parser.add_argument(
        "--output-dir",
        type=pathlib.Path,
        default=pathlib.Path("results"),
        help="Directory where a JSON file with results will be created.",
    )
    parser.add_argument("--latent", type=int, default=5, help="Latent space dimension.")
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
    parser.add_argument(
        "--scvi-layers",
        type=int,
        default=1,
        help="Number of scVI layers."
    )
    parser.add_argument(
        "--scvi-hidden",
        type=int,
        default=128,
        help="Number of neurons per layer in scVI."
    )

    return parser


def get_results(args) -> Results:
    log(f"Running method {args.method}...")

    if args.method == "bbknn":
        return run_bbknn(data_path=args.data, args=args)
    elif args.method == "scvi":
        return run_scvi(data_path=args.data, args=args)
    elif args.method == "cansig":
        return run_cansig(data_path=args.data, args=args)
    elif args.method == "scanorama":
        return run_scanorama(data_path=args.data, args=args)
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

    log(f"Saving results to {output_path}...")

    with open(output_path, "w") as fp:
        fp.write(results.json())

    log("Run finished.")


if __name__ == "__main__":
    main()
