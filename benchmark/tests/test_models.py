"""Smoke tests for the integration models."""
import anndata
import numpy as np
import pytest

from benchmark.models import (run_bbknn, BBKNNConfig, SCVIConfig, run_scvi,
                              ScanoramaConfig, run_scanorama, CombatConfig, run_combat,
                              DescConfig, run_desc, MNNConfig, run_mnn)


@pytest.fixture
def adata():
    adata = anndata.AnnData(X=np.random.negative_binomial(1000, 0.5, size=(100, 2000)))
    adata.layers["counts"] = adata.X.copy()
    adata.obs["sample_id"] = ["batch_1"] * 50 + ["batch_2"] * 50
    adata.strings_to_categoricals()
    return adata


def test_bbknn(adata):
    config = BBKNNConfig()
    adata = run_bbknn(adata, config)
    assert "distances" in adata.obsp.keys()
    assert "connectivities" in adata.obsp.keys()
    assert "neighbors" in adata.uns


def test_scvi(adata):
    config = SCVIConfig(max_epochs=2)
    adata = run_scvi(adata, config)
    assert config.latent_key in adata.obsm_keys()


def test_scanorama(adata):
    config = ScanoramaConfig()
    adata = run_scanorama(adata, config)
    assert config.latent_key in adata.obsm_keys()


def test_combat(adata):
    config = CombatConfig()
    adata = run_combat(adata, config)
    assert config.latent_key in adata.obsm_keys()

def test_desc(adata):
    config = DescConfig()
    adata = run_desc(adata, config)
    assert config.latent_key in adata.obsm_keys()

def test_mnn(adata):
    config = MNNConfig()
    adata = run_mnn(adata, config)
    assert config.latent_key in adata.obsm_keys()