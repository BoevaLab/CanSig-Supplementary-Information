"""A simple script demonstrating how to use the Dhaka implementation."""
import dhaka.api as dh


def main(n_cells: int = 100, n_latent: int = 5) -> None:
    # Generate some example data
    adata = dh.example_data(n_cells=n_cells, n_genes=40)
    # Generate run config
    config = dh.DhakaConfig(n_latent=n_latent, epochs=2)

    # Learn the representations
    adata = dh.run_dhaka(adata, config=config, key_added="X_dhaka")

    # Check that they indeed exist
    assert adata.obsm["X_dhaka"].shape == (n_cells, n_latent)


if __name__ == "__main__":
    main()
