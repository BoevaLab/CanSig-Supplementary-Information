import numpy as np
import pytorch_lightning as pl
import torch
import torch.utils.data as torchdata

import dhaka.api as dh


def construct_dataset(n_genes: int, batch_size: int = 30) -> torchdata.DataLoader:
    data = dh.example_data(n_cells=280, n_genes=2 * n_genes)

    X = dh.normalize(data, pseudocounts=1, n_top_genes=n_genes)
    dataset = dh.NumpyArrayDataset(X)

    return torchdata.DataLoader(dataset=dataset, shuffle=True, batch_size=batch_size)


def tensor_to_numpy(t: torch.Tensor) -> np.ndarray:
    return t.detach().cpu().numpy()


def main() -> None:
    n_genes: int = 100
    batch_size: int = 30

    adata = dh.example_data(n_cells=35, n_genes=2 * n_genes)
    dataset = dh.NumpyArrayDataset(
        dh.normalize(adata, pseudocounts=1, n_top_genes=n_genes)
    )

    train_dataloader = torchdata.DataLoader(dataset=dataset, shuffle=True, batch_size=batch_size)

    model = dh.Dhaka(n_genes=n_genes)

    trainer = pl.Trainer(max_epochs=5, gradient_clip_val=2, gradient_clip_algorithm="norm")
    trainer.fit(model, train_dataloaders=train_dataloader)

    prediction_dataloader = torchdata.DataLoader(dataset=dataset, shuffle=False, batch_size=batch_size)
    model.eval()

    representations_all = np.concatenate([
        tensor_to_numpy(model.representations(batch)) for batch in prediction_dataloader
    ])

    print(representations_all.shape)

if __name__ == "__main__":
    main()
