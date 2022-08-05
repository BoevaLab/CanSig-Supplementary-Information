import pandas as pd

import simul.cna.profiles as profiles


def test_smoke() -> None:
    mock_genome_df = pd.DataFrame(
        {
            "chromosome": ["chr1"] * 10 + ["chr4"] * 12,
            "start": list(range(1, 11)) + list(range(1, 13)),
        },
        index=[f"gene_{i}" for i in range(10 + 12)],
    )

    mock_generator = profiles.CNVProfileGenerator(
        genome=profiles.Genome(mock_genome_df),
        chromosomes_gain=["chr1"],
        chromosomes_loss=["chr4"],
        min_region_length=1,
        max_region_length=9,
    )

    sample = mock_generator.generate_subclone()
    assert len(sample) == len(mock_genome_df)
