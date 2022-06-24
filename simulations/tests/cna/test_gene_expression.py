import numpy as np
import numpy.testing as nptest

import pytest

import sim.cna.gene_expression as gex


@pytest.mark.parametrize("fill", [1.0, 2.0])
def test_create_changes_vector(fill: float) -> None:
    mask = np.asarray([True, False, True, False, False], dtype=bool)
    changes = np.asarray([2., 3., 5., 2., 4.])

    desired = np.asarray([2., fill, 5., fill, fill])
    output = gex._create_changes_vector(mask=mask, change=changes, fill=fill)

    nptest.assert_equal(output, desired)
