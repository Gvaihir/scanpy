import pytest
import numpy as np

from anndata import AnnData
from scipy.sparse import csr_matrix
from scipy import sparse

import scanpy as sc
from scanpy.tests.helpers import check_rep_mutation, check_rep_results
from anndata.tests.helpers import assert_equal, asarray

X_total = [[1, 0], [3, 0], [5, 6]]


@pytest.mark.parametrize('typ', [np.array, csr_matrix], ids=lambda x: x.__name__)
@pytest.mark.parametrize('dtype', ['float32', 'int64'])
def test_tfidf_transform(typ, dtype):
    adata = AnnData(typ(X_total), dtype=dtype)
    sc.pp.tfidf_transform(
        adata=adata,
        tfidf_method='tf-logidf',
        norm='l2',
        layer=None,
        inplace=True,
    )
    assert np.allclose(np.ravel(adata.X.sum(axis=1)), [1.0, 1.0, 1.3388])
    adata = AnnData(typ(X_total), dtype=dtype)
    sc.pp.tfidf_transform(
        adata=adata,
        tfidf_method='logtf-logidf',
        norm='l2',
        layer=None,
        inplace=True,
    )
    assert np.allclose(np.ravel(adata.X.sum(axis=1)), [1.0, 1.0, 1.35875])


@pytest.mark.parametrize('typ', [asarray, csr_matrix], ids=lambda x: x.__name__)
@pytest.mark.parametrize('dtype', ['float32', 'int64'])
def test_tfidf_transform_rep(typ, dtype):
    # Test that layer kwarg works
    X = typ(sparse.random(100, 50, format="csr", density=0.2, dtype=dtype))
    check_rep_mutation(sc.pp.normalize_total, X, fields=["layer"])
    check_rep_results(sc.pp.normalize_total, X, fields=["layer"])


@pytest.mark.parametrize('typ', [np.array, csr_matrix], ids=lambda x: x.__name__)
@pytest.mark.parametrize('dtype', ['float32', 'int64'])
def test_tfidf_transform_layers(typ, dtype):
    adata = AnnData(typ(X_total), dtype=dtype)
    adata.layers["layer"] = adata.X.copy()
    sc.pp.tfidf_transform(adata, layer="layer")
    assert np.allclose(np.ravel(adata.layers["layer"].sum(axis=1)), [1.0, 1.0, 1.3388])


@pytest.mark.parametrize('typ', [np.array, csr_matrix], ids=lambda x: x.__name__)
@pytest.mark.parametrize('dtype', ['float32', 'int64'])
def test_tfidf_transform_view(typ, dtype):
    adata = AnnData(typ(X_total), dtype=dtype)
    v = adata[:, :]

    sc.pp.tfidf_transform(v)
    sc.pp.tfidf_transform(adata)

    assert not v.is_view
    assert_equal(adata, v)
