from typing import Optional, Dict

import numpy as np
from anndata import AnnData
from sklearn.feature_extraction.text import TfidfTransformer
from .. import logging as logg
from .._utils import view_to_actual
from scanpy.get import _get_obs_rep, _set_obs_rep


def _tfidf_transform_inner(untransformed_x, norm, tfidf_method, inplace=False):
    untransformed_x = untransformed_x if inplace else untransformed_x.copy()
    method_switch = {
        "tf-logidf": {"sublinear_tf": False},
        "logtf-logidf": {"sublinear_tf": True},
    }
    transformer = TfidfTransformer(norm=norm, **method_switch.get(tfidf_method))
    transformed = transformer.fit_transform(untransformed_x)
    return transformed


def tfidf_transform(
    adata: AnnData,
    tfidf_method: str = 'tf-logidf',
    norm: str = 'l2',
    layer: Optional[str] = None,
    inplace: bool = True,
) -> Optional[Dict[str, np.ndarray]]:
    """\
    Transform counts to TF-IDF representation for scATAC data to perform LSA

    Similar approach used in ArchR package [Granja21]

    Params
    ------
    adata
        The annotated data matrix of shape `n_obs` × `n_vars`.
        Rows correspond to cells and columns to genes.
    tfidf_method
        `tf-logidf` (default) - adapted from [Cusanovich15], `logtf-logidf` -
        alternative method with sublinear tf transform
    norm
        Vector norm. `l2` (default) uses Euclidean distance, `l1` uses absolute distance
    layer
        Layer to normalize instead of `X`. If `None`, `X` is normalized.
    inplace
        Whether to update `adata` or return dictionary with normalized copies of
        `adata.X` and `adata.layers`.

    Returns
    -------
    Returns dictionary with transformed copies of `adata.X` and `adata.layers`
    or updates `adata` with transformed version of the original
    `adata.X` and `adata.layers`, depending on `inplace`.

    Example
    --------
    >>> from anndata import AnnData
    >>> import scanpy as sc
    >>> sc.settings.verbosity = 2
    >>> np.set_printoptions(precision=2)
    >>> adata = AnnData(np.array([
    ...    [3, 3, 3, 6, 6],
    ...    [1, 1, 1, 2, 2],
    ...    [1, 22, 1, 2, 2],
    ... ]))
    >>> adata.X
    array([[ 3.,  3.,  3.,  6.,  6.],
           [ 1.,  1.,  1.,  2.,  2.],
           [ 1., 22.,  1.,  2.,  2.]], dtype=float32)
    >>> X_tfidf = sc.pp.tfidf_transform(adata, inplace=False)['X']
    >>> X_tfidf

    >>> X_tfidf = sc.pp.tfidf_transform(
    ...     adata, tfidf_method='logtf-logidf', inplace=False
    ... )['X']
    Using an alternative transform method with sublinear term frequency:
    >>> X_tfidf
    """

    accepted_tfidf_methods = ['tf-logidf', 'logtf-logidf']

    if tfidf_method not in accepted_tfidf_methods:
        raise ValueError(
            f'tfidf_transform(): provided tfidf_method {tfidf_method}. '
            f'Accepted methods are {accepted_tfidf_methods}'
        )

    view_to_actual(adata)
    X = _get_obs_rep(adata, layer=layer)
    msg = f'performing TF-IDF tranformation using method {tfidf_method}'
    start = logg.info(msg)

    if inplace:
        _set_obs_rep(
            adata, _tfidf_transform_inner(X, norm, tfidf_method, inplace), layer=layer
        )
    else:
        # not recarray because need to support sparse
        dat = dict(
            X=_tfidf_transform_inner(X, norm, tfidf_method, inplace),
            tfidf_method=tfidf_method,
        )

    logg.info(
        '    finished ({time_passed})',
        time=start,
    )

    if not inplace:
        return dat
