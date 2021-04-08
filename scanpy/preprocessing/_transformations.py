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
