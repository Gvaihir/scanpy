from typing import Union, Callable

import numpy as np

from .. import settings
from .._compat import Literal


def ndcs_npcs():
    n_dcs = 15  # default number of diffusion components
    n_pcs = (
        settings.N_PCS
    )  # Backwards compat, constants should be defined in only one place.
    return n_dcs, n_pcs


def method_type():
    return Literal['umap', 'gauss', 'rapids']


def metric_fn():
    return Callable[[np.ndarray, np.ndarray], float]


def metric():
    metric_sparse_capable = Literal[
        'cityblock', 'cosine', 'euclidean', 'l1', 'l2', 'manhattan'
    ]
    metric_scipy_spatial = Literal[
        'braycurtis',
        'canberra',
        'chebyshev',
        'correlation',
        'dice',
        'hamming',
        'jaccard',
        'kulsinski',
        'mahalanobis',
        'minkowski',
        'rogerstanimoto',
        'russellrao',
        'seuclidean',
        'sokalmichener',
        'sokalsneath',
        'sqeuclidean',
        'yule',
    ]
    return Union[metric_sparse_capable, metric_scipy_spatial]
