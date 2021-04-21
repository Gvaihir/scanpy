from typing import Optional, List, Tuple, Callable, Any, Iterable, Dict, Union
import numpy as np
import numba as nb
from anndata import AnnData
from scipy.sparse import issparse, find, csr_matrix
from itertools import product

from .neighbors_config import method_type, metric, metric_fn
from .._utils import _doc_params, NeighborsView
from ..tools._utils import doc_use_rep, doc_n_pcs
from .Neighbors import NeighborsError

_Method = method_type()
_Metric = metric()
_MetricFn = metric_fn()


@_doc_params(n_pcs=doc_n_pcs, use_rep=doc_use_rep)
def weighted_neighbors(
    adata_collection: List[AnnData],
    joint: bool = False,
    key_added: Optional[str] = None,
) -> AnnData:
    """\
    Compute a combined neighborhood graph based on weighted kNN.

    Parameters
    ----------
    adata_collection
        List of annotated data matrix.
    joint
        If `True` uses all modalities to calculate weights at once. Else, computes w-kNN
        for the first pair of modalities, then recursively add new modality to computed
        w-kNN graph
        key_added
    If not specified, the neighbors data is stored in .uns['neighbors'],
    distances and connectivities are stored in .obsp['distances'] and
    .obsp['connectivities'] respectively.
    If specified, the neighbors data is added to .uns[key_added],
    distances are stored in .obsp[key_added+'_distances'] and
    connectivities in .obsp[key_added+'_connectivities'].

    Returns
    -------
    New AnnData with w-kNN attribute
    """
    for adata in adata_collection:
        if not knn_computed(adata):
            raise NeighborsError(
                "weighted_neighbors(): some AnnData objects don't "
                "have pre-computed neighbor graphs"
            )


def calculate_neighborhood_weights(
    adata_collection: List[AnnData],
    joint: bool = False,
    key_added: Optional[str] = None,
) -> Iterable[Tuple[float, ...]]:
    """\
    Computes weights for every modality for each cell.

    Parameters
    ----------
    See weighted_neighbors()
    Returns
    -------
    Generator of tuples with weights for each modality in the order of adata_collection
    argument
    """
    # TODO need L2Normalization?
    if joint:
        pass


# For now creating a common method - will see if it results in an overhead
def impute_cell_via_neighbors(
    this_pcs_hash: Dict, cross_cell_neighbors: Union[np.array, csr_matrix]
) -> np.ndarray:
    closest_neighbors_indexes, _ = find_neighbors(
        cell_neighbors=cross_cell_neighbors,
        filter_func=generic_filter,
        statement='>0',
    )
    this_neighbors_barcodes = list(this_pcs_hash.keys())[closest_neighbors_indexes]
    cross_neighbors_pcs = [this_pcs_hash.get(x, -1) for x in this_neighbors_barcodes]
    if any([x == -1 for x in cross_neighbors_pcs]):
        raise NeighborsError(
            "cross_modality_impute(): some cell barcodes are not found in "
            "other assays"
        )  # TODO barcodes to the error
    return np.mean(cross_neighbors_pcs, axis=1)


def find_neighbors(cell_neighbors, filter_func, **kwargs):
    if issparse(cell_neighbors):
        closest_neighbors = zip(find(cell_neighbors)[1], find(cell_neighbors)[2])
    else:
        closest_neighbors = filter_large_array(cell_neighbors, filter_func, **kwargs)
    closest_neighbors = sorted(closest_neighbors, key=lambda x: x[1])
    return list(zip(*closest_neighbors))


def barcodes_features_hash(
    barcodes: List[str], features: Union[np.array, csr_matrix]
) -> Dict:
    out = {bc: features[idx] for idx, bc in enumerate(barcodes)}
    return out


def knn_computed(adata: AnnData) -> bool:
    try:
        _ = NeighborsView(adata)
        return True
    except KeyError:
        return False


def modality_product(adata_collection: List[AnnData]) -> List[Tuple[Any, ...]]:
    cross_modalities = list(
        product(range(len(adata_collection)), range(len(adata_collection)))
    )
    return cross_modalities


@nb.jit
def filter_large_array(
    input_array: np.array, filter_func: Callable, **kwargs
) -> np.array:
    counter = 0
    for idx, element in enumerate(input_array):
        if filter_func(element, **kwargs):
            counter += 1
    result = np.empty(counter, dtype=input_array.dtype)
    counter = 0
    for idx, element in enumerate(input_array):
        if filter_func(element, **kwargs):
            result[counter] = idx, element
            counter += 1
    return result


@nb.jit
def generic_filter(element: Any, expression: str):
    return eval(f'{element}{expression}')
