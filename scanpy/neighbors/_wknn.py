from typing import Optional, List, Tuple, Callable, Any, Generator, Dict
import numpy as np
import numba as nb
from anndata import AnnData
from scipy.sparse import issparse, find

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
) -> Generator[Tuple[float, ...]]:
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
    pass


def within_modality_impute(
    pcs: np.array, neighbors: np.array
) -> Generator[Tuple[float, ...]]:
    for idx, this_cell_neighbors in enumerate(neighbors):
        closest_neighbors_indexes = find_neighbors_index(
            cell_neighbors=this_cell_neighbors,
            filter_func=generic_filter,
            statement='>0',
        )
        neighbors_pcs = pcs[idx][closest_neighbors_indexes]
        yield np.mean(neighbors_pcs, axis=1)


def cross_modality_impute(
    neighbors: np.array, this_barcodes: List, cross_hash_pcs: Dict
) -> Generator[Tuple[float, ...]]:
    for idx, this_cell_neighbors in enumerate(neighbors):
        closest_neighbors_indexes = find_neighbors_index(
            cell_neighbors=this_cell_neighbors,
            filter_func=generic_filter,
            statement='>0',
        )
        this_neighbors_barcodes = this_barcodes[closest_neighbors_indexes]
        cross_neighbors_pcs = [
            cross_hash_pcs.get(x, -1) for x in this_neighbors_barcodes
        ]
        if any([x == -1 for x in cross_neighbors_pcs]):
            raise NeighborsError(
                "cross_modality_impute(): some cell barcodes are not "
                "found in other assays"
            )  # TODO barcodes to the error
        yield np.mean(cross_neighbors_pcs, axis=1)


def find_neighbors_index(cell_neighbors, filter_func, **kwargs):
    if issparse(cell_neighbors):
        closest_neighbors_indexes = find(cell_neighbors)[1]
    else:
        closest_neighbors_indexes = filter_large_array(
            cell_neighbors, filter_func, **kwargs
        )
    return closest_neighbors_indexes


def hash_pcs(barcodes: List[str], pcs: np.array) -> Dict:
    out = {bc: pcs[idx] for idx, bc in enumerate(barcodes)}
    return out


def knn_computed(adata: AnnData) -> bool:
    try:
        _ = NeighborsView(adata)
        return True
    except KeyError:
        return False


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
            result[counter] = idx
            counter += 1
    return result


@nb.jit
def generic_filter(element: Any, expression: str):
    return eval(f'{element}{expression}')
