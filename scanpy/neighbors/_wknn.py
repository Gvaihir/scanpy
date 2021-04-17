from typing import Optional, List

from anndata import AnnData

from .neighbors_config import method_type, metric, metric_fn
from .._utils import _doc_params
from ..tools._utils import doc_use_rep, doc_n_pcs

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

    Returns
    -------
    New AnnData with w-kNN attribute
    """
    pass


def knn_computed(adata: AnnData) -> bool:
    pass
