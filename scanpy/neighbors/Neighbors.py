from types import MappingProxyType
from typing import Union, Optional, Any, Mapping

import numpy as np
import scipy
from anndata import AnnData
from scipy.sparse import issparse, csr_matrix

from .OnFlySymMatrix import OnFlySymMatrix
from .neighbors_config import method_type, metric
from .neighbors_utils import (
    backwards_compat_get_full_eval,
    backwards_compat_get_full_X_diffmap,
    get_indices_distances_from_dense_matrix,
    get_sparse_matrix_from_indices_distances_numpy,
    compute_neighbors_rapids,
    compute_neighbors_umap,
    make_forest_dict,
    compute_connectivities_umap,
    get_indices_distances_from_sparse_matrix,
)
from .. import _utils
from .. import logging as logg
from .._compat import Literal
from .._utils import _doc_params, AnyRandom, NeighborsView
from ..tools._utils import _choose_representation, doc_use_rep, doc_n_pcs

RPForestDict = Mapping[str, Mapping[str, np.ndarray]]

_Method = method_type()
_Metric = metric()


class NeighborsError(Exception):
    """Handles exceptions related to neighbors"""


class Neighbors(object):
    """\
    Data represented as graph of nearest neighbors.

    Represent a data matrix as a graph of nearest neighbor relations (edges)
    among data points (nodes).

    Parameters
    ----------
    adata
        Annotated data object.
    n_dcs
        Number of diffusion components to use.
    neighbors_key
        Where to look in .uns and .obsp for neighbors data
    """

    def __init__(
        self,
        adata: AnnData,
        n_dcs: Optional[int] = None,
        neighbors_key: Optional[str] = None,
    ):
        self._adata = adata
        self._init_iroot()
        # use the graph in adata
        info_str = ''
        self.knn: Optional[bool] = None
        self._distances: Union[np.ndarray, csr_matrix, None] = None
        self._connectivities: Union[np.ndarray, csr_matrix, None] = None
        self._transitions_sym: Union[np.ndarray, csr_matrix, None] = None
        self._number_connected_components: Optional[int] = None
        self._rp_forest: Optional[RPForestDict] = None
        if neighbors_key is None:
            neighbors_key = 'neighbors'
        if neighbors_key in adata.uns:
            neighbors = NeighborsView(adata, neighbors_key)
            if 'distances' in neighbors:
                self.knn = issparse(neighbors['distances'])
                self._distances = neighbors['distances']
            if 'connectivities' in neighbors:
                self.knn = issparse(neighbors['connectivities'])
                self._connectivities = neighbors['connectivities']
            if 'rp_forest' in neighbors:
                self._rp_forest = neighbors['rp_forest']
            if 'params' in neighbors:
                self.n_neighbors = neighbors['params']['n_neighbors']
            else:

                def count_nonzero(a: Union[np.ndarray, csr_matrix]) -> int:
                    return a.count_nonzero() if issparse(a) else np.count_nonzero(a)

                # estimating n_neighbors
                if self._connectivities is None:
                    self.n_neighbors = int(
                        count_nonzero(self._distances) / self._distances.shape[0]
                    )
                else:
                    self.n_neighbors = int(
                        count_nonzero(self._connectivities)
                        / self._connectivities.shape[0]
                        / 2
                    )
            info_str += '`.distances` `.connectivities` '
            self._number_connected_components = 1
            if issparse(self._connectivities):
                from scipy.sparse.csgraph import connected_components

                self._connected_components = connected_components(self._connectivities)
                self._number_connected_components = self._connected_components[0]
        if 'X_diffmap' in adata.obsm_keys():
            self._eigen_values = backwards_compat_get_full_eval(adata)
            self._eigen_basis = backwards_compat_get_full_X_diffmap(adata)
            if n_dcs is not None:
                if n_dcs > len(self._eigen_values):
                    raise ValueError(
                        'Cannot instantiate using `n_dcs`={}. '
                        'Compute diffmap/spectrum with more components first.'.format(
                            n_dcs
                        )
                    )
                self._eigen_values = self._eigen_values[:n_dcs]
                self._eigen_basis = self._eigen_basis[:, :n_dcs]
            self.n_dcs = len(self._eigen_values)
            info_str += '`.eigen_values` `.eigen_basis` `.distances_dpt`'
        else:
            self._eigen_values = None
            self._eigen_basis = None
            self.n_dcs = None
        if info_str != '':
            logg.debug(f'    initialized {info_str}')

    @property
    def rp_forest(self) -> Optional[RPForestDict]:
        return self._rp_forest

    @property
    def distances(self) -> Union[np.ndarray, csr_matrix, None]:
        """Distances between data points (sparse matrix)."""
        return self._distances

    @property
    def connectivities(self) -> Union[np.ndarray, csr_matrix, None]:
        """Connectivities between data points (sparse matrix)."""
        return self._connectivities

    @property
    def transitions(self) -> Union[np.ndarray, csr_matrix]:
        """Transition matrix (sparse matrix).

        Is conjugate to the symmetrized transition matrix via::

            self.transitions = self.Z *  self.transitions_sym / self.Z

        where ``self.Z`` is the diagonal matrix storing the normalization of the
        underlying kernel matrix.

        Notes
        -----
        This has not been tested, in contrast to `transitions_sym`.
        """
        if issparse(self.Z):
            Zinv = self.Z.power(-1)
        else:
            Zinv = np.diag(1.0 / np.diag(self.Z))
        return self.Z @ self.transitions_sym @ Zinv

    @property
    def transitions_sym(self) -> Union[np.ndarray, csr_matrix, None]:
        """Symmetrized transition matrix (sparse matrix).

        Is conjugate to the transition matrix via::

            self.transitions_sym = self.Z /  self.transitions * self.Z

        where ``self.Z`` is the diagonal matrix storing the normalization of the
        underlying kernel matrix.
        """
        return self._transitions_sym

    @property
    def eigen_values(self):
        """Eigen values of transition matrix (numpy array)."""
        return self._eigen_values

    @property
    def eigen_basis(self):
        """Eigen basis of transition matrix (numpy array)."""
        return self._eigen_basis

    @property
    def distances_dpt(self):
        """DPT distances (on-fly matrix).

        This is yields [Haghverdi16]_, Eq. 15 from the supplement with the
        extensions of [Wolf19]_, supplement on random-walk based distance
        measures.
        """
        return OnFlySymMatrix(self._get_dpt_row, shape=self._adata.shape)

    def to_igraph(self):
        """Generate igraph from connectivities."""
        return _utils.get_igraph_from_adjacency(self.connectivities)

    @_doc_params(n_pcs=doc_n_pcs, use_rep=doc_use_rep)
    def compute_neighbors(
        self,
        n_neighbors: int = 30,
        knn: bool = True,
        n_pcs: Optional[int] = None,
        use_rep: Optional[str] = None,
        method: _Method = 'umap',
        random_state: AnyRandom = 0,
        write_knn_indices: bool = False,
        metric: _Metric = 'euclidean',
        metric_kwds: Mapping[str, Any] = MappingProxyType({}),
    ) -> None:
        """\
        Compute distances and connectivities of neighbors.

        Parameters
        ----------
        n_neighbors
             Use this number of nearest neighbors.
        knn
             Restrict result to `n_neighbors` nearest neighbors.
        {n_pcs}
        {use_rep}

        Returns
        -------
        Writes sparse graph attributes `.distances` and `.connectivities`.
        Also writes `.knn_indices` and `.knn_distances` if
        `write_knn_indices==True`.
        """
        from sklearn.metrics import pairwise_distances

        start_neighbors = logg.debug('computing neighbors')
        if n_neighbors > self._adata.shape[0]:  # very small datasets
            n_neighbors = 1 + int(0.5 * self._adata.shape[0])
            logg.warning(f'n_obs too small: adjusting to `n_neighbors = {n_neighbors}`')
        if method == 'umap' and not knn:
            raise ValueError('`method = \'umap\' only with `knn = True`.')
        if method == 'rapids' and metric != 'euclidean':
            raise ValueError(
                "`method` 'rapids' only supports the 'euclidean' `metric`."
            )
        if method not in {'umap', 'gauss', 'rapids'}:
            raise ValueError("`method` needs to be 'umap', 'gauss', or 'rapids'.")
        if self._adata.shape[0] >= 10000 and not knn:
            logg.warning('Using high n_obs without `knn=True` takes a lot of memory...')
        # do not use the cached rp_forest
        self._rp_forest = None
        self.n_neighbors = n_neighbors
        self.knn = knn
        X = _choose_representation(self._adata, use_rep=use_rep, n_pcs=n_pcs)
        # neighbor search
        use_dense_distances = (metric == 'euclidean' and X.shape[0] < 8192) or not knn
        if use_dense_distances:
            _distances = pairwise_distances(X, metric=metric, **metric_kwds)
            knn_indices, knn_distances = get_indices_distances_from_dense_matrix(
                _distances, n_neighbors
            )
            if knn:
                self._distances = get_sparse_matrix_from_indices_distances_numpy(
                    knn_indices, knn_distances, X.shape[0], n_neighbors
                )
            else:
                self._distances = _distances
        elif method == 'rapids':
            knn_indices, knn_distances = compute_neighbors_rapids(X, n_neighbors)
        else:
            # non-euclidean case and approx nearest neighbors
            if X.shape[0] < 4096:
                X = pairwise_distances(X, metric=metric, **metric_kwds)
                metric = 'precomputed'
            knn_indices, knn_distances, forest = compute_neighbors_umap(
                X, n_neighbors, random_state, metric=metric, metric_kwds=metric_kwds
            )
            # very cautious here
            try:
                if forest:
                    self._rp_forest = make_forest_dict(forest)
            except Exception:  # TODO catch the correct exception
                pass
        # write indices as attributes
        if write_knn_indices:
            self.knn_indices = knn_indices
            self.knn_distances = knn_distances
        start_connect = logg.debug('computed neighbors', time=start_neighbors)
        if not use_dense_distances or method in {'umap', 'rapids'}:
            # we need self._distances also for method == 'gauss' if we didn't
            # use dense distances
            self._distances, self._connectivities = compute_connectivities_umap(
                knn_indices,
                knn_distances,
                self._adata.shape[0],
                self.n_neighbors,
            )
        # overwrite the umap connectivities if method is 'gauss'
        # self._distances is unaffected by this
        if method == 'gauss':
            self._compute_connectivities_diffmap()
        logg.debug('computed connectivities', time=start_connect)
        self._number_connected_components = 1
        if issparse(self._connectivities):
            from scipy.sparse.csgraph import connected_components

            self._connected_components = connected_components(self._connectivities)
            self._number_connected_components = self._connected_components[0]

    def _compute_connectivities_diffmap(self, density_normalize=True):
        # init distances
        if self.knn:
            Dsq = self._distances.power(2)
            indices, distances_sq = get_indices_distances_from_sparse_matrix(
                Dsq, self.n_neighbors
            )
        else:
            Dsq = np.power(self._distances, 2)
            indices, distances_sq = get_indices_distances_from_dense_matrix(
                Dsq, self.n_neighbors
            )

        # exclude the first point, the 0th neighbor
        indices = indices[:, 1:]
        distances_sq = distances_sq[:, 1:]

        # choose sigma, the heuristic here doesn't seem to make much of a difference,
        # but is used to reproduce the figures of Haghverdi et al. (2016)
        if self.knn:
            # as the distances are not sorted
            # we have decay within the n_neighbors first neighbors
            sigmas_sq = np.median(distances_sq, axis=1)
        else:
            # the last item is already in its sorted position through argpartition
            # we have decay beyond the n_neighbors neighbors
            sigmas_sq = distances_sq[:, -1] / 4
        sigmas = np.sqrt(sigmas_sq)

        # compute the symmetric weight matrix
        if not issparse(self._distances):
            Num = 2 * np.multiply.outer(sigmas, sigmas)
            Den = np.add.outer(sigmas_sq, sigmas_sq)
            W = np.sqrt(Num / Den) * np.exp(-Dsq / Den)
            # make the weight matrix sparse
            if not self.knn:
                mask = W > 1e-14
                W[~mask] = 0
            else:
                # restrict number of neighbors to ~k
                # build a symmetric mask
                mask = np.zeros(Dsq.shape, dtype=bool)
                for i, row in enumerate(indices):
                    mask[i, row] = True
                    for j in row:
                        if i not in set(indices[j]):
                            W[j, i] = W[i, j]
                            mask[j, i] = True
                # set all entries that are not nearest neighbors to zero
                W[~mask] = 0
        else:
            W = (
                Dsq.copy()
            )  # need to copy the distance matrix here; what follows is inplace
            for i in range(len(Dsq.indptr[:-1])):
                row = Dsq.indices[Dsq.indptr[i] : Dsq.indptr[i + 1]]
                num = 2 * sigmas[i] * sigmas[row]
                den = sigmas_sq[i] + sigmas_sq[row]
                W.data[Dsq.indptr[i] : Dsq.indptr[i + 1]] = np.sqrt(num / den) * np.exp(
                    -Dsq.data[Dsq.indptr[i] : Dsq.indptr[i + 1]] / den
                )
            W = W.tolil()
            for i, row in enumerate(indices):
                for j in row:
                    if i not in set(indices[j]):
                        W[j, i] = W[i, j]
            W = W.tocsr()

        self._connectivities = W

    def compute_transitions(self, density_normalize: bool = True):
        """\
        Compute transition matrix.

        Parameters
        ----------
        density_normalize
            The density rescaling of Coifman and Lafon (2006): Then only the
            geometry of the data matters, not the sampled density.

        Returns
        -------
        Makes attributes `.transitions_sym` and `.transitions` available.
        """
        start = logg.info('computing transitions')
        W = self._connectivities
        # density normalization as of Coifman et al. (2005)
        # ensures that kernel matrix is independent of sampling density
        if density_normalize:
            # q[i] is an estimate for the sampling density at point i
            # it's also the degree of the underlying graph
            q = np.asarray(W.sum(axis=0))
            if not issparse(W):
                Q = np.diag(1.0 / q)
            else:
                Q = scipy.sparse.spdiags(1.0 / q, 0, W.shape[0], W.shape[0])
            K = Q @ W @ Q
        else:
            K = W

        # z[i] is the square root of the row sum of K
        z = np.sqrt(np.asarray(K.sum(axis=0)))
        if not issparse(K):
            self.Z = np.diag(1.0 / z)
        else:
            self.Z = scipy.sparse.spdiags(1.0 / z, 0, K.shape[0], K.shape[0])
        self._transitions_sym = self.Z @ K @ self.Z
        logg.info('    finished', time=start)

    def compute_eigen(
        self,
        n_comps: int = 15,
        sym: Optional[bool] = None,
        sort: Literal['decrease', 'increase'] = 'decrease',
    ):
        """\
        Compute eigen decomposition of transition matrix.

        Parameters
        ----------
        n_comps
            Number of eigenvalues/vectors to be computed, set `n_comps = 0` if
            you need all eigenvectors.
        sym
            Instead of computing the eigendecomposition of the assymetric
            transition matrix, computed the eigendecomposition of the symmetric
            Ktilde matrix.

        Returns
        -------
        Writes the following attributes.

        eigen_values : numpy.ndarray
            Eigenvalues of transition matrix.
        eigen_basis : numpy.ndarray
             Matrix of eigenvectors (stored in columns).  `.eigen_basis` is
             projection of data matrix on right eigenvectors, that is, the
             projection on the diffusion components.  these are simply the
             components of the right eigenvectors and can directly be used for
             plotting.
        """
        np.set_printoptions(precision=10)
        if self._transitions_sym is None:
            raise ValueError('Run `.compute_transitions` first.')
        matrix = self._transitions_sym
        # compute the spectrum
        if n_comps == 0:
            evals, evecs = scipy.linalg.eigh(matrix)
        else:
            n_comps = min(matrix.shape[0] - 1, n_comps)
            # ncv = max(2 * n_comps + 1, int(np.sqrt(matrix.shape[0])))
            ncv = None
            which = 'LM' if sort == 'decrease' else 'SM'
            # it pays off to increase the stability with a bit more precision
            matrix = matrix.astype(np.float64)
            evals, evecs = scipy.sparse.linalg.eigsh(
                matrix, k=n_comps, which=which, ncv=ncv
            )
            evals, evecs = evals.astype(np.float32), evecs.astype(np.float32)
        if sort == 'decrease':
            evals = evals[::-1]
            evecs = evecs[:, ::-1]
        logg.info(
            '    eigenvalues of transition matrix\n'
            '    {}'.format(str(evals).replace('\n', '\n    '))
        )
        if self._number_connected_components > len(evals) / 2:
            logg.warning('Transition matrix has many disconnected components!')
        self._eigen_values = evals
        self._eigen_basis = evecs

    def _init_iroot(self):
        self.iroot = None
        # set iroot directly
        if 'iroot' in self._adata.uns:
            if self._adata.uns['iroot'] >= self._adata.n_obs:
                logg.warning(
                    f'Root cell index {self._adata.uns["iroot"]} does not '
                    f'exist for {self._adata.n_obs} samples. It’s ignored.'
                )
            else:
                self.iroot = self._adata.uns['iroot']
            return
        # set iroot via xroot
        xroot = None
        if 'xroot' in self._adata.uns:
            xroot = self._adata.uns['xroot']
        elif 'xroot' in self._adata.var:
            xroot = self._adata.var['xroot']
        # see whether we can set self.iroot using the full data matrix
        if xroot is not None and xroot.size == self._adata.shape[1]:
            self._set_iroot_via_xroot(xroot)

    def _get_dpt_row(self, i):
        mask = None
        if self._number_connected_components > 1:
            label = self._connected_components[1][i]
            mask = self._connected_components[1] == label
        row = sum(
            (
                self.eigen_values[j]
                / (1 - self.eigen_values[j])
                * (self.eigen_basis[i, j] - self.eigen_basis[:, j])
            )
            ** 2
            # account for float32 precision
            for j in range(0, self.eigen_values.size)
            if self.eigen_values[j] < 0.9994
        )
        # thanks to Marius Lange for pointing Alex to this: we will likely remove the
        # contributions from the stationary state below when making backwards compat
        # breaking changes, they originate from an early implementation in 2015 they
        # never seem to have deteriorated results, but also other distance measures (
        # see e.g. PAGA paper) don't have it, which makes sense
        row += sum(
            (self.eigen_basis[i, k] - self.eigen_basis[:, k]) ** 2
            for k in range(0, self.eigen_values.size)
            if self.eigen_values[k] >= 0.9994
        )
        if mask is not None:
            row[~mask] = np.inf
        return np.sqrt(row)

    def _set_pseudotime(self):
        """Return pseudotime with respect to root point."""
        self.pseudotime = self.distances_dpt[self.iroot].copy()
        self.pseudotime /= np.max(self.pseudotime[self.pseudotime < np.inf])

    def _set_iroot_via_xroot(self, xroot):
        """Determine the index of the root cell.

        Given an expression vector, find the observation index that is closest
        to this vector.

        Parameters
        ----------
        xroot : np.ndarray
            Vector that marks the root cell, the vector storing the initial
            condition, only relevant for computing pseudotime.
        """
        if self._adata.shape[1] != xroot.size:
            raise ValueError(
                'The root vector you provided does not have the ' 'correct dimension.'
            )
        # this is the squared distance
        dsqroot = 1e10
        iroot = 0
        for i in range(self._adata.shape[0]):
            diff = self._adata.X[i, :] - xroot
            dsq = diff @ diff
            if dsq < dsqroot:
                dsqroot = dsq
                iroot = i
                if np.sqrt(dsqroot) < 1e-10:
                    break
        logg.debug(f'setting root index to {iroot}')
        if self.iroot is not None and iroot != self.iroot:
            logg.warning(f'Changing index of iroot from {self.iroot} to {iroot}.')
        self.iroot = iroot
