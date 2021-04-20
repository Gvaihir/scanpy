from typing import List

import numpy as np
import pytest
from anndata import AnnData

import scanpy as sc

n_neighbors = 5
key = 'test'


@pytest.fixture
def adata_collection() -> List[AnnData]:
    return [x for x in sc.datasets.pbmc_tea()]  # TODO ingest


def test_neighbors_key_added(adata_collection):
    joint_adata = sc.pp.weighted_neighbors(adata_collection, joint=False, key_added=key)

    conns_key = joint_adata.uns[key]['connectivities_key']
    dists_key = joint_adata.uns[key]['distances_key']

    assert joint_adata.uns['neighbors']['params'] == joint_adata.uns[key]['params']
    assert np.allclose(
        joint_adata.obsp['connectivities'].toarray(),
        joint_adata.obsp[conns_key].toarray(),
    )
    assert np.allclose(
        joint_adata.obsp['distances'].toarray(), joint_adata.obsp[dists_key].toarray()
    )


# test functions with neighbors_key and obsp
@pytest.mark.parametrize('joint', [True, False])
@pytest.mark.parametrize('field', ['neighbors_key', 'obsp'])
def test_neighbors_key_obsp(adata_collection, joint, field):
    joint_adata_key = sc.pp.weighted_neighbors(
        adata_collection, joint=joint, key_added='test'
    )

    joint_adata = sc.pp.weighted_neighbors(
        adata_collection,
        joint=joint,
    )

    if field == 'neighbors_key':
        arg = {field: key}
    else:
        arg = {field: joint_adata_key.uns[key]['connectivities_key']}

    sc.tl.draw_graph(joint_adata_key, layout='fr', random_state=1)
    sc.tl.draw_graph(joint_adata, layout='fr', random_state=1, **arg)

    assert (
        joint_adata.uns['draw_graph']['params']
        == joint_adata_key.uns['draw_graph']['params']
    )
    assert np.allclose(
        joint_adata.obsm['X_draw_graph_fr'], joint_adata_key.obsm['X_draw_graph_fr']
    )

    sc.tl.leiden(joint_adata, random_state=0)
    sc.tl.leiden(joint_adata_key, random_state=0, **arg)

    assert (
        joint_adata.uns['leiden']['params'] == joint_adata_key.uns['leiden']['params']
    )
    assert np.all(joint_adata.obs['leiden'] == joint_adata_key.obs['leiden'])

    sc.tl.louvain(joint_adata, random_state=0)
    sc.tl.louvain(joint_adata_key, random_state=0, **arg)

    assert (
        joint_adata.uns['louvain']['params'] == joint_adata_key.uns['louvain']['params']
    )
    assert np.all(joint_adata.obs['louvain'] == joint_adata_key.obs['louvain'])

    # no obsp in umap, paga
    if field == 'neighbors_key':
        sc.tl.umap(joint_adata, random_state=0)
        sc.tl.umap(joint_adata_key, random_state=0, neighbors_key=key)

        assert (
            joint_adata.uns['umap']['params'] == joint_adata_key.uns['umap']['params']
        )
        assert np.allclose(joint_adata.obsm['X_umap'], joint_adata_key.obsm['X_umap'])

        sc.tl.paga(joint_adata, groups='leiden')
        sc.tl.paga(joint_adata_key, groups='leiden', neighbors_key=key)

        assert np.allclose(
            joint_adata.uns['paga']['connectivities'].toarray(),
            joint_adata_key.uns['paga']['connectivities'].toarray(),
        )
        assert np.allclose(
            joint_adata.uns['paga']['connectivities_tree'].toarray(),
            joint_adata_key.uns['paga']['connectivities_tree'].toarray(),
        )
