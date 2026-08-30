import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist

from niftsy.tabular.knn import knn_retrieval_step1, knn_retrieval_with_text_blocks


def _unit_rows(a):
    norms = np.linalg.norm(a, axis=1, keepdims=True)
    norms[norms == 0.0] = 1.0
    return a / norms


def test_matches_brute_force_reference():
    rng = np.random.default_rng(0)
    cols = ["z0", "z1", "z2", "z3"]
    real_z = pd.DataFrame(rng.normal(size=(50, 4)), columns=cols)
    syn_z = pd.DataFrame(rng.normal(size=(10, 4)), columns=cols)

    nn_idx, nn_dist = knn_retrieval_step1(real_z, syn_z, k=3)

    # knn_retrieval_step1 L2-row-normalizes real_z/syn_z before computing
    # distance (i.e. it ranks by cosine similarity, not raw Euclidean
    # distance) -- the brute-force reference must apply the same
    # normalization to be an unambiguous comparison.
    ref_dist = cdist(_unit_rows(syn_z.to_numpy()), _unit_rows(real_z.to_numpy()))
    ref = np.argsort(ref_dist, axis=1)[:, :3]

    assert nn_idx.shape == (10, 3)
    assert nn_dist.shape == (10, 3)
    # nearest neighbor (rank 0) must match the brute-force reference exactly
    assert (nn_idx[:, 0] == ref[:, 0]).all()


def test_text_blocks_with_zero_beta_matches_tabular_only():
    # beta=0 must make the blended distance degenerate to pure tabular
    # distance -- proves the blending math doesn't distort results when a
    # text block is given zero weight.
    rng = np.random.default_rng(1)
    cols = ["z0", "z1", "z2", "z3"]
    real_z = pd.DataFrame(rng.normal(size=(50, 4)), columns=cols)
    syn_z = pd.DataFrame(rng.normal(size=(10, 4)), columns=cols)

    text_cols = [f"hash_{i}" for i in range(8)]
    real_text = pd.DataFrame(rng.normal(size=(50, 8)), columns=text_cols)
    syn_text = pd.DataFrame(rng.normal(size=(10, 8)), columns=text_cols)

    tabular_only_idx, _ = knn_retrieval_step1(real_z, syn_z, k=3)
    # knn_retrieval_with_text_blocks additionally normalizes the tabular block
    # by its feature dimension, so distances differ by a constant scale factor
    # -- but that doesn't change the ranking, which is what beta=0 must preserve.
    blended_idx, _ = knn_retrieval_with_text_blocks(
        real_z, syn_z, k=3, alpha=1.0, text_blocks=[(real_text, syn_text, 0.0)],
    )

    assert (blended_idx == tabular_only_idx).all()
