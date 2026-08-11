import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist

from niftsy.tabular.knn import knn_retrieval_step1


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
