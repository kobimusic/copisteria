import numpy as np

from copista_evidence.train import collate_np


def test_collate_mixed_features():
    a = {"fam": np.zeros(3, np.int16), "feat": np.ones((3, 4), np.float16), "cand": np.ones(3, np.int8)}
    b = {"fam": np.zeros(2, np.int16)}
    out = collate_np([b, a])
    assert out["feat"].shape == (2, 3, 4) and out["feat"][0].sum() == 0 and out["feat"][1].sum() == 12
    assert out["cand"][0].sum() == 0
