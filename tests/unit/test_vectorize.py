import pandas as pd

from niftsy.text.vectorize import prepare_text_vector_features


def test_deterministic_and_shape():
    s = pd.Series(["hello world", "hello there", ""])
    out1 = prepare_text_vector_features(s, hash_dim=8)
    out2 = prepare_text_vector_features(s, hash_dim=8)
    assert out1.shape == (3, 8)
    assert out1.equals(out2)          # deterministic
    assert (out1.iloc[2] == 0).all()  # empty string -> zero vector
