import pandas as pd

from niftsy.tabular.npgc import NPGC_special


def _sample_df():
    return pd.DataFrame({
        "age": [22, 35, 41, 29, 60, 33, 45, 27],
        "income": [30000, 52000, 61000, 40000, 90000, 48000, 70000, 39000],
        "category": ["a", "b", "a", "c", "b", "a", "c", "b"],
    })


def test_fit_sample_shape_and_dtypes():
    df = _sample_df()
    model = NPGC_special(epsilon=1.0, enforce_min_max_values=True)
    model.fit(df)
    synth, z_correlated = model.sample(num_rows=20, seed=0)
    assert len(synth) == 20
    assert set(synth.columns) == set(df.columns)
    assert synth["category"].isin(df["category"].unique()).all()
    assert z_correlated.shape == (20, len(df.columns))


def test_save_load_round_trip(tmp_path):
    df = _sample_df()
    model = NPGC_special(epsilon=1.0)
    model.fit(df)
    path = tmp_path / "model.pkl"
    model.save(path)

    loaded = NPGC_special()   # load() is an instance method, not a factory
    loaded.load(path)
    synth, _ = loaded.sample(num_rows=5, seed=0)
    assert len(synth) == 5
