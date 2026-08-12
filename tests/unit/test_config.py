from niftsy.config import GenerationConfig


def test_yaml_round_trip(tmp_path):
    cfg = GenerationConfig()  # defaults
    path = tmp_path / "cfg.yml"
    cfg.to_yaml(path)
    loaded = GenerationConfig.from_yaml(path)
    assert loaded == cfg


def test_bad_provider_rejected():
    import pytest
    with pytest.raises(ValueError):
        GenerationConfig(llm={"provider": "not-a-real-provider"})


def test_dataset_specific_fields_round_trip(tmp_path):
    cfg = GenerationConfig(
        text_columns=["bio", "notes"],
        target_column="income",
        feature_weights={"age": 2.0},
    )
    path = tmp_path / "cfg.yml"
    cfg.to_yaml(path)
    loaded = GenerationConfig.from_yaml(path)
    assert loaded == cfg
