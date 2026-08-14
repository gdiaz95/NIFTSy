import niftsy.cli as cli_module
from niftsy.cli import _build_model_menu


def test_build_model_menu_lists_local_before_recommended_without_duplicates():
    options = _build_model_menu(
        default_model="gemini-3.1-flash-lite-preview",
        local_models=[("gpt2-large", "3.3G"), ("Qwen/Qwen2.5-7B-Instruct", "15.2G")],
    )
    assert options[0] == ("gpt2-large  (3.3G, already downloaded)", "gpt2-large")
    assert options[1] == (
        "Qwen/Qwen2.5-7B-Instruct  (15.2G, already downloaded)",
        "Qwen/Qwen2.5-7B-Instruct",
    )
    models = [m for _, m in options]
    assert models[-2:] == ["gemini-3.1-flash-lite-preview", "Qwen/Qwen2.5-1.5B-Instruct"]


def test_build_model_menu_dedupes_default_model_already_cached_locally():
    options = _build_model_menu(
        default_model="gemini-3.1-flash-lite-preview",
        local_models=[("gemini-3.1-flash-lite-preview", "1G")],
    )
    models = [m for _, m in options]
    assert models.count("gemini-3.1-flash-lite-preview") == 1


def test_build_model_menu_dedupes_recommended_local_already_cached():
    options = _build_model_menu(
        default_model="gemini-3.1-flash-lite-preview",
        local_models=[("Qwen/Qwen2.5-1.5B-Instruct", "3.1G")],
    )
    models = [m for _, m in options]
    assert models.count("Qwen/Qwen2.5-1.5B-Instruct") == 1


def test_build_model_menu_empty_local_cache_still_offers_recommendations():
    options = _build_model_menu(default_model="gemini-3.1-flash-lite-preview", local_models=[])
    assert [m for _, m in options] == [
        "gemini-3.1-flash-lite-preview",
        "Qwen/Qwen2.5-1.5B-Instruct",
    ]


class _FakeRepo:
    def __init__(self, repo_id, repo_type, size_on_disk_str):
        self.repo_id, self.repo_type, self.size_on_disk_str = repo_id, repo_type, size_on_disk_str


class _FakeCacheInfo:
    def __init__(self, repos):
        self.repos = repos


def test_local_cached_models_filters_to_model_type_and_sorts(monkeypatch):
    fake_info = _FakeCacheInfo([
        _FakeRepo("Qwen/Qwen2.5-7B-Instruct", "model", "15.2G"),
        _FakeRepo("some/dataset", "dataset", "1G"),
        _FakeRepo("gpt2-large", "model", "3.3G"),
    ])
    monkeypatch.setattr(cli_module, "scan_cache_dir", lambda: fake_info)
    assert cli_module._local_cached_models() == [
        ("gpt2-large", "3.3G"),
        ("Qwen/Qwen2.5-7B-Instruct", "15.2G"),
    ]


def test_local_cached_models_returns_empty_list_on_scan_failure(monkeypatch):
    def _raise():
        raise RuntimeError("Cache directory not found")

    monkeypatch.setattr(cli_module, "scan_cache_dir", _raise)
    assert cli_module._local_cached_models() == []


def test_run_setup_reports_clean_error_on_eof_instead_of_crashing(monkeypatch, capsys):
    # Regression test: piped/redirected stdin running out mid-wizard (e.g. an
    # undercounted input script, or a real Ctrl-D) used to surface as a raw
    # EOFError traceback instead of the project's usual clean error message.
    def _raise_eof(_args):
        raise EOFError()

    monkeypatch.setattr(cli_module, "_run_setup_impl", _raise_eof)
    exit_code = cli_module._run_setup(args=None)
    assert exit_code == 1
    assert "unexpected end of input" in capsys.readouterr().err
