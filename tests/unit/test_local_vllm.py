import pytest
from huggingface_hub.utils import GatedRepoError, RepositoryNotFoundError

from niftsy.exceptions import NiftsyError
from niftsy.llm import local_vllm as local_vllm_module


def _make_backend_raising(monkeypatch, exc):
    def _fake_llm(**kwargs):
        raise exc

    monkeypatch.setattr(local_vllm_module, "LLM", _fake_llm)
    with pytest.raises(NiftsyError) as excinfo:
        local_vllm_module.LocalVLLMBackend(model="meta-llama/Llama-3.3-70B-Instruct", gpu_index=0)
    return excinfo.value


def test_gated_repo_error_class_is_detected_via_isinstance(monkeypatch):
    # Message deliberately has none of the marker substrings, isolating the
    # isinstance-based detection path.
    exc = GatedRepoError("403 Client Error for repo XYZ (Request ID: abc123)")
    error = _make_backend_raising(monkeypatch, exc)
    assert "requires Hugging Face authentication" in str(error)
    assert "meta-llama/Llama-3.3-70B-Instruct" in str(error)
    assert "HUGGINGFACE_HUB_TOKEN" in str(error)
    assert error.__cause__ is exc


def test_gated_repo_wrapped_as_plain_oserror_by_transformers(monkeypatch):
    # Mirrors transformers.utils.hub.py's actual wrapping: a plain OSError,
    # not the huggingface_hub exception type.
    exc = OSError(
        "You are trying to access a gated repo.\nMake sure to have access to it at "
        "https://huggingface.co/meta-llama/Llama-3.3-70B-Instruct."
    )
    error = _make_backend_raising(monkeypatch, exc)
    assert "requires Hugging Face authentication" in str(error)


def test_repository_not_found_error_class_is_detected(monkeypatch):
    exc = RepositoryNotFoundError("401 Client Error. Repository Not Found for url: ...")
    error = _make_backend_raising(monkeypatch, exc)
    assert "was not found on Hugging Face Hub" in str(error)
    assert "typos" in str(error)


def test_not_found_wrapped_as_plain_oserror_by_transformers(monkeypatch):
    exc = OSError(
        "Qwen/Qwen2.5-Typo-Instruct is not a local folder and is not a valid model "
        "identifier listed on 'https://huggingface.co/models'"
    )
    error = _make_backend_raising(monkeypatch, exc)
    assert "was not found on Hugging Face Hub" in str(error)


def test_network_failure_during_download(monkeypatch):
    exc = OSError(
        "There was a specific connection error when trying to load "
        "meta-llama/Llama-3.3-70B-Instruct:\nMax retries exceeded with url: ..."
    )
    error = _make_backend_raising(monkeypatch, exc)
    assert "network error" in str(error)
    assert "internet connection" in str(error)


def test_oom_message_unchanged(monkeypatch):
    exc = RuntimeError("CUDA error: out of memory")
    error = _make_backend_raising(monkeypatch, exc)
    assert "does not fit in one GPU" in str(error)


def test_unclassified_error_is_wrapped_not_bare_raised(monkeypatch):
    exc = ValueError("some totally novel internal vllm bug")
    error = _make_backend_raising(monkeypatch, exc)
    assert "Failed to load local model" in str(error)
    assert "ValueError" in str(error)
    assert "some totally novel internal vllm bug" in str(error)
    assert error.__cause__ is exc
