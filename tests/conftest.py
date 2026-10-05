"""pytest configuration for ScoringBench tests."""

import logging
import gc
import pytest
import torch

_CUDA_IS_AVAILABLE = torch.cuda.is_available


def pytest_addoption(parser):
    parser.addoption("--cuda", action="store_true", help="Run scoring pipelines and device-aware tests on CUDA.")


def pytest_configure(config):
    if config.getoption("--cuda") and not _CUDA_IS_AVAILABLE():
        raise pytest.UsageError("--cuda requires an available CUDA device")


@pytest.fixture
def scoring_device(pytestconfig):
    return torch.device("cuda" if pytestconfig.getoption("--cuda") else "cpu")


@pytest.fixture(autouse=True)
def configure_scoring_device(pytestconfig, monkeypatch):
    if not pytestconfig.getoption("--cuda"):
        yield
        return
    from scoringbench.univariate.metrics import _compute_device

    monkeypatch.setattr(torch.cuda, "is_available", _CUDA_IS_AVAILABLE)
    _compute_device.cache_clear()
    assert _compute_device().type == "cuda"
    yield
    _compute_device.cache_clear()

# Configure pytest logging to show INFO and DEBUG levels during test runs
logging.basicConfig(
    level=logging.DEBUG,
    format='%(levelname)-8s | %(name)s | %(message)s'
)

# Optionally reduce noise from specific libraries
logging.getLogger("pytorch_lightning").setLevel(logging.WARNING)
logging.getLogger("lightning_fabric").setLevel(logging.WARNING)
logging.getLogger("pytabkit").setLevel(logging.WARNING)


# ============================================================================
# GPU/CUDA Cleanup Fixtures
# ============================================================================

@pytest.fixture(autouse=True)
def cleanup_gpu():
    """Automatically clean up GPU memory between tests to prevent OOM issues.
    
    This fixture runs after each test to:
    1. Force garbage collection
    2. Empty PyTorch CUDA cache (if available)
    3. Prevent memory fragmentation across tests
    
    This is crucial for tests using deep learning models (e.g., TabPFN)
    where GPU memory can be exhausted when tests run sequentially.
    """
    yield
    
    # Cleanup after test
    gc.collect()
    
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass  # PyTorch/CUDA not available or another error
