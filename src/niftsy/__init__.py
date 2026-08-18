import os

# Guard against OpenBLAS thread-oversubscription segfaults in numpy-heavy
# paths (KNN retrieval, NPGC fitting/sampling). setdefault so an explicit
# override in the environment is still respected.
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

from niftsy.config import GenerationConfig, LLMConfig, PromptConfig, TabularConfig
from niftsy.exceptions import NiftsyError
from niftsy.pipeline import (
    GenerationResult,
    SyntheticDataGenerator,
    generate_synthetic_dataset,
)

__all__ = [
    "SyntheticDataGenerator", "generate_synthetic_dataset", "GenerationResult",
    "GenerationConfig", "TabularConfig", "LLMConfig", "PromptConfig", "NiftsyError",
]
