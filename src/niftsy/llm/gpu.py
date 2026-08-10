from __future__ import annotations

import logging
import subprocess
from dataclasses import dataclass

from niftsy.exceptions import NiftsyError

LOGGER = logging.getLogger(__name__)

_NO_GPU_MESSAGE = "No CUDA GPU detected; use provider='gemini' or 'openai' instead."


@dataclass
class GPUInfo:
    index: int
    total_mb: int
    used_mb: int
    utilization: int


def detect_gpus() -> list[GPUInfo]:
    cmd = [
        "nvidia-smi",
        "--query-gpu=index,memory.total,memory.used,utilization.gpu",
        "--format=csv,noheader,nounits",
    ]

    try:
        output = subprocess.check_output(cmd, text=True).strip()
    except FileNotFoundError as exc:
        raise NiftsyError(_NO_GPU_MESSAGE) from exc
    except subprocess.CalledProcessError as exc:
        raise NiftsyError(_NO_GPU_MESSAGE) from exc

    if not output:
        raise NiftsyError(_NO_GPU_MESSAGE)

    gpus = []
    for line in output.splitlines():
        idx, total, used, util = [x.strip() for x in line.split(",")]
        gpus.append(
            GPUInfo(
                index=int(idx),
                total_mb=int(total),
                used_mb=int(used),
                utilization=int(util),
            )
        )
    return sorted(gpus, key=lambda gpu: gpu.index)


def select_free_gpu(required_utilization: float) -> GPUInfo:
    gpus = detect_gpus()

    free_candidates = [
        gpu for gpu in gpus if gpu.utilization == 0 and gpu.used_mb <= 256
    ]

    if not free_candidates:
        raise NiftsyError("All GPUs are currently busy/full. No free GPU available.")

    selected = free_candidates[0]
    max_usable_mb = int(selected.total_mb * required_utilization)
    LOGGER.info(
        "Selected GPU %s | total=%sMB | used=%sMB | target_utilization=%s (~%sMB usable)",
        selected.index, selected.total_mb, selected.used_mb, required_utilization, max_usable_mb,
    )
    return selected
