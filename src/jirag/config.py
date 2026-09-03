"""Central project configuration and reproducibility utilities."""

from __future__ import annotations

import os
import random
from copy import deepcopy

import numpy as np

try:
    import torch
except ImportError:  # The project runtime supplies PyTorch; utilities remain importable without it.
    torch = None


BASE_CONFIG = {
    "project_name": "jiRAG",
    "project_version": "0.1.0",
    "project_ref": "main",
    "random_seed": 42,
    "dataset_version": "final_1000_v1",
    "dataset_filename": "jira_rag_master_FINAL_1000.csv",
    "target_clean_tickets": 1000,
    "prompt_version": "v1",
    "split_version": "v1_seed42",
    "index_version": "v1",
    "model_version": "base",
    "train_ratio": 0.75,
    "validation_ratio": 0.15,
    "test_ratio": 0.10,
    "stratification_columns": ["family", "solution_type"],
    "default_chunking_strategy": "one_ticket_one_document",
}


def get_base_config() -> dict:
    """Return an isolated mutable copy of the versioned base configuration."""
    return deepcopy(BASE_CONFIG)


def set_global_seed(seed: int) -> None:
    """Seed Python, NumPy and PyTorch for repeatable project experiments."""
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)

    if torch is None:
        return

    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True, warn_only=True)
