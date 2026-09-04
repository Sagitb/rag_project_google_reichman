"""Reusable infrastructure for the jiRAG project."""

from .artifacts import calculate_sha256, save_config_snapshot, write_run_metadata
from .config import BASE_CONFIG, set_global_seed
from .data_pipeline import ensure_validated_dataset, load_dataset_contract
from .documents import ensure_rag_documents, load_jsonl
from .eda import ensure_eda_artifacts
from .embedding_selection import ensure_embedding_model_selection
from .generation import (
    GroundedGenerator,
    build_generation_identity,
    ensure_generation_artifact,
    load_generation_contract,
    load_system_prompt,
    validate_citations,
    validate_grounded_result,
)
from .evaluation import (
    BGEReranker,
    ensure_generated_answers,
    ensure_retrieval_contexts,
    evaluate_retrieval_contexts,
    fingerprint,
    load_evaluation_contract,
    load_gold_benchmark,
    score_generation,
    select_retriever,
)
from .paths import build_project_paths, create_artifact_directories
from .retrieval import RetrievalEngine, load_retrieval_contract, results_to_dataframe
from .splitting import ensure_dataset_splits
from .vector_store import ensure_vector_store
from .qlora_data import ensure_qlora_datasets, load_qlora_contract
from .qlora_training import (
    load_adapter_generator,
    load_qlora_processor,
    token_diagnostics,
    train_or_load_adapter,
    training_history_rows,
)
from .qlora_evaluation import (
    comparison_rows,
    failure_analysis,
    score_qlora_generation,
    select_adapter,
)

__all__ = [
    "BASE_CONFIG",
    "build_project_paths",
    "calculate_sha256",
    "create_artifact_directories",
    "ensure_validated_dataset",
    "ensure_eda_artifacts",
    "ensure_embedding_model_selection",
    "ensure_generation_artifact",
    "ensure_dataset_splits",
    "ensure_rag_documents",
    "ensure_vector_store",
    "load_dataset_contract",
    "load_generation_contract",
    "load_jsonl",
    "load_retrieval_contract",
    "load_system_prompt",
    "build_generation_identity",
    "GroundedGenerator",
    "RetrievalEngine",
    "results_to_dataframe",
    "save_config_snapshot",
    "set_global_seed",
    "validate_citations",
    "validate_grounded_result",
    "write_run_metadata",
    "BGEReranker",
    "ensure_generated_answers",
    "ensure_retrieval_contexts",
    "evaluate_retrieval_contexts",
    "fingerprint",
    "load_evaluation_contract",
    "load_gold_benchmark",
    "score_generation",
    "select_retriever",
    "ensure_qlora_datasets",
    "load_qlora_contract",
    "load_adapter_generator",
    "load_qlora_processor",
    "token_diagnostics",
    "train_or_load_adapter",
    "training_history_rows",
    "comparison_rows",
    "failure_analysis",
    "score_qlora_generation",
    "select_adapter",
]
