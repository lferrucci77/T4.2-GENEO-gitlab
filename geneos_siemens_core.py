import os
import time
from functools import lru_cache
from dataclasses import dataclass, field
from threading import Lock
from typing import List, Tuple, Dict, Any, Sequence
from concurrent.futures import ThreadPoolExecutor

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, minimize

# =============================================================================
# Core numerical configuration
# =============================================================================
# CRITICAL_THRESHOLD = 55.0
# Historical threshold that binarized the target as
# c(t)=1 if critical_value < threshold else 0.
# It is intentionally disabled so GENEO now approximates the original
# critical function loaded from the target column in every execution mode.
GREEDY_LIMIT = 16           # Use the exact non-greedy assignment up to this many experiments
WEIGHTS_SUM = 1.0
TRUST_CONSTR_MAXITER = 1500
TRUST_CONSTR_GTOL = 1e-10
TRUST_CONSTR_XTOL = 1e-12
TRUST_CONSTR_BARRIER_TOL = 1e-14
TRUST_CONSTR_INITIAL_BARRIER_PARAMETER = 1e-8
TRUST_CONSTR_INITIAL_BARRIER_TOLERANCE = 1e-8
CENTERED_L2_NORM_EPS = 1e-12
FEATURE_SELECTION_INITIAL_FEATURES = 1
FEATURE_SELECTION_STOP_RELATIVE_IMPROVEMENT = 0.005
FEATURE_SELECTION_PRUNING_MAX_RELATIVE_LOSS_INCREASE = 0.05
FEATURE_SELECTION_FORWARD_REQUIRED_RELATIVE_IMPROVEMENT_PER_FEATURE = 0.001
FEATURE_SELECTION_FORWARD_MEDIUM_REJECT_MARGIN_PER_FEATURE = 0.005
FEATURE_SELECTION_FORWARD_LARGE_REJECT_MARGIN_PER_FEATURE = 0.05
FEATURE_SELECTION_LOW_RELATIVE_IMPROVEMENT = 0.15
FEATURE_SELECTION_HIGH_RELATIVE_IMPROVEMENT = 0.3
FEATURE_SELECTION_PROXY_MAX_RELATIVE_WORSENING_ADD = 0.15
FEATURE_SELECTION_PROXY_MAX_RELATIVE_WORSENING_REMOVE = 0.9
FEATURE_SELECTION_PROXY_STRONG_ACCEPT_RELATIVE_IMPROVEMENT = 0.015
FEATURE_SELECTION_DELAYED_LOW_RELATIVE_IMPROVEMENT = 0.01
FEATURE_SELECTION_DELAYED_HIGH_RELATIVE_IMPROVEMENT = 0.05
FEATURE_SELECTION_DELAYED_MEDIUM_RELATIVE_WORSENING = 0.001
FEATURE_SELECTION_DELAYED_LARGE_RELATIVE_WORSENING = 0.05
LEVEL2_ACTIVE_FEATURE_COPY_MAX_BYTES = 256 * 1024 * 1024
LEVEL2_GRAM_CACHE_MAX_BYTES = 256 * 1024 * 1024
LEVEL2_GRAM_CACHE_INCREMENTAL_MIN_POOL_SIZE = 64

@dataclass
class GeneoRuntimeState:
    # This is the exact in-memory state needed by Method 1 to continue
    # training without rereading past batches.
    target_column_name: str
    surviving_feature_names: Tuple[str, ...]
    surviving_mean_correlations: np.ndarray
    frozen_select_all_features: bool
    sxx: np.ndarray
    sxc: np.ndarray
    scc: float
    total_rows_seen: int
    last_weights: np.ndarray
    # This is the 1-based output folder index for the latest completed request.
    completed_requests_count: int


@dataclass
class GeneoLevel1Statistics:
    mean_correlations: np.ndarray


@dataclass
class GeneoComputedBatchStatistics:
    level1_statistics: GeneoLevel1Statistics | None
    invalid_feature_mask: np.ndarray
    level2_feature_scales_by_experiment: np.ndarray
    worker_count: int = 1
    experiment_blocks: List[List[int]] = field(default_factory=list)


class GeneoCoreValidationError(ValueError):
    """Raised when the numerical core receives an invalid batch."""


class GeneoCoreExecutionError(RuntimeError):
    """Raised when the numerical core cannot complete a batch update."""


def freeze_runtime_array(array: np.ndarray) -> np.ndarray:
    array.setflags(write=False)
    return array


def build_runtime_state(
    target_column_name: str,
    surviving_feature_names: Sequence[str],
    surviving_mean_correlations: np.ndarray,
    frozen_select_all_features: bool,
    sxx: np.ndarray,
    sxc: np.ndarray,
    scc: float,
    total_rows_seen: int,
    last_weights: np.ndarray,
    completed_requests_count: int
) -> GeneoRuntimeState:
    return GeneoRuntimeState(
        target_column_name=target_column_name,
        surviving_feature_names=tuple(surviving_feature_names),
        surviving_mean_correlations=freeze_runtime_array(surviving_mean_correlations),
        frozen_select_all_features=bool(frozen_select_all_features),
        sxx=freeze_runtime_array(sxx),
        sxc=freeze_runtime_array(sxc),
        scc=scc,
        total_rows_seen=total_rows_seen,
        last_weights=freeze_runtime_array(last_weights),
        completed_requests_count=completed_requests_count
    )


@dataclass
class GeneoPreparedBatch:
    features_data: np.ndarray
    critical_data: np.ndarray
    experiment_ranges: List[Tuple[int, int]]
    worker_count: int
    experiment_blocks: List[List[int]]
    total_rows: int
    level1_statistics: GeneoLevel1Statistics | None
    invalid_feature_mask: np.ndarray
    level2_feature_scales_by_experiment: np.ndarray


@dataclass
class GeneoCoreExecutionState:
    runtime_state: GeneoRuntimeState | None
    feature_names: Sequence[str]
    features_data: np.ndarray | None
    level1_features_data: np.ndarray | None
    critical_data: np.ndarray | None
    mean_correlations: np.ndarray
    weights: np.ndarray
    solver_started_at: float
    solver_seconds: float
    debug_seconds: float
    request_index: int
    continual_learning_enabled: bool
    is_initial_training: bool
    response_status: str
    optimizer_success: bool
    optimizer_status: int | None
    optimizer_message: str
    optimizer_total_loss: float | None


@dataclass
class EvaluatedFeatureSet:
    active_indices: Tuple[int, ...]
    weights: np.ndarray
    solver_started_at: float
    solver_seconds: float
    optimizer_success: bool
    optimizer_status: int | None
    optimizer_message: str
    optimizer_total_loss: float | None

    @property
    def feature_count(self) -> int:
        return len(self.active_indices)


@dataclass
class Level2GramCache:
    pool_indices: np.ndarray
    positions_by_feature: np.ndarray
    experiment_grams: np.ndarray

# =============================================================================
# Experiment partitioning and preprocessing helpers
# =============================================================================
def build_experiment_ranges(experiment_lengths: List[int]) -> List[Tuple[int, int]]:
    experiment_ranges: List[Tuple[int, int]] = []
    start_row = 0

    # Each request may contain multiple experiments with variable lengths.
    # We map them once to row ranges and then process the whole batch.
    for experiment_length in experiment_lengths:
        end_row = start_row + experiment_length
        experiment_ranges.append((start_row, end_row))
        start_row = end_row

    return experiment_ranges


def normalize_level1_experiment_slice(
    critical_slice: np.ndarray,
    feature_slice: np.ndarray,
    feature_norms: np.ndarray,
    zero_norm_feature_mask: np.ndarray,
    experiment_index: int
) -> None:
    # Main semantics: first center the critical kernel and every feature within
    # the current experiment, then L2-normalize them in place so level 2 reuses
    # the same transformed signals.
    critical_slice -= np.mean(critical_slice, dtype=np.float64)
    feature_slice -= np.mean(feature_slice, axis=0, dtype=np.float64)

    critical_norm = np.linalg.norm(critical_slice)
    if critical_norm <= CENTERED_L2_NORM_EPS:
        raise GeneoCoreValidationError(
            "Target column has zero or near-zero L2 norm after centering within experiment "
            f"{experiment_index + 1}; the critical kernel cannot be normalized"
        )
    np.einsum("ij,ij->j", feature_slice, feature_slice, dtype=np.float64, out=feature_norms)
    np.sqrt(feature_norms, out=feature_norms)
    np.less_equal(feature_norms, CENTERED_L2_NORM_EPS, out=zero_norm_feature_mask)

    if np.any(zero_norm_feature_mask):
        valid_feature_mask = ~zero_norm_feature_mask
        if np.any(valid_feature_mask):
            feature_slice[:, valid_feature_mask] /= feature_norms[valid_feature_mask]
        feature_slice[:, zero_norm_feature_mask] = 0.0
    else:
        feature_slice /= feature_norms
    critical_slice /= critical_norm


def fill_scalar_level2_feature_scales(
    feature_target_products: np.ndarray,
    out: np.ndarray
) -> None:
    np.divide(feature_target_products, 3.0, out=out)


def process_experiment_block(
    exp_block: Sequence[int],
    experiment_ranges: List[Tuple[int, int]],
    critical_data: np.ndarray,
    features_data: np.ndarray,
    level2_feature_scales_by_experiment: np.ndarray,
    num_features: int,
    compute_level1: bool,
) -> GeneoComputedBatchStatistics:
    # LVL1 scores each feature through the per-experiment pre-normalized LVL2
    # vector z^(0)_{e,j}(t). Since every normalized feature slice has unit L2
    # norm, ||z^(0)_{e,j}||_2 simplifies to |alpha_j|, so the worker can
    # accumulate that value directly and the main thread only needs to average
    # across experiments. The same pass also stores the per-experiment alpha_j
    # values. The final LVL2 z slices are built later, after LVL1 pruning, so
    # D_j is computed only on the surviving features.
    # The in-place L2 normalization still runs even when compute_level1=False.
    # Incremental updates reuse the same alpha collection and then finalize z
    # on the frozen surviving feature set.
    local_mean_correlations = (
        np.zeros(num_features, dtype=np.float64)
        if compute_level1
        else None
    )
    # True only for features that remain zero-norm in every experiment of this
    # worker block. Features that are zero only in some experiments contribute
    # zero there, but stay eligible globally.
    local_invalid_feature_mask = np.ones(num_features, dtype=bool)

    # Reused scratch buffers
    feature_norms = np.empty(num_features, dtype=np.float64)
    feature_target_products = np.empty(num_features, dtype=np.float64)
    zero_norm_feature_mask = np.empty(num_features, dtype=bool)

    for experiment_idx in exp_block:
        start, end = experiment_ranges[experiment_idx]
        critical_slice = critical_data[start:end]
        feature_slice = features_data[start:end, :]

        normalize_level1_experiment_slice(
            critical_slice,
            feature_slice,
            feature_norms,
            zero_norm_feature_mask,
            experiment_idx
        )
        local_invalid_feature_mask &= zero_norm_feature_mask
        np.matmul(feature_slice.T, critical_slice, out=feature_target_products)

        # LVL2 first builds z^(0)_j(t) = alpha_j x_j(t), with
        # alpha_j = m_j / 3 and m_j = x_j^T c. Store alpha now, then
        # normalize the experiment block after LVL1 pruning.
        fill_scalar_level2_feature_scales(
            feature_target_products,
            feature_target_products
        )
        level2_feature_scales_by_experiment[experiment_idx, :] = feature_target_products
        if local_mean_correlations is not None:
            np.abs(feature_target_products, out=feature_norms)
            local_mean_correlations += feature_norms

    return GeneoComputedBatchStatistics(
        level1_statistics=(
            None
            if local_mean_correlations is None
            else GeneoLevel1Statistics(
                mean_correlations=local_mean_correlations
            )
        ),
        invalid_feature_mask=local_invalid_feature_mask,
        level2_feature_scales_by_experiment=level2_feature_scales_by_experiment
    )


def build_experiment_worker_blocks(
    experiment_ranges: List[Tuple[int, int]],
    worker_count: int | None = None
) -> Tuple[int, List[List[int]]]:
    num_experiments = len(experiment_ranges)
    resolved_worker_count = (
        min(os.cpu_count() or 1, num_experiments)
        if worker_count is None
        else worker_count
    )
    return (
        resolved_worker_count,
        assign_experiment_blocks(experiment_ranges, resolved_worker_count)
    )


def compute_batch_statistics(
    experiment_ranges: List[Tuple[int, int]],
    critical_data: np.ndarray,
    features_data: np.ndarray,
    compute_level1: bool = True,
    worker_count: int | None = None,
    experiment_blocks: List[List[int]] | None = None,
) -> GeneoComputedBatchStatistics:
    num_experiments = len(experiment_ranges)
    if experiment_blocks is None:
        worker_count, experiment_blocks = build_experiment_worker_blocks(
            experiment_ranges,
            worker_count
        )
    elif worker_count is None:
        worker_count = len(experiment_blocks)
    num_features = features_data.shape[1]
    level2_feature_scales_by_experiment = np.zeros(
        (num_experiments, num_features),
        dtype=np.float64
    )

    def process_block(block: Sequence[int]) -> GeneoComputedBatchStatistics:
        return process_experiment_block(
            block,
            experiment_ranges,
            critical_data,
            features_data,
            level2_feature_scales_by_experiment,
            num_features,
            compute_level1,
        )

    computed_batch_statistics = GeneoComputedBatchStatistics(
        level1_statistics=(
            None
            if not compute_level1
            else GeneoLevel1Statistics(
                mean_correlations=np.zeros(num_features, dtype=np.float64)
            )
        ),
        invalid_feature_mask=np.ones(num_features, dtype=bool),
        level2_feature_scales_by_experiment=level2_feature_scales_by_experiment,
        worker_count=worker_count,
        experiment_blocks=experiment_blocks
    )

    def merge_partial_result(partial_result: GeneoComputedBatchStatistics) -> None:
        computed_batch_statistics.invalid_feature_mask &= partial_result.invalid_feature_mask
        if compute_level1:
            computed_batch_statistics.level1_statistics.mean_correlations += (
                partial_result.level1_statistics.mean_correlations
            )

    if worker_count == 1:
        merge_partial_result(process_block(experiment_blocks[0]))
    else:
        with ThreadPoolExecutor(max_workers=worker_count) as executor:
            partial_results = executor.map(process_block, experiment_blocks)
            for partial_result in partial_results:
                merge_partial_result(partial_result)

    if compute_level1:
        computed_batch_statistics.level1_statistics.mean_correlations /= num_experiments

    return computed_batch_statistics


def active_indices_cover_all_columns(
    num_features: int,
    active_indices: np.ndarray
) -> bool:
    return (
        active_indices.shape[0] == num_features
        and np.array_equal(active_indices, np.arange(num_features, dtype=active_indices.dtype))
    )


def estimate_array_bytes(shape: Tuple[int, ...]) -> int:
    item_count = 1
    for dimension in shape:
        item_count *= int(dimension)
    return item_count * np.dtype(np.float64).itemsize


def select_active_feature_columns_for_statistics(
    base_features_data: np.ndarray,
    active_indices: np.ndarray
) -> np.ndarray | None:
    num_rows, num_features = base_features_data.shape
    if active_indices_cover_all_columns(num_features, active_indices):
        return base_features_data

    active_copy_bytes = estimate_array_bytes((num_rows, int(active_indices.shape[0])))
    if active_copy_bytes > LEVEL2_ACTIVE_FEATURE_COPY_MAX_BYTES:
        return None

    return base_features_data[:, active_indices]


def copy_active_feature_columns(
    base_features_data: np.ndarray,
    active_indices: np.ndarray
) -> np.ndarray:
    if active_indices_cover_all_columns(base_features_data.shape[1], active_indices):
        return base_features_data.copy()
    return base_features_data[:, active_indices]


def fill_level2_gram_cache_for_experiment_block(
    base_features_data: np.ndarray,
    experiment_ranges: List[Tuple[int, int]],
    pool_indices: np.ndarray,
    experiment_grams: np.ndarray,
    experiment_block: Sequence[int]
) -> None:
    for experiment_idx in experiment_block:
        start, end = experiment_ranges[experiment_idx]
        feature_slice = base_features_data[start:end, pool_indices]
        np.matmul(feature_slice.T, feature_slice, out=experiment_grams[experiment_idx])


def maybe_build_level2_gram_cache(
    base_features_data: np.ndarray,
    experiment_ranges: List[Tuple[int, int]],
    pool_indices: np.ndarray,
    worker_count: int,
    experiment_blocks: List[List[int]]
) -> Level2GramCache | None:
    pool_size = int(pool_indices.shape[0])
    cache_bytes = estimate_array_bytes((len(experiment_ranges), pool_size, pool_size))
    if cache_bytes > LEVEL2_GRAM_CACHE_MAX_BYTES:
        return None

    experiment_grams = np.empty(
        (len(experiment_ranges), pool_size, pool_size),
        dtype=np.float64
    )
    if worker_count == 1:
        fill_level2_gram_cache_for_experiment_block(
            base_features_data,
            experiment_ranges,
            pool_indices,
            experiment_grams,
            experiment_blocks[0]
        )
    else:
        with ThreadPoolExecutor(max_workers=worker_count) as executor:
            futures = [
                executor.submit(
                    fill_level2_gram_cache_for_experiment_block,
                    base_features_data,
                    experiment_ranges,
                    pool_indices,
                    experiment_grams,
                    experiment_block
                )
                for experiment_block in experiment_blocks
            ]
            for future in futures:
                future.result()

    positions_by_feature = np.full(base_features_data.shape[1], -1, dtype=np.int64)
    positions_by_feature[pool_indices] = np.arange(pool_size, dtype=np.int64)
    return Level2GramCache(
        pool_indices=pool_indices.copy(),
        positions_by_feature=positions_by_feature,
        experiment_grams=experiment_grams
    )


def maybe_expand_level2_gram_cache(
    existing_cache: Level2GramCache | None,
    base_features_data: np.ndarray,
    experiment_ranges: List[Tuple[int, int]],
    pool_indices: np.ndarray,
    worker_count: int,
    experiment_blocks: List[List[int]]
) -> Level2GramCache | None:
    if existing_cache is None:
        return maybe_build_level2_gram_cache(
            base_features_data,
            experiment_ranges,
            pool_indices,
            worker_count,
            experiment_blocks
        )

    pool_size = int(pool_indices.shape[0])
    existing_pool_size = int(existing_cache.pool_indices.shape[0])
    if existing_pool_size >= pool_size:
        return existing_cache

    if (
        pool_size < LEVEL2_GRAM_CACHE_INCREMENTAL_MIN_POOL_SIZE
        or existing_pool_size < LEVEL2_GRAM_CACHE_INCREMENTAL_MIN_POOL_SIZE
    ):
        return maybe_build_level2_gram_cache(
            base_features_data,
            experiment_ranges,
            pool_indices,
            worker_count,
            experiment_blocks
        )

    if not np.array_equal(existing_cache.pool_indices, pool_indices[:existing_pool_size]):
        return maybe_build_level2_gram_cache(
            base_features_data,
            experiment_ranges,
            pool_indices,
            worker_count,
            experiment_blocks
        )

    cache_bytes = estimate_array_bytes((len(experiment_ranges), pool_size, pool_size))
    if cache_bytes > LEVEL2_GRAM_CACHE_MAX_BYTES:
        return None

    added_indices = pool_indices[existing_pool_size:]
    added_count = int(added_indices.shape[0])
    experiment_grams = np.empty(
        (len(experiment_ranges), pool_size, pool_size),
        dtype=np.float64
    )
    experiment_grams[:, :existing_pool_size, :existing_pool_size] = (
        existing_cache.experiment_grams
    )

    def expand_experiment_block(experiment_block: Sequence[int]) -> None:
        cross_gram = np.empty((existing_pool_size, added_count), dtype=np.float64)
        added_gram = np.empty((added_count, added_count), dtype=np.float64)
        for experiment_idx in experiment_block:
            start, end = experiment_ranges[experiment_idx]
            existing_feature_slice = base_features_data[start:end, existing_cache.pool_indices]
            added_feature_slice = base_features_data[start:end, added_indices]
            np.matmul(existing_feature_slice.T, added_feature_slice, out=cross_gram)
            np.matmul(added_feature_slice.T, added_feature_slice, out=added_gram)
            experiment_grams[experiment_idx, :existing_pool_size, existing_pool_size:] = cross_gram
            experiment_grams[experiment_idx, existing_pool_size:, :existing_pool_size] = cross_gram.T
            experiment_grams[experiment_idx, existing_pool_size:, existing_pool_size:] = added_gram

    if worker_count == 1:
        expand_experiment_block(experiment_blocks[0])
    else:
        with ThreadPoolExecutor(max_workers=worker_count) as executor:
            futures = [
                executor.submit(expand_experiment_block, experiment_block)
                for experiment_block in experiment_blocks
            ]
            for future in futures:
                future.result()

    positions_by_feature = np.full(base_features_data.shape[1], -1, dtype=np.int64)
    positions_by_feature[pool_indices] = np.arange(pool_size, dtype=np.int64)
    return Level2GramCache(
        pool_indices=pool_indices.copy(),
        positions_by_feature=positions_by_feature,
        experiment_grams=experiment_grams
    )


def compute_level2_denominators_by_experiment(
    level2_feature_scales_by_experiment: np.ndarray,
    active_indices: np.ndarray
) -> np.ndarray:
    if active_indices_cover_all_columns(
        level2_feature_scales_by_experiment.shape[1],
        active_indices
    ):
        return np.max(np.abs(level2_feature_scales_by_experiment), axis=1)

    denominators = np.zeros(level2_feature_scales_by_experiment.shape[0], dtype=np.float64)
    for feature_index in active_indices:
        np.maximum(
            denominators,
            np.abs(level2_feature_scales_by_experiment[:, feature_index]),
            out=denominators
        )
    return denominators


def compute_level2_active_beta(
    active_alpha: np.ndarray,
    experiment_level2_denominator: float
) -> np.ndarray:
    active_feature_count = active_alpha.shape[0]
    return (active_feature_count * active_alpha) / experiment_level2_denominator


def compute_level2_active_batch_statistics_from_gram_cache(
    gram_cache: Level2GramCache,
    level2_feature_scales_by_experiment: np.ndarray,
    active_indices: np.ndarray,
    level2_denominators_by_experiment: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, float]:
    active_positions = gram_cache.positions_by_feature[active_indices]
    num_active_features = int(active_indices.shape[0])
    active_sxx = np.zeros((num_active_features, num_active_features), dtype=np.float64)
    active_sxc = np.zeros(num_active_features, dtype=np.float64)
    active_scc = float(gram_cache.experiment_grams.shape[0])
    temp_sxx = np.empty((num_active_features, num_active_features), dtype=np.float64)
    active_position_index = np.ix_(active_positions, active_positions)

    for experiment_idx in range(gram_cache.experiment_grams.shape[0]):
        active_alpha = level2_feature_scales_by_experiment[experiment_idx, active_indices]
        experiment_level2_denominator = level2_denominators_by_experiment[experiment_idx]
        if experiment_level2_denominator <= CENTERED_L2_NORM_EPS:
            continue

        active_beta = compute_level2_active_beta(
            active_alpha,
            experiment_level2_denominator
        )
        active_sxc += 3.0 * active_alpha * active_beta

        temp_sxx[:, :] = gram_cache.experiment_grams[experiment_idx][active_position_index]
        temp_sxx *= active_beta[:, None]
        temp_sxx *= active_beta[None, :]
        active_sxx += temp_sxx

    return active_sxx, active_sxc, active_scc


def build_level2_active_batch_arrays(
    base_features_data: np.ndarray,
    experiment_ranges: List[Tuple[int, int]],
    level2_feature_scales_by_experiment: np.ndarray,
    active_indices: np.ndarray,
    level2_denominators_by_experiment: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    active_features_data = copy_active_feature_columns(base_features_data, active_indices)
    active_sxc = np.zeros(active_indices.shape[0], dtype=np.float64)

    for experiment_idx, (start, end) in enumerate(experiment_ranges):
        active_alpha = level2_feature_scales_by_experiment[experiment_idx, active_indices]
        experiment_level2_denominator = level2_denominators_by_experiment[experiment_idx]
        if experiment_level2_denominator <= CENTERED_L2_NORM_EPS:
            active_features_data[start:end, :] = 0.0
            continue

        active_beta = compute_level2_active_beta(
            active_alpha,
            experiment_level2_denominator
        )
        active_features_data[start:end, :] *= active_beta
        active_sxc += 3.0 * active_alpha * active_beta

    return active_features_data, active_sxc


def compute_level2_active_batch_statistics_for_experiment_block(
    base_features_data: np.ndarray,
    active_features_data: np.ndarray | None,
    experiment_ranges: List[Tuple[int, int]],
    level2_feature_scales_by_experiment: np.ndarray,
    active_indices: np.ndarray,
    level2_denominators_by_experiment: np.ndarray,
    experiment_block: Sequence[int]
) -> Tuple[np.ndarray, np.ndarray, float]:
    num_active_features = active_indices.shape[0]
    active_sxx = np.zeros((num_active_features, num_active_features), dtype=np.float64)
    active_sxc = np.zeros(num_active_features, dtype=np.float64)
    active_scc = float(len(experiment_block))
    temp_sxx = np.empty((num_active_features, num_active_features), dtype=np.float64)

    for experiment_idx in experiment_block:
        start, end = experiment_ranges[experiment_idx]
        active_alpha = level2_feature_scales_by_experiment[experiment_idx, active_indices]
        experiment_level2_denominator = level2_denominators_by_experiment[experiment_idx]
        if experiment_level2_denominator <= CENTERED_L2_NORM_EPS:
            continue

        active_beta = compute_level2_active_beta(
            active_alpha,
            experiment_level2_denominator
        )
        active_sxc += 3.0 * active_alpha * active_beta

        feature_slice = (
            active_features_data[start:end, :]
            if active_features_data is not None
            else base_features_data[start:end, active_indices]
        )
        np.matmul(feature_slice.T, feature_slice, out=temp_sxx)
        temp_sxx *= active_beta[:, None]
        temp_sxx *= active_beta[None, :]
        active_sxx += temp_sxx

    return active_sxx, active_sxc, active_scc


def compute_level2_active_batch_statistics(
    base_features_data: np.ndarray,
    experiment_ranges: List[Tuple[int, int]],
    level2_feature_scales_by_experiment: np.ndarray,
    active_indices: np.ndarray,
    level2_denominators_by_experiment: np.ndarray,
    worker_count: int,
    experiment_blocks: List[List[int]]
) -> Tuple[np.ndarray, np.ndarray, float]:
    active_features_data = select_active_feature_columns_for_statistics(
        base_features_data,
        active_indices
    )

    def process_block(experiment_block: Sequence[int]) -> Tuple[np.ndarray, np.ndarray, float]:
        return compute_level2_active_batch_statistics_for_experiment_block(
            base_features_data,
            active_features_data,
            experiment_ranges,
            level2_feature_scales_by_experiment,
            active_indices,
            level2_denominators_by_experiment,
            experiment_block
        )

    if worker_count == 1:
        return process_block(experiment_blocks[0])

    num_active_features = active_indices.shape[0]
    active_sxx = np.zeros((num_active_features, num_active_features), dtype=np.float64)
    active_sxc = np.zeros(num_active_features, dtype=np.float64)
    active_scc = 0.0

    with ThreadPoolExecutor(max_workers=worker_count) as executor:
        partial_results = executor.map(process_block, experiment_blocks)
        for partial_sxx, partial_sxc, partial_scc in partial_results:
            active_sxx += partial_sxx
            active_sxc += partial_sxc
            active_scc += partial_scc

    return active_sxx, active_sxc, active_scc


def compute_level2_active_batch_statistics_for_active_mask(
    base_features_data: np.ndarray,
    experiment_ranges: List[Tuple[int, int]],
    level2_feature_scales_by_experiment: np.ndarray,
    active_feature_mask: np.ndarray,
    worker_count: int,
    experiment_blocks: List[List[int]]
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    active_indices = np.flatnonzero(active_feature_mask)
    level2_denominators_by_experiment = compute_level2_denominators_by_experiment(
        level2_feature_scales_by_experiment,
        active_indices
    )
    active_sxx, active_sxc, active_scc = compute_level2_active_batch_statistics(
        base_features_data,
        experiment_ranges,
        level2_feature_scales_by_experiment,
        active_indices,
        level2_denominators_by_experiment,
        worker_count,
        experiment_blocks
    )
    return active_indices, active_sxx, active_sxc, active_scc


def build_full_level2_feature_data_for_active_mask(
    base_features_data: np.ndarray,
    experiment_ranges: List[Tuple[int, int]],
    level2_feature_scales_by_experiment: np.ndarray,
    active_feature_mask: np.ndarray
) -> np.ndarray:
    finalized_features_data = np.zeros_like(base_features_data)
    active_indices = np.flatnonzero(active_feature_mask)
    level2_denominators_by_experiment = compute_level2_denominators_by_experiment(
        level2_feature_scales_by_experiment,
        active_indices
    )
    active_features_data, _ = build_level2_active_batch_arrays(
        base_features_data,
        experiment_ranges,
        level2_feature_scales_by_experiment,
        active_indices,
        level2_denominators_by_experiment
    )
    finalized_features_data[:, active_indices] = active_features_data
    return finalized_features_data


def base_assign_experiment_blocks(experiment_ranges: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
    indexed_lengths = sorted(
        (
            (experiment_idx, end - start)
            for experiment_idx, (start, end) in enumerate(experiment_ranges)
        ),
        key=lambda item: item[1],
        reverse=True
    )
    return indexed_lengths


def assign_experiment_blocks_greedy(experiment_ranges: List[Tuple[int, int]], workers: int) -> List[List[int]]:
    """
    Greedy serpentine assignment for indivisible experiments.
    We sort experiments by length, then assign one experiment per worker at each
    round. Odd rounds take the largest remaining experiments, even rounds take
    the smallest remaining ones, so a worker that got a large experiment in one
    round receives a small one in the next round.
    """
    if workers <= 1 or len(experiment_ranges) <= 1:
        return [list(range(len(experiment_ranges)))]

    indexed_lengths = base_assign_experiment_blocks(experiment_ranges)

    experiment_blocks = [[] for _ in range(workers)]
    left = 0
    right = len(indexed_lengths) - 1
    take_largest = True

    while left <= right:
        for worker_idx in range(workers):
            if left > right:
                break

            if take_largest:
                experiment_idx, _ = indexed_lengths[left]
                left += 1
            else:
                experiment_idx, _ = indexed_lengths[right]
                right -= 1

            experiment_blocks[worker_idx].append(experiment_idx)

        take_largest = not take_largest

    return experiment_blocks


def assign_experiment_blocks_exact(experiment_ranges: List[Tuple[int, int]], workers: int) -> List[List[int]]:
    """
    Exact branch-and-bound assignment that minimizes
    max(worker_loads) - min(worker_loads).
    """
    if workers <= 1 or len(experiment_ranges) <= 1:
        return [list(range(len(experiment_ranges)))]

    indexed_lengths = base_assign_experiment_blocks(experiment_ranges)

    greedy_blocks = assign_experiment_blocks_greedy(experiment_ranges, workers)
    greedy_loads = [
        sum(
            experiment_ranges[experiment_idx][1] - experiment_ranges[experiment_idx][0]
            for experiment_idx in block
        )
        for block in greedy_blocks
    ]

    remaining_sum = [0] * (len(experiment_ranges) + 1)
    for pos in range(len(experiment_ranges) - 1, -1, -1):
        remaining_sum[pos] = remaining_sum[pos + 1] + indexed_lengths[pos][1]

    best_blocks = [block.copy() for block in greedy_blocks]
    best_range = max(greedy_loads) - min(greedy_loads)
    best_max_load = max(greedy_loads)
    visited_states = set()

    def lower_bound_for_loads(item_pos: int, loads: List[int]) -> Tuple[int, int]:
        current_max_load = max(loads)
        current_min_load = min(loads)
        residual_load = remaining_sum[item_pos]
        best_possible_min_load = min(
            current_min_load + residual_load,
            (sum(loads) + residual_load) // workers
        )
        return max(0, current_max_load - best_possible_min_load), current_max_load

    def dfs(item_pos: int, loads: List[int], blocks: List[List[int]]) -> None:
        nonlocal best_blocks, best_range, best_max_load

        state = (item_pos, tuple(sorted(loads)))
        if state in visited_states:
            return
        visited_states.add(state)

        lower_bound_range, current_max_load = lower_bound_for_loads(item_pos, loads)
        if lower_bound_range > best_range:
            return
        if lower_bound_range == best_range and current_max_load >= best_max_load:
            return

        if item_pos == len(indexed_lengths):
            current_min_load = min(loads)
            current_range = current_max_load - current_min_load
            if current_range < best_range or (
                current_range == best_range and current_max_load < best_max_load
            ):
                best_range = current_range
                best_max_load = current_max_load
                best_blocks = [block.copy() for block in blocks]
            return

        experiment_idx, exp_length = indexed_lengths[item_pos]
        seen_loads = set()
        target_load = (sum(loads) + remaining_sum[item_pos]) / workers
        worker_order = sorted(
            range(workers),
            key=lambda idx: (
                abs((loads[idx] + exp_length) - target_load),
                loads[idx],
                len(blocks[idx])
            )
        )
        for worker_idx in worker_order:
            if loads[worker_idx] in seen_loads:
                continue
            seen_loads.add(loads[worker_idx])

            updated_load = loads[worker_idx] + exp_length
            original_load = loads[worker_idx]
            loads[worker_idx] = updated_load
            projected_lower_bound, projected_max_load = lower_bound_for_loads(item_pos + 1, loads)
            if projected_lower_bound > best_range:
                loads[worker_idx] = original_load
                continue
            if projected_lower_bound == best_range and projected_max_load >= best_max_load:
                loads[worker_idx] = original_load
                continue

            blocks[worker_idx].append(experiment_idx)
            dfs(item_pos + 1, loads, blocks)
            blocks[worker_idx].pop()
            loads[worker_idx] = original_load

            if loads[worker_idx] == 0:
                break

    dfs(0, [0] * workers, [[] for _ in range(workers)])
    return best_blocks

def assign_experiment_blocks(experiment_ranges: List[Tuple[int, int]], workers: int) -> List[List[int]]:
    if len(experiment_ranges) <= GREEDY_LIMIT:
        return assign_experiment_blocks_exact(experiment_ranges, workers)
    return assign_experiment_blocks_greedy(experiment_ranges, workers)

def make_feasible_initial_weights(initial_weights: np.ndarray) -> np.ndarray:
    w0 = initial_weights.copy()
    w0 = np.nan_to_num(w0, nan=0.0, posinf=0.0, neginf=0.0, copy=False)
    w0 = np.maximum(w0, 0.0)
    w0_sum = w0.sum()
    if w0_sum > 0.0:
        w0 *= WEIGHTS_SUM / w0_sum
        return w0

    return np.full(w0.shape[0], WEIGHTS_SUM / w0.shape[0], dtype=np.float64)


def compute_level2_quadratic_loss(
    sxx: np.ndarray,
    sxc: np.ndarray,
    scc: float,
    weights: np.ndarray,
    total_rows_seen: int
) -> float:
    residual_core = sxx @ weights - sxc
    residual_squared = float(weights @ residual_core - sxc @ weights + scc)
    residual_squared = max(residual_squared, 0.0)
    return residual_squared / float(total_rows_seen)


@lru_cache(maxsize=None)
def get_level2_trust_constr_static_inputs(
    num_features: int
) -> Tuple[np.ndarray, Bounds, LinearConstraint]:
    simplex_row = np.ones(num_features, dtype=np.float64)
    lower_bounds = np.zeros(num_features, dtype=np.float64)
    upper_bounds = np.full(num_features, np.inf, dtype=np.float64)
    bounds = Bounds(lower_bounds, upper_bounds)
    simplex_constraint = LinearConstraint(
        simplex_row,
        WEIGHTS_SUM,
        WEIGHTS_SUM
    )
    return simplex_row, bounds, simplex_constraint


def optimize_level2_weights(
    sxx: np.ndarray,
    sxc: np.ndarray,
    scc: float,
    total_rows_seen: int,
    initial_weights: np.ndarray,
    debug: bool
) -> Tuple[Any, np.ndarray, float, float, float]:
    num_features = sxx.shape[0]

    # Rebuild the exact global quadratic objective from the cumulative
    # sufficient statistics collected so far, and report the final loss as MSE.
    residual_scale_squared = float(total_rows_seen)

    def compute_residual_terms(w: np.ndarray) -> Tuple[np.ndarray, float]:
        residual_core = sxx @ w - sxc
        residual_squared = float(w @ residual_core - sxc @ w + scc)
        residual_squared = max(residual_squared, 0.0)
        return residual_core, residual_squared

    def geneo_level2_jacobian(w: np.ndarray) -> np.ndarray:
        residual_core = sxx @ w - sxc
        return (2.0 * residual_core) / residual_scale_squared

    def geneo_level2_objective_global(w_raw: np.ndarray) -> float:
        _, residual_squared = compute_residual_terms(w_raw)
        return residual_squared / residual_scale_squared

    def geneo_level2_hessian(w: np.ndarray) -> np.ndarray:
        return (2.0 * sxx) / residual_scale_squared

    _, non_negative_bounds, simplex_constraint = (
        get_level2_trust_constr_static_inputs(num_features)
    )
    solver_started_at = time.perf_counter()
    result = minimize(
        geneo_level2_objective_global,
        initial_weights,
        method="trust-constr",
        jac=geneo_level2_jacobian,
        hess=geneo_level2_hessian,
        bounds=non_negative_bounds,
        constraints=[simplex_constraint],
        options={
            "maxiter": TRUST_CONSTR_MAXITER,
            "gtol": TRUST_CONSTR_GTOL,
            "xtol": TRUST_CONSTR_XTOL,
            "barrier_tol": TRUST_CONSTR_BARRIER_TOL,
            "initial_barrier_parameter": (
                TRUST_CONSTR_INITIAL_BARRIER_PARAMETER
            ),
            "initial_barrier_tolerance": (
                TRUST_CONSTR_INITIAL_BARRIER_TOLERANCE
            ),
            "verbose": 0,
        }
    )
    solver_seconds = time.perf_counter() - solver_started_at
    if debug:
        print(
            "trust-constr optimization finished with status "
            f"{result.status} and message: {result.message}"
        )
    return (
        result,
        result.x,
        solver_started_at,
        solver_seconds,
        solver_seconds
    )


def is_optimizer_hard_failure(result: Any) -> bool:
    return not bool(getattr(result, "success", False))


def extract_optimizer_metadata(result: Any) -> Tuple[bool, int | None, str, float | None]:
    return (
        not is_optimizer_hard_failure(result),
        int(getattr(result, "status", -1)),
        str(getattr(result, "message", "")),
        float(getattr(result, "fun", np.nan))
    )


def build_valid_level1_feature_mask(invalid_feature_mask: np.ndarray) -> np.ndarray:
    keep_mask = ~invalid_feature_mask
    if not np.any(keep_mask):
        raise GeneoCoreValidationError(
            "No valid features remain because every feature has zero L2 norm "
            "in every experiment of the current batch"
        )

    return keep_mask


def build_min_correlation_feature_mask(
    valid_feature_mask: np.ndarray,
    mean_correlations: np.ndarray,
    min_correlation_threshold: float
) -> np.ndarray:
    keep_mask = valid_feature_mask & (
        mean_correlations >= min_correlation_threshold
    )
    if not np.any(keep_mask):
        raise GeneoCoreValidationError(
            "No valid features remain after applying "
            f"MIN_CORRELATION_THRESHOLD={min_correlation_threshold}"
        )

    return keep_mask


class GeneoCoreModel:
    def __init__(
        self,
        target_column_name: str,
        runtime_state: GeneoRuntimeState | None = None
    ):
        self.target_column_name = target_column_name
        self.runtime_state = runtime_state
        self._state_lock = Lock()

    def _prepare_batch_arrays(
        self,
        data_matrix: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        features_data = data_matrix[:, :-1]
        critical_data = data_matrix[:, -1]

        # The threshold-based binarization has been disabled on purpose:
        # keep the original critical function values when loading the batch.
        # The target is still L2-normalized later, per experiment, before
        # contributing to level 1 and level 2 statistics.
        #
        # Historical behavior left here for traceability:
        # critical_data[:] = critical_data < CRITICAL_THRESHOLD
        return features_data, critical_data

    def _prepare_batch(
        self,
        data_matrix: np.ndarray,
        experiment_ranges: List[Tuple[int, int]],
        compute_level1: bool = True,
    ) -> GeneoPreparedBatch:
        features_data, critical_data = self._prepare_batch_arrays(data_matrix)
        # Shared batch preparation always performs the in-place per-experiment
        # centering/L2 normalization inside compute_batch_statistics and stores
        # the pre-pruning alpha values later used to build LVL2 features on the
        # surviving feature set.
        computed_batch_statistics = compute_batch_statistics(
            experiment_ranges,
            critical_data,
            features_data,
            compute_level1,
        )
        return GeneoPreparedBatch(
            features_data=features_data,
            critical_data=critical_data,
            experiment_ranges=experiment_ranges,
            worker_count=computed_batch_statistics.worker_count,
            experiment_blocks=computed_batch_statistics.experiment_blocks,
            total_rows=data_matrix.shape[0],
            level1_statistics=computed_batch_statistics.level1_statistics,
            invalid_feature_mask=computed_batch_statistics.invalid_feature_mask,
            level2_feature_scales_by_experiment=(
                computed_batch_statistics.level2_feature_scales_by_experiment
            )
        )

    def _select_level2_feature_set(
        self,
        prepared_batch: GeneoPreparedBatch,
        mean_correlations: np.ndarray,
        eligible_feature_mask: np.ndarray,
        debug: bool
    ) -> Tuple[EvaluatedFeatureSet, np.ndarray | None, np.ndarray, np.ndarray, float, float, float, float]:
        eligible_indices = np.flatnonzero(eligible_feature_mask)
        ranked_indices_array = eligible_indices[
            np.argsort(mean_correlations[eligible_indices], kind="stable")[::-1]
        ]
        ranked_count = int(ranked_indices_array.size)
        rank_positions = np.full(mean_correlations.shape[0], ranked_count + 1, dtype=np.int64)
        for rank_position, feature_index in enumerate(ranked_indices_array):
            rank_positions[int(feature_index)] = rank_position

        base_features_data = prepared_batch.features_data
        level2_feature_scales = prepared_batch.level2_feature_scales_by_experiment
        statistics_cache: Level2GramCache | None = None
        last_weights_by_feature = np.zeros(mean_correlations.shape[0], dtype=np.float64)
        evaluated_sets: Dict[Tuple[int, ...], EvaluatedFeatureSet] = {}
        first_solver_started_at: float | None = None
        total_solver_seconds = 0.0
        debug_seconds = 0.0

        def get_statistics_cache(active_indices_array: np.ndarray) -> Level2GramCache | None:
            nonlocal statistics_cache

            pool_size = int(np.max(rank_positions[active_indices_array])) + 1
            if (
                statistics_cache is not None
                and statistics_cache.pool_indices.shape[0] >= pool_size
            ):
                return statistics_cache

            previous_pool_size = (
                0
                if statistics_cache is None
                else int(statistics_cache.pool_indices.shape[0])
            )
            expanded_statistics_cache = maybe_expand_level2_gram_cache(
                statistics_cache,
                base_features_data,
                prepared_batch.experiment_ranges,
                ranked_indices_array[:pool_size],
                prepared_batch.worker_count,
                prepared_batch.experiment_blocks
            )
            if debug:
                if expanded_statistics_cache is None:
                    print(
                        "Level-2 feature-selection Gram cache skipped: "
                        f"{pool_size} feature pool"
                    )
                else:
                    cache_action = "built" if previous_pool_size == 0 else "expanded"
                    print(
                        f"Level-2 feature-selection Gram cache {cache_action}: "
                        f"{pool_size} feature pool, "
                        f"{len(prepared_batch.experiment_ranges)} experiments"
                    )

            if expanded_statistics_cache is None:
                return None

            statistics_cache = expanded_statistics_cache
            return statistics_cache

        def compute_candidate_statistics(
            active_indices_array: np.ndarray
        ) -> Tuple[np.ndarray, np.ndarray, float]:
            level2_denominators_by_experiment = compute_level2_denominators_by_experiment(
                level2_feature_scales,
                active_indices_array
            )
            active_statistics_cache = get_statistics_cache(active_indices_array)
            if active_statistics_cache is None:
                return compute_level2_active_batch_statistics(
                    base_features_data,
                    prepared_batch.experiment_ranges,
                    level2_feature_scales,
                    active_indices_array,
                    level2_denominators_by_experiment,
                    prepared_batch.worker_count,
                    prepared_batch.experiment_blocks
                )

            return compute_level2_active_batch_statistics_from_gram_cache(
                active_statistics_cache,
                level2_feature_scales,
                active_indices_array,
                level2_denominators_by_experiment
            )

        def canonicalize(indices: Sequence[int]) -> Tuple[int, ...]:
            return tuple(
                sorted(
                    {int(feature_index) for feature_index in indices},
                    key=lambda feature_index: int(rank_positions[feature_index])
                )
            )

        def pruning_loss_is_acceptable(
            candidate: EvaluatedFeatureSet,
            reference: EvaluatedFeatureSet
        ) -> bool:
            if (
                candidate.optimizer_total_loss is None
                or reference.optimizer_total_loss is None
            ):
                return False

            reference_loss = reference.optimizer_total_loss
            candidate_loss = candidate.optimizer_total_loss

            if reference_loss <= 0.0:
                return candidate_loss <= reference_loss

            relative_loss_increase = (
                candidate_loss - reference_loss
            ) / abs(reference_loss)

            return (
                relative_loss_increase
                <= FEATURE_SELECTION_PRUNING_MAX_RELATIVE_LOSS_INCREASE
            )

        def losses_are_equivalent(
            left: EvaluatedFeatureSet,
            right: EvaluatedFeatureSet
        ) -> bool:
            if (
                left.optimizer_total_loss is None
                or right.optimizer_total_loss is None
            ):
                return False
            loss_scale = max(
                abs(left.optimizer_total_loss),
                abs(right.optimizer_total_loss)
            )
            if loss_scale <= 0.0:
                return left.optimizer_total_loss == right.optimizer_total_loss
            equivalent_loss_delta = (
                FEATURE_SELECTION_STOP_RELATIVE_IMPROVEMENT * loss_scale
            )
            return (
                abs(left.optimizer_total_loss - right.optimizer_total_loss)
                <= equivalent_loss_delta
            )

        def loss_is_better(
            candidate: EvaluatedFeatureSet,
            reference: EvaluatedFeatureSet
        ) -> bool:
            relative_improvement = compute_relative_loss_improvement(
                candidate,
                reference
            )
            if relative_improvement is None:
                return False
            return relative_improvement >= FEATURE_SELECTION_STOP_RELATIVE_IMPROVEMENT

        def compute_relative_loss_improvement(
            candidate: EvaluatedFeatureSet,
            reference: EvaluatedFeatureSet
        ) -> float | None:
            if (
                candidate.optimizer_total_loss is None
                or reference.optimizer_total_loss is None
            ):
                return None

            reference_loss = reference.optimizer_total_loss
            candidate_loss = candidate.optimizer_total_loss
            if reference_loss == 0.0:
                if candidate_loss < reference_loss:
                    return float("inf")
                return 0.0

            return (
                (reference_loss - candidate_loss)
                / abs(reference_loss)
            )

        def compute_proxy_relative_improvement(
            proxy_loss: float,
            reference_loss: float
        ) -> float:
            if reference_loss == 0.0:
                if proxy_loss < reference_loss:
                    return float("inf")
                if proxy_loss > reference_loss:
                    return -float("inf")
                return 0.0

            return (reference_loss - proxy_loss) / abs(reference_loss)

        def choose_better_or_pruned(
            current_best: EvaluatedFeatureSet,
            candidate: EvaluatedFeatureSet
        ) -> EvaluatedFeatureSet:
            if loss_is_better(candidate, current_best):
                return candidate

            if (
                candidate.feature_count < current_best.feature_count
                and pruning_loss_is_acceptable(candidate, current_best)
            ):
                return candidate

            if (
                losses_are_equivalent(candidate, current_best)
                and candidate.feature_count < current_best.feature_count
            ):
                return candidate

            return current_best

        def choose_better(
            current_best: EvaluatedFeatureSet,
            candidate: EvaluatedFeatureSet
        ) -> EvaluatedFeatureSet:
            if loss_is_better(candidate, current_best):
                return candidate
            if losses_are_equivalent(candidate, current_best):
                if candidate.feature_count < current_best.feature_count:
                    return candidate
            return current_best

        def select_weighted(
            indices: Sequence[int],
            count: int,
            highest: bool
        ) -> Tuple[int, ...]:
            index_array = np.array(indices, dtype=np.int64)
            if count <= 0 or index_array.size == 0:
                return ()
            count = min(count, index_array.size)

            weights = last_weights_by_feature[index_array]
            order_values = (
                -weights
                if highest
                else weights
            )
            partition_positions = np.argpartition(order_values, count - 1)[:count]
            selected_indices = index_array[partition_positions]
            selected_weights = weights[partition_positions]

            if highest:
                order = np.lexsort(
                    (
                        selected_indices,
                        -selected_weights
                    )
                )
            else:
                order = np.lexsort(
                    (
                        selected_indices,
                        selected_weights
                    )
                )
            return tuple(int(feature_index) for feature_index in selected_indices[order])

        def lowest_weighted(indices: Sequence[int], count: int) -> Tuple[int, ...]:
            return select_weighted(indices, count, highest=False)

        def highest_weighted(indices: Sequence[int], count: int) -> Tuple[int, ...]:
            return select_weighted(indices, count, highest=True)

        def difference_preserving_order(
            indices: Sequence[int],
            excluded_indices: Sequence[int]
        ) -> Tuple[int, ...]:
            if not excluded_indices:
                return tuple(int(feature_index) for feature_index in indices)
            excluded_mask = np.zeros(mean_correlations.shape[0], dtype=bool)
            excluded_mask[np.array(excluded_indices, dtype=np.int64)] = True
            index_array = np.array(indices, dtype=np.int64)
            return tuple(
                int(feature_index)
                for feature_index in index_array[~excluded_mask[index_array]]
            )

        def evaluate(
            active_indices: Sequence[int],
            precomputed_statistics: Tuple[np.ndarray, np.ndarray, float] | None = None
        ) -> EvaluatedFeatureSet:
            nonlocal first_solver_started_at, total_solver_seconds

            active_indices_key = canonicalize(active_indices)
            cached_evaluation = evaluated_sets.get(active_indices_key)
            if cached_evaluation is not None:
                return cached_evaluation

            active_indices_array = np.array(active_indices_key, dtype=np.int64)
            if precomputed_statistics is None:
                batch_sxx, batch_sxc, batch_scc = compute_candidate_statistics(active_indices_array)
            else:
                batch_sxx, batch_sxc, batch_scc = precomputed_statistics

            historical_weights = last_weights_by_feature[active_indices_array]
            initial_weights = (
                make_feasible_initial_weights(historical_weights)
                if np.any(historical_weights > 0.0)
                else make_feasible_initial_weights(batch_sxc)
            )

            try:
                (
                    result,
                    w_opt,
                    solver_started_at,
                    solver_seconds,
                    _
                ) = optimize_level2_weights(
                    batch_sxx,
                    batch_sxc,
                    batch_scc,
                    prepared_batch.total_rows,
                    initial_weights,
                    debug
                )
            except Exception as exc:
                raise GeneoCoreExecutionError(
                    "Level-2 feature selection raised an exception during "
                    f"candidate optimization: {exc}"
                ) from exc

            (
                optimizer_success,
                optimizer_status,
                optimizer_message,
                optimizer_total_loss
            ) = extract_optimizer_metadata(result)

            if is_optimizer_hard_failure(result):
                raise GeneoCoreExecutionError(
                    "Level-2 feature selection failed during candidate optimization "
                    f"with optimizer status {optimizer_status}: {optimizer_message}"
                )

            if first_solver_started_at is None:
                first_solver_started_at = solver_started_at
            total_solver_seconds += solver_seconds
            last_weights_by_feature[active_indices_array] = w_opt

            evaluated_feature_set = EvaluatedFeatureSet(
                active_indices=active_indices_key,
                weights=w_opt,
                solver_started_at=solver_started_at,
                solver_seconds=solver_seconds,
                optimizer_success=optimizer_success,
                optimizer_status=optimizer_status,
                optimizer_message=optimizer_message,
                optimizer_total_loss=optimizer_total_loss
            )
            evaluated_sets[active_indices_key] = evaluated_feature_set
            if debug:
                print(
                    "Feature-selection candidate: "
                    f"{evaluated_feature_set.feature_count} features, "
                    f"loss={optimizer_total_loss}"
                )
            return evaluated_feature_set

        def compute_proxy_evaluation_loss(
            active_indices: Sequence[int]
        ) -> Tuple[float, Tuple[np.ndarray, np.ndarray, float]]:
            active_indices_key = canonicalize(active_indices)
            active_indices_array = np.array(active_indices_key, dtype=np.int64)
            batch_statistics = compute_candidate_statistics(active_indices_array)
            batch_sxx, batch_sxc, batch_scc = batch_statistics

            projected_weights = make_feasible_initial_weights(
                last_weights_by_feature[active_indices_array]
            )
            proxy_loss = compute_level2_quadratic_loss(
                batch_sxx,
                batch_sxc,
                batch_scc,
                projected_weights,
                prepared_batch.total_rows
            )
            return proxy_loss, batch_statistics

        def should_evaluate_candidate_with_solver(
            candidate_indices: Sequence[int],
            reference_evaluation: EvaluatedFeatureSet,
            screening_mode: str
        ) -> Tuple[bool, Tuple[np.ndarray, np.ndarray, float] | None, float | None]:
            reference_loss = reference_evaluation.optimizer_total_loss
            if reference_loss is None or reference_loss <= 0.0:
                return True, None, None

            proxy_loss, batch_statistics = compute_proxy_evaluation_loss(candidate_indices)

            proxy_relative_improvement = compute_proxy_relative_improvement(
                proxy_loss,
                reference_loss
            )

            if screening_mode == "remove":
                max_proxy_worsening = (
                    FEATURE_SELECTION_PROXY_MAX_RELATIVE_WORSENING_REMOVE
                )
            else:
                max_proxy_worsening = (
                    FEATURE_SELECTION_PROXY_MAX_RELATIVE_WORSENING_ADD
                )

            should_evaluate = (
                proxy_relative_improvement >= -max_proxy_worsening
            )

            return should_evaluate, batch_statistics, proxy_loss

        def propose_candidate(
            current_indices: Tuple[int, ...],
            direction: str,
            step: int,
            available_to_add: Tuple[int, ...],
            pool_indices: Tuple[int, ...],
            removable_indices: Sequence[int] | None = None
        ) -> Tuple[Tuple[int, ...], Tuple[int, ...], Tuple[int, ...], int] | None:
            attempted_step = step
            current_indices_set = set(current_indices)
            pool_indices_set = set(pool_indices)
            removable_indices_set = (
                None
                if removable_indices is None
                else {int(feature_index) for feature_index in removable_indices}
            )

            while attempted_step >= 1:
                if direction == "down":
                    if len(current_indices) <= 1:
                        return None
                    removable_source = (
                        current_indices
                        if removable_indices_set is None
                        else tuple(
                            feature_index
                            for feature_index in current_indices
                            if feature_index in removable_indices_set
                        )
                    )
                    if not removable_source:
                        return None
                    remove_count = min(
                        attempted_step,
                        len(removable_source),
                        len(current_indices) - 1
                    )
                    if remove_count <= 0:
                        return None
                    removed_indices = lowest_weighted(removable_source, remove_count)
                    candidate_indices = difference_preserving_order(
                        current_indices,
                        removed_indices
                    )
                    added_indices: Tuple[int, ...] = ()
                else:
                    add_source = tuple(
                        feature_index
                        for feature_index in available_to_add
                        if feature_index not in current_indices_set
                    )
                    if not add_source:
                        add_source = tuple(
                            feature_index
                            for feature_index in pool_indices
                            if feature_index not in current_indices_set
                        )
                    if not add_source:
                        return None

                    add_count = min(attempted_step, len(add_source))
                    added_indices = highest_weighted(add_source, add_count)
                    candidate_indices = canonicalize((*current_indices, *added_indices))
                    removed_indices = ()

                if (
                    candidate_indices
                    and set(candidate_indices).issubset(pool_indices_set)
                    and candidate_indices != current_indices
                    and candidate_indices not in evaluated_sets
                ):
                    return candidate_indices, removed_indices, added_indices, attempted_step

                attempted_step //= 2

            return None

        def bisect_inside_pool(
            current_evaluation: EvaluatedFeatureSet,
            reference_evaluation: EvaluatedFeatureSet,
            pool_indices: Tuple[int, ...],
            best_evaluation: EvaluatedFeatureSet,
            removable_indices: Sequence[int] | None = None,
            initial_step: int | None = None
        ) -> EvaluatedFeatureSet:
            direction = "down"
            current = current_evaluation
            previous = current_evaluation
            step = (
                initial_step
                if initial_step is not None
                else (
                    current_evaluation.feature_count
                    - reference_evaluation.feature_count
                ) // 2
            )
            available_to_add: Tuple[int, ...] = ()

            loss_best_evaluation = best_evaluation

            while step >= 1:
                proposed = propose_candidate(
                    current.active_indices,
                    direction,
                    step,
                    available_to_add,
                    pool_indices,
                    removable_indices
                )
                if proposed is None:
                    break

                candidate_indices, removed_indices, added_indices, used_step = proposed
                screening_mode = "remove" if direction == "down" else "add"

                (
                    should_evaluate_candidate,
                    precomputed_statistics,
                    proxy_loss
                ) = should_evaluate_candidate_with_solver(
                    candidate_indices,
                    previous,
                    screening_mode
                )
                if not should_evaluate_candidate:
                    if debug:
                        print(
                            "Feature-selection bisection candidate skipped by "
                            "proxy screening: "
                            f"{len(candidate_indices)} features, "
                            f"screening_mode={screening_mode}, "
                            f"proxy_loss={proxy_loss}, "
                            f"reference_loss={previous.optimizer_total_loss}"
                        )
                    if direction == "down":
                        available_to_add = removed_indices
                        direction = "up"
                    else:
                        direction = "down"
                    step = used_step // 2
                    continue

                candidate = evaluate(candidate_indices, precomputed_statistics)
                loss_best_evaluation = choose_better(
                    loss_best_evaluation,
                    candidate
                )

                if loss_is_better(candidate, best_evaluation):
                    best_evaluation = candidate
                elif (
                    candidate.feature_count < best_evaluation.feature_count
                    and pruning_loss_is_acceptable(candidate, loss_best_evaluation)
                ):
                    best_evaluation = candidate
                elif (
                    losses_are_equivalent(candidate, best_evaluation)
                    and candidate.feature_count < best_evaluation.feature_count
                ):
                    best_evaluation = candidate
                candidate_is_better = loss_is_better(candidate, previous)
                candidate_is_equivalent = losses_are_equivalent(candidate, previous)
                candidate_is_pruning_acceptable = (
                    direction == "down"
                    and candidate.feature_count < previous.feature_count
                    and pruning_loss_is_acceptable(candidate, loss_best_evaluation)
                )
                keep_candidate_as_current = True

                if (
                    direction == "down"
                    and candidate.feature_count < previous.feature_count
                    and not candidate_is_better
                    and not candidate_is_equivalent
                    and not candidate_is_pruning_acceptable
                ):
                    if debug:
                        print(
                            "Feature-selection bisection stopped: "
                            "downward pruning candidate rejected"
                        )
                    break

                if candidate_is_better or candidate_is_pruning_acceptable:
                    if direction == "down":
                        available_to_add = removed_indices
                    else:
                        added_indices_set = set(added_indices)
                        available_to_add = tuple(
                            feature_index
                            for feature_index in available_to_add
                            if feature_index not in added_indices_set
                        )
                elif candidate_is_equivalent:
                    if direction == "down":
                        available_to_add = removed_indices
                    else:
                        added_indices_set = set(added_indices)
                        available_to_add = tuple(
                            feature_index
                            for feature_index in available_to_add
                            if feature_index not in added_indices_set
                        )
                        direction = "down"
                        keep_candidate_as_current = False
                else:
                    if direction == "down":
                        available_to_add = removed_indices
                        direction = "up"
                    else:
                        direction = "down"
                    keep_candidate_as_current = False

                if keep_candidate_as_current:
                    current = candidate
                    previous = candidate
                step = used_step // 2

            return best_evaluation

        next_rank_position = min(FEATURE_SELECTION_INITIAL_FEATURES, ranked_count)
        delayed_feature_blocks: List[Tuple[int, ...]] = []

        def take_next_ranked_features(
            count: int,
            current_indices: Sequence[int]
        ) -> Tuple[int, ...]:
            nonlocal next_rank_position

            current_index_set = {int(feature_index) for feature_index in current_indices}
            selected: List[int] = []
            while next_rank_position < ranked_count and len(selected) < count:
                feature_index = int(ranked_indices_array[next_rank_position])
                next_rank_position += 1
                if feature_index in current_index_set:
                    continue
                selected.append(feature_index)
            return tuple(selected)

        def compute_required_forward_improvement(attempted_step: int) -> float:
            return (
                FEATURE_SELECTION_FORWARD_REQUIRED_RELATIVE_IMPROVEMENT_PER_FEATURE
                * attempted_step
            )

        def update_step_size_after_accept(
            relative_improvement: float,
            attempted_step: int
        ) -> int:
            if relative_improvement >= FEATURE_SELECTION_HIGH_RELATIVE_IMPROVEMENT:
                return max(1, attempted_step * 2)
            if relative_improvement >= FEATURE_SELECTION_LOW_RELATIVE_IMPROVEMENT:
                return max(1, attempted_step)
            return max(1, attempted_step // 2)
         
        def update_step_size_after_reject(
            relative_improvement: float | None,
            attempted_step: int
        ) -> int:
            required_forward_improvement = compute_required_forward_improvement(
                attempted_step
            )

            reject_margin = (
                float("inf")
                if relative_improvement is None
                else required_forward_improvement - relative_improvement
            )

            medium_reject_margin = (
                FEATURE_SELECTION_FORWARD_MEDIUM_REJECT_MARGIN_PER_FEATURE
                * attempted_step
            )

            large_reject_margin = (
                FEATURE_SELECTION_FORWARD_LARGE_REJECT_MARGIN_PER_FEATURE
                * attempted_step
            )

            if reject_margin >= large_reject_margin:
                return max(1, attempted_step * 2)

            if reject_margin >= medium_reject_margin:
                return max(1, attempted_step)

            return max(1, attempted_step // 2)

        def refine_current_accepted_set(
            accepted_evaluation: EvaluatedFeatureSet
        ) -> EvaluatedFeatureSet:
            if accepted_evaluation.feature_count <= 1:
                return accepted_evaluation

            refined_evaluation = bisect_inside_pool(
                accepted_evaluation,
                accepted_evaluation,
                accepted_evaluation.active_indices,
                accepted_evaluation,
                initial_step=max(1, accepted_evaluation.feature_count // 2)
            )
            return choose_better_or_pruned(accepted_evaluation, refined_evaluation)

        initial_size = min(FEATURE_SELECTION_INITIAL_FEATURES, ranked_count)
        current_evaluation = evaluate(ranked_indices_array[:initial_size])
        best_evaluation = current_evaluation
        step_size = max(2, current_evaluation.feature_count)

        while True:
            new_feature_indices = take_next_ranked_features(
                step_size,
                current_evaluation.active_indices
            )
            if not new_feature_indices:
                break

            attempted_step = len(new_feature_indices)
            candidate_evaluation = evaluate(
                (*current_evaluation.active_indices, *new_feature_indices)
            )

            relative_improvement = compute_relative_loss_improvement(
                candidate_evaluation,
                current_evaluation
            )

            required_forward_improvement = compute_required_forward_improvement(attempted_step)

            if (
                relative_improvement is not None
                and relative_improvement  >= required_forward_improvement             
            ):
                current_evaluation = candidate_evaluation
                best_evaluation = choose_better(best_evaluation, current_evaluation)
                step_size = update_step_size_after_accept(
                    relative_improvement,
                    attempted_step
                )
                continue

            delayed_feature_blocks.append(tuple(new_feature_indices))
            
            next_step_size = update_step_size_after_reject(
                relative_improvement,
                attempted_step
            )
            if debug:
                if (
                    relative_improvement is not None
                    and relative_improvement >= 0.0
                ):
                    delayed_reason = (
                        "candidate improvement below forward acceptance "
                        "threshold"
                    )
                else:
                    delayed_reason = "candidate loss worsened"

                reject_margin = (
                    float("inf")
                    if relative_improvement is None
                    else required_forward_improvement - relative_improvement
                )
                print(
                    "Feature-selection delayed "
                    f"{len(new_feature_indices)} feature(s): "
                    f"{delayed_reason}; "
                    f"relative_improvement={relative_improvement}, "
                    f"required_forward_improvement={required_forward_improvement}, "
                    f"reject_margin={reject_margin}, "
                    f"next_step={next_step_size}"
                )
            step_size = next_step_size

        current_evaluation = refine_current_accepted_set(current_evaluation)
        best_evaluation = choose_better_or_pruned(best_evaluation, current_evaluation)
        if debug:
            print(
                "Feature-selection forward pass refined accepted features "
                "by bisection"
            )

        def order_delayed_features(
            delayed_indices: Tuple[int, ...]
        ) -> Tuple[int, ...]:
            if not delayed_indices:
                return ()

            delayed_array = np.array(delayed_indices, dtype=np.int64)

            weights = last_weights_by_feature[delayed_array]
            order = np.argsort(-weights, kind="stable")

            return tuple(int(feature_index) for feature_index in delayed_array[order])
        
        def build_candidate_weights_with_added_features(
            candidate_indices: Tuple[int, ...],
            current_indices: Tuple[int, ...],
            current_weights: np.ndarray,
            added_feature_indices: Sequence[int]
        ) -> Tuple[np.ndarray, np.ndarray]:
            candidate_weights = np.zeros(len(candidate_indices), dtype=np.float64)
            positions_by_index = {
                feature_index: position
                for position, feature_index in enumerate(candidate_indices)
            }
            for weight_position, feature_index in enumerate(current_indices):
                candidate_weights[positions_by_index[feature_index]] = (
                    current_weights[weight_position]
                )

            added_basis_vector = np.zeros_like(candidate_weights)
            added_indices_array = np.array(added_feature_indices, dtype=np.int64)
            added_weights = np.maximum(last_weights_by_feature[added_indices_array], 0.0)
            added_weight_sum = float(np.sum(added_weights))
            if added_weight_sum > 0.0:
                added_weight_distribution = WEIGHTS_SUM * added_weights / added_weight_sum
            else:
                added_weight_distribution = np.full(
                    added_indices_array.size,
                    WEIGHTS_SUM / added_indices_array.size,
                    dtype=np.float64
                )
            for added_feature_index, added_weight in zip(
                added_indices_array,
                added_weight_distribution
            ):
                added_position = positions_by_index[int(added_feature_index)]
                added_basis_vector[added_position] = added_weight
            return candidate_weights, added_basis_vector

        def compute_best_one_dimensional_delayed_proxy(
            batch_sxx: np.ndarray,
            batch_sxc: np.ndarray,
            batch_scc: float,
            current_weights_on_candidate: np.ndarray,
            added_basis_vector: np.ndarray
        ) -> Tuple[np.ndarray, float, float]:
            search_direction = added_basis_vector - current_weights_on_candidate
            quadratic_term = float(search_direction @ (batch_sxx @ search_direction))
            linear_term = 2.0 * float(
                current_weights_on_candidate @ (batch_sxx @ search_direction)
                - batch_sxc @ search_direction
            )

            candidate_lambdas = [0.0, 1.0]
            if quadratic_term > CENTERED_L2_NORM_EPS:
                unconstrained_lambda = -linear_term / (2.0 * quadratic_term)
                candidate_lambdas.append(float(np.clip(unconstrained_lambda, 0.0, 1.0)))

            best_lambda = 0.0
            best_weights = current_weights_on_candidate
            best_loss = compute_level2_quadratic_loss(
                batch_sxx,
                batch_sxc,
                batch_scc,
                best_weights,
                prepared_batch.total_rows
            )
            for candidate_lambda in candidate_lambdas[1:]:
                candidate_weights = (
                    current_weights_on_candidate
                    + candidate_lambda * search_direction
                )
                candidate_loss = compute_level2_quadratic_loss(
                    batch_sxx,
                    batch_sxc,
                    batch_scc,
                    candidate_weights,
                    prepared_batch.total_rows
                )
                if candidate_loss < best_loss:
                    best_lambda = candidate_lambda
                    best_weights = candidate_weights
                    best_loss = candidate_loss

            return best_weights, best_loss, best_lambda

        def compute_relative_proxy_loss_improvement(
            candidate_loss: float,
            reference_loss: float
        ) -> float:
            if reference_loss == 0.0:
                if candidate_loss < reference_loss:
                    return float("inf")
                if candidate_loss > reference_loss:
                    return -float("inf")
                return 0.0

            return (reference_loss - candidate_loss) / abs(reference_loss)

        def update_delayed_step_size_after_accept(
            relative_improvement: float,
            attempted_step: int
        ) -> int:
            if relative_improvement >= FEATURE_SELECTION_DELAYED_HIGH_RELATIVE_IMPROVEMENT:
                return max(1, attempted_step * 2)
            if relative_improvement >= FEATURE_SELECTION_DELAYED_LOW_RELATIVE_IMPROVEMENT:
                return max(1, attempted_step)
            return max(1, attempted_step // 2)

        def update_delayed_step_size_after_reject(
            relative_proxy_improvement: float,
            attempted_step: int
        ) -> int:
            if relative_proxy_improvement >= 0.0:
                return max(1, attempted_step)

            relative_worsening = -relative_proxy_improvement

            if relative_worsening >= FEATURE_SELECTION_DELAYED_LARGE_RELATIVE_WORSENING:
                return max(1, attempted_step * 2)

            if relative_worsening >= FEATURE_SELECTION_DELAYED_MEDIUM_RELATIVE_WORSENING:
                return max(1, attempted_step)

            return max(1, attempted_step // 2)

        def retry_delayed_features_with_proxy(
            base_evaluation: EvaluatedFeatureSet,
            delayed_indices: Tuple[int, ...]
        ) -> EvaluatedFeatureSet:
            if not delayed_indices:
                return base_evaluation

            ordered_delayed_indices = order_delayed_features(delayed_indices)
            current_proxy_indices = base_evaluation.active_indices
            current_proxy_weights = base_evaluation.weights.copy()
            current_proxy_loss = base_evaluation.optimizer_total_loss
            if current_proxy_loss is None:
                current_proxy_array = np.array(current_proxy_indices, dtype=np.int64)
                current_proxy_statistics = compute_candidate_statistics(current_proxy_array)
                current_proxy_loss = compute_level2_quadratic_loss(
                    current_proxy_statistics[0],
                    current_proxy_statistics[1],
                    current_proxy_statistics[2],
                    current_proxy_weights,
                    prepared_batch.total_rows
                )

            accepted_any_delayed_feature = False
            final_proxy_statistics: Tuple[np.ndarray, np.ndarray, float] | None = None
            current_active_feature_set = set(current_proxy_indices)
            next_delayed_position = 0
            step_size = min(2, len(ordered_delayed_indices))

            while next_delayed_position < len(ordered_delayed_indices):
                selected_delayed_features: List[int] = []
                while (
                    next_delayed_position < len(ordered_delayed_indices)
                    and len(selected_delayed_features) < step_size
                ):
                    delayed_feature_index = ordered_delayed_indices[next_delayed_position]
                    next_delayed_position += 1
                    if delayed_feature_index in current_active_feature_set:
                        continue
                    selected_delayed_features.append(delayed_feature_index)

                if not selected_delayed_features:
                    break

                candidate_indices = canonicalize(
                    (*current_proxy_indices, *selected_delayed_features)
                )
                candidate_indices_array = np.array(candidate_indices, dtype=np.int64)
                batch_statistics = compute_candidate_statistics(candidate_indices_array)
                batch_sxx, batch_sxc, batch_scc = batch_statistics
                current_weights_on_candidate, added_basis_vector = (
                    build_candidate_weights_with_added_features(
                        candidate_indices,
                        current_proxy_indices,
                        current_proxy_weights,
                        selected_delayed_features
                    )
                )
                (
                    proxy_weights,
                    proxy_loss,
                    proxy_lambda
                ) = compute_best_one_dimensional_delayed_proxy(
                    batch_sxx,
                    batch_sxc,
                    batch_scc,
                    current_weights_on_candidate,
                    added_basis_vector
                )

                relative_proxy_improvement = compute_relative_proxy_loss_improvement(
                    proxy_loss,
                    current_proxy_loss
                )
                attempted_step = len(selected_delayed_features)

                if (
                    relative_proxy_improvement
                    >= FEATURE_SELECTION_PROXY_STRONG_ACCEPT_RELATIVE_IMPROVEMENT
                ):
                    current_proxy_indices = candidate_indices
                    current_proxy_weights = proxy_weights
                    current_proxy_loss = proxy_loss
                    final_proxy_statistics = batch_statistics
                    current_active_feature_set.update(selected_delayed_features)
                    accepted_any_delayed_feature = True
                    last_weights_by_feature[candidate_indices_array] = proxy_weights
                    step_size = update_delayed_step_size_after_accept(
                        relative_proxy_improvement,
                        attempted_step
                    )
                    if debug:
                        print(
                            "Feature-selection delayed block accepted by "
                            "proxy screening: "
                            f"{attempted_step} feature(s), "
                            f"lambda={proxy_lambda}, proxy_loss={proxy_loss}, "
                            f"relative_improvement={relative_proxy_improvement}, "
                            f"next_step={step_size}"
                        )
                    continue

                step_size = update_delayed_step_size_after_reject(
                    relative_proxy_improvement,
                    attempted_step
                )
                if debug:
                    print(
                        "Feature-selection delayed block discarded by "
                        "proxy screening: "
                        f"{attempted_step} feature(s), "
                        f"proxy_loss={proxy_loss}, reference_loss={current_proxy_loss}, "
                        f"relative_improvement={relative_proxy_improvement}, "
                        f"next_step={step_size}"
                    )

            if not accepted_any_delayed_feature:
                return base_evaluation

            if debug:
                print(
                        "Feature-selection delayed proxy selected "
                        f"{len(current_proxy_indices) - base_evaluation.feature_count} "
                        "feature(s); running one final optimizer evaluation"
                )

            return evaluate(current_proxy_indices, final_proxy_statistics)

        delayed_feature_indices = canonicalize(
            feature_index
            for delayed_block in delayed_feature_blocks
            for feature_index in delayed_block
        )
        current_evaluation = retry_delayed_features_with_proxy(
            current_evaluation,
            delayed_feature_indices
        )
        best_evaluation = choose_better_or_pruned(best_evaluation, current_evaluation)
        if current_evaluation.feature_count > 1:
            refined_evaluation = bisect_inside_pool(
                current_evaluation,
                current_evaluation,
                current_evaluation.active_indices,
                best_evaluation,
                initial_step=max(1, current_evaluation.feature_count // 2)
            )
            best_evaluation = choose_better_or_pruned(best_evaluation, refined_evaluation)
            if debug:
                print(
                    "Feature-selection stopped: "
                    "delayed features refined by final bisection"
                )

        final_indices_array = np.array(best_evaluation.active_indices, dtype=np.int64)
        final_sxx, final_sxc, final_scc = compute_candidate_statistics(final_indices_array)

        if debug:
            debug_started_at = time.perf_counter()
            final_denominators_by_experiment = compute_level2_denominators_by_experiment(
                level2_feature_scales,
                final_indices_array
            )
            final_features_data, _ = build_level2_active_batch_arrays(
                base_features_data,
                prepared_batch.experiment_ranges,
                level2_feature_scales,
                final_indices_array,
                final_denominators_by_experiment
            )
            debug_seconds += time.perf_counter() - debug_started_at
        else:
            final_features_data = None

        return (
            best_evaluation,
            final_features_data,
            final_sxx,
            final_sxc,
            final_scc,
            first_solver_started_at or best_evaluation.solver_started_at,
            total_solver_seconds if total_solver_seconds > 0.0 else best_evaluation.solver_seconds,
            debug_seconds
        )

    def _optimize_fixed_level2_feature_set(
        self,
        prepared_batch: GeneoPreparedBatch,
        active_feature_indices: np.ndarray,
        debug: bool
    ) -> Tuple[EvaluatedFeatureSet, np.ndarray | None, np.ndarray, np.ndarray, float, float]:
        level2_denominators_by_experiment = compute_level2_denominators_by_experiment(
            prepared_batch.level2_feature_scales_by_experiment,
            active_feature_indices
        )
        batch_sxx, batch_sxc, batch_scc = compute_level2_active_batch_statistics(
            prepared_batch.features_data,
            prepared_batch.experiment_ranges,
            prepared_batch.level2_feature_scales_by_experiment,
            active_feature_indices,
            level2_denominators_by_experiment,
            prepared_batch.worker_count,
            prepared_batch.experiment_blocks
        )
        initial_weights = make_feasible_initial_weights(batch_sxc)

        try:
            (
                result,
                w_opt,
                solver_started_at,
                _,
                mean_solver_seconds
            ) = optimize_level2_weights(
                batch_sxx,
                batch_sxc,
                batch_scc,
                prepared_batch.total_rows,
                initial_weights,
                debug
            )
        except Exception as exc:
            raise GeneoCoreExecutionError(
                "Level-2 optimization raised an exception: "
                f"{exc}"
            ) from exc

        (
            optimizer_success,
            optimizer_status,
            optimizer_message,
            optimizer_total_loss
        ) = extract_optimizer_metadata(result)

        if is_optimizer_hard_failure(result):
            raise GeneoCoreExecutionError(
                "Level-2 optimization failed with optimizer status "
                f"{optimizer_status}: {optimizer_message}"
            )

        selected_feature_set = EvaluatedFeatureSet(
            active_indices=tuple(int(feature_index) for feature_index in active_feature_indices),
            weights=w_opt,
            solver_started_at=solver_started_at,
            solver_seconds=mean_solver_seconds,
            optimizer_success=optimizer_success,
            optimizer_status=optimizer_status,
            optimizer_message=optimizer_message,
            optimizer_total_loss=optimizer_total_loss
        )

        debug_seconds = 0.0
        if debug:
            debug_started_at = time.perf_counter()
            active_features_data, _ = build_level2_active_batch_arrays(
                prepared_batch.features_data,
                prepared_batch.experiment_ranges,
                prepared_batch.level2_feature_scales_by_experiment,
                active_feature_indices,
                level2_denominators_by_experiment
            )
            debug_seconds += time.perf_counter() - debug_started_at
        else:
            active_features_data = None

        return (
            selected_feature_set,
            active_features_data,
            batch_sxx,
            batch_sxc,
            batch_scc,
            debug_seconds
        )

    def _initialize_from_prepared_batch(
        self,
        prepared_batch: GeneoPreparedBatch,
        feature_names: List[str],
        select_all_features: bool,
        min_correlation_threshold: float,
        debug: bool,
        persist_state: bool,
        request_index: int
    ) -> GeneoCoreExecutionState:
        if prepared_batch.level1_statistics is None:
            if not select_all_features:
                raise GeneoCoreExecutionError(
                    "Feature selection requires level-1 statistics, but the "
                    "prepared batch does not provide them"
                )
            mean_correlations = np.zeros(
                prepared_batch.features_data.shape[1],
                dtype=np.float64
            )
        else:
            mean_correlations = prepared_batch.level1_statistics.mean_correlations.copy()
        invalid_feature_mask = prepared_batch.invalid_feature_mask.copy()
        if np.any(invalid_feature_mask):
            # Features that are identically zero in every experiment of the
            # current batch must never participate in LVL2 optimization.
            mean_correlations[invalid_feature_mask] = 0.0
            prepared_batch.features_data[:, invalid_feature_mask] = 0.0

        debug_seconds = 0.0
        active_feature_mask = build_valid_level1_feature_mask(invalid_feature_mask)
        if select_all_features:
            active_feature_indices = np.flatnonzero(active_feature_mask)
            (
                selected_feature_set,
                active_features_data,
                batch_sxx,
                batch_sxc,
                batch_scc,
                feature_debug_seconds
            ) = self._optimize_fixed_level2_feature_set(
                prepared_batch,
                active_feature_indices,
                debug
            )
            debug_seconds += feature_debug_seconds
            solver_started_at = selected_feature_set.solver_started_at
            solver_seconds = selected_feature_set.solver_seconds
            if debug:
                print(
                    "Feature-selection skipped: "
                    f"using all {selected_feature_set.feature_count} valid features"
                )
        elif min_correlation_threshold > 0.0:
            threshold_feature_mask = build_min_correlation_feature_mask(
                active_feature_mask,
                mean_correlations,
                min_correlation_threshold
            )
            active_feature_indices = np.flatnonzero(threshold_feature_mask)
            (
                selected_feature_set,
                active_features_data,
                batch_sxx,
                batch_sxc,
                batch_scc,
                feature_debug_seconds
            ) = self._optimize_fixed_level2_feature_set(
                prepared_batch,
                active_feature_indices,
                debug
            )
            debug_seconds += feature_debug_seconds
            solver_started_at = selected_feature_set.solver_started_at
            solver_seconds = selected_feature_set.solver_seconds
            if debug:
                print(
                    "Adaptive feature-selection skipped: "
                    f"using {selected_feature_set.feature_count} feature(s) "
                    "with mean correlation >= "
                    f"{min_correlation_threshold}"
                )
        else:
            (
                selected_feature_set,
                active_features_data,
                batch_sxx,
                batch_sxc,
                batch_scc,
                solver_started_at,
                solver_seconds,
                feature_debug_seconds
            ) = (
                self._select_level2_feature_set(
                    prepared_batch,
                    mean_correlations,
                    active_feature_mask,
                    debug
                )
            )
            debug_seconds += feature_debug_seconds
            active_feature_indices = np.array(selected_feature_set.active_indices, dtype=np.int64)
        active_feature_names = [
            feature_names[feature_index]
            for feature_index in active_feature_indices
        ]
        if debug:
            debug_started_at = time.perf_counter()
            active_level1_features_data = prepared_batch.features_data[:, active_feature_indices]
            response_critical_data = prepared_batch.critical_data.copy()
            debug_seconds += time.perf_counter() - debug_started_at
        else:
            active_level1_features_data = None
            response_critical_data = None
        active_mean_correlations = mean_correlations[active_feature_indices]

        response_feature_names = active_feature_names
        response_features_data = active_features_data
        response_level1_features_data = active_level1_features_data
        response_mean_correlations = active_mean_correlations
        response_weights = selected_feature_set.weights

        (
            optimizer_success,
            optimizer_status,
            optimizer_message,
            optimizer_total_loss
        ) = (
            selected_feature_set.optimizer_success,
            selected_feature_set.optimizer_status,
            selected_feature_set.optimizer_message,
            selected_feature_set.optimizer_total_loss
        )

        runtime_state: GeneoRuntimeState | None = None
        if persist_state:
            runtime_state = build_runtime_state(
                target_column_name=self.target_column_name,
                surviving_feature_names=active_feature_names,
                surviving_mean_correlations=active_mean_correlations,
                frozen_select_all_features=select_all_features,
                sxx=batch_sxx,
                sxc=batch_sxc,
                scc=batch_scc,
                total_rows_seen=prepared_batch.total_rows,
                last_weights=response_weights,
                completed_requests_count=request_index
            )
            self.runtime_state = runtime_state

        # The incremental state only needs the cumulative sufficient statistics.
        # Keep the normalized batch arrays only when debug artifacts will use them.
        return GeneoCoreExecutionState(
            runtime_state=runtime_state,
            feature_names=response_feature_names,
            features_data=response_features_data if debug else None,
            level1_features_data=response_level1_features_data if debug else None,
            critical_data=response_critical_data,
            mean_correlations=response_mean_correlations,
            weights=response_weights,
            solver_started_at=solver_started_at,
            solver_seconds=solver_seconds,
            debug_seconds=debug_seconds,
            request_index=request_index,
            continual_learning_enabled=persist_state,
            is_initial_training=True,
            response_status="ok",
            optimizer_success=optimizer_success,
            optimizer_status=optimizer_status,
            optimizer_message=optimizer_message,
            optimizer_total_loss=optimizer_total_loss
        )

    def initialize(
        self,
        data_matrix: np.ndarray,
        experiment_ranges: List[Tuple[int, int]],
        feature_names: List[str],
        select_all_features: bool,
        min_correlation_threshold: float,
        debug: bool
    ) -> GeneoCoreExecutionState:
        # Initial continual-learning bootstrap shares the standalone LVL2
        # path: collect alpha values first, then either run adaptive feature
        # selection or keep every valid feature before computing D_j.
        # Initialization runs on a private core model instance that is not
        # published to the global registry until the whole request succeeds.
        # Keep the expensive level-2 setup outside _state_lock so only true
        # incremental updates are serialized.
        return self._initialize_from_prepared_batch(
            self._prepare_batch(
                data_matrix,
                experiment_ranges,
                compute_level1=not select_all_features,
            ),
            feature_names,
            select_all_features,
            min_correlation_threshold,
            debug,
            persist_state=True,
            request_index=1
        )

    def run_standalone(
        self,
        data_matrix: np.ndarray,
        experiment_ranges: List[Tuple[int, int]],
        feature_names: List[str],
        select_all_features: bool,
        min_correlation_threshold: float,
        debug: bool,
        request_index: int
    ) -> GeneoCoreExecutionState:
        # Single-pass / standalone execution matches continual bootstrap at
        # LVL2: feature selection is decided before D_j is computed.
        return self._initialize_from_prepared_batch(
            self._prepare_batch(
                data_matrix,
                experiment_ranges,
                compute_level1=not select_all_features,
            ),
            feature_names,
            select_all_features,
            min_correlation_threshold,
            debug,
            persist_state=False,
            request_index=request_index
        )

    def update(
        self,
        data_matrix: np.ndarray,
        loaded_feature_names: Sequence[str],
        experiment_ranges: List[Tuple[int, int]],
        debug: bool
    ) -> GeneoCoreExecutionState:
        with self._state_lock:
            current_state = self.runtime_state
            if current_state is None:
                raise GeneoCoreExecutionError(
                    "Continual update requested before the model runtime state was initialized"
                )

            loaded_feature_names_tuple = tuple(loaded_feature_names)
            current_feature_names = current_state.surviving_feature_names
            if loaded_feature_names_tuple != current_feature_names:
                loaded_feature_indices = {
                    feature_name: feature_index
                    for feature_index, feature_name in enumerate(loaded_feature_names_tuple)
                }
                try:
                    current_feature_indices = [
                        loaded_feature_indices[feature_name]
                        for feature_name in current_feature_names
                    ]
                except KeyError as exc:
                    raise GeneoCoreValidationError(
                        "Continual update CSV does not match the current surviving feature set"
                    ) from exc

                aligned_data_matrix = np.empty(
                    (data_matrix.shape[0], len(current_feature_indices) + 1),
                    dtype=data_matrix.dtype
                )
                aligned_data_matrix[:, :-1] = data_matrix[:, current_feature_indices]
                aligned_data_matrix[:, -1] = data_matrix[:, -1]
                data_matrix = aligned_data_matrix

            prepared_batch = self._prepare_batch(
                data_matrix,
                experiment_ranges,
                compute_level1=False,
            )
            # Incremental continual-learning updates keep using the frozen
            # surviving feature set from the initial training. For the new
            # batch, D_j is computed on the currently non-flat surviving
            # features; features that are flat in this batch contribute zero
            # without rewriting the full feature matrix.
            batch_active_feature_mask = ~prepared_batch.invalid_feature_mask
            frozen_mean_correlations = current_state.surviving_mean_correlations
            current_sxx = current_state.sxx
            current_sxc = current_state.sxc
            current_last_weights = current_state.last_weights

            (
                batch_active_feature_indices,
                batch_active_sxx,
                batch_active_sxc,
                batch_active_scc
            ) = compute_level2_active_batch_statistics_for_active_mask(
                prepared_batch.features_data,
                experiment_ranges,
                prepared_batch.level2_feature_scales_by_experiment,
                batch_active_feature_mask,
                prepared_batch.worker_count,
                prepared_batch.experiment_blocks
            )

            total_rows_seen = current_state.total_rows_seen + prepared_batch.total_rows

            updated_sxx = current_sxx.copy()
            if batch_active_feature_indices.size > 0:
                active_index_grid = np.ix_(
                    batch_active_feature_indices,
                    batch_active_feature_indices
                )
                updated_sxx[active_index_grid] += batch_active_sxx
            updated_sxc = current_sxc.copy()
            if batch_active_feature_indices.size > 0:
                updated_sxc[batch_active_feature_indices] += batch_active_sxc
            updated_scc = current_state.scc + batch_active_scc
            response_status = "ok_state_unchanged"
            request_index = current_state.completed_requests_count
            w_opt = current_last_weights
            next_runtime_state = current_state

            try:
                (
                    result,
                    candidate_w_opt,
                    solver_started_at,
                    solver_seconds,
                    mean_solver_seconds
                ) = optimize_level2_weights(
                    updated_sxx,
                    updated_sxc,
                    updated_scc,
                    total_rows_seen,
                    make_feasible_initial_weights(current_last_weights),
                    debug
                )
                if current_state.frozen_select_all_features:
                    solver_seconds = mean_solver_seconds
            except Exception as exc:
                optimizer_success = False
                optimizer_status = None
                optimizer_message = (
                    "Continual update skipped because level-2 optimization raised "
                    f"an exception: {str(exc)}"
                )
                optimizer_total_loss = None
                solver_started_at = time.perf_counter()
                solver_seconds = 0.0
            else:
                (
                    optimizer_success,
                    optimizer_status,
                    optimizer_message,
                    optimizer_total_loss
                ) = extract_optimizer_metadata(result)

                if not is_optimizer_hard_failure(result):
                    w_opt = candidate_w_opt
                    next_runtime_state = build_runtime_state(
                        target_column_name=current_state.target_column_name,
                        surviving_feature_names=current_feature_names,
                        surviving_mean_correlations=frozen_mean_correlations,
                        frozen_select_all_features=current_state.frozen_select_all_features,
                        sxx=updated_sxx,
                        sxc=updated_sxc,
                        scc=updated_scc,
                        total_rows_seen=total_rows_seen,
                        last_weights=w_opt,
                        completed_requests_count=current_state.completed_requests_count + 1
                    )
                    self.runtime_state = next_runtime_state
                    request_index = next_runtime_state.completed_requests_count
                    response_status = "ok"
                else:
                    optimizer_message = (
                        "Continual update skipped because level-2 optimization failed "
                        f"with optimizer status {optimizer_status}: {optimizer_message}"
                    )

            # As in the standalone path, the update logic only needs the current
            # batch arrays for debug artifact generation, not for future updates.
            debug_seconds = 0.0
            if debug:
                debug_started_at = time.perf_counter()
                debug_features_data = build_full_level2_feature_data_for_active_mask(
                    prepared_batch.features_data,
                    experiment_ranges,
                    prepared_batch.level2_feature_scales_by_experiment,
                    batch_active_feature_mask
                )
                debug_critical_data = prepared_batch.critical_data.copy()
                debug_seconds += time.perf_counter() - debug_started_at
            else:
                debug_features_data = None
                debug_critical_data = None
            return GeneoCoreExecutionState(
                runtime_state=next_runtime_state,
                feature_names=current_feature_names,
                features_data=debug_features_data,
                level1_features_data=debug_features_data,
                critical_data=debug_critical_data,
                mean_correlations=frozen_mean_correlations,
                weights=w_opt,
                solver_started_at=solver_started_at,
                solver_seconds=solver_seconds,
                debug_seconds=debug_seconds,
                request_index=request_index,
                continual_learning_enabled=True,
                is_initial_training=False,
                response_status=response_status,
                optimizer_success=optimizer_success,
                optimizer_status=optimizer_status,
                optimizer_message=optimizer_message,
                optimizer_total_loss=optimizer_total_loss
            )
