import os
import re
import math
import time
import hashlib
import json
import csv
import io
import shutil
import uuid
import warnings
from contextlib import suppress
from dataclasses import dataclass
from threading import Event, Lock
from typing import IO, List, Tuple, Dict, Any, Sequence
import numpy as np
from pydantic import AliasChoices, BaseModel, ConfigDict, Field, ValidationError, field_validator

# The numerical core configuration now lives in geneos_core.py.

_MATPLOTLIB_PYPLOT: Any | None = None

def get_pyplot() -> Any:
    global _MATPLOTLIB_PYPLOT
    if _MATPLOTLIB_PYPLOT is None:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as pyplot
        _MATPLOTLIB_PYPLOT = pyplot
    return _MATPLOTLIB_PYPLOT

GENEO_SUPPORTED_MODES = {"kafka", "api"}
GENEO_DEFAULT_MODE = "api"
KAFKA_DEFAULT_BOOTSTRAP_SERVERS = ""
KAFKA_DEFAULT_TRAINING_TOPIC = ""
KAFKA_DEFAULT_EXPLANATION_TOPIC = ""
KAFKA_DEFAULT_GROUP_ID = ""
KAFKA_DEFAULT_SECURITY_PROTOCOL = "PLAINTEXT"
KAFKA_SUPPORTED_SECURITY_PROTOCOLS = {"PLAINTEXT", "SSL"}

# =============================================================================
# FastAPI application metadata
# =============================================================================
GENEO_API_DESCRIPTION = (
    "Two-level GENEO analysis for CSV time-series batches.\n\n"
    "`/analyze` receives multipart `config` JSON plus a separate `dataset_file` "
    "CSV. Service settings are read from environment variables at "
    "startup. With continual learning enabled, the first request selects "
    "features and later requests for the same experiment reuse them."
)

API_REFERENCE_FILE_PATH = os.path.abspath(__file__)
SERVICE_CONFIG_BOOLEAN_ENV_FIELD_NAMES = {
    "DEBUG_MODE": "debug_mode",
    "CONTINUAL_LEARNING_MODE": "continual_learning_mode",
    "SELECT_ALL_FEATURES": "select_all_features",
}
SERVICE_CONFIG_DEFAULT_VALUES = {
    "debug_mode": False,
    "continual_learning_mode": True,
    "select_all_features": False,
    "min_correlation_threshold": 0.0,
}
GENEO_DEFAULT_HOST = "0.0.0.0:8000"

GENEO_CONFIG_BOOTSTRAP_EXAMPLE_DATA = {
    "experiment_name": "exp_bootstrap",
    "target_column_name": "critical_signal",
    "experiments_rows": [120, 120, 120]
}

GENEO_CONFIG_INCREMENTAL_EXAMPLE_DATA = {
    "experiment_name": "exp_bootstrap",
    "target_column_name": "critical_signal",
    "experiments_rows": [120, 120]
}

GENEO_SERVICE_CONFIG_EXAMPLE_DATA = {
    **SERVICE_CONFIG_DEFAULT_VALUES,
}

GENEO_CONFIG_BOOTSTRAP_EXAMPLE = json.dumps(
    GENEO_CONFIG_BOOTSTRAP_EXAMPLE_DATA,
    indent=2
)
GENEO_CONFIG_INCREMENTAL_EXAMPLE = json.dumps(
    GENEO_CONFIG_INCREMENTAL_EXAMPLE_DATA,
    indent=2
)
GENEO_CONFIG_FORM_DESCRIPTION = (
    "API JSON string with `experiment_name`, `target_column_name`, and "
    "`experiments_rows`. Upload the CSV separately as `dataset_file`; service "
    "settings are environment variables, not request fields."
)

GENEO_DATASET_FILE_FORM_DESCRIPTION = (
    "CSV multipart upload. The target column must match `target_column_name`, "
    "and rows must equal `sum(experiments_rows)`."
)


from geneos_core import (
    GeneoCoreExecutionError,
    GeneoCoreExecutionState,
    GeneoCoreModel,
    GeneoCoreValidationError,
    build_experiment_ranges,
)

@dataclass
class GeneoModelEntry:
    # The first request creates a placeholder entry and initializes it.
    # Later requests either wait for readiness or reuse the published core model.
    ready_event: Event
    core_model: Any = None
    is_initializing: bool = True


class GeneoModelRegistry:
    def __init__(self):
        self._entries: Dict[str, GeneoModelEntry] = {}
        self._lock = Lock()

    def acquire_or_wait(self, experiment_name: str) -> Tuple[GeneoModelEntry, bool]:
        while True:
            with self._lock:
                entry = self._entries.get(experiment_name)
                if entry is None:
                    # Reserve this experiment name immediately so concurrent
                    # requests wait instead of starting a second initialization.
                    entry = GeneoModelEntry(ready_event=Event())
                    self._entries[experiment_name] = entry
                    return entry, True
                if not entry.is_initializing:
                    return entry, False
                ready_event = entry.ready_event

            # Another request is still publishing the initial immutable state for
            # this experiment name. Wait until it becomes ready or is discarded.
            ready_event.wait()

    def publish_initialized(
        self,
        experiment_name: str,
        entry: GeneoModelEntry,
        core_model: "GeneoCoreModel"
    ) -> None:
        with self._lock:
            if self._entries.get(experiment_name) is entry:
                entry.core_model = core_model
                entry.is_initializing = False
                entry.ready_event.set()

    def discard_initializing(self, experiment_name: str, entry: GeneoModelEntry) -> None:
        with self._lock:
            if self._entries.get(experiment_name) is entry:
                self._entries.pop(experiment_name, None)
                entry.ready_event.set()


MODEL_REGISTRY = GeneoModelRegistry()

class StandaloneOutputRegistry:
    def __init__(self):
        self._next_indices: Dict[str, int] = {}
        self._lock = Lock()

    def reserve_next(self, config_file_path: str, experiment_name: str) -> int:
        output_root_dir = build_experiment_output_root_dir(config_file_path, experiment_name)
        with self._lock:
            next_index = self._next_indices.get(output_root_dir)
            if next_index is None:
                next_index = get_next_standalone_request_index(config_file_path, experiment_name)
            self._next_indices[output_root_dir] = next_index + 1
            return next_index


STANDALONE_OUTPUT_REGISTRY = StandaloneOutputRegistry()

# =============================================================================
# Configuration model
# =============================================================================
class GeneoConfig(BaseModel):
    """Validated analysis configuration loaded from the API multipart config field."""

    # Request fields are intentionally strict: deployment flags and legacy
    # thresholds must not be sent inside the API config payload.
    model_config = ConfigDict(
        extra="forbid",
        json_schema_extra={
            "examples": [
                GENEO_CONFIG_BOOTSTRAP_EXAMPLE_DATA,
                GENEO_CONFIG_INCREMENTAL_EXAMPLE_DATA
            ]
        }
    )

    experiment_name: str = Field(
        description=(
            "Experiment identifier used as the continual-learning key."
        )
    )
    target_column_name: str = Field(
        description=(
            "CSV target column used to build the critical kernel."
        )
    )
    experiments_rows: List[int] = Field(
        description=(
            "Rows per experiment, in dataset order."
        )
    )

    @field_validator("experiment_name")
    @classmethod
    def validate_experiment_name(cls, value: str) -> str:
        normalized_value = value.strip()
        if not normalized_value:
            raise ValueError("experiment_name cannot be empty")
        return normalized_value

    @field_validator("experiments_rows")
    @classmethod
    def validate_experiments_rows(cls, value: List[int]) -> List[int]:
        if not value:
            raise ValueError("experiments_rows cannot be empty")
        if any(experiment_length <= 0 for experiment_length in value):
            raise ValueError("experiments_rows must contain only positive integers")
        return value


class GeneoServiceConfig(BaseModel):
    """Informational schema for service settings loaded at server startup."""

    model_config = ConfigDict(
        extra="forbid",
        json_schema_extra={
            "example": GENEO_SERVICE_CONFIG_EXAMPLE_DATA
        }
    )

    debug_mode: bool = Field(
        default=False,
        description=(
            "Loaded from DEBUG_MODE. If True, save debug plots and files."
        )
    )
    continual_learning_mode: bool = Field(
        default=True,
        description=(
            "Loaded from CONTINUAL_LEARNING_MODE. If True, reuse the in-memory "
            "model for incremental updates."
        )
    )
    select_all_features: bool = Field(
        default=False,
        description=(
            "Loaded from SELECT_ALL_FEATURES. If True, skip adaptive feature "
            "selection and keep all valid features."
        )
    )
    min_correlation_threshold: float = Field(
        default=0.0,
        ge=0.0,
        description=(
            "Loaded from MIN_CORRELATION_THRESHOLD only when SELECT_ALL_FEATURES "
            "is false. If greater than 0, use threshold-based level-1 feature "
            "selection instead of adaptive pruning."
        )
    )


class GeneoFeatureResponse(BaseModel):
    name: str = Field(description="Feature name as loaded from the CSV header.")
    weight: float = Field(
        description="Optimized non-negative level-2 weight associated with the feature."
    )
    mean_correlation: float = Field(
        description=(
            "Frozen level-1 weighted mean correlation used for ranking and feature scaling."
        )
    )


class GeneoAnalyzeResponse(BaseModel):
    explanationId: str = Field(
        description="Request-scoped identifier generated by the API."
    )
    timestamp: int = Field(
        description="Unix timestamp in seconds when the response was produced."
    )
    features: List[GeneoFeatureResponse] = Field(
        description=(
            "Active features returned by the run, with final weights."
        )
    )

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "explanationId": "a1b2c3d4e5f678901234567890abcdef",
                "timestamp": 1777286400,
                "features": [
                    {
                        "name": "sensor_1",
                        "weight": 0.31,
                        "mean_correlation": 0.82
                    },
                    {
                        "name": "sensor_2",
                        "weight": 0.69,
                        "mean_correlation": 0.91
                    }
                ]
            }
        }
    )


class GeneoErrorResponse(BaseModel):
    detail: str = Field(description="Human-readable error message.")

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "detail": "Missing 'config' part in multipart request"
            }
        }
    )


class GeneoKafkaTrainingFile(BaseModel):
    """Single CSV file descriptor inside a Kafka training batch."""

    model_config = ConfigDict(extra="ignore")

    file_name: str | None = Field(
        default=None,
        validation_alias=AliasChoices("file_name", "fileName"),
        description="Original CSV file name, when supplied by the upstream service."
    )
    download_url: str = Field(
        validation_alias="downloadUrl",
        description="HTTP(S) URL of the CSV to analyze."
    )

    @field_validator("download_url")
    @classmethod
    def validate_download_url(cls, value: str) -> str:
        normalized_value = value.strip()
        if not normalized_value:
            raise ValueError("file downloadUrl cannot be empty")
        if not normalized_value.startswith("https://"):
            raise ValueError("file downloadUrl must start with 'https://'")
        return normalized_value


class GeneoKafkaTrainingMessage(BaseModel):
    """Validated Kafka message used to start a GENEOS run."""

    model_config = ConfigDict(extra="ignore")

    use_case: str = Field(
        validation_alias="uc",
        description="Kafka uc identifier used to derive experiment_name."
    )
    scenario: str = Field(description="Scenario identifier used to derive experiment_name.")
    timestamp: str = Field(description="Upstream batch timestamp.")
    total_files: int = Field(description="Expected number of CSV files in the batch.")
    files: List[GeneoKafkaTrainingFile] = Field(description="CSV files belonging to the batch.")

    @field_validator("use_case", "scenario", "timestamp")
    @classmethod
    def validate_required_text(cls, value: str) -> str:
        normalized_value = value.strip()
        if not normalized_value:
            raise ValueError("required Kafka message fields cannot be empty")
        return normalized_value

    @field_validator("total_files")
    @classmethod
    def validate_total_files(cls, value: int) -> int:
        if value <= 0:
            raise ValueError("total_files must be a positive integer")
        return value

    @field_validator("files")
    @classmethod
    def validate_files(cls, value: List[GeneoKafkaTrainingFile]) -> List[GeneoKafkaTrainingFile]:
        if not value:
            raise ValueError("files cannot be empty")
        return value

    def validate_file_count(self) -> None:
        if len(self.files) != self.total_files:
            raise ValueError(
                f"files contains {len(self.files)} items, expected total_files={self.total_files}"
            )

    @property
    def experiment_name(self) -> str:
        return f"{self.use_case}_{self.scenario}"


@dataclass(frozen=True)
class KafkaRuntimeConfig:
    bootstrap_servers: str
    training_topic: str
    explanation_topic: str
    group_id: str
    security_protocol: str
    ssl_certfile: str | None
    ssl_keyfile: str | None
    ssl_cafile: str | None


@dataclass(frozen=True)
class DownloadedKafkaTrainingFile:
    file_name: str | None
    download_url: str
    csv_file: io.BytesIO


@dataclass(frozen=True)
class DownloadedKafkaTrainingBatch:
    experiment_name: str
    timestamp: str
    files: List[DownloadedKafkaTrainingFile]


class ServiceConfigurationError(RuntimeError):
    """Raised when deployment-time service configuration is invalid."""


def raise_http_exception(status_code: int, detail: str) -> None:
    from fastapi import HTTPException

    raise HTTPException(status_code=status_code, detail=detail)


def load_runtime_mode() -> str:
    mode = os.getenv("MODE", GENEO_DEFAULT_MODE).strip().lower()
    if mode not in GENEO_SUPPORTED_MODES:
        supported_modes = ", ".join(sorted(GENEO_SUPPORTED_MODES))
        raise ServiceConfigurationError(
            f"Invalid MODE value {mode!r}; expected one of: {supported_modes}"
        )
    return mode


def read_environment_value(env_name: str, default_value: str) -> str:
    raw_value = os.getenv(env_name)
    return default_value if raw_value is None else raw_value.strip()


def read_non_empty_environment_value(env_name: str, default_value: str) -> str:
    value = read_environment_value(env_name, default_value)
    if not value:
        raise ServiceConfigurationError(f"Environment variable {env_name} cannot be empty")
    return value


def load_kafka_runtime_config() -> KafkaRuntimeConfig:
    security_protocol = read_non_empty_environment_value(
        "KAFKA_SECURITY_PROTOCOL",
        KAFKA_DEFAULT_SECURITY_PROTOCOL
    ).upper()
    if security_protocol not in KAFKA_SUPPORTED_SECURITY_PROTOCOLS:
        supported_protocols = ", ".join(sorted(KAFKA_SUPPORTED_SECURITY_PROTOCOLS))
        raise ServiceConfigurationError(
            f"Invalid KAFKA_SECURITY_PROTOCOL value {security_protocol!r}; "
            f"expected one of: {supported_protocols}"
        )

    ssl_certfile = os.getenv("KAFKA_SSL_CERTFILE")
    ssl_keyfile = os.getenv("KAFKA_SSL_KEYFILE")
    ssl_cafile = os.getenv("KAFKA_SSL_CAFILE")

    if security_protocol == "SSL":
        missing_ssl_values = [
            env_name
            for env_name, value in (
                ("KAFKA_SSL_CERTFILE", ssl_certfile),
                ("KAFKA_SSL_KEYFILE", ssl_keyfile),
                ("KAFKA_SSL_CAFILE", ssl_cafile),
            )
            if value is None or not value.strip()
        ]
        if missing_ssl_values:
            raise ServiceConfigurationError(
                "Missing Kafka SSL configuration: " + ", ".join(missing_ssl_values)
            )

    return KafkaRuntimeConfig(
        bootstrap_servers=read_non_empty_environment_value(
            "KAFKA_BOOTSTRAP_SERVERS",
            KAFKA_DEFAULT_BOOTSTRAP_SERVERS
        ),
        training_topic=read_non_empty_environment_value(
            "KAFKA_TRAINING_TOPIC",
            KAFKA_DEFAULT_TRAINING_TOPIC
        ),
        explanation_topic=read_non_empty_environment_value(
            "KAFKA_EXPLANATION_TOPIC",
            KAFKA_DEFAULT_EXPLANATION_TOPIC
        ),
        group_id=read_environment_value("KAFKA_GROUP_ID", KAFKA_DEFAULT_GROUP_ID),
        security_protocol=security_protocol,
        ssl_certfile=None if ssl_certfile is None else ssl_certfile.strip(),
        ssl_keyfile=None if ssl_keyfile is None else ssl_keyfile.strip(),
        ssl_cafile=None if ssl_cafile is None else ssl_cafile.strip(),
    )


def build_confluent_common_config(config: KafkaRuntimeConfig) -> Dict[str, str]:
    kafka_config = {
        "bootstrap.servers": config.bootstrap_servers,
    }
    if config.security_protocol == "SSL":
        kafka_config.update(
            {
                "security.protocol": "ssl",
                "ssl.certificate.location": config.ssl_certfile or "",
                "ssl.key.location": config.ssl_keyfile or "",
                "ssl.ca.location": config.ssl_cafile or "",
            }
        )
    return kafka_config


def build_confluent_consumer_config(config: KafkaRuntimeConfig) -> Dict[str, str]:
    consumer_config = build_confluent_common_config(config)
    consumer_config["auto.offset.reset"] = "earliest"
    if config.group_id.strip():
        consumer_config["group.id"] = config.group_id.strip()
    return consumer_config


def debug_print(message: str, debug_enabled: bool) -> None:
    if debug_enabled:
        print(message)


def stringify_exception(exc: Exception) -> str:
    detail = getattr(exc, "detail", None)
    if detail:
        return str(detail)
    return str(exc)


# =============================================================================
# Error handling helpers
# =============================================================================
def parse_request_config(config_payload: str) -> GeneoConfig:
    try:
        return GeneoConfig.model_validate_json(config_payload)
    except ValidationError as exc:
        raise_http_exception(status_code=400, detail=f"Invalid configuration payload: {str(exc)}")


def parse_bool_environment_value(env_name: str, raw_value: str) -> bool:
    normalized_value = raw_value.strip().lower()
    if normalized_value in {"1", "true", "yes", "y", "on"}:
        return True
    if normalized_value in {"0", "false", "no", "n", "off"}:
        return False
    raise ServiceConfigurationError(
        f"Invalid boolean value for environment variable {env_name}: {raw_value!r}"
    )


def parse_non_negative_float_environment_value(env_name: str, raw_value: str) -> float:
    try:
        value = float(raw_value.strip())
    except ValueError as exc:
        raise ServiceConfigurationError(
            f"Invalid numeric value for environment variable {env_name}: {raw_value!r}"
        ) from exc

    if not math.isfinite(value):
        raise ServiceConfigurationError(
            f"Environment variable {env_name} must be a finite number: {raw_value!r}"
        )
    if value < 0.0:
        raise ServiceConfigurationError(
            f"Environment variable {env_name} must be non-negative: {raw_value!r}"
        )
    return value


def load_service_config() -> GeneoServiceConfig:
    config_data: Dict[str, Any] = dict(SERVICE_CONFIG_DEFAULT_VALUES)

    for env_name, field_name in SERVICE_CONFIG_BOOLEAN_ENV_FIELD_NAMES.items():
        raw_value = os.getenv(env_name)
        if raw_value is not None:
            config_data[field_name] = parse_bool_environment_value(env_name, raw_value)

    if not bool(config_data["select_all_features"]):
        raw_threshold = os.getenv("MIN_CORRELATION_THRESHOLD")
        if raw_threshold is not None:
            config_data["min_correlation_threshold"] = (
                parse_non_negative_float_environment_value(
                    "MIN_CORRELATION_THRESHOLD",
                    raw_threshold
                )
            )

    try:
        return GeneoServiceConfig.model_validate(config_data)
    except ValidationError as exc:
        raise ServiceConfigurationError(f"Invalid service configuration: {str(exc)}") from exc


SERVICE_CONFIG: GeneoServiceConfig | None = None


def get_service_config() -> GeneoServiceConfig:
    global SERVICE_CONFIG
    if SERVICE_CONFIG is None:
        SERVICE_CONFIG = load_service_config()
    return SERVICE_CONFIG


def load_uvicorn_bind_config() -> Tuple[str, int]:
    endpoint = os.getenv("GENEO_HOST", GENEO_DEFAULT_HOST).strip()
    if not endpoint:
        raise ServiceConfigurationError(
            "Environment variable GENEO_HOST cannot be empty"
        )

    normalized_endpoint = endpoint.replace("http://", "").replace("https://", "")
    host, separator, raw_port = normalized_endpoint.rpartition(":")
    if not separator or not host or not raw_port:
        raise ServiceConfigurationError(
            "Environment variable GENEO_HOST must have format <host>:<port>"
        )
    if host.startswith("[") and host.endswith("]"):
        host = host[1:-1]

    try:
        port = int(raw_port)
    except ValueError as exc:
        raise ServiceConfigurationError(
            f"Port in environment variable GENEO_HOST must be an integer: {raw_port!r}"
        ) from exc

    if port < 1 or port > 65535:
        raise ServiceConfigurationError(
            "Port in environment variable GENEO_HOST must be between 1 and 65535"
        )

    return host, port


def safe_csv_load(operation):
    try:
        return operation()
    except ValueError as exc:
        raise_http_exception(status_code=400, detail=str(exc))

def windows_long_path(path: str) -> str:
    """
    Bypass Windows MAX_PATH limitations by using the '\\\\?\\' prefix.
    Matplotlib/PIL sometimes raises FileNotFoundError when the path is too long.
    This function expects/returns an absolute path.
    """
    abs_path = os.path.abspath(path)

    if os.name != "nt":
        return abs_path

    if abs_path.startswith("\\\\?\\"):
        return abs_path

    # UNC path handling: \\server\share\... -> \\?\UNC\server\share\...
    if abs_path.startswith("\\\\"):
        return "\\\\?\\UNC\\" + abs_path.lstrip("\\")
    return "\\\\?\\" + abs_path

# =============================================================================
# Filename helpers (readable but short + safe)
# =============================================================================
def sanitize_filename(name: str) -> str:
    # Replace Windows-invalid characters with underscore
    name = re.sub(r'[<>:"/\\|?*\x00-\x1f{}]', "_", name)
    # Collapse repeated underscores/spaces
    name = re.sub(r"[\s_]+", "_", name).strip("_")
    return name

def short_hash(text: str, n: int = 8) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()[:n]

def build_experiment_output_dirname(experiment_name: str) -> str:
    # Keep the directory name readable while making collisions between
    # different raw experiment names practically impossible.
    safe_name = sanitize_filename(experiment_name) or "experiment"
    return f"{safe_name}__{short_hash(experiment_name)}"

def shorten_label_like_name(s: str) -> str:
    """
    Make Prometheus-like metric names shorter but still readable.
    """
    s = s.replace("exported_job=", "ej=")
    s = s.replace("instance=", "i=")
    s = s.replace("job=", "j=")
    s = s.replace("device=", "d=")
    s = s.replace("label", "l")  # label1 -> l1, label2 -> l2

    # Make separators filesystem-friendly
    s = s.replace(":", "_")
    s = s.replace(",", "_")
    return s

def make_readable_safe_filename(feature_name: str, suffix: str, max_len: int = 100) -> str:
    """
    Returns a readable, Windows-safe filename that still points to the feature name.
    Format:
      <sanitized_short_prefix>__<hash>_<suffix>.png
    """
    readable = shorten_label_like_name(feature_name)
    readable = sanitize_filename(readable)

    h = short_hash(feature_name, n=8)
    tail = f"__{h}_{suffix}.png"

    # Leave room for the tail
    keep = max_len - len(tail)
    if keep < 20:
        keep = 20

    return f"{readable[:keep].rstrip('_')}{tail}"

# =============================================================================
# CSV loading helpers
# =============================================================================
class CsvLoader:
    def __init__(self, csv_file: IO[bytes]):
        self.csv_file = csv_file

    def _reset_stream(self) -> None:
        try:
            self.csv_file.seek(0)
        except (AttributeError, OSError, ValueError) as exc:
            raise ValueError(f"Unable to read uploaded CSV: {exc}") from exc

    def _open_reader(self) -> Tuple[io.TextIOWrapper, Any]:
        self._reset_stream()
        text_stream = io.TextIOWrapper(self.csv_file, encoding="utf-8-sig", newline="")
        return text_stream, csv.reader(text_stream)

    def _read_header(self, reader: Any) -> Tuple[List[str], Dict[str, int]]:
        try:
            header = next(reader)
        except StopIteration as exc:
            raise ValueError("Uploaded CSV is empty") from exc

        if not header:
            raise ValueError("CSV header is empty")

        return header, {column_name: column_index for column_index, column_name in enumerate(header)}

    def _iter_data_rows(self, reader: Any, header_len: int):
        data_row_index = 0
        for csv_row_number, row in enumerate(reader, start=2):
            if not row or not any(cell.strip() for cell in row):
                continue

            if len(row) > header_len:
                raise ValueError(
                    f"CSV row {csv_row_number} has {len(row)} columns, expected {header_len}"
                )

            if len(row) < header_len:
                row = row + [""] * (header_len - len(row))

            data_row_index += 1
            yield data_row_index, row

    def _resolve_selected_indices(
        self,
        header_index: Dict[str, int],
        selected_columns: List[str]
    ) -> List[int]:
        missing_columns = [
            column_name
            for column_name in selected_columns
            if column_name not in header_index
        ]
        if missing_columns:
            raise ValueError(
                "CSV is missing required columns: " + ", ".join(missing_columns)
            )
        return [header_index[column_name] for column_name in selected_columns]

    def _ensure_expected_row_count(self, row_count: int, expected_rows: int) -> None:
        if row_count == 0:
            raise ValueError("CSV contains no data rows (excluding header)")
        if row_count != expected_rows:
            raise ValueError(
                f"CSV contains {row_count} data rows, expected {expected_rows}"
            )

    def _load_selected_matrix_with_loadtxt(
        self,
        selected_indices: List[int],
        selected_columns: List[str],
        expected_rows: int
    ) -> np.ndarray:
        text_stream: io.TextIOWrapper | None = None
        try:
            self._reset_stream()
            text_stream = io.TextIOWrapper(self.csv_file, encoding="utf-8-sig", newline="")
            with warnings.catch_warnings():
                warnings.filterwarnings(
                    "ignore",
                    message="loadtxt: input contained no data",
                    category=UserWarning
                )
                data_matrix = np.loadtxt(
                    text_stream,
                    delimiter=",",
                    skiprows=1,
                    usecols=tuple(selected_indices),
                    dtype=np.float64,
                    ndmin=2,
                    comments=None
                )
        except UnicodeDecodeError as exc:
            raise ValueError(
                f"Error decoding uploaded CSV as UTF-8: {exc}"
            ) from exc
        except ValueError as exc:
            raise ValueError(
                "CSV contains invalid or missing numeric data in the selected columns "
                f"({', '.join(selected_columns)}): {exc}"
            ) from exc
        finally:
            if text_stream is not None:
                with suppress(ValueError, OSError):
                    text_stream.detach()

        if data_matrix.ndim != 2 or data_matrix.shape[1] < 2:
            raise ValueError(
                "Data matrix must be 2-dimensional and contain at least one feature column plus the target column"
            )
        if data_matrix.shape[1] != len(selected_columns):
            raise ValueError(
                f"CSV loader returned {data_matrix.shape[1]} columns, "
                f"expected {len(selected_columns)}"
            )
        self._ensure_expected_row_count(int(data_matrix.shape[0]), expected_rows)
        return data_matrix

    def load_bootstrap_matrix(
        self,
        critical_col_name: str,
        expected_rows: int
    ) -> Tuple[np.ndarray, List[str]]:
        text_stream, reader = self._open_reader()
        try:
            header, header_index = self._read_header(reader)
            data_rows = self._iter_data_rows(reader, len(header))
            try:
                _, first_row = next(data_rows)
            except StopIteration as exc:
                raise ValueError("CSV contains no data rows (excluding header)") from exc

            feature_names = [
                column_name
                for column_index, column_name in enumerate(header)
                if column_name != critical_col_name
                and is_numeric_csv_value(first_row[column_index])
            ]
            if not feature_names:
                raise ValueError("No features found in the dataset (after excluding the critical column)")

            selected_columns = feature_names + [critical_col_name]
            selected_indices = self._resolve_selected_indices(header_index, selected_columns)
        except UnicodeDecodeError as exc:
            raise ValueError(
                f"Error decoding uploaded CSV as UTF-8: {exc}"
            ) from exc
        finally:
            with suppress(ValueError, OSError):
                text_stream.detach()

        return (
            self._load_selected_matrix_with_loadtxt(
                selected_indices,
                selected_columns,
                expected_rows
            ),
            feature_names
        )

    def load_selected_timeseries(
        self,
        critical_col_name: str,
        feature_names: Sequence[str],
        expected_rows: int
    ) -> np.ndarray:
        text_stream, reader = self._open_reader()
        try:
            header, header_index = self._read_header(reader)
            selected_columns = list(feature_names) + [critical_col_name]
            selected_indices = self._resolve_selected_indices(header_index, selected_columns)
        except UnicodeDecodeError as exc:
            raise ValueError(
                f"Error decoding uploaded CSV as UTF-8: {exc}"
            ) from exc
        finally:
            with suppress(ValueError, OSError):
                text_stream.detach()

        return self._load_selected_matrix_with_loadtxt(
            selected_indices,
            selected_columns,
            expected_rows
        )


def is_numeric_csv_value(value: Any) -> bool:
    if value is None:
        return False
    if isinstance(value, str):
        value = value.strip()
        if value == "":
            return False
    try:
        parsed_value = float(value)
    except (TypeError, ValueError):
        return False
    return math.isfinite(parsed_value)


def load_incremental_timeseries(
    loader: CsvLoader,
    request_config: GeneoConfig,
    core_model: "GeneoCoreModel",
    expected_rows: int
) -> Tuple[np.ndarray, Tuple[str, ...]]:
    runtime_state = core_model.runtime_state

    if request_config.target_column_name != runtime_state.target_column_name:
        raise ValueError(
            "target_column_name does not match the one used during the initial training "
            f"for experiment '{request_config.experiment_name}'"
        )

    # Continual-learning requests only read the frozen surviving features
    # plus the critical column. Previously excluded features stay excluded.
    return loader.load_selected_timeseries(
        runtime_state.target_column_name,
        runtime_state.surviving_feature_names,
        expected_rows
    ), runtime_state.surviving_feature_names

# Core numerical implementation lives in a separate module.


def build_request_output_leaf_name(request_index: int, continual_learning_enabled: bool) -> str:
    if continual_learning_enabled:
        return str(request_index)
    return f"unico_{request_index}"


def build_request_output_dir(
    config_file_path: str,
    experiment_name: str,
    request_index: int,
    continual_learning_enabled: bool
) -> str:
    return os.path.join(
        os.path.dirname(config_file_path),
        "geneo_plots_optimized",
        build_experiment_output_dirname(experiment_name),
        build_request_output_leaf_name(request_index, continual_learning_enabled),
    )


def build_experiment_output_root_dir(config_file_path: str, experiment_name: str) -> str:
    return os.path.join(
        os.path.dirname(config_file_path),
        "geneo_plots_optimized",
        build_experiment_output_dirname(experiment_name),
    )


def reset_continual_output_dirs(config_file_path: str, experiment_name: str) -> None:
    output_root_dir = build_experiment_output_root_dir(config_file_path, experiment_name)
    if not os.path.isdir(output_root_dir):
        return

    for child_name in os.listdir(output_root_dir):
        child_path = os.path.join(output_root_dir, child_name)
        if os.path.isdir(child_path) and child_name.isdigit():
            shutil.rmtree(windows_long_path(child_path))


def get_next_standalone_request_index(config_file_path: str, experiment_name: str) -> int:
    output_root_dir = build_experiment_output_root_dir(config_file_path, experiment_name)
    if not os.path.isdir(output_root_dir):
        return 1

    next_index = 1
    for child_name in os.listdir(output_root_dir):
        match = re.fullmatch(r"unico_(\d+)", child_name)
        if match is None:
            continue
        next_index = max(next_index, int(match.group(1)) + 1)
    return next_index


def build_feature_response(
    feature_names: Sequence[str],
    weights: np.ndarray,
    mean_correlations: np.ndarray
) -> List[GeneoFeatureResponse]:
    return [
        GeneoFeatureResponse(
            name=feature_name,
            weight=float(weights[index]),
            mean_correlation=float(mean_correlations[index])
        )
        for index, feature_name in enumerate(feature_names)
    ]


def save_critical_vs_approximated_plot(
    output_dir: str,
    critical_data: np.ndarray,
    features_data: np.ndarray,
    weights: np.ndarray
) -> None:
    plt = get_pyplot()
    time_axis = np.arange(len(critical_data))
    approximated_critical_kernels = features_data @ weights

    plt.figure(figsize=(12, 4))
    plt.plot(time_axis, critical_data, label="Centered and L2-normalized critical function c_bar(t)", color="red")
    plt.plot(
        time_axis,
        approximated_critical_kernels,
        label="LVL2 approximation c_hat(t) = sum_i w_i z_i(t)",
        color="green"
    )
    plt.title("Normalized critical function vs LVL2 approximation")
    plt.xlabel("Time")
    plt.legend()
    plt.tight_layout()
    plt.savefig(windows_long_path(os.path.join(output_dir, "critical_vs_approximated.png")))
    plt.close()


def save_request_outputs(
    output_dir: str,
    execution_state: GeneoCoreExecutionState
) -> None:
    plt = get_pyplot()
    os.makedirs(output_dir, exist_ok=True)
    feature_names = execution_state.feature_names
    critical_data = execution_state.critical_data
    features_data = execution_state.features_data
    level1_features_data = execution_state.level1_features_data
    if critical_data is None or features_data is None or level1_features_data is None:
        raise ValueError("Debug artifact generation requires retained batch arrays")
    weights = execution_state.weights

    # Initial-training runs keep the full original artifact set. Continual
    # updates only save the global comparison plot plus the updated weights.
    if execution_state.is_initial_training:
        time_axis = np.arange(len(critical_data))
        for index, feature_name in enumerate(feature_names):
            plt.figure(figsize=(12, 4))
            plt.plot(time_axis, critical_data, label="Centered and L2-normalized critical function c_bar(t)", color="red")
            plt.plot(
                time_axis,
                level1_features_data[:, index],
                label="Centered and L2-normalized time series x_i(t)",
                color="blue"
            )
            plt.title(feature_name)
            plt.xlabel("Time")
            plt.legend()
            plt.tight_layout()
            plt.savefig(
                windows_long_path(
                    os.path.join(output_dir, make_readable_safe_filename(feature_name, "combined"))
                )
            )
            plt.close()

    save_critical_vs_approximated_plot(output_dir, critical_data, features_data, weights)

    if execution_state.is_initial_training:
        weights_path = os.path.join(output_dir, "feature_mean_correlations_and_weights.txt")
        with open(windows_long_path(weights_path), "w", encoding="utf-8") as output_file:
            output_file.write("feature_name;mean_correlation;weight\n")
            for index, feature_name in enumerate(feature_names):
                output_file.write(
                    f"{feature_name};{execution_state.mean_correlations[index]};{weights[index]}\n"
                )
    else:
        weights_path = os.path.join(output_dir, "feature_weights.txt")
        with open(windows_long_path(weights_path), "w", encoding="utf-8") as output_file:
            output_file.write("feature_name;weight\n")
            for index, feature_name in enumerate(feature_names):
                output_file.write(f"{feature_name};{weights[index]}\n")


def try_save_request_outputs(
    config_file_path: str,
    experiment_name: str,
    output_dir: str,
    execution_state: GeneoCoreExecutionState,
    debug_enabled: bool
) -> None:
    # Centralize the per-request debug decision here so the main flow can call this
    # helper unconditionally whenever the numerical result was accepted.
    # Debug artifacts must never invalidate a numerically successful request.
    # If filesystem writes fail, log the problem and keep the HTTP response OK.
    if not debug_enabled:
        return

    if execution_state.is_initial_training and execution_state.continual_learning_enabled:
        try:
            reset_continual_output_dirs(config_file_path, experiment_name)
        except Exception as exc:
            print(
                "WARNING: Failed to reset continual debug output directories for "
                f"experiment '{experiment_name}': {exc}"
            )

    try:
        save_request_outputs(
            output_dir,
            execution_state
        )
    except Exception as exc:
        print(
            "WARNING: Failed to save debug artifacts for "
            f"experiment '{experiment_name}', request {execution_state.request_index}: {exc}"
        )


def release_execution_debug_arrays(execution_state: GeneoCoreExecutionState) -> None:
    # Once plots/debug artifacts are handled, the HTTP response only needs
    # feature names, correlations, weights, and optimizer metadata.
    execution_state.features_data = None
    execution_state.level1_features_data = None
    execution_state.critical_data = None


def save_statistics_file(
    output_dir: str,
    load_seconds: float,
    total_compute_seconds: float,
    pre_solver_seconds: float,
    solver_seconds: float,
    final_loss: float | None
) -> None:
    os.makedirs(output_dir, exist_ok=True)
    loss_value = "" if final_loss is None else str(float(final_loss))
    with open(
        windows_long_path(os.path.join(output_dir, "statistics")),
        "w",
        encoding="utf-8"
    ) as statistics_file:
        statistics_file.write(
            "load_seconds,total_compute_seconds,pre_solver_seconds,solver_seconds,final_loss\n"
        )
        statistics_file.write(
            f"{float(load_seconds)},{float(total_compute_seconds)},{float(pre_solver_seconds)},{float(solver_seconds)},{loss_value}\n"
        )


def try_save_statistics_file(
    output_dir: str,
    load_seconds: float,
    total_compute_seconds: float,
    pre_solver_seconds: float,
    solver_seconds: float,
    final_loss: float | None
) -> None:
    try:
        save_statistics_file(
            output_dir,
            load_seconds,
            total_compute_seconds,
            pre_solver_seconds,
            solver_seconds,
            final_loss
        )
    except Exception as exc:
        print(f"WARNING: Failed to save statistics file in '{output_dir}': {exc}")


def should_save_statistics_file(execution_state: GeneoCoreExecutionState) -> bool:
    # Standalone and initialization requests always create a fresh committed
    # output directory. Incremental updates reuse the latest committed request
    # index when the optimizer result is not applied, so skip statistics in
    # that case to avoid overwriting the previous successful request metadata.
    return execution_state.response_status == "ok"


def compute_total_compute_seconds(
    compute_start: float,
    compute_end: float,
    execution_state: GeneoCoreExecutionState
) -> float:
    return max(
        0.0,
        float(compute_end - compute_start) - float(execution_state.debug_seconds)
    )


def finalize_request_execution(
    config_file_path: str,
    experiment_name: str,
    execution_state: GeneoCoreExecutionState,
    target_column_name: str,
    debug_enabled: bool
) -> Tuple[int, str]:
    request_index = execution_state.request_index
    output_dir = build_request_output_dir(
        config_file_path,
        experiment_name,
        request_index,
        execution_state.continual_learning_enabled
    )
    if execution_state.optimizer_success:
        try_save_request_outputs(
            config_file_path,
            experiment_name,
            output_dir,
            execution_state,
            debug_enabled
        )
    release_execution_debug_arrays(execution_state)
    
    # =========================
    # DEBUG EXPORT SELECTED FEATURES
    # =========================
    if debug_enabled:
        try:
            export_path = os.path.join(output_dir, "selected_features.json")
            payload = {
                "selected_features": execution_state.feature_names,
                "target": target_column_name
            }
            with open(export_path, "w", encoding="utf-8") as f:
                json.dump(payload, f, indent=2)
        except Exception as exc:
            print(f"WARNING: failed to export selected features JSON: {exc}")


    return request_index, output_dir


def custom_openapi(api_app: Any) -> Dict[str, Any]:
    from fastapi.openapi.utils import get_openapi

    if api_app.openapi_schema is not None:
        return api_app.openapi_schema

    openapi_schema = get_openapi(
        title=api_app.title,
        version=api_app.version,
        description=api_app.description,
        routes=api_app.routes
    )

    components = openapi_schema.setdefault("components", {})
    schemas = components.setdefault("schemas", {})
    schemas["GeneoConfig"] = GeneoConfig.model_json_schema(
        ref_template="#/components/schemas/{model}"
    )
    schemas["GeneoConfig"]["description"] = (
        "JSON string sent in the multipart `config` field. The CSV is uploaded "
        "separately as `dataset_file`."
    )
    schemas["GeneoServiceConfig"] = GeneoServiceConfig.model_json_schema(
        ref_template="#/components/schemas/{model}"
    )
    schemas["GeneoServiceConfig"]["description"] = (
        "Informational only. Set these values with environment variables before "
        "startup. Service flags use local defaults when their environment "
        "variables are not defined."
    )

    analyze_post = openapi_schema.get("paths", {}).get("/analyze", {}).get("post")
    if analyze_post is not None:
        request_body = analyze_post.get("requestBody", {})
        multipart_content = request_body.get("content", {}).get("multipart/form-data", {})
        multipart_schema_ref = multipart_content.get("schema", {}).get("$ref")
        if isinstance(multipart_schema_ref, str) and multipart_schema_ref.startswith("#/components/schemas/"):
            multipart_schema_name = multipart_schema_ref.rsplit("/", 1)[-1]
            multipart_schema = schemas.get(multipart_schema_name, {})
            multipart_properties = multipart_schema.get("properties", {})
            config_property = multipart_properties.get("config")
            if isinstance(config_property, dict):
                config_property["description"] = GENEO_CONFIG_FORM_DESCRIPTION
                config_property["example"] = GENEO_CONFIG_BOOTSTRAP_EXAMPLE
                config_property["examples"] = [
                    GENEO_CONFIG_BOOTSTRAP_EXAMPLE,
                    GENEO_CONFIG_INCREMENTAL_EXAMPLE
                ]
                config_property["x-related-schema"] = "#/components/schemas/GeneoConfig"
            dataset_file_property = multipart_properties.get("dataset_file")
            if isinstance(dataset_file_property, dict):
                dataset_file_property["description"] = GENEO_DATASET_FILE_FORM_DESCRIPTION
        multipart_content["encoding"] = {
            "config": {
                "contentType": "application/json"
            },
            "dataset_file": {
                "contentType": "text/csv"
            }
        }
        multipart_content["examples"] = {
            "bootstrap": {
                "summary": "Initial training / bootstrap",
                "description": (
                    "First request for an experiment; feature selection runs here."
                ),
                "value": {
                    "config": GENEO_CONFIG_BOOTSTRAP_EXAMPLE
                }
            },
            "incremental_update": {
                "summary": "Incremental continual-learning update",
                "description": (
                    "Same experiment after bootstrap; reuses the selected features."
                ),
                "value": {
                    "config": GENEO_CONFIG_INCREMENTAL_EXAMPLE
                }
            }
        }

    api_app.openapi_schema = openapi_schema
    return api_app.openapi_schema


# =============================================================================
# Main API endpoint
# =============================================================================
ANALYZE_ROUTE_PATH = "/analyze"
ANALYZE_ROUTE_OPTIONS = dict(
    summary="Analyze time-series batch with GENEO",
    description=(
        "Multipart request with `config` JSON and `dataset_file` CSV. The JSON "
        "contains `experiment_name`, `target_column_name`, and `experiments_rows`; "
        "the CSV rows must equal `sum(experiments_rows)`.\n\n"
        "Service flags come from environment variables at "
        "startup. In continual learning, bootstrap selects features and later "
        "updates reuse them."
    ),
    response_model=GeneoAnalyzeResponse,
    responses={
        400: {
            "model": GeneoErrorResponse,
            "description": "Invalid configuration payload or malformed CSV input.",
            "content": {
                "application/json": {
                    "examples": {
                        "missing_config": {
                            "summary": "Missing config field",
                            "value": {"detail": "Missing 'config' part in multipart request"}
                        },
                        "missing_dataset_file": {
                            "summary": "Missing dataset_file field",
                            "value": {"detail": "Missing 'dataset_file' part in multipart request"}
                        },
                        "row_mismatch": {
                            "summary": "CSV row count mismatch",
                            "value": {"detail": "CSV contains 241 data rows, expected 240"}
                        }
                    }
                }
            }
        },
        500: {
            "model": GeneoErrorResponse,
            "description": "Internal core/optimizer error while processing the batch.",
            "content": {
                "application/json": {
                    "examples": {
                        "optimizer_failure": {
                            "summary": "Level-2 optimizer hard failure",
                            "value": {
                                "detail": (
                                    "Level-2 optimization failed during the initial "
                                    "training with trust-constr status 4: Constraint violation"
                                )
                            }
                        }
                    }
                }
            }
        }
    }
)
GENEO_CONFIG_FORM_OPENAPI_EXAMPLES = {
    "bootstrap": {
        "summary": "Initial training / bootstrap",
        "description": (
            "First request for an experiment; selects features unless "
            "`SELECT_ALL_FEATURES=true`."
        ),
        "value": GENEO_CONFIG_BOOTSTRAP_EXAMPLE
    },
    "incremental_update": {
        "summary": "Incremental continual-learning update",
        "description": (
            "Same experiment after bootstrap; requires "
            "`CONTINUAL_LEARNING_MODE=true`."
        ),
        "value": GENEO_CONFIG_INCREMENTAL_EXAMPLE
    }
}


def run_geneo_analysis(
    request_config: GeneoConfig,
    csv_file: IO[bytes]
) -> GeneoAnalyzeResponse:
    total_wall_start = time.perf_counter()
    t_load_start = time.perf_counter()
    config_file_path = API_REFERENCE_FILE_PATH

    try:
        service_config = get_service_config()
        loader = CsvLoader(csv_file)

        experiment_name = request_config.experiment_name
        debug_enabled = service_config.debug_mode
        continual_learning_enabled = service_config.continual_learning_mode
        select_all_features = service_config.select_all_features
        min_correlation_threshold = service_config.min_correlation_threshold
        experiment_ranges = build_experiment_ranges(request_config.experiments_rows)
        batch_total_rows = experiment_ranges[-1][1]
        num_experiments = len(experiment_ranges)
        request_index = None
        output_dir = None
        execution_state = None

        if not continual_learning_enabled:
            standalone_request_index = STANDALONE_OUTPUT_REGISTRY.reserve_next(
                config_file_path,
                experiment_name
            )
            core_model = GeneoCoreModel(request_config.target_column_name)
            try:
                data_matrix, feature_names = safe_csv_load(
                    lambda: loader.load_bootstrap_matrix(
                        request_config.target_column_name,
                        batch_total_rows
                    )
                )

                t_load_end = time.perf_counter()
                t_compute_start = time.perf_counter()
                execution_state = core_model.run_standalone(
                    data_matrix,
                    experiment_ranges,
                    feature_names,
                    select_all_features,
                    min_correlation_threshold,
                    debug_enabled,
                    standalone_request_index
                )
                t_compute_end = time.perf_counter()
                request_index, output_dir = finalize_request_execution(
                    config_file_path,
                    experiment_name,
                    execution_state,
                    request_config.target_column_name,
                    debug_enabled
                )
                if should_save_statistics_file(execution_state):
                    try_save_statistics_file(
                        output_dir,
                        t_load_end - t_load_start,
                        compute_total_compute_seconds(
                            t_compute_start,
                            t_compute_end,
                            execution_state
                        ),
                        execution_state.solver_started_at - t_compute_start,
                        execution_state.solver_seconds,
                        execution_state.optimizer_total_loss
                    )
            except GeneoCoreValidationError as exc:
                raise_http_exception(status_code=400, detail=str(exc))
            except GeneoCoreExecutionError as exc:
                raise_http_exception(status_code=500, detail=str(exc))
        else:
            entry, should_initialize = MODEL_REGISTRY.acquire_or_wait(experiment_name)

            if should_initialize:
                core_model = GeneoCoreModel(request_config.target_column_name)
                try:
                    data_matrix, feature_names = safe_csv_load(
                        lambda: loader.load_bootstrap_matrix(
                            request_config.target_column_name,
                            batch_total_rows
                        )
                    )

                    t_load_end = time.perf_counter()
                    t_compute_start = time.perf_counter()
                    execution_state = core_model.initialize(
                        data_matrix,
                        experiment_ranges,
                        feature_names,
                        select_all_features,
                        min_correlation_threshold,
                        debug_enabled
                    )
                    t_compute_end = time.perf_counter()
                    request_index, output_dir = finalize_request_execution(
                        config_file_path,
                        experiment_name,
                        execution_state,
                        request_config.target_column_name,
                        debug_enabled
                    )
                    if should_save_statistics_file(execution_state):
                        try_save_statistics_file(
                            output_dir,
                            t_load_end - t_load_start,
                            compute_total_compute_seconds(
                                t_compute_start,
                                t_compute_end,
                                execution_state
                            ),
                            execution_state.solver_started_at - t_compute_start,
                            execution_state.solver_seconds,
                            execution_state.optimizer_total_loss
                        )
                    # Publish the ready state only after request-1 artifacts are ready,
                    # so later requests see a fully initialized model/output layout.
                    MODEL_REGISTRY.publish_initialized(experiment_name, entry, core_model)
                except GeneoCoreValidationError as exc:
                    MODEL_REGISTRY.discard_initializing(experiment_name, entry)
                    raise_http_exception(status_code=400, detail=str(exc))
                except GeneoCoreExecutionError as exc:
                    MODEL_REGISTRY.discard_initializing(experiment_name, entry)
                    raise_http_exception(status_code=500, detail=str(exc))
                except Exception:
                    MODEL_REGISTRY.discard_initializing(experiment_name, entry)
                    raise
            else:
                existing_core_model = entry.core_model

                data_matrix, loaded_feature_names = safe_csv_load(
                    lambda: load_incremental_timeseries(
                        loader,
                        request_config,
                        existing_core_model,
                        batch_total_rows
                    )
                )

                t_load_end = time.perf_counter()
                t_compute_start = time.perf_counter()
                try:
                    execution_state = existing_core_model.update(
                        data_matrix,
                        loaded_feature_names,
                        experiment_ranges,
                        debug_enabled
                    )
                    t_compute_end = time.perf_counter()
                except GeneoCoreValidationError as exc:
                    raise_http_exception(status_code=400, detail=str(exc))
                except GeneoCoreExecutionError as exc:
                    raise_http_exception(status_code=500, detail=str(exc))

                request_index, output_dir = finalize_request_execution(
                    config_file_path,
                    experiment_name,
                    execution_state,
                    request_config.target_column_name,
                    debug_enabled
                )
                if should_save_statistics_file(execution_state):
                    try_save_statistics_file(
                        output_dir,
                        t_load_end - t_load_start,
                        compute_total_compute_seconds(
                            t_compute_start,
                            t_compute_end,
                            execution_state
                        ),
                        execution_state.solver_started_at - t_compute_start,
                        execution_state.solver_seconds,
                        execution_state.optimizer_total_loss
                    )

        num_features = len(execution_state.feature_names)
        total_wall_end = time.perf_counter()
        explanation_id = str(uuid.uuid4()).replace("-", "")[:32]
        timestamp = int(time.time())

        response = GeneoAnalyzeResponse(
            explanationId=explanation_id,
            timestamp=timestamp,
            features=build_feature_response(
                execution_state.feature_names,
                execution_state.weights,
                execution_state.mean_correlations
            )
        )

        print(f"status: {execution_state.response_status}")
        print(f"experiment_name: {experiment_name}")
        print(f"debug_mode: {debug_enabled}")
        print(f"continual_learning_mode: {continual_learning_enabled}")
        print(f"select_all_features: {select_all_features}")
        print(f"min_correlation_threshold: {min_correlation_threshold}")
        print(f"request_index: {int(request_index)}")
        print(f"output_dir: {output_dir}")
        print(f"num_features: {int(num_features)}")
        print(f"num_experiments: {int(num_experiments)}")
        print(f"optimizer_success: {execution_state.optimizer_success}")
        print(f"optimizer_status: {execution_state.optimizer_status}")
        print(f"optimizer_message: {execution_state.optimizer_message}")
        print(f"total_loss: {execution_state.optimizer_total_loss}")
        print(f"csv_load_seconds: {float(t_load_end - t_load_start)}")
        print(
            "total_compute_seconds_excluding_load_and_debug_io: "
            f"{compute_total_compute_seconds(t_compute_start, t_compute_end, execution_state)}"
        )
        print(f"debug_seconds_excluded_from_total_compute: {float(execution_state.debug_seconds)}")
        print(
            "pre_solver_seconds_excluding_load_and_plots: "
            f"{float(execution_state.solver_started_at - t_compute_start)}"
        )
        print(f"solver_seconds: {float(execution_state.solver_seconds)}")
        print(f"total_wall_seconds: {float(total_wall_end - total_wall_start)}")

        return response
    finally:
        pass


async def analyze_geneo(
    config: str | None,
    dataset_file: Any | None
) -> GeneoAnalyzeResponse:
    if config is None:
        raise_http_exception(status_code=400, detail="Missing 'config' part in multipart request")
    if dataset_file is None:
        raise_http_exception(status_code=400, detail="Missing 'dataset_file' part in multipart request")

    try:
        request_config = parse_request_config(config)
        return run_geneo_analysis(request_config, dataset_file.file)
    finally:
        if dataset_file is not None:
            await dataset_file.close()


def create_api_app() -> Any:
    from fastapi import FastAPI, File, Form, UploadFile

    api_app = FastAPI(
        title="GENEO Time Series Analyzer (2-Level)",
        description=GENEO_API_DESCRIPTION,
        version="1.1"
    )

    @api_app.post(ANALYZE_ROUTE_PATH, **ANALYZE_ROUTE_OPTIONS)
    async def analyze_geneo_api(
        config: str | None = Form(
            default=None,
            description=GENEO_CONFIG_FORM_DESCRIPTION,
            openapi_examples=GENEO_CONFIG_FORM_OPENAPI_EXAMPLES
        ),
        dataset_file: UploadFile | None = File(
            default=None,
            description=GENEO_DATASET_FILE_FORM_DESCRIPTION
        )
    ) -> GeneoAnalyzeResponse:
        return await analyze_geneo(config, dataset_file)

    def api_openapi() -> Dict[str, Any]:
        return custom_openapi(api_app)

    api_app.openapi = api_openapi
    return api_app


def read_http_csv_as_stream(download_url: str) -> io.BytesIO:
    from urllib.request import Request, urlopen

    request = Request(
        download_url,
        headers={"User-Agent": "GENEOS/1.1"}
    )
    with urlopen(request, timeout=60) as response:
        return io.BytesIO(response.read())


def decode_kafka_message_value(raw_value: Any) -> Dict[str, Any]:
    if raw_value is None:
        raise ValueError("Invalid Kafka message: empty value")
    if isinstance(raw_value, bytes):
        message_text = raw_value.decode("utf-8")
    elif isinstance(raw_value, str):
        message_text = raw_value
    elif isinstance(raw_value, dict):
        return raw_value
    else:
        raise ValueError("Invalid Kafka message: value must be JSON bytes or string")

    try:
        decoded_value = json.loads(message_text)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid Kafka message: value is not valid JSON ({exc})") from exc

    if not isinstance(decoded_value, dict):
        raise ValueError("Invalid Kafka message: JSON value must be an object")
    return decoded_value


def process_kafka_training_message(
    payload: Dict[str, Any]
) -> DownloadedKafkaTrainingBatch:
    training_message = GeneoKafkaTrainingMessage.model_validate(payload)
    training_message.validate_file_count()

    downloaded_files: List[DownloadedKafkaTrainingFile] = []
    for file_index, training_file in enumerate(training_message.files, start=1):
        try:
            csv_file = read_http_csv_as_stream(training_file.download_url)
        except Exception as exc:
            file_label = training_file.file_name or training_file.download_url
            raise RuntimeError(
                f"Failed to download training file {file_index}/"
                f"{training_message.total_files} ({file_label!r}): {exc}"
            ) from exc

        downloaded_files.append(
            DownloadedKafkaTrainingFile(
                file_name=training_file.file_name,
                download_url=training_file.download_url,
                csv_file=csv_file
            )
        )

    return DownloadedKafkaTrainingBatch(
        experiment_name=training_message.experiment_name,
        timestamp=training_message.timestamp,
        files=downloaded_files
    )


app: Any | None = None


def run_api_mode() -> None:
    global app
    try:
        get_service_config()
        host, port = load_uvicorn_bind_config()
    except ServiceConfigurationError as exc:
        raise SystemExit(f"API configuration error: {exc}") from exc

    import uvicorn

    app = create_api_app()
    bind_endpoint = f"{host}:{port}"
    try:
        uvicorn.run(app, host=host, port=port)
    except SystemExit as exc:
        if exc.code in (0, None):
            raise
        raise SystemExit(
            f"API server startup error for GENEO_HOST={bind_endpoint!r}: "
            f"Uvicorn exited with status {exc.code}"
        ) from exc
    except Exception as exc:
        raise SystemExit(
            f"API server startup error for GENEO_HOST={bind_endpoint!r}: "
            f"{stringify_exception(exc)}"
        ) from exc


def run_kafka_mode() -> None:
    try:
        service_config = get_service_config()
        kafka_config = load_kafka_runtime_config()

        from confluent_kafka import Consumer

        consumer = Consumer(build_confluent_consumer_config(kafka_config))
        consumer.subscribe([kafka_config.training_topic])
    except Exception as exc:
        raise SystemExit(f"Kafka mode configuration error: {stringify_exception(exc)}") from exc

    debug_enabled = service_config.debug_mode
    debug_print(
        f"Listening on Kafka topic '{kafka_config.training_topic}'",
        debug_enabled
    )

    try:
        while True:
            message = consumer.poll(1.0)
            if message is None:
                continue
            if message.error():
                debug_print(f"Kafka consumer error: {message.error()}", debug_enabled)
                continue

            try:
                payload = decode_kafka_message_value(message.value())
                downloaded_batch = process_kafka_training_message(payload)
                debug_print(
                    f"Downloaded {len(downloaded_batch.files)} Kafka training file(s) "
                    f"for experiment '{downloaded_batch.experiment_name}'; "
                    "GENEOS core processing is not enabled for batch messages yet.",
                    debug_enabled
                )
                continue
            except Exception as exc:
                error_message = stringify_exception(exc)
                debug_print(f"GENEOS Kafka error: {error_message}", debug_enabled)
                continue
    except KeyboardInterrupt:
        debug_print("GENEOS Kafka worker interrupted.", debug_enabled)
    finally:
        consumer.close()


def main() -> None:
    try:
        mode = load_runtime_mode()
    except ServiceConfigurationError as exc:
        raise SystemExit(f"Runtime mode configuration error: {exc}") from exc

    if mode == "api":
        run_api_mode()
        return
    if mode == "kafka":
        run_kafka_mode()
        return


if __name__ == "__main__":
    main()
