import argparse
import json
import os
import sys
from typing import Any

import requests


API_URL = "http://127.0.0.1:8000/analyze"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Read a local GENEOS request JSON, use its dataset_file entry to load "
            "the CSV from disk, and send the API config plus CSV as multipart form-data."
        )
    )
    parser.add_argument(
        "config_file_path",
        help=(
            "Path to the local JSON file. It contains experiment_name, dataset_file, "
            "target_column_name, and experiments_rows."
        )
    )
    parser.add_argument(
        "--url",
        default=API_URL,
        help=f"REST API endpoint. Default: {API_URL}"
    )
    return parser


def load_json_file(config_file_path: str) -> dict[str, Any]:
    try:
        with open(config_file_path, "r", encoding="utf-8") as config_file:
            config_data = json.load(config_file)
    except FileNotFoundError as exc:
        raise ValueError(f"Configuration file not found: {config_file_path}") from exc
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON configuration file: {exc}") from exc
    except UnicodeDecodeError as exc:
        raise ValueError(f"Configuration file is not valid UTF-8: {exc}") from exc

    if not isinstance(config_data, dict):
        raise ValueError("Configuration file must contain a JSON object")

    return config_data


def ensure_type(
    config_data: dict[str, Any],
    field_name: str,
    expected_type: type | tuple[type, ...]
) -> None:
    if field_name not in config_data:
        raise ValueError(f"Missing required configuration field: {field_name}")

    field_value = config_data[field_name]

    if expected_type == bool:
        if not isinstance(field_value, bool):
            raise ValueError(f"Field '{field_name}' must be of type bool")
        return

    if expected_type == list:
        if not isinstance(field_value, list):
            raise ValueError(f"Field '{field_name}' must be of type list")
        return

    if isinstance(expected_type, tuple):
        if isinstance(field_value, bool) and bool not in expected_type:
            expected_names = ", ".join(item.__name__ for item in expected_type)
            raise ValueError(f"Field '{field_name}' must be one of: {expected_names}")
        if not isinstance(field_value, expected_type):
            expected_names = ", ".join(item.__name__ for item in expected_type)
            raise ValueError(f"Field '{field_name}' must be one of: {expected_names}")
        return

    if not isinstance(expected_type, tuple) and not isinstance(field_value, expected_type):
        raise ValueError(f"Field '{field_name}' must be of type {expected_type.__name__}")


def validate_config_shape(config_data: dict[str, Any]) -> None:
    # dataset_file is a local-client convenience field. It is allowed here so
    # this helper can open the CSV, but it is stripped before sending the API
    # config payload.
    expected_fields = {
        "experiment_name",
        "dataset_file",
        "target_column_name",
        "experiments_rows"
    }
    extra_fields = set(config_data) - expected_fields
    if extra_fields:
        extra_field_names = ", ".join(sorted(extra_fields))
        raise ValueError(
            "Configuration file must contain only experiment_name, dataset_file, "
            "target_column_name, and experiments_rows. Unexpected field(s): "
            f"{extra_field_names}"
        )

    ensure_type(config_data, "experiment_name", str)
    ensure_type(config_data, "dataset_file", str)
    ensure_type(config_data, "target_column_name", str)
    ensure_type(config_data, "experiments_rows", list)

    experiments_rows = config_data["experiments_rows"]
    if not all(isinstance(value, int) and not isinstance(value, bool) for value in experiments_rows):
        raise ValueError("Field 'experiments_rows' must contain only integers")


def resolve_path_from_reference(path_value: str, reference_file_path: str) -> str:
    expanded_path = os.path.expanduser(path_value)
    if os.path.isabs(expanded_path):
        return os.path.abspath(expanded_path)

    reference_dir = os.path.dirname(os.path.abspath(reference_file_path))
    candidate = os.path.abspath(os.path.join(reference_dir, expanded_path))
    if os.path.exists(candidate):
        return candidate

    current_dir = reference_dir
    while True:
        parent_dir = os.path.dirname(current_dir)
        if parent_dir == current_dir:
            break

        candidate = os.path.abspath(os.path.join(parent_dir, expanded_path))
        if os.path.exists(candidate):
            return candidate
        current_dir = parent_dir

    return os.path.abspath(os.path.join(reference_dir, expanded_path))


def build_request_parts(config_data: dict[str, Any], config_file_path: str):
    dataset_file_path = resolve_path_from_reference(config_data["dataset_file"], config_file_path)
    if not os.path.isfile(dataset_file_path):
        raise ValueError(f"Dataset file not found: {dataset_file_path}")

    # FastAPI receives dataset_file as a separate multipart file field, so the
    # JSON config part must match GeneoConfig and must not contain dataset_file.
    api_config_data = dict(config_data)
    api_config_data.pop("dataset_file", None)
    config_payload = json.dumps(api_config_data, ensure_ascii=False)
    dataset_file_name = os.path.basename(dataset_file_path)
    return config_payload, dataset_file_path, dataset_file_name


def send_request(api_url: str, config_payload: str, dataset_file_path: str, dataset_file_name: str) -> requests.Response:
    with open(dataset_file_path, "rb") as dataset_file:
        response = requests.post(
            api_url,
            files={
                "config": (None, config_payload, "application/json"),
                "dataset_file": (dataset_file_name, dataset_file, "text/csv"),
            },
            timeout=600,
        )
    return response


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    config_file_path = os.path.abspath(os.path.expanduser(args.config_file_path))

    try:
        config_data = load_json_file(config_file_path)
        validate_config_shape(config_data)
        config_payload, dataset_file_path, dataset_file_name = build_request_parts(
            config_data,
            config_file_path
        )
        response = send_request(args.url, config_payload, dataset_file_path, dataset_file_name)
        response.raise_for_status()
    except ValueError as exc:
        print(f"Configuration error: {exc}")
        return 1
    except requests.exceptions.HTTPError as exc:
        response = exc.response
        response_text = "" if response is None else response.text
        print(f"HTTP error occurred: {exc} - {response_text}")
        return 1
    except requests.exceptions.RequestException as exc:
        print(f"Request error occurred: {exc}")
        return 1
    except Exception as exc:
        print(f"Unexpected error occurred: {exc}")
        return 1

    print("Status code:", response.status_code)
    try:
        print("Response JSON:", json.dumps(response.json(), indent=4))
    except ValueError:
        print("Response text:", response.text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
