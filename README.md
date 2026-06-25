# Continual GENEOS (UNIPI)

# GENEO Siemens Service

GENEOs are a tool to reduce the invariance of persistence diagrams in Topological Data Analysis. Their use was later extended to machine
learning, since they are particularly useful for approximating observers and decomposing neural networks. GENEOs can be combined through
operations such as concatenation, direct product, and convex combination. Thanks to their ability to significantly reduce the number of
required parameters - and consequently the number of training examples - GENEOs have proven useful in various applications.

The component is implemented as a Python service.


The numerical implementation lives in `geneos_siemens_core.py`; `geneos_siemens.py` contains the API adapter, runtime configuration, CSV loading, request registry, and optional debug artifact generation.

## Requirements

The current runtime dependencies are the ones declared in [requirements.txt](./requirements.txt):

```txt
numpy
scipy
fastapi>=0.100
uvicorn
python-multipart
pydantic>=2
requests
minio
confluent-kafka
```

## Local Installation

The service is contained in this folder and can be installed directly from here.

```bash
cd code/GENEOS
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## Running the API Locally

The script starts from its `main` function. Since the default runtime mode is Kafka, the FastAPI server must be enabled explicitly with `MODE=api`.
`GENEO_HOST` contains both host and port and defaults to `0.0.0.0:8000`.

```bash
cd code/GENEOS
MODE=api GENEO_HOST=0.0.0.0:8000 python geneos_siemens.py
```

Interactive API documentation is then available at:

- Swagger UI: `http://127.0.0.1:8000/docs`
- ReDoc: `http://127.0.0.1:8000/redoc`
- OpenAPI JSON: `http://127.0.0.1:8000/openapi.json`

## Docker

A Dockerfile is available in this directory and starts the same script main:

```bash
python geneos_siemens.py
```

The Dockerfile sets `GENEO_HOST` as a single `<host>:<port>` value. It also expects MinIO build arguments and stores them as environment variables inside the image. The Python service currently does not read the MinIO values directly; they are kept for the integration deployment contract.

Build the image:

```bash
cd code/GENEOS
docker build \
  --build-arg GENEO_HOST=0.0.0.0:8000 \
  -t registry/dice_geneos:vx.x.x .
```

Run the API container:

```bash
docker run --rm \
  -e MODE=api \
  -e GENEO_HOST=0.0.0.0:8000 \
  -p 8000:8000 \
  registry/dice_geneos:vx.x.x
```

To use another API port, keep `GENEO_HOST` and the port mapping aligned:

```bash
docker build \
  --build-arg GENEO_HOST=0.0.0.0:9000 \
  -t registry/dice_geneos:vx.x.x .

docker run --rm \
  -e MODE=api \
  -e GENEO_HOST=0.0.0.0:9000 \
  -p 9000:9000 \
  registry/dice_geneos:vx.x.x
```

If you want to preserve debug artifacts generated inside the container, mount a volume for `/app/geneo_plots_optimized`.

## Networking & Resources

- **Ports & Protocols**: `GENEO_HOST` is used only in `MODE=api`; default API bind address is `0.0.0.0:8000`. 
- **Resource Requirements**:
  - **CPU**: `250m` request. No hard limit is required by the code; the numerical core uses a number of worker threads based on the available logical CPUs and the number of experiment blocks.
  - **Memory**: `512Mi` request / `4Gi` limit
- **ML Model**: No external ML model is loaded. The service uses the internal two-level GENEO numerical model.
- **Persistent Storage**: None required. Debug/statistics artifacts are written locally when generated, as reported in previous paragraph.

## API Overview

### POST `/analyze`

Runs the GENEO analysis on a CSV dataset and returns the surviving features with their final weights.

The endpoint consumes `multipart/form-data` with two parts:

- `config`: JSON string
- `dataset_file`: the CSV file with the dataset, or the real-time batch represented as a CSV dataset file

## Request Format

### Multipart fields

- `config`: JSON string
- `dataset_file`: CSV file to be uploaded

The JSON sent in the FastAPI multipart `config` field contains only:

```json
{
  "experiment_name": "siemens_exp_bootstrap",
  "target_column_name": "critical_signal",
  "experiments_rows": [120, 120, 120]
}
```

When using the local `GeneosRequest.py` helper, the local JSON file also contains
`dataset_file` so the helper can open the CSV. The helper removes `dataset_file`
before sending the API `config` payload.

```json
{
  "experiment_name": "siemens_exp_bootstrap",
  "dataset_file": ".\\code\\GENEOS\\Realistic.csv",
  "target_column_name": "critical_signal",
  "experiments_rows": [120, 120, 120]
}
```

```bash
python GeneosRequest.py test_configuration_realistic.json
```

### `config` schema

The `config` field must contain a JSON object with the following structure:

```json
{
  "experiment_name": "siemens_exp_bootstrap",
  "target_column_name": "critical_signal",
  "experiments_rows": [120, 120, 120]
}
```

Field meaning:

- `experiment_name`: a description or an identifier of the experiment. In continual-learning mode this is also the in-memory model key.
- `target_column_name`: the exact name of the CSV column that represents the target critical function.
- `experiments_rows`: array containing the number of rows for each experiment, in dataset order. Values must be positive integers and the sum must match the number of non-empty CSV data rows.

Service settings are not part of the FastAPI request payload. They are loaded
once when the server starts from environment variables. The service does not
read a local configuration file.

## Environment Variables

### Runtime mode and API binding

| Variable | Default value | Meaning |
|---|---:|---|
| `MODE` | `kafka` | Startup mode. Supported values are `kafka` and `api`. 'api' must be used for test purposes |
| `GENEO_HOST` | `0.0.0.0:8000` | API bind endpoint in `<host>:<port>` format, used only in `MODE=api`. Optional `http://` or `https://` prefixes are stripped before binding. |

`GENEO_PORT` is not read by the current Python scripts. Use `GENEO_HOST=<host>:<port>` instead.

### Core analysis configuration

| Variable | Default value | Meaning |
|---|---:|---|
| `DEBUG_MODE` | `false` | Boolean flag. When `true`, save plots and per-request debug artifacts on disk. |
| `SELECT_ALL_FEATURES` | `false` | Boolean flag. When `true`, skip adaptive level-2 feature selection and use every valid non-flat feature. |
| `MIN_CORRELATION_THRESHOLD` | `0.0` | Finite non-negative float. Read only when `SELECT_ALL_FEATURES=false`; values greater than `0` keep valid features whose level-1 mean correlation is at least the threshold and skip adaptive pruning. |

## CSV Input Requirements

The uploaded dataset must satisfy the following conditions:

- UTF-8 compatible CSV with header row.
- Row order represents the time axis.
- Columns represent covariates, except the target column.
- The target column named by `target_column_name` must be present.
- The total number of non-empty data rows must equal `sum(experiments_rows)`.
- Selected values must be numeric and finite.
- During bootstrap, the service discovers numeric feature columns excluding the target column by inspecting the first non-empty data row.

Additional behavior worth noting:

- Empty CSV rows are ignored.
- Rows with more columns than the header are rejected.
- Rows with fewer columns than the header are padded, but missing selected numeric values are rejected by the numeric loader.
- During continual updates, a different column order is accepted if all surviving feature names and the target column are present.

## Example Request

Example `curl` request:

```bash
curl -X POST "http://127.0.0.1:8000/analyze" \
  -F 'config={
  "experiment_name":"siemens_exp_bootstrap",
  "target_column_name":"critical_signal",
  "experiments_rows":[120,120,120]
}' \
  -F "dataset_file=@Realistic.csv;type=text/csv"
```

The same endpoint can be tested interactively from Swagger UI at `/docs`.

## Response Format

Successful responses have the following shape:

```json
{
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
```

Field meaning:

- `explanationId`: unique request-scoped identifier generated by the API.
- `timestamp`: Unix timestamp in seconds.
- `features`: ordered list of surviving features, where each item contains:
- `name`: feature name from the CSV header.
- `weight`: final non-negative weight associated with the feature.
- `mean_correlation`: mean correlation for that feature. Only if adaptive of fixed pruning are set.

## Error Responses

The API currently documents and returns these main error categories:

- `400 Bad Request`: invalid `config`, malformed CSV, missing multipart fields, inconsistent row counts, or numerical validation errors caused by the input batch.
- `422 Unprocessable Entity`: FastAPI validation error on the multipart request itself.
- `500 Internal Server Error`: numerical core or optimizer execution failure.

Typical `400` error example:

```json
{
  "detail": "CSV contains 241 data rows, expected 240"
}
```

Typical `500` error example:

```json
{
  "detail": "Level-2 optimization failed during the initial training with trust-constr status 4: Constraint violation"
}
```

## Output Artifacts

Committed requests write a `statistics` file when the response status is `ok`.
When `DEBUG_MODE=true`, the service may also generate:

- per-feature comparison plots during initial training
- `critical_vs_approximated.png`
- text files with weights and correlations

The `statistics` file contains `load_seconds`, `total_compute_seconds`,
`pre_solver_seconds`, `solver_seconds`, and `final_loss`.
`total_compute_seconds` excludes input loading, output/debug artifact generation,
and core debug-array preparation.

Artifacts are stored under:

```txt
geneo_plots_optimized/<experiment_name__hash>/<request_index_or_unico_n>/
```

Standalone runs use `unico_<n>` directories. Continual-learning runs use numeric request directories such as `1`, `2`, `3`, and so on.

NB: Kafka message ingestion exists, but the current Kafka worker does not yet call the numerical GENEO core or publish explanations. The FastAPI `POST /analyze` path is the current execution path for full GENEO analysis.
