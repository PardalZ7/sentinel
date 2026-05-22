# Sentinel

Sentinel is a Python library for anomaly detection in event-driven architectures. It is fully domain-agnostic — plug-and-play via a `sentinel.json` configuration file, no application code changes required.

---

## Architecture Overview

Sentinel has two complementary layers:

### SentinelAgent
Monitors a single application flow (input + output queues):
- Correlates input/output messages by configurable `input_correlation_field` / `output_correlation_field` (dot-notation and array traversal supported)
- Supports three correlation modes: `normal` (1-to-1), `grouping` (N inputs → 1 output), and `splitting` (1 input → N outputs)
- Detects **timeout anomalies** when input has no matching output within the TTL
- Runs **multiple detectors in parallel** (Strategy Pattern via `IDetector`), each with its own algorithm, phase, champion/challenger cycle, and persisted model
- An event is flagged anomalous if **any** detector signals anomaly; the reported score is the worst across all detectors
- Publishes `ProcessedEvent` and `HeartbeatEvent` to a SentinelCortex via gRPC
- Operates in **standalone mode** if gRPC fails — never blocks message processing

### SentinelCortex
Manages multiple Agents and/or other Cortex nodes (hierarchical):
- Receives events via gRPC server
- Maintains an aggregated state vector per agent
- Detects **silence** (agent stopped reporting)
- Runs a **Dual Network** in parallel:
  - **Causal Chain**: correlates anomaly sequences in time to identify root cause
  - **Autoencoder (PyTorch)**: detects systemic patterns invisible to individual agents
- Emits alerts to a configurable alarm queue

```
┌─────────────┐           ┌──────────────────────────┐    gRPC    ┌─────────────────┐
│ Application │──SQS/SNS──│ SentinelAgent             │──────────▶│ SentinelCortex  │
│  (SQS/SNS)  │           │  ┌─────────────────────┐  │           │ (Dual Network)  │
└─────────────┘           │  │ IsolationForest det  │  │           └────────┬────────┘
                          │  │ cVAE detector        │  │                    │ alerts
                          │  │ MAF detector         │  │              alarm_queue (SQS)
                          │  └─────────────────────┘  │
                          └──────────────────────────┘
```

---

## Quick Start

### Install

```bash
pip install sentinel
```

### Create a config file

```json
{
  "sentinel": {
    "storage_path": "~/.sentinel",
    "redis": { "host": "localhost", "port": 6379 },
    "alarm_queue": { "type": "sqs", "resource": "sentinel-alarms" },
    "model": {
      "max_versions": 10,
      "isolation_forest": { "n_estimators": 100, "anomaly_threshold": -0.1, "training_window": 500 },
      "autoencoder": { "hidden_dim": 32, "training_window": 1000 }
    },
    "agents": [
      {
        "name": "agent01",
        "input":  { "type": "sqs", "resource": "my-input-queue" },
        "output": { "type": "sqs", "resource": "my-output-queue" },
        "input_correlation_field": "correlationId",
        "output_correlation_field": "correlationId",
        "correlation_mode": "normal",
        "correlation_ttl_s": 300,
        "cortex": [{ "host": "localhost", "port": 50051 }],
        "reporting": { "heartbeat_interval_s": 30, "send_all_events": false },
        "detectors": [
          {
            "name": "if_baseline",
            "algorithm": "isolation_forest",
            "phase": "TRAINING",
            "training_window": 500,
            "anomaly_threshold": -0.1
          },
          {
            "name": "cvae_transform",
            "algorithm": "cvae",
            "phase": "TRAINING",
            "training_window": 500,
            "anomaly_threshold": 0.5,
            "hidden_dim": 32,
            "latent_dim": 8,
            "field_map": {
              "input": ["fieldA", "fieldB"],
              "output": ["derivedField"]
            }
          }
        ]
      }
    ],
    "cortex": [
      {
        "name": "cortex01",
        "grpc_port": 50051,
        "inputs": ["agent01"],
        "silence_threshold_s": 60
      }
    ]
  }
}
```

> **No detectors configured?** Sentinel will automatically synthesise a single `isolation_forest` detector from the global `model.isolation_forest` config — existing `sentinel.json` files work without any changes.

### Start Sentinel

```bash
sentinel start --config sentinel.json
```

---

## Detectors

Each agent can run multiple detectors in parallel. Every detector maintains its own training buffer, test buffer, champion model, and phase independently.

### IsolationForest (`isolation_forest`)

**Purpose:** Detects **operational anomalies** — deviations in processing latency, payload sizes, and schema hash. Does not inspect payload field values. Fast to converge (~500 samples). Recommended as a baseline for every agent.

**Features (extracted automatically):**
- `processing_latency_ms`
- `input_size_bytes`
- `output_size_bytes`
- `payload_schema_hash` (integer hash of the output's top-level keys)

**Score direction:** `score < threshold` → anomaly. Threshold is negative (IsolationForest returns scores in [-1, 1] where more negative = more abnormal).

**When to use:** Always. Pair with a semantic detector for agents that perform data transformations.

---

### cVAE — Conditional Variational Autoencoder (`cvae`)

**Purpose:** Detects **semantic transformation anomalies** — learns the conditional distribution P(output_fields | input_fields). Flags events where the output is semantically inconsistent with the input.

**Example:** An enrichment service that computes `result = value01 + value02` starts returning wrong sums. The IsolationForest won't catch this (latency and sizes are normal), but the cVAE will, because the output no longer matches what the model learned for those input values.

**Requires `field_map`** — the field names must match actual fields in the monitored system's payloads. Without `field_map`, startup will fail with a `ValueError`.

```json
"field_map": {
  "input": ["value01", "value02"],
  "output": ["result", "stringSize"]
}
```

**Score direction:** `score > threshold` → anomaly. Score is reconstruction MSE (≥ 0); threshold is positive.

**When to use:** Agents monitoring 1-to-1 transformation services (enrichers, validators, calculators).

---

### MAF — Masked Autoregressive Flow (`maf`)

**Purpose:** Detects **joint distribution anomalies** — learns the joint density P(x₁, x₂, ..., xₙ) across all numeric fields using a MADE (Masked Autoencoder for Distribution Estimation). Flags events where the multivariate combination of metrics is anomalous, even if each individual metric looks normal.

**Example:** A batcher produces outputs where latency is in range, record count is in range, and total size is in range — but the combination of all three simultaneously is unusual. MAF catches this where IsolationForest would not.

**`field_map` is optional.** If omitted, MAF auto-discovers all numeric fields from the payload bodies at first fit. The field order is fixed after fitting and stored in the model.

**Score direction:** `score > threshold` → anomaly. Score is `-log_likelihood` (higher = more anomalous).

**When to use:** Agents monitoring aggregation services (batchers) or end-to-end pipeline spans, where individual metrics are insufficient.

---

### NRI — Neural Relational Inference (`nri`)

**Purpose (planned):** Detects anomalies in the dependency graph between fields — learns which fields normally influence which, and flags when that relational structure changes.

**Current status:** Stable interface stub. `fit()` raises `NotImplementedError`; `score()` returns `0.0`. Safe to include in a config without causing errors — it simply does not contribute to detection until implemented.

---

## Configuration Reference

### Root (`sentinel.*`)

| Field | Default | Description |
|-------|---------|-------------|
| `storage_path` | `~/.sentinel` | Base directory for model files |
| `redis.host` | `localhost` | Redis host |
| `redis.port` | `6379` | Redis port |
| `alarm_queue.type` | `sqs` | Transport type for alarms |
| `alarm_queue.resource` | `""` | Queue URL or topic ARN |

### Model (`sentinel.model.*`)

| Field | Default | Description |
|-------|---------|-------------|
| `max_versions` | `5` | Maximum model versions per detector to keep on disk |
| `auto_rollback` | `false` | Revert to previous model if performance degrades |
| `isolation_forest.n_estimators` | `100` | Number of trees (used when synthesising detector from global config) |
| `isolation_forest.contamination` | `"auto"` | Expected outlier fraction |
| `isolation_forest.anomaly_threshold` | `-0.1` | Score threshold |
| `isolation_forest.training_window` | `500` | Events to accumulate before fitting |
| `autoencoder.hidden_dim` | `64` | Cortex autoencoder hidden layer size |
| `autoencoder.training_window` | `1000` | Cortex autoencoder training events |
| `autoencoder.learning_rate` | `0.001` | Adam optimizer learning rate |
| `autoencoder.epochs` | `50` | Training epochs |

### Agent (`sentinel.agents[*].*`)

| Field | Default | Description |
|-------|---------|-------------|
| `name` | — | Unique agent identifier |
| `input.resource` | — | Input queue URL |
| `output.resource` | — | Output queue URL |
| `input_correlation_field` | `correlationId` | Dot-notation field path to extract the correlation ID from input messages |
| `output_correlation_field` | `correlationId` | Dot-notation or array path to extract the correlation ID(s) from output messages |
| `correlation_mode` | `normal` | `normal` (1→1), `grouping` (N→1), `splitting` (1→N) |
| `correlation_ttl_s` | `300` | Seconds before an unmatched input expires (TIMEOUT anomaly) |
| `cortex` | `[]` | List of Cortex gRPC endpoints: `[{"host": "...", "port": 50051}]` |
| `reporting.heartbeat_interval_s` | `30` | Heartbeat interval in seconds |
| `reporting.send_all_events` | `false` | Send all events upstream (auto-true during TRAINING) |
| `detectors` | `[]` | List of `DetectorConfig`. If empty, an IsolationForest detector is synthesised automatically. |

### Detector (`sentinel.agents[*].detectors[*].*`)

| Field | Default | Description |
|-------|---------|-------------|
| `name` | — | Unique detector name within the agent |
| `algorithm` | `isolation_forest` | `isolation_forest` \| `cvae` \| `maf` \| `nri` |
| `phase` | `TRAINING` | Initial phase: `TRAINING` or `INFERENCE` |
| `training_window` | `500` | Events to accumulate before fitting the model |
| `test_buffer_size` | `100` | Hold-out test buffer size for FP-rate evaluation |
| `anomaly_threshold` | `-0.1` | Score threshold. **Negative** for IF (`score < threshold` → anomaly). **Positive** for cVAE/MAF (`score > threshold` → anomaly). |
| `field_map.input` | `null` | Input payload field names. **Required** for `cvae`. Optional for `maf`. |
| `field_map.output` | `null` | Output payload field names. **Required** for `cvae`. Optional for `maf`. |
| `n_estimators` | `100` | `isolation_forest` only: number of trees |
| `contamination` | `"auto"` | `isolation_forest` only: expected fraction of anomalies in training data |
| `hidden_dim` | `32` | `cvae` / `maf` only: hidden layer dimension |
| `latent_dim` | `8` | `cvae` only: latent space dimension |
| `learning_rate` | `0.001` | `cvae` / `maf` only: Adam optimizer learning rate |
| `epochs` | `30` | `cvae` / `maf` only: training epochs per fit |

### Cortex (`sentinel.cortex[*].*`)

| Field | Default | Description |
|-------|---------|-------------|
| `name` | — | Unique cortex identifier |
| `grpc_port` | `50051` | Port for the gRPC server |
| `inputs` | `[]` | Agent/cortex names this cortex monitors |
| `silence_threshold_s` | `60` | Seconds of silence before a SILENCE alert |
| `parent_cortex` | `[]` | Parent cortex gRPC endpoints (hierarchical setup) |
| `test_buffer_size` | `200` | Hold-out test buffer size |
| `training_sample_interval_s` | `5` | Minimum seconds between training samples |
| `adaptation_mode` | `ema` | `ema` or `fine_tuning` |

---

## Phase Management

Each detector has an independent phase. The agent's aggregate phase is that of its first detector (for heartbeat and dashboard display).

| Phase | Behavior |
|-------|----------|
| `TRAINING` | Accumulates events and trains the model when `training_window` is reached. All events sent upstream. |
| `INFERENCE` | Uses the trained model to score events. Only anomalies sent upstream (unless `send_all_events=true`). |

### Change phase via CLI (all detectors in the agent)

```bash
sentinel set-phase agent01 INFERENCE
sentinel set-phase agent01 TRAINING
```

### Change phase via API (per detector)

```bash
# All detectors in agent01
curl -X POST http://localhost:8888/api/agents/agent01/phase \
  -H "Content-Type: application/json" -d '{"phase": "INFERENCE"}'

# Only the cvae_transform detector
curl -X POST http://localhost:8888/api/agents/agent01/detectors/cvae_transform/phase \
  -H "Content-Type: application/json" -d '{"phase": "INFERENCE"}'
```

---

## Model Storage

Models are stored under `storage_path` with one subdirectory per detector:

```
~/.sentinel/
└── agents/
    └── agent01/
        ├── if_baseline/
        │   ├── manifest.json
        │   └── v20250101T120000.joblib
        └── cvae_transform/
            ├── manifest.json
            └── v20250101T120000.joblib
```

The legacy single-model layout (`agents/{agent}/*.joblib`) is preserved for backward compatibility.

---

## Transport Adapters

### SQS/SNS (AWS)

Built-in support via `aiobotocore`. For local development with LocalStack:

```bash
docker run -p 4566:4566 localstack/localstack:3
export AWS_DEFAULT_REGION=us-east-1
export AWS_ACCESS_KEY_ID=test
export AWS_SECRET_ACCESS_KEY=test
```

### Extensibility

Implement `ITransport` from `sentinel.domain.ports.transport`:

```python
from sentinel.domain.ports.transport import ITransport

class MyTransport(ITransport):
    async def receive(self): ...
    async def ack(self, message_id): ...
    async def nack(self, message_id): ...
    async def publish(self, topic, payload): ...
```

---

## Development

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"

pytest tests/unit          # unit tests (no external dependencies)
pytest tests/integration   # requires Docker (Redis + LocalStack)

bash scripts/generate_proto.sh   # regenerate gRPC stubs
```

The generated stubs in `sentinel/adapters/grpc/generated/` are committed — CI does not require `protoc`.

---

## Project Structure

```
sentinel/
├── domain/
│   ├── models.py
│   ├── errors.py
│   └── ports/
│       ├── transport.py
│       ├── correlation_store.py
│       ├── model_store.py
│       ├── reporter.py
│       ├── alert_sink.py
│       └── detector.py            ← IDetector + DetectorState (Strategy interface)
├── agent/
│   ├── use_cases/
│   │   ├── process_message.py     ← multi-detector scoring and training loop
│   │   ├── train.py
│   │   ├── champion.py
│   │   ├── run_test.py
│   │   └── set_phase.py
│   ├── detectors/
│   │   ├── isolation_forest_detector.py
│   │   ├── cvae_detector.py
│   │   ├── maf_detector.py
│   │   ├── nri_detector.py
│   │   └── factory.py
│   ├── isolation_forest.py        ← low-level sklearn wrapper (legacy, preserved)
│   └── correlator.py
├── cortex/
│   ├── use_cases/
│   └── dual_network.py
├── adapters/
│   ├── transport/
│   ├── store/
│   ├── model_store/               ← DiskModelStore with per-detector paths
│   ├── grpc/
│   └── alerting/
├── config/
├── logging/
└── runtime/
```

---

## License

Sentinel is licensed under the Apache License, Version 2.0.

See the [LICENSE](./LICENSE) and [NOTICE](./NOTICE) files for details.
