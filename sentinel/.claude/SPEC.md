# Sentinel — Technical Specification (SPEC)

## Stack Tecnológico

| Componente | Tecnologia |
|------------|-----------|
| Linguagem | Python 3.11+ |
| Concorrência | asyncio |
| gRPC | grpcio + grpcio-tools (stubs commitados) |
| Redis client | aioredis |
| AWS client | aiobotocore |
| ML — Agent (operacional) | scikit-learn IsolationForest |
| ML — Agent (semântico) | PyTorch (cVAE, MAF/MADE, NRI stub) |
| ML — Cortex | PyTorch (Autoencoder) |
| Persistência de modelo | joblib (IsolationForest) + torch.save (cVAE/MAF) |
| Config | Pydantic v2 + python-dotenv |
| Logs | structlog (stdout JSON \| pretty) |
| Testes | pytest + pytest-asyncio + pytest-docker |
| Build | pyproject.toml (PEP 517) |
| CLI | Click |

---

## Estrutura de Pastas

```
sentinel/                              # root do projeto
├── sentinel/                          # pacote Python pip-instalável
│   ├── domain/                        # núcleo — zero dependências externas
│   │   ├── models.py                  # dataclasses/Pydantic de domínio
│   │   ├── errors.py                  # exceções de domínio
│   │   └── ports/                     # interfaces abstratas (ABCs)
│   │       ├── transport.py           # ITransport
│   │       ├── correlation_store.py   # ICorrelationStore
│   │       ├── model_store.py         # IModelStore
│   │       ├── reporter.py            # IReporter
│   │       ├── alert_sink.py          # IAlertSink
│   │       └── detector.py            # IDetector + DetectorState (Strategy)
│   │
│   ├── agent/
│   │   ├── use_cases/
│   │   │   ├── process_message.py     # orquestra correlação → score (todos os detectors) → publish
│   │   │   ├── train.py               # accumulate_sample + run_training(IDetector)
│   │   │   ├── champion.py            # evaluate_fp_rate + run_selection(IDetector)
│   │   │   ├── run_test.py            # run_detector_test(IDetector, DetectorState)
│   │   │   └── set_phase.py           # transição por detector ou todos
│   │   ├── detectors/                 # implementações concretas de IDetector
│   │   │   ├── __init__.py
│   │   │   ├── isolation_forest_detector.py
│   │   │   ├── cvae_detector.py
│   │   │   ├── maf_detector.py
│   │   │   ├── nri_detector.py
│   │   │   └── factory.py             # create_detector(DetectorConfig) → IDetector
│   │   ├── isolation_forest.py        # funções sklearn de baixo nível (legado, mantido)
│   │   └── correlator.py              # lógica de correlação via ICorrelationStore
│   │
│   ├── cortex/
│   │   ├── use_cases/
│   │   │   ├── aggregate.py           # monta/atualiza vetor de estado
│   │   │   ├── causal_chain.py        # detecta propagação temporal de anomalias
│   │   │   └── autoencoder.py         # PyTorch: treina e infere padrão sistêmico
│   │   └── dual_network.py            # orquestra causal + autoencoder → Alert
│   │
│   ├── adapters/
│   │   ├── transport/
│   │   │   ├── sqs_sns.py             # ITransport → AWS SQS/SNS via aiobotocore
│   │   │   └── kafka.py               # ITransport → stub Kafka (futuro)
│   │   ├── store/
│   │   │   └── redis_store.py         # ICorrelationStore via aioredis
│   │   ├── model_store/
│   │   │   └── disk_store.py          # IModelStore: joblib + manifest.json por detector
│   │   ├── grpc/
│   │   │   ├── proto/
│   │   │   │   └── sentinel.proto
│   │   │   ├── generated/             # stubs commitados — NÃO editar manualmente
│   │   │   │   ├── sentinel_pb2.py
│   │   │   │   └── sentinel_pb2_grpc.py
│   │   │   ├── server.py              # Cortex gRPC server (asyncio)
│   │   │   └── client.py              # Agent/Cortex client com fallback standalone
│   │   └── alerting/
│   │       └── queue_sink.py          # IAlertSink → publica em alarm_queue
│   │
│   ├── config/
│   │   ├── schema.py                  # Pydantic v2: SentinelConfig, AgentConfig, DetectorConfig
│   │   ├── loader.py                  # carrega JSON/YAML + env vars + aplica migrações
│   │   └── migrations/
│   │       ├── registry.py
│   │       └── v1_to_v2.py
│   │
│   ├── logging/
│   │   └── logger.py
│   │
│   └── runtime/
│       ├── launcher.py                # wiring: config → detectors → AgentRunner → asyncio.run
│       └── cli.py                     # CLI Click
│
├── tests/
│   ├── unit/
│   │   ├── agent/
│   │   │   ├── test_correlator.py
│   │   │   ├── test_isolation_forest.py
│   │   │   ├── test_process_message.py
│   │   │   ├── test_detectors.py      # testa todos os 4 algoritmos + factory
│   │   │   └── test_use_cases.py      # champion, train, set_phase, run_test
│   │   └── cortex/
│   ├── integration/
│   └── fixtures/
│
├── scripts/
│   └── generate_proto.sh
│
├── pyproject.toml
├── CLAUDE.md
└── README.md
```

---

## Domain Models

```python
# domain/models.py

@dataclass
class RawMessage:
    id: str
    body: dict
    received_at: datetime

@dataclass
class ProcessedEvent:
    agent_name: str
    correlation_id: str
    input_received_at: datetime
    output_sent_at: datetime
    processing_latency_ms: float
    anomaly_score: float           # pior score entre detectores em INFERENCE
    is_anomaly: bool               # any(det.is_anomaly(score))
    anomaly_type: str | None       # TIMEOUT | SCORE | None
    payload_schema_hash: str
    input_size_bytes: int
    output_size_bytes: int
    input_body: dict | None        # incluído apenas em anomalias (para debugging)
    output_body: dict | None       # incluído apenas em anomalias (para debugging)

@dataclass
class HeartbeatEvent:
    agent_name: str
    timestamp: datetime
    message_rate_per_sec: float
    error_rate_last_window: float
    last_message_timestamp: datetime
    model_phase: ModelPhase        # fase do primeiro detector (backward compat)

@dataclass
class Alert:
    source: str
    severity: AlertSeverity
    alert_type: AlertType
    affected_agents: list[str]
    description: str
    timestamp: datetime
    diagnosis: dict
```

---

## Port Contracts

```python
# domain/ports/detector.py  ← NOVO

@dataclass
class DetectorState:
    name: str
    algorithm: str
    phase: ModelPhase
    model: Any                         # objeto opaco — gerenciado pelo IDetector
    training_buffer: list[dict]
    test_buffer: list[dict] = field(default_factory=list)
    test_sample_rate: float = 0.0
    last_test_result: dict = field(default_factory=dict)
    champion_fp_rate: float = 1.0
    challengers_rejected: int = 0

class IDetector(ABC):
    # Propriedades obrigatórias
    @property
    def name(self) -> str: ...
    @property
    def algorithm(self) -> str: ...
    @property
    def training_window(self) -> int: ...
    @property
    def test_buffer_size(self) -> int: ...

    # Métodos obrigatórios
    def extract_features(self, event_dict: dict) -> list[float]: ...
    def fit(self, samples: list[dict]) -> Any: ...          # retorna modelo treinado
    def score(self, model: Any, event_dict: dict) -> float: ...
    def is_anomaly(self, score: float) -> bool: ...         # encapsula direção do score

# event_dict disponível para todos os detectores:
# {
#   "processing_latency_ms": float,
#   "input_size_bytes": int,
#   "output_size_bytes": int,
#   "payload_schema_hash": str,
#   "_input_body": dict,    ← campos de payload do input (para cVAE, MAF)
#   "_output_body": dict,   ← campos de payload do output (para cVAE, MAF)
# }
```

```python
# domain/ports/model_store.py  (métodos adicionados)

class IModelStore(ABC):
    # Legado (single model por agent)
    async def save_version(self, name: str, model: Any) -> str: ...
    async def load(self, name: str, version: str | None = None) -> Any: ...
    async def list_versions(self, name: str) -> list[VersionMeta]: ...
    async def rollback(self, name: str) -> None: ...

    # Por detector (novo, recomendado)
    async def save_detector_version(self, agent_name: str, detector_name: str, model: Any) -> str: ...
    async def load_detector(self, agent_name: str, detector_name: str, version: str | None = None) -> Any: ...
    async def list_detector_versions(self, agent_name: str, detector_name: str) -> list[VersionMeta]: ...
```

---

## Algoritmos de Detecção

### IsolationForest (`isolation_forest`)

**Finalidade:** Detecta anomalias **operacionais** — desvios em latência de processamento, tamanhos de payload e schema hash. Não acessa os campos do payload; usa apenas as métricas do fluxo. Converge rápido (~500 amostras) e é o baseline recomendado para todo agent.

**Features extraídas automaticamente:**
- `processing_latency_ms` — tempo entre input e output
- `input_size_bytes` — tamanho em bytes do payload de input
- `output_size_bytes` — tamanho em bytes do payload de output
- `payload_schema_hash` — hash inteiro das chaves do output (detecta schema drift)

**Direção do score:** `score < threshold` → anomalia. Threshold é negativo (default: -0.1) porque o IsolationForest retorna scores entre -1 e 1 onde valores mais negativos indicam maior anormalidade.

**Parâmetros de configuração:**
| Parâmetro | Default | Descrição |
|-----------|---------|-----------|
| `anomaly_threshold` | `-0.1` | Limiar de score; valores abaixo indicam anomalia |
| `training_window` | `500` | Número de amostras antes do primeiro fit |
| `test_buffer_size` | `100` | Tamanho do buffer hold-out para avaliação de FP |
| `n_estimators` | `100` | Número de árvores da floresta |
| `contamination` | `"auto"` | Fração esperada de anomalias no treino |

---

### cVAE — Conditional Variational Autoencoder (`cvae`)

**Finalidade:** Detecta anomalias **semânticas de transformação** — aprende a distribuição condicional P(output_fields | input_fields). Sinaliza quando o output é semanticamente inconsistente com o input, por exemplo: um serviço que calcula um campo derivado (`result = value01 + value02`) começa a retornar valores incorretos.

**Arquitetura:**
- Encoder: MLP `[input_dim + output_dim → hidden_dim → mu, logvar]`
- Decoder: MLP `[latent_dim + input_dim → hidden_dim → output_dim]`
- Loss: ELBO = MSE(reconstrução, real) + 0.1 × KL(N(mu,var) ‖ N(0,I))
- Inferência: usa mu diretamente (sem amostragem), score = MSE(decoder(mu, input), actual_output)

**Normalização:** média e desvio-padrão calculados no fit e armazenados no modelo; aplicados no score.

**Direção do score:** `score > threshold` → anomalia. Score é MSE de reconstrução (≥ 0); threshold é positivo.

**Requer `field_map`** — sem ele, o construtor lança `ValueError`. Os campos devem existir nos payloads do sistema monitorado.

**Parâmetros de configuração:**
| Parâmetro | Default | Descrição |
|-----------|---------|-----------|
| `field_map.input` | — | **Obrigatório.** Lista de campos do `_input_body` usados como condição |
| `field_map.output` | — | **Obrigatório.** Lista de campos do `_output_body` a reconstruir |
| `anomaly_threshold` | `0.5` | Limiar de MSE; valores acima indicam anomalia |
| `training_window` | `500` | Número de amostras antes do primeiro fit |
| `test_buffer_size` | `100` | Tamanho do buffer hold-out |
| `hidden_dim` | `32` | Dimensão das camadas ocultas do encoder/decoder |
| `latent_dim` | `8` | Dimensão do espaço latente |
| `learning_rate` | `0.001` | Taxa de aprendizado do Adam |
| `epochs` | `30` | Épocas de treinamento por fit |

---

### MAF — Masked Autoregressive Flow (`maf`)

**Finalidade:** Detecta anomalias na **distribuição conjunta** de todos os campos numéricos — aprende P(x₁, x₂, ..., xₙ) via MADE (Masked Autoencoder for Distribution Estimation). Sinaliza quando a combinação multivariada de métricas é anômala, mesmo que cada métrica individualmente pareça normal. Ideal para agents que agregam múltiplos fluxos (batcher) ou cobrem o pipeline completo end-to-end.

**Arquitetura:** MADE de 2 camadas com máscaras autoregressivas. Assume distribuição Gaussiana condicional; score = -log_likelihood(x).

**Descoberta de campos:** se `field_map` for omitido, o MAF extrai automaticamente todos os campos numéricos dos payloads (`_input_body` + `_output_body`) no primeiro fit, incluindo as métricas operacionais. A ordem dos campos é fixada no fit e persistida no modelo.

**Direção do score:** `score > threshold` → anomalia. Score é `-log_likelihood` (maior = mais anômalo).

**Parâmetros de configuração:**
| Parâmetro | Default | Descrição |
|-----------|---------|-----------|
| `field_map.input` | `null` | Opcional. Campos do `_input_body` a incluir |
| `field_map.output` | `null` | Opcional. Campos do `_output_body` a incluir |
| `anomaly_threshold` | `5.0` | Limiar de -log_likelihood; valores acima indicam anomalia |
| `training_window` | `500` | Número de amostras antes do primeiro fit |
| `test_buffer_size` | `100` | Tamanho do buffer hold-out |
| `hidden_dim` | `64` | Dimensão das camadas ocultas do MADE |
| `learning_rate` | `0.001` | Taxa de aprendizado do Adam |
| `epochs` | `30` | Épocas de treinamento por fit |

---

### NRI — Neural Relational Inference (`nri`)

**Finalidade (prevista):** Detectar anomalias no **grafo de dependências** entre campos — aprende quais campos normalmente influenciam quais e detecta quando essa estrutura de relacionamentos muda (ex: um campo que sempre derivava de outro passa a ser independente).

**Status atual:** Stub com interface estável.
- `fit()` → lança `NotImplementedError`
- `score()` → retorna `0.0`
- `is_anomaly()` → retorna `False`

Pode ser incluído em um agent sem causar erros; simplesmente não contribui para detecção até ser implementado.

**Parâmetros de configuração:**
| Parâmetro | Default | Descrição |
|-----------|---------|-----------|
| `anomaly_threshold` | `0.5` | Reservado para implementação futura |
| `training_window` | `500` | Reservado para implementação futura |
| `test_buffer_size` | `100` | Reservado para implementação futura |

---

## Configuração Completa (`sentinel.json`)

```json
{
  "sentinel": {
    "storage_path": "~/.sentinel",
    "redis": { "host": "localhost", "port": 6379 },
    "alarm_queue": { "type": "sqs", "resource": "sentinel-alarms" },

    "model": {
      "max_versions": 10,
      "auto_rollback": true,
      "isolation_forest": {
        "n_estimators": 100,
        "contamination": "auto",
        "anomaly_threshold": -0.1,
        "training_window": 500
      },
      "autoencoder": {
        "hidden_dim": 64,
        "training_window": 1000,
        "learning_rate": 0.001,
        "epochs": 50
      }
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
            "test_buffer_size": 100,
            "anomaly_threshold": -0.1,
            "n_estimators": 100,
            "contamination": "auto"
          },
          {
            "name": "cvae_transform",
            "algorithm": "cvae",
            "phase": "TRAINING",
            "training_window": 500,
            "test_buffer_size": 100,
            "anomaly_threshold": 0.5,
            "hidden_dim": 32,
            "latent_dim": 8,
            "learning_rate": 0.001,
            "epochs": 30,
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

> **Backward compat:** se `detectors` for omitido no agent, o launcher sintetiza automaticamente um `DetectorConfig(algorithm="isolation_forest")` a partir de `model.isolation_forest`. Arquivos `sentinel.json` existentes sem `detectors` continuam funcionando sem alteração.

### Defaults dos campos de `DetectorConfig`

| Campo | Default | Descrição |
|-------|---------|-----------|
| `name` | — | Identificador único dentro do agent |
| `algorithm` | `"isolation_forest"` | `isolation_forest` \| `cvae` \| `maf` \| `nri` |
| `phase` | `"TRAINING"` | Fase inicial do detector |
| `training_window` | `500` | Amostras antes do primeiro fit |
| `test_buffer_size` | `100` | Tamanho do buffer hold-out para FP rate |
| `anomaly_threshold` | `-0.1` | Limiar (semântica difere por algoritmo — ver seção acima) |
| `field_map` | `null` | Mapeamento de campos; obrigatório para `cvae`, opcional para `maf` |
| `n_estimators` | `100` | Número de árvores (apenas `isolation_forest`) |
| `contamination` | `"auto"` | Fração de anomalias esperada (apenas `isolation_forest`) |
| `hidden_dim` | `32` | Dimensão oculta (apenas `cvae` e `maf`) |
| `latent_dim` | `8` | Dimensão latente (apenas `cvae`) |
| `learning_rate` | `0.001` | Taxa de aprendizado (apenas `cvae` e `maf`) |
| `epochs` | `30` | Épocas de treino por fit (apenas `cvae` e `maf`) |

---

## AgentState

```python
@dataclass
class AgentState:
    agent_name: str
    detectors: dict[str, DetectorState]   # detector_name → DetectorState
    message_count: int = 0
    error_count: int = 0

    # Propriedades backward-compat (usadas por heartbeat e dashboard)
    @property
    def phase(self) -> ModelPhase:
        """Fase do primeiro detector, ou COLD se sem detectores."""

    @property
    def champion_fp_rate(self) -> float:
        """Melhor (menor) FP rate entre todos os detectores."""

    @property
    def challengers_rejected(self) -> int:
        """Total de challengers rejeitados em todos os detectores."""
```

---

## Redis Key Schema

| Padrão | Exemplo | TTL |
|--------|---------|-----|
| `{agent_name}:{correlation_id}` | `agent01:abc-123` | 300s (configurável) |

---

## Versionamento de Modelos em Disco

```
~/.sentinel/
├── agents/
│   └── agent01/
│       ├── if_baseline/               ← novo layout por detector
│       │   ├── manifest.json
│       │   └── v20250101T120000.joblib
│       └── cvae_transform/
│           ├── manifest.json
│           └── v20250101T120000.joblib
└── cortex/
    └── cortex01/
        ├── manifest.json
        ├── v20250101T120000.pt
        └── v20250101T120000.meta.json
```

O layout legado `agents/{agent_name}/*.joblib` (sem subdiretório de detector) é preservado para compatibilidade com instalações existentes.

`manifest.json` por detector:
```json
{
  "current": "v20250101T120000",
  "versions": [
    { "id": "v20250101T120000", "created_at": "...", "performance_score": null }
  ]
}
```

---

## Fluxo de Dados — Agent (multi-detector)

```
[Transport Input] → receive() → RawMessage
    → store_input(redis, key, {body, received_at, size_bytes}, ttl)

[Transport Output] → receive() → RawMessage
    → retrieve_input(redis, key)
        → None: emit ProcessedEvent(TIMEOUT)
        → Found: build event_dict = {
              processing_latency_ms, input_size_bytes, output_size_bytes,
              payload_schema_hash,
              _input_body,    ← payload completo do input
              _output_body,   ← payload completo do output
          }

    → Para cada detector em INFERENCE:
        score = detector.score(model, event_dict)
        if detector.is_anomaly(score): detected_anomaly = True

    → Para cada detector em TRAINING:
        accumulate_sample(training_buffer, event_dict)
        if should_train(buffer, training_window):
            challenger = detector.fit(buffer)
            result = run_selection(detector, challenger, champion, test_buffer)
            if result.is_new_champion:
                model_store.save_detector_version(agent, detector_name, model)

    → anomaly_score = pior score entre detectores que sinalizaram
    → emit ProcessedEvent(is_anomaly=any, score=worst)
```

---

## Dashboard API — Endpoints de Detector

| Método | Path | Descrição |
|--------|------|-----------|
| `POST` | `/api/agents/{name}/phase` | Muda fase de **todos** os detectores do agent |
| `POST` | `/api/agents/{name}/detectors/{det}/phase` | Muda fase de um detector específico |
| `POST` | `/api/agents/{name}/detectors/{det}/test-rate` | Define sample rate do test buffer |
| `POST` | `/api/agents/{name}/detectors/{det}/run-test` | Executa avaliação do test buffer |

SSE events relacionados a detectores:
- `detector_update` — campos: `agent`, `detector`, + campos atualizados (`champion_fp_rate`, `test_sample_rate`, etc.)
- `phase_change` — campo `detector` presente quando a mudança é específica de um detector

---

## gRPC Protocol

```protobuf
syntax = "proto3";

service SentinelCortex {
  rpc ReportEvent(ProcessedEventProto) returns (Ack);
  rpc ReportHeartbeat(HeartbeatProto) returns (Ack);
}

message ProcessedEventProto {
  string agent_name = 1;
  string correlation_id = 2;
  int64  input_received_at_ms = 3;
  int64  output_sent_at_ms = 4;
  float  processing_latency_ms = 5;
  float  anomaly_score = 6;
  bool   is_anomaly = 7;
  string anomaly_type = 8;
  string payload_schema_hash = 9;
  int32  input_size_bytes = 10;
  int32  output_size_bytes = 11;
}

message HeartbeatProto {
  string agent_name = 1;
  int64  timestamp_ms = 2;
  float  message_rate_per_sec = 3;
  float  error_rate_last_window = 4;
  int64  last_message_timestamp_ms = 5;
  string model_phase = 6;
}

message Ack { bool ok = 1; }
```

---

## Tratamento de Falhas

| Cenário | Comportamento |
|---------|---------------|
| Redis indisponível | Agent pausa consumo, polling com backoff até reconectar |
| gRPC Cortex falhou | Agent entra em standalone mode, loga warning, continua operando |
| TTL de correlação expira | Emite `ProcessedEvent(type=TIMEOUT)`, remove chave |
| Detector sem modelo em INFERENCE | Scoring ignorado para esse detector; outros detectores continuam |
| Modelo novo pior (FP rate) | Challenger rejeitado, champion mantido; `challengers_rejected` incrementado |
| `fit()` lança NotImplementedError | Loga warning `detector_training_not_implemented`, training_buffer descartado |
| `score()` lança exceção | Loga warning `scoring_failed`, detector ignorado no evento |
| cVAE sem `field_map` | `ValueError` no construtor — impede startup antes de processar mensagens |
| Campo do `field_map` ausente no payload | Feature zerada (0.0); detector continua operando |

---

## CLI

```bash
sentinel start --config sentinel.json          # sobe agents e cortex configurados
sentinel set-phase agent01 INFERENCE           # transiciona TODOS os detectores do agent
sentinel status                                # exibe estado de cada agent/cortex
```

---

## Testes

```bash
pytest tests/unit                  # unitários (sem dependências externas)
pytest tests/integration           # requer Docker (Redis + LocalStack)
```

Cobertura unitária principal:
- `test_detectors.py` — todos os 4 algoritmos: fit, score, is_anomaly direction, factory
- `test_use_cases.py` — champion selection, training trigger, set_phase por detector, run_test
- `test_process_message.py` — pipeline completo com mock detectors, multi-detector anomaly propagation

---

## Geração de Stubs gRPC

```bash
python -m grpc_tools.protoc \
  -I sentinel/adapters/grpc/proto \
  --python_out=sentinel/adapters/grpc/generated \
  --grpc_python_out=sentinel/adapters/grpc/generated \
  sentinel/adapters/grpc/proto/sentinel.proto
```

Stubs commitados — CI não precisa de `protoc` instalado.
