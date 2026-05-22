# Sentinel — Full Context

> Documento de referência completo. Combina arquitetura do sistema, use case e topologia de observação.

---

## PARTE 1 — SENTINEL: ARQUITETURA E ALGORITMOS

### 1.1 Conceito Central

Sentinel é um sistema modular de detecção de anomalias para arquiteturas orientadas a mensagens (SQS/SNS/Kafka). Opera em dois níveis:

- **SentinelAgent** — monitora fluxos individuais (input→output correlacionados)
- **SentinelCortex** — agrega sinais de múltiplos agents, detecta cascatas e falhas sistêmicas

A filosofia central é que anomalias existem em múltiplas camadas simultaneamente, e nenhum algoritmo sozinho captura todas elas:

| Nível | Algoritmo | O que detecta |
|-------|-----------|---------------|
| Operacional | IsolationForest | Latência anormal, tamanho de payload, schema drift |
| Semântico | cVAE | Output inconsistente com input (relação quebrada) |
| Distribuição | MAF | Comportamento conjunto de campos (correlações inter-campo) |
| Relacional | NRI (futuro) | Grafos de dependência entre campos |
| Causal | Cortex — Causal Chain | Cascata temporal entre agents |
| Sistêmico | Cortex — Autoencoder | Falha coordenada system-wide |

---

### 1.2 SentinelAgent — Arquitetura Multi-Detector

#### Fluxo de Processamento por Evento

```
Input Message → store_input(Redis, ttl configurável)
                    ↓
Output Message → retrieve_input(Redis) → build event_dict
                    ↓
         [Para cada detector independentemente]
                    ↓
    INFERENCE:  score(model) → is_anomaly? → flag se SIM
    TRAINING:   accumulate_sample → treina ao atingir training_window
                    ↓
         Agregação: is_anomaly = ANY(detector flagou)
                    worst_score = MAX(scores flagados)
                    ↓
         Publica ProcessedEvent → Cortex via gRPC
```

#### Estrutura do event_dict (compartilhado por todos os detectores)

```python
{
    "processing_latency_ms": float,    # tempo entre input e output
    "input_size_bytes": int,
    "output_size_bytes": int,
    "payload_schema_hash": str,        # hash das chaves do output
    "_input_body": dict,               # payload bruto do input
    "_output_body": dict,              # payload bruto do output
}
```

#### Modos de Correlação

| Modo | Padrão | Comportamento |
|------|--------|---------------|
| `normal` | 1→1 | Chave Redis deletada após match |
| `grouping` | N→1 | Múltiplos inputs → 1 output agregado |
| `splitting` | 1→N | 1 input → múltiplos outputs (chave mantida até TTL) |

---

### 1.3 Ciclo Champion/Challenger

Cada detector mantém estado independente e nunca promove um modelo pior:

```
TRAINING phase:
  1. Acumula amostras no training_buffer
  2. Desvia ~10% para test_buffer (hold-out)
  3. Ao atingir training_window:
     a) fit challenger no training_buffer
     b) mede FP rate do challenger no test_buffer
     c) FP rate < champion atual → new champion, salva em disco
        FP rate ≥ champion atual → rejeita, continua treinando
     d) limpa training_buffer

INFERENCE phase:
  - Só score, sem update de modelo
  - Transição automática via auto_infer_fp_threshold
```

**FP (False Positive)** = amostra normal do hold-out flagada como anomalia. Só modelos com melhoria demonstrável chegam à inferência.

---

### 1.4 Os 4 Algoritmos de Detecção

#### A. IsolationForest (Baseline Operacional)

**Propósito:** Anomalias em latência, tamanhos de payload e schema drift. Não requer conhecimento de domínio.

**Features:**
- `processing_latency_ms`
- `input_size_bytes`
- `output_size_bytes`
- `payload_schema_hash`

**Algoritmo:** scikit-learn IsolationForest — ensemble de árvores aleatórias que isolam pontos anômalos.

**Score:** `score < threshold (-0.1)` → anomalia

**Por quê:** Converge rápido (~500 amostras), sem hiperparâmetros críticos, multivariado por natureza.

---

#### B. cVAE — Conditional Variational Autoencoder (Semântico)

**Propósito:** Detectar quando campos de output são semanticamente inconsistentes com o input. Ex: `result = value01 + value02` começa retornando somas erradas.

**Arquitetura:**
```
Encoder:  [input_fields + output_fields] → MLP → (mu, log_var)
Decoder:  [latent_vector + input_fields] → MLP → output_hat

Loss: ELBO = MSE(output_hat, actual_output) + 0.1 * KL(q(z) || p(z))

Inferência: score = MSE(decode(mu, input_fields), actual_output)
```

**Requisito:** `field_map` obrigatório — especifica quais campos de input e output monitorar.

**Score:** `score > threshold` → anomalia (threshold padrão: 0.5)

**Por quê:** Captura relações aprendidas input→output. Ideal para transformações com lógica de negócio.

---

#### C. MAF — Masked Autoregressive Flow (Distribuição Conjunta)

**Propósito:** Detectar anomalias na distribuição conjunta de múltiplos campos. Captura correlações inter-campo invisíveis a análises univariadas.

**Algoritmo:** MADE (Masked Autoencoder for Distribution Estimation)
- 2-layer autoregressive MLP com máscaras garantindo `x_i` depende só de `x_{<i}`
- Modela Gaussianas condicionais: `p(x_i | x_{<i}) = N(μ_i, σ_i)`
- Score: `-log P(x) = -Σ log N(x_i; μ_i, σ_i)` (negative log-likelihood)

**Field discovery:**
- Sem `field_map`: auto-descobre todos os campos numéricos no primeiro fit
- Com `field_map`: usa apenas os campos especificados
- Ordem de campos fixada no primeiro fit e persistida no modelo

**Score:** `score > threshold` → anomalia (threshold padrão: 5.0)

**Por quê:** Captura correlações entre campos. Mais data-hungry (~1500-2000 amostras) mas poderoso para pipelines end-to-end.

---

#### D. NRI — Neural Relational Inference (Futuro)

**Propósito (planejado):** Aprender grafos explícitos de dependência entre campos — quais campos normalmente influenciam quais. Detectar quando relações quebram.

**Status atual:** Interface estável, implementação pendente.
- `fit()` → `NotImplementedError` (logado, não crasha)
- `score()` → 0.0 (safe no-op)
- `is_anomaly()` → sempre False

---

### 1.5 SentinelCortex — Dual Network

#### Visão Geral

```
Eventos/Heartbeats dos agents (gRPC)
              ↓
    [Silence Detection] ← agent sem resposta?
              ↓
    ┌─────────────────────────────────┐
    │         DUAL NETWORK            │
    ├──────────────┬──────────────────┤
    │  Causal Chain│   Autoencoder    │
    │  (temporal)  │   (sistêmico)    │
    └──────┬───────┴──────┬───────────┘
           │               │
    ┌──────▼───────────────▼──────────┐
    │         Alert Fusion            │
    │  Ambos    → CRITICAL            │
    │  Causal   → WARNING             │
    │  Sistêmico→ WARNING             │
    │  Nenhum   → sem alerta          │
    └─────────────┬───────────────────┘
                  ↓
           Publica alerta
         (sentinel-alarms queue)
```

#### 1. Causal Chain Analysis

```python
def analyze_causality(anomaly_window, window_s=30.0):
    # Coleta eventos anômalos dos últimos 30s
    # Ordena por timestamp (crescente)
    # Primeiro agent = root cause
    # Demais = vítimas em cascata
    # Confidence = len(sequence) / total_events
    
    return CausalDiagnosis(
        root_agent=events_sorted[0].agent_name,
        affected_agents=[e.agent_name for e in events_sorted[1:]],
        confidence=min(1.0, len(sequence) / total_events),
        sequence=ordered_unique_agent_names,
    )
```

#### 2. Autoencoder Sistêmico

```python
class SentinelAutoencoder(nn.Module):
    # Encoder: Linear(input_dim=4, hidden_dim=64) + ReLU
    # Decoder: Linear(hidden_dim, input_dim)

# Vetor de estado por agent:
state_vector = [
    message_rate,       # eventos/seg
    error_rate,         # fração de eventos anômalos
    anomaly_score,      # score máximo reportado
    is_inference_mode,  # 0=TRAINING, 1=INFERENCE
]

# Estado agregado = média de todos os vectors
# Anomalia sistêmica: reconstruction_error > baseline_error * 2.0
```

#### Modos de Adaptação do Cortex

| Modo | Mecanismo | Quando usar |
|------|-----------|-------------|
| `fine_tuning` | Re-treina a cada N amostras com replay buffer | Cortex setoriais (deriva gradual) |
| `ema` | Atualização suave via média exponencial (alpha=0.005) | Cortex de topo (estabilidade > sensibilidade) |
| `sliding_window` | Retreina com janela deslizante de amostras | Ambientes muito não-estacionários |

---

### 1.6 Persistência e Versionamento

```
~/.sentinel/
├── agents/{agent_name}/
│   └── {detector_name}/
│       ├── manifest.json          ← version atual + histórico
│       ├── v20250101T120000.joblib    (IsolationForest / MAF)
│       └── v20250101T120035.pt        (cVAE — PyTorch state_dict)
│
└── cortex/{cortex_name}/
    ├── manifest.json
    ├── v20250101T120000.pt
    └── v20250101T120000.meta.json
```

- Cada detector mantém histórico independente de versões
- Rollback por detector sem afetar outros
- Max 10 versões por detector (configurável)
- Auto-rollback habilitado por padrão

---

### 1.7 Arquitetura Hexagonal — Interfaces Plugáveis

| Interface | Implementações |
|-----------|----------------|
| `ITransport` | SQS, SNS, Kafka |
| `ICorrelationStore` | Redis |
| `IModelStore` | Disco (por versão) |
| `IReporter` | gRPC client, Dashboard SSE, Standalone |
| `IAlertSink` | SQS queue, SNS topic |

Adicionar algoritmo = implementar `IDetector` + registrar no factory. Zero mudanças em agent/cortex.

---

### 1.8 Resiliência

| Cenário | Comportamento |
|---------|---------------|
| Redis indisponível | Agent pausa, retry com backoff |
| Cortex gRPC inacessível | Agent entra em standalone mode (continua operando) |
| Correlation timeout (TTL expirado) | `ProcessedEvent(type=TIMEOUT, is_anomaly=True)` |
| Detector scoring falha | Loga warning, outros detectores continuam |
| `fit()` raises `NotImplementedError` | Loga warning, buffer limpo, continua |
| Campo ausente no `field_map` | Feature default 0.0, detector continua |
| cVAE sem `field_map` | `ValueError` na inicialização (fail fast) |

---

## PARTE 2 — USE CASE: OS 4 APPS

### 2.1 Topologia do Pipeline de Aplicação

```
APP01 (Producer)
  │
  └──→ SNS01 ──→ SQS01 ──→ APP02 (Enricher)
                              │
                              └──→ SNS02 ──┬──→ SQS02 ──→ APP03 (Batcher) ──→ SNS03 ──→ SQS04
                                           │
                                           └──→ SQS03 ──→ APP04 (Validator) ──→ SNS04 ──→ SQS05
```

APP02 faz **fan-out**: APP03 e APP04 consomem o mesmo SNS02 em paralelo (diamond dependency).

---

### 2.2 APP01 — Producer (porta 3001)

**Papel:** Gera o stream contínuo de eventos que alimenta o pipeline.

**Comportamento:**
- Timer configurable (padrão: 1000ms)
- A cada tick: cria UUID `correlationId`, gera `value01` e `value02` (inteiros 1-1000) e `someString` (string aleatória 10-50 chars)
- Publica para SNS01

**Output normal:** `{ correlationId, someString, value01, value02 }`

**Erro simulado:** omite `value02` → **schema drift** detectável via `payload_schema_hash`

**API de controle:**
- `GET /config` → `{ intervalMs, errorRate, running }`
- `POST /config` → atualiza `intervalMs`, `errorRate`
- `POST /state` → `{ running: true/false }`
- `GET /log` → últimos 30 eventos

---

### 2.3 APP02 — Enricher (porta 3002)

**Papel:** Consome mensagens, adiciona campos computados, republica. É o coração semântico do pipeline.

**Comportamento:**
- Consome SQS01 (long-poll, 20s, batch 10)
- Calcula `result = value01 + value02`
- Calcula `stringSize = someString.length`
- Preserva `correlationId`
- Publica para SNS02

**Output normal:** `{ correlationId, someString, value01, value02, result, stringSize }`

**Erros simulados:**
- **Tipo 1:** `result = value01 + value02 + random(1-100)` → soma errada
- **Tipo 2:** `stringSize = len(someString) + random(1-100)` → tamanho errado

> Este é o app mais interessante para o cVAE: a relação input→output é matematicamente definida e verificável.

---

### 2.4 APP03 — Batcher (porta 3003)

**Papel:** Acumula 20 mensagens individuais em buffer e emite um único batch aggregado.

**Comportamento:**
- Consome SQS02 (subscriber do SNS02)
- Mantém buffer em memória, `BATCH_SIZE = 20`
- Ao atingir 20 itens: publica batch e deleta todas as 20 mensagens do SQS02
- Output: `{ records: [...20 items], fileName: "batch_${timestamp}.txt" }`
- Cada item no records: `{ someString, correlationId }`

**Erros simulados:**
- **Tipo 1:** batch incompleto — slice aleatório para 1-19 itens (perda de mensagens)
- **Tipo 2:** um item duplicado no records (duplicata)

**Implicação para correlação:** cria padrão **N:1** — 20 `correlationId` chegam, apenas 1 output sai.

---

### 2.5 APP04 — Validator (porta 3004)

**Papel:** Adiciona campo de validação semântica ao payload enriquecido.

**Comportamento:**
- Consome SQS03 (subscriber do SNS02, paralelo com APP03)
- Lógica: `valid = (stringSize % 2 === 0)` — true se tamanho par, false se ímpar
- Publica para SNS04

**Output normal:** `{ ...todos campos anteriores, valid: boolean }`

**Erros simulados:**
- **Tipo 1:** lógica invertida — `valid = (stringSize % 2 !== 0)`
- **Tipo 2:** campo `valid` omitido → schema drift

---

### 2.6 Mapa de Erros vs Detecção Esperada

| App | Erro Simulado | Mecanismo de Detecção |
|-----|---------------|-----------------------|
| APP01 | `value02` ausente | IsolationForest (schema_hash muda) |
| APP02 | `result` com offset (soma errada) | cVAE (relação value01+value02→result quebrada) |
| APP02 | `stringSize` com offset | cVAE (relação len→stringSize quebrada) |
| APP03 | Batch incompleto (1-19 records) | MAF (distribuição conjunta de campos anômala) + IsolationForest (payload size) |
| APP03 | Registro duplicado | MAF (distribuição anômala) |
| APP04 | `valid` invertido | cVAE (output inconsistente com input) |
| APP04 | `valid` ausente | IsolationForest (schema_hash muda) |

---

## PARTE 3 — TOPOLOGIA DO SENTINEL PARA O USE CASE

### 3.1 Infraestrutura de Observação (Filas Espelho)

O Sentinel **não interfere** no pipeline de aplicação. Todas as observações são feitas via filas espelho — subscribers adicionais dos tópicos SNS:

```
SNS01 ──┬──→ SQS01 (APP02 consome)
        └──→ sentinel-a01-in  (AGENT01 consome — espelha input do APP02)
        └──→ sentinel-a04-in  (AGENT04 consome — espelha input do pipeline)

SNS02 ──┬──→ SQS02 (APP03 consome)
        ├──→ SQS03 (APP04 consome)
        ├──→ sentinel-a01-out (AGENT01 consome — espelha output do APP02)
        ├──→ sentinel-a02-in  (AGENT02 consome — espelha input do APP03)
        └──→ sentinel-a03-in  (AGENT03 consome — espelha input do APP04)

SNS03 ──┬──→ SQS04 (consumidor externo)
        └──→ sentinel-a02-out (AGENT02 consome — espelha output do APP03)

SNS04 ──┬──→ SQS05 (consumidor externo)
        ├──→ sentinel-a03-out (AGENT03 consome — espelha output do APP04)
        └──→ sentinel-a04-out (AGENT04 consome — espelha output do pipeline)
```

**8 filas espelho no total:** sentinel-a0{1-4}-in e sentinel-a0{1-4}-out

---

### 3.2 AGENT01 — Observa APP02 (Enricher)

**Trecho observado:** `SNS01 → [APP02] → SNS02`
**Filas:** `sentinel-a01-in` (input), `sentinel-a01-out` (output)
**Correlação:** 1:1 via `correlationId` · TTL 90s
**Cortex:** reporta para `cortex11` (porta 50051)

#### Detectores

**`if_baseline`** — Isolation Forest
- Training window: 1000 amostras
- Auto-infer FP threshold: 5%
- Anomaly threshold: -0.1
- **Objetivo:** Baseline operacional — detecta latência anormal, mudança de tamanho de payload e schema drift (ex: `value02` ausente no input ou campo sumindo do output)

**`cvae_enrichment`** — Conditional VAE
- Training window: 500 amostras
- Auto-infer FP threshold: 8%
- Anomaly threshold: 0.5
- Hidden dim: 32, Latent dim: 8, Epochs: 30, LR: 0.001
- **Field map:** inputs `[value01, value02, len(someString)]` → outputs `[result, stringSize]`
- **Objetivo:** Verificar se `result = value01 + value02` e `stringSize = len(someString)`. Aprende a transformação matemática e dispara quando o APP02 retorna valores incorretos.

---

### 3.3 AGENT02 — Observa APP03 (Batcher)

**Trecho observado:** `SNS02 → [APP03] → SNS03`
**Filas:** `sentinel-a02-in` (input), `sentinel-a02-out` (output)
**Correlação:** N:1 grouping via `records[*].correlationId` · TTL **1800s (30 min)**
**Cortex:** reporta para `cortex11` (porta 50051)

> TTL de 30 minutos porque o batch só sai quando 20 mensagens acumulam — e com intervalos de 1s no APP01, isso leva ~20 segundos em condições normais, mas pode variar muito.

#### Detectores

**`if_batch_stats`** — Isolation Forest
- Training window: 1000 amostras
- Auto-infer FP threshold: 8%
- Anomaly threshold: -0.1
- **Objetivo:** Baseline operacional do batch — detecta latência de processamento anormal e tamanho de payload incomum (batch muito pequeno ou muito grande).

**`maf_joint`** — Masked Autoregressive Flow
- Training window: 1500 amostras
- Auto-infer FP threshold: 10%
- Anomaly threshold: 2.0
- Hidden dim: 64, Epochs: 30, LR: 0.001
- **Field map:** ausente → auto-descobre todos os campos numéricos no primeiro fit
- **Objetivo:** Aprender a distribuição conjunta normal de todos os campos numéricos do batch (count de records, valores numéricos, tamanhos). Detecta quando o batch chega com número anormal de records (perda de mensagens) ou com distribuição estranha por duplicatas. MAF captura correlações entre campos que IsolationForest não captura individualmente.

---

### 3.4 AGENT03 — Observa APP04 (Validator)

**Trecho observado:** `SNS02 → [APP04] → SNS04`
**Filas:** `sentinel-a03-in` (input), `sentinel-a03-out` (output)
**Correlação:** 1:1 via `correlationId` · TTL 90s
**Cortex:** reporta para `cortex11` (porta 50051) **E** `cortex12` (porta 50052)

> Único agent com dois cortex — serve como ponte entre os grupos setoriais. Eventos do AG03 alimentam tanto a análise upstream (cortex11) quanto a downstream (cortex12).

#### Detectores

**`if_latency`** — Isolation Forest
- Training window: 1000 amostras
- Auto-infer FP threshold: 5%
- Anomaly threshold: -0.1
- **Objetivo:** Baseline operacional do validator — detecta latência anormal e schema drift (ex: campo `valid` sumindo do output).

**`cvae_validation`** — Conditional VAE
- Training window: 500 amostras
- Auto-infer FP threshold: 10%
- Anomaly threshold: **0.1** (5x mais sensível que cvae_enrichment)
- Hidden dim: 32, Latent dim: 8, Epochs: 30, LR: 0.001
- **Field map:** inputs `[value01, value02, result, stringSize]` → outputs `[validationStatus]`
- **Objetivo:** Verificar se a lógica de validação está correta. Aprende a distribuição esperada de `validationStatus` dado os inputs. Threshold mais baixo (0.1) porque a lógica é binária — qualquer inversão é clara violação.

---

### 3.5 AGENT04 — Observa Pipeline End-to-End

**Trecho observado:** `SNS01 → [APP02 → APP03/APP04] → SNS04`
**Filas:** `sentinel-a04-in` (espelha SNS01), `sentinel-a04-out` (espelha SNS04)
**Correlação:** 1:1 via `correlationId` · TTL **300s (5 min)**
**Cortex:** reporta para `cortex12` (porta 50052) **E** `cortex21` (porta 50053)

> Alta taxa de TIMEOUT esperada: mensagens que passam pelo APP03 (batcher N:1) nunca chegam ao SNS04 com o mesmo `correlationId` individual — apenas 1 em cada 20 sai pelo caminho do validator (APP04). O AGENT04 observa a rota APP04, então ~95% das mensagens vão expirar (rota via APP03 não chega ao SNS04 individual).

#### Detectores

**`if_e2e`** — Isolation Forest
- Training window: 1000 amostras
- Auto-infer FP threshold: 8%
- Anomaly threshold: -0.1
- **Objetivo:** Baseline de latência total ponta-a-ponta (SNS01 até SNS04). Captura quando qualquer app no meio adiciona delay anormal. Também detecta schema drift no output final.

**`maf_e2e`** — Masked Autoregressive Flow
- Training window: 1000 amostras
- Auto-infer FP threshold: 12% (mais relaxado — end-to-end é naturalmente mais ruidoso)
- Anomaly threshold: 2.0
- Hidden dim: 64, Epochs: 30, LR: 0.001
- **Field map:** ausente → auto-descobre campos numéricos do output final
- **Objetivo:** Aprender a distribuição conjunta de todo o output final (que contém todos os campos acumulados ao longo do pipeline). Detecta anomalias sistêmicas visíveis apenas quando se compara input original com output final — ex: `value01 + value02` no input vs `result` no output (cross-stage semantic check).

---

### 3.6 Hierarquia de Cortex

```
                        CORTEX21 (porta 50053)
                        "Sistema Completo"
                        Adaptation: EMA (alpha=0.005)
                        Silence threshold: 60s
                        ┌──────────┴──────────┐
                   CORTEX11              CORTEX12
                (porta 50051)         (porta 50052)
                "Upstream"            "Downstream"
                Adaptation: fine_tuning  Adaptation: fine_tuning
                Silence: 45s             Silence: 45s
               ┌────┬────┐             ┌────┐
            AG01  AG02  AG03         AG03  AG04
                          └─────────────┘
                          (AG03 reporta para ambos)
```

#### CORTEX11 — Sectoral Upstream (porta 50051)

**Recebe de:** AGENT01, AGENT02, AGENT03
**Reporta para:** CORTEX21

**Objetivo:** Detectar cascatas entre enricher, batcher e validator. Se o APP02 começa a ter erros de cálculo (AG01 detecta), espera-se que AG02 e AG03 logo também vejam anomalias — o cortex11 identifica AG01 como root cause via causal chain temporal.

**Adaptation fine_tuning:**
- Retreina a cada 100 amostras
- Epochs: 8, LR: 0.0001
- Replay buffer: 500 amostras, ratio 0.3
- Garante que o autoencoder sistêmico se adapte a mudanças graduais do pipeline sem esquecer padrões históricos (replay evita catastrophic forgetting)

---

#### CORTEX12 — Sectoral Downstream (porta 50052)

**Recebe de:** AGENT03, AGENT04
**Reporta para:** CORTEX21

**Objetivo:** Detectar anomalias na parte final do pipeline (validação e e2e). Liga a perspectiva do validator (AG03) com a perspectiva end-to-end (AG04). Se AG04 detecta anomalia mas AG03 não, indica que o problema está no path do batcher, não do validator.

**Adaptation fine_tuning:** mesma configuração que cortex11.

---

#### CORTEX21 — Top-Level Coordinator (porta 50053)

**Recebe de:** AGENT04, CORTEX11, CORTEX12 (cortex como super-agents)
**Não reporta para ninguém — é o topo da hierarquia**
**Publica alertas em:** fila SQS `sentinel-alarms`

**Objetivo:** Visão sistêmica completa do pipeline. Correlaciona anomalias setoriais (cortex11/12) com anomalias end-to-end (ag04). Detecta quando falhas coordenadas afetam todo o sistema simultaneamente.

**Adaptation EMA (alpha=0.005):**
- Atualização muito suave a cada evento: `baseline = 0.005 * new_value + 0.995 * baseline`
- Prioriza estabilidade sobre sensibilidade — o cortex de topo não deve ser volátil
- Atualiza a cada 15 amostras (vs 5s dos cortex setoriais)

---

### 3.7 Tabela Completa de Configuração

| Agent | Trecho Observado | Modo | TTL | Detector | Algoritmo | Threshold | Auto-infer FP |
|-------|-----------------|------|-----|----------|-----------|-----------|---------------|
| AG01 | SNS01→SNS02 (APP02) | 1:1 | 90s | if_baseline | IsolationForest | -0.1 | 5% |
| AG01 | SNS01→SNS02 (APP02) | 1:1 | 90s | cvae_enrichment | cVAE | 0.5 | 8% |
| AG02 | SNS02→SNS03 (APP03) | N:1 | 1800s | if_batch_stats | IsolationForest | -0.1 | 8% |
| AG02 | SNS02→SNS03 (APP03) | N:1 | 1800s | maf_joint | MAF | 2.0 | 10% |
| AG03 | SNS02→SNS04 (APP04) | 1:1 | 90s | if_latency | IsolationForest | -0.1 | 5% |
| AG03 | SNS02→SNS04 (APP04) | 1:1 | 90s | cvae_validation | cVAE | **0.1** | 10% |
| AG04 | SNS01→SNS04 (E2E) | 1:1 | 300s | if_e2e | IsolationForest | -0.1 | 8% |
| AG04 | SNS01→SNS04 (E2E) | 1:1 | 300s | maf_e2e | MAF | 2.0 | 12% |

| Cortex | Porta | Recebe de | Adaptation | Silence |
|--------|-------|-----------|------------|---------|
| cortex11 | 50051 | AG01, AG02, AG03 | fine_tuning | 45s |
| cortex12 | 50052 | AG03, AG04 | fine_tuning | 45s |
| cortex21 | 50053 | AG04, cortex11, cortex12 | EMA α=0.005 | 60s |

---

### 3.8 Fluxo Completo de um Evento Anômalo (Exemplo: APP02 retorna soma errada)

```
1. APP01 publica: { correlationId: "abc", value01: 10, value02: 5, someString: "hello" }
   → SNS01 → SQS01 (APP02) + sentinel-a01-in (AG01) + sentinel-a04-in (AG04)

2. APP02 processa e retorna: { correlationId: "abc", result: 57, stringSize: 5, ... }
   (resultado correto seria 15 — offset de 42 injetado)
   → SNS02 → sentinel-a01-out (AG01) + sentinel-a02-in (AG02) + sentinel-a03-in (AG03)

3. AGENT01 correlaciona "abc":
   - if_baseline: latência normal, schema normal → score OK
   - cvae_enrichment: MSE(decode(μ, [10, 5, 5]), [57, 5]) >> 0.5 → ANOMALY ✓
   - ProcessedEvent(is_anomaly=True, score=X) → gRPC → CORTEX11

4. CORTEX11 recebe evento do AG01:
   - Causal chain: AG01 é único agente anômalo na janela de 30s → root_agent=AG01
   - Autoencoder: error_rate do AG01 sobe → reconstruction_error aumenta
   - Se error_rate alto o suficiente → is_systemic=True
   - Alert fusion: WARNING ou CRITICAL → reporta para CORTEX21

5. CORTEX21 recebe sinal do CORTEX11:
   - Correlaciona com AG04 (e2e) — se AG04 também está vendo anomalias → CRITICAL
   - Publica alerta em sentinel-alarms: { severity, root_agent: "ag01", affected: [...], ... }

6. Paralelamente, AGENT04 recebe output final via SNS04:
   - maf_e2e: distribuição conjunta do output final anômala (result não condiz com value01+value02)
   - Reporta para CORTEX12 e CORTEX21 diretamente
```

---

### 3.9 Configurações Globais

```json
{
  "schema_version": 1,
  "storage_path": "~/.sentinel/usecase",
  "redis": { "host": "localhost", "port": 6379 },
  "aws": { "region": "us-east-1", "endpoint": "http://localhost:4566" },
  "dashboard": { "host": "0.0.0.0", "port": 8888 },
  "alarm_queue": { "type": "sqs", "url": "http://localhost:4566/000000000000/sentinel-alarms" },
  "model": {
    "max_versions": 10,
    "auto_rollback": true,
    "isolation_forest": {
      "n_estimators": 100,
      "contamination": "auto",
      "anomaly_threshold": -0.1,
      "training_window": 1000
    },
    "autoencoder": {
      "hidden_dim": 32,
      "training_window": 2000,
      "learning_rate": 0.001,
      "epochs": 30
    }
  }
}
```

---

### 3.10 Dashboard e Reset

**Sentinel Dashboard:** `http://localhost:8888`
- Cards de status por agent (score, anomaly rate, heartbeat, phase)
- Switching manual TRAINING ↔ INFERENCE por agent/detector
- Feed de alertas do cortex (CAUSAL, SYSTEMIC, SILENCE, TIMEOUT)
- Sparklines de score em tempo real

**Apps Dashboard:** `http://localhost:3000`
- Controle de `intervalMs`, `errorRate`, `delayMs` por app
- Log de eventos recentes de cada app
- Purge de filas

**Reset (`shared/reset.js`):**
- Purga filas SQS01-05 (pipeline de aplicação)
- Purga filas sentinel-a0{1-4}-in/out (observação)
- Flush do Redis (correlações pendentes)
- Usado quando dados de treino estão contaminados por timeouts excessivos

---

*Gerado em 2026-04-20. Baseado na análise completa do código-fonte em `c:\projects\Septimus\sentinel`.*
