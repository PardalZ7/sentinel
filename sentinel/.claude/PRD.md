# Sentinel — Product Requirements Document (PRD)

## Visão Geral

Sentinel é uma biblioteca Python pip-instalável que implementa uma rede de detecção de anomalias para arquiteturas orientadas a eventos. É **totalmente agnóstica ao use case** — funciona como plug-and-play via arquivo de configuração, sem necessidade de alterar o código da aplicação monitorada.

---

## Problema

Sistemas de mensageria distribuída (SNS/SQS, Kafka, RabbitMQ) são difíceis de monitorar:
- Anomalias de dados (campo ausente, valor incorreto, schema drift) não geram erros explícitos
- Gargalos temporais (serviço parou de responder) são detectados tarde
- Correlação entre causa raiz e efeito em cascata é manual e lenta
- Regras de validação explícitas não escalam com a complexidade do sistema

---

## Solução

Sentinel aprende o comportamento normal do sistema e detecta desvios automaticamente, sem regras programadas.

### Duas camadas:

#### SentinelAgent
- Monitora **uma aplicação/fluxo** (input + output)
- Correlaciona mensagens de entrada e saída via `correlationId`
- Executa **múltiplos detectores em paralelo** (Strategy Pattern), cada um com algoritmo, fase, ciclo de champion/challenger e persistência independentes
- Em inferência, emite `AnomalyReport` quando **qualquer** detector sinaliza anomalia
- Detecta timeouts (input sem output após TTL)
- Publica eventos e heartbeats para o Cortex via gRPC

#### SentinelCortex
- Gerencia **múltiplos Agents e/ou outros Cortex** (hierárquico)
- Detecta gargalos temporais (silêncio de um agente)
- Executa **Dual Network** em paralelo:
  - **Causal Chain**: correlaciona anomalias no tempo → root cause
  - **Autoencoder (PyTorch)**: detecta padrões sistêmicos invisíveis individualmente
- Emite alertas para uma fila de alarmes configurável
- Pode reportar para um Cortex pai (age como super-agente)

---

## Princípios

- **Agnóstico ao domínio**: zero conhecimento sobre o negócio da aplicação monitorada
- **Plug-and-play**: configuração via `sentinel.json` / `sentinel.yaml`, sem código
- **Strategy Pattern**: cada algoritmo de detecção implementa `IDetector`; o agent é o contexto
- **Hexagonal Architecture**: domínio isolado, adapters plugáveis
- **Transport plugável**: SQS/SNS hoje, Kafka/RabbitMQ amanhã
- **Modular**: sem classes gigantes, funções focadas, design patterns explícitos
- **Assíncrono**: asyncio em toda a stack

---

## Fases de Operação

| Fase | Comportamento |
|------|---------------|
| `TRAINING` | Acumula mensagens, treina modelo, `send_all_events=true` automático |
| `INFERENCE` | Usa modelo treinado para scoring, publica só anomalias (configurável) |

A fase pode ser controlada **por detector individualmente** ou para todos os detectores do agent de uma vez. A transição é sempre controlada **explicitamente pelo usuário** via dashboard, API ou CLI.

---

## Detectores Disponíveis

### IsolationForest (`isolation_forest`)
Detecta anomalias **operacionais**: desvios em latência de processamento, tamanhos de payload e schema hash. Converge rápido (~500 amostras) e não requer configuração de campos. É o baseline recomendado para todo agent.

### cVAE — Conditional Variational Autoencoder (`cvae`)
Detecta anomalias **semânticas de transformação**: aprende a distribuição condicional P(output_fields | input_fields). Sinaliza quando o output é semanticamente inconsistente com o input — por exemplo, quando um serviço de enriquecimento calcula incorretamente um campo derivado. Requer `field_map` configurado no `sentinel.json`.

### MAF — Masked Autoregressive Flow (`maf`)
Detecta anomalias na **distribuição conjunta** de todos os campos numéricos: aprende P(x₁, x₂, ..., xₙ) via MADE (Masked Autoencoder for Distribution Estimation). Sinaliza quando a combinação multivariada de métricas é anômala, mesmo que cada métrica individualmente pareça normal. `field_map` é opcional — sem ele, descobre automaticamente todos os campos numéricos.

### NRI — Neural Relational Inference (`nri`)
Detecta **anomalias em grafos de dependência** entre campos: aprende quais campos normalmente influenciam quais e detecta quando essa estrutura de relacionamentos muda. Interface estável implementada como stub — `fit()` lança `NotImplementedError`. Previsto para implementação futura.

---

## Requisitos Funcionais

### Agent
- [x] Consumir mensagens de input via transport plugável
- [x] Consumir mensagens de output via transport plugável
- [x] Correlacionar input+output por `correlation_field` (dot-notation) via Redis
- [x] Suportar três modos de correlação: `normal` (1→1), `grouping` (N→1), `splitting` (1→N)
- [x] TTL configurável por correlação (padrão: 300s) — expiração = anomalia TIMEOUT
- [x] Suportar múltiplos detectores por agent (Strategy Pattern via `IDetector`)
- [x] Cada detector tem fase, buffer de treinamento, test buffer e champion independentes
- [x] Treinar cada detector com janela configurável de mensagens
- [x] Champion/challenger por detector — novo modelo só promovido se FP rate melhorar
- [x] Calcular anomaly score por detector; `is_anomaly = any(det.is_anomaly(score))`
- [x] Score reportado = pior score entre detectores em INFERENCE
- [x] Emitir `ProcessedEvent` (normal) e `AnomalyReport` (anomalia) via gRPC
- [x] Emitir `HeartbeatEvent` periodicamente (intervalo configurável)
- [x] Operar em modo standalone se gRPC falhar (sem bloqueio)
- [x] Versionar modelos em disco por detector: `agents/{agent}/{detector}/manifest.json`
- [x] Backward compat: agent sem `detectors` sintetiza IsolationForest a partir do config global

### Detectores
- [x] `IsolationForest`: features operacionais (latência, tamanhos, schema hash)
- [x] `cVAE`: aprende P(output_fields | input_fields); requer `field_map` no sentinel.json
- [x] `MAF`: aprende distribuição conjunta; `field_map` opcional
- [x] `NRI`: stub com interface estável; `fit()` → NotImplementedError; `score()` → 0.0

### Config
- [x] Suporte a `sentinel.json` e `sentinel.yaml`
- [x] `DetectorConfig` com `field_map` para mapeamento de campos de payload
- [x] Migração automática de schema entre versões
- [x] Valores default para todos os campos opcionais
- [x] CLI: `sentinel start --config sentinel.json`

### Dashboard
- [x] SSE snapshot inclui `detectors[]` por agent
- [x] Endpoints por detector: `POST /api/agents/{name}/detectors/{det}/phase`
- [x] Endpoint de agent muda todos os detectores (backward compat)

---

## Requisitos Não-Funcionais

- Python 3.11+
- Assíncrono: asyncio puro
- Logs estruturados (stdout por ora, módulo extensível)
- Testes unitários + integração (pytest + docker fixtures)
- Stubs gRPC commitados no repo (gerados via `scripts/generate_proto.sh`)
- Instalável via `pip install sentinel`

---

## Fora de Escopo (v1)

- Implementação do NRI (stub apenas)
- Autenticação gRPC (sem mTLS)
- Alertas via webhook (v2)
- Métricas Prometheus (v2)
- Frontend do dashboard com sub-linha por detector (v2)
