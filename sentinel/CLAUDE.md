# CLAUDE.md — Sentinel

## O que é este projeto

Sentinel é uma biblioteca Python para detecção de anomalias em arquiteturas orientadas a eventos. É agnóstica ao domínio — configurada via `sentinel.json` sem alterar o código da aplicação monitorada.

Documentação completa: `.claude/PRD.md` (requisitos) e `.claude/SPEC.md` (especificação técnica).

---

## Estrutura do Projeto

```
sentinel/                          # pacote Python
├── domain/                        # núcleo — zero dependências externas
│   ├── models.py                  # RawMessage, ProcessedEvent, Alert, etc.
│   ├── errors.py
│   └── ports/                     # interfaces ABCs
│       ├── detector.py            # IDetector + DetectorState (Strategy Pattern)
│       └── ...                    # ITransport, ICorrelationStore, IModelStore, etc.
├── agent/                         # lógica do SentinelAgent
│   ├── use_cases/                 # process_message, train, champion, run_test, set_phase
│   ├── detectors/                 # implementações concretas de IDetector
│   │   ├── isolation_forest_detector.py
│   │   ├── cvae_detector.py
│   │   ├── maf_detector.py
│   │   ├── nri_detector.py        # stub
│   │   └── factory.py
│   ├── isolation_forest.py        # funções sklearn de baixo nível (legado, mantido)
│   └── correlator.py
├── cortex/                        # lógica do SentinelCortex
│   ├── use_cases/                 # aggregate, causal_chain, autoencoder
│   └── dual_network.py
├── adapters/                      # implementações concretas das portas
│   ├── transport/                 # sqs_sns.py, kafka.py (stub)
│   ├── store/                     # redis_store.py
│   ├── model_store/               # disk_store.py (paths por detector)
│   ├── grpc/                      # proto, generated (commitados), server, client
│   └── alerting/                  # queue_sink.py
├── config/                        # schema Pydantic v2, loader, migrations
├── logging/                       # logger.py (structlog, stdout)
└── runtime/                       # launcher.py, cli.py (Click)
```

---

## Convenções

### Arquitetura
- **Hexagonal**: domínio não importa adapters; adapters implementam ports
- Sem classes gigantes — cada módulo tem responsabilidade única
- Use cases são funções assíncronas puras, não classes

### Nomenclatura
- Ports (interfaces): prefixo `I` — `ITransport`, `ICorrelationStore`
- Adapters: nome do recurso — `SqsSnsTransport`, `RedisCorrelationStore`
- Use cases: verbo no infinitivo — `process_message()`, `train_model()`, `set_phase()`
- Models: substantivos — `RawMessage`, `ProcessedEvent`, `Alert`

### Python
- Python 3.11+ (`match`, `tomllib`, typing melhorado)
- asyncio em toda a stack
- Pydantic v2 para todos os modelos de configuração e domínio
- `structlog` para logs — nunca `print()` direto
- Type hints obrigatórios em toda função pública

### Redis
- Chave de correlação: `{agent_name}:{correlation_id}`
- TTL padrão: 300s (configurável por agent)

### gRPC
- Stubs em `adapters/grpc/generated/` são commitados — não editar manualmente
- Para regenerar: `bash scripts/generate_proto.sh`

### Modelos em disco
- Path novo (por detector): `{storage_path}/agents/{agent_name}/{detector_name}/manifest.json`
- Path legado (sem detector): `{storage_path}/agents/{agent_name}/manifest.json` — preservado para backward compat
- Cortex: `{storage_path}/cortex/{cortex_name}/manifest.json`
- Manifest em `manifest.json` — lista versões e aponta para a atual

---

## Comandos Úteis

```bash
# Instalar dependências de desenvolvimento
pip install -e ".[dev]"

# Regenerar stubs gRPC
bash scripts/generate_proto.sh

# Rodar testes unitários
pytest tests/unit

# Rodar testes de integração (requer Docker)
pytest tests/integration

# CLI
sentinel start --config sentinel.json
sentinel set-phase agent01 INFERENCE
sentinel status
```

---

## Decisões de Design Importantes

1. **Transport plugável desde o início** — `ITransport` abstrai SQS/SNS/Kafka. Adapters vivem em `adapters/transport/`.
2. **Agent opera standalone** se gRPC falhar — nunca bloqueia o processamento principal.
3. **Agent pausa** (não descarta) se Redis cair — correlações não são perdidas.
4. **Fase explícita** — transição TRAINING↔INFERENCE é sempre controlada pelo usuário.
5. **Em TRAINING**, `send_all_events` é forçado `true` — Cortex precisa dos dados normais.
6. **Cortex Dual Network** — Causal Chain + Autoencoder rodam em paralelo, resultado fundido pelo aggregator.
7. **Cortex é simétrico** — expõe gRPC server (recebe de filhos) e gRPC client (reporta para pai).
8. **Alarmes** vão para uma fila dedicada (`alarm_queue`) configurada na raiz do Sentinel.
9. **Strategy Pattern para detectores** — `IDetector` é o contrato; cada algoritmo é uma estratégia concreta; `AgentRunner` é o contexto. Adicionar um novo algoritmo = criar um arquivo em `agent/detectors/` + registrar na factory.
10. **Detectores são stateless** — todo estado de runtime fica em `DetectorState` (dataclass). `IDetector` contém apenas a lógica do algoritmo.
11. **field_map é configuração do usecase** — nomes de campos do payload são específicos do sistema monitorado; nunca são hardcoded no Sentinel.
12. **Backward compat via síntese** — agent sem `detectors` no config recebe um detector IsolationForest sintetizado automaticamente pelo launcher a partir de `model.isolation_forest`.
13. **Anomalia = any()** — um evento é anômalo se qualquer detector em INFERENCE o sinalizar. Score reportado = pior score entre os que sinalizaram.

---

## Fluxo de Treinamento Completo

### Agents (multi-detector)

1. Cada detector inicia em `TRAINING` (ou na fase definida no `sentinel.json`). O buffer acumula apenas correlações **bem-sucedidas** (timeouts não contam).
2. Ao atingir `training_window` amostras, o detector é treinado. Se o novo modelo (challenger) tem FP rate menor que o champion atual no test buffer, o champion é substituído e o modelo salvo em `{storage_path}/agents/{name}/{detector_name}/manifest.json`.
3. A **transição para `INFERENCE` é manual** — feita via dashboard, API ou CLI. Pode ser feita por detector individualmente ou para todos de uma vez.
4. Em restart, cada detector carrega seu modelo salvo do disco e retoma em `INFERENCE` automaticamente.
5. O `event_dict` passado a todos os detectores inclui `_input_body` e `_output_body` — detectores operacionais (IF) os ignoram; detectores semânticos (cVAE, MAF) os usam para extrair campos do payload.

> **Atenção:** `total_messages` no dashboard conta apenas correlações bem-sucedidas. Timeouts não são adicionados ao buffer de treinamento de nenhum detector.

> **Atenção:** `total_messages` no dashboard conta apenas correlações bem-sucedidas. Mensagens que geraram TIMEOUT não são adicionadas ao buffer de treinamento.

### Cortex (Autoencoder PyTorch)

1. O Cortex só começa a acumular dados quando **todos** os agents declarados em `inputs` estiverem em `INFERENCE`.
2. Após `training_window` amostras (default: 1000, configurável em `model.autoencoder.training_window`), o autoencoder é treinado e salvo em `{storage_path}/cortex/{name}/`.
3. O Cortex transiciona para `INFERENCE` automaticamente após o treinamento.
4. Em restart, o modelo `.pt` é carregado do disco e o Cortex retoma em `INFERENCE`.

### Sequência correta para um export com modelos treinados

```
1. Produza ≥ training_window correlações bem-sucedidas por agent
2. No dashboard: mude cada agent para INFERENCE
3. Aguarde o Cortex acumular amostras e treinar automaticamente
4. Dashboard → Export → state.json terá versões e baseline_error preenchidos
```

