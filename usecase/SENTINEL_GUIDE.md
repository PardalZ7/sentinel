# Guia de Experimentação — Sentinel + Usecase

Este guia conduz você do zero até experimentos controlados de anomalia, explicando o que observar e como interpretar os resultados.

---

## Arquitetura de Observação

```
APP01 ──► SNS01 ──► SQS01 ──► APP02 ──► SNS02 ──┬──► SQS02 ──► APP03 ──► SNS03 ──► SQS04
                │                                 │
                │                                 └──► SQS03 ──► APP04 ──► SNS04 ──► SQS05
                │
  Sentinel observa via filas paralelas (sem impacto nas apps):
                │
        ┌───────┴────────────────────────────────────────────────────────────────────┐
        │  sentinel-a01-in (←SNS01)   sentinel-a01-out (←SNS02)  → agent01 → cortex11
        │  sentinel-a02-in (←SNS02)   sentinel-a02-out (←SNS03)  → agent02 → cortex11
        │  sentinel-a03-in (←SNS02)   sentinel-a03-out (←SNS04)  → agent03 → cortex11, cortex12
        │  sentinel-a04-in (←SNS01)   sentinel-a04-out (←SNS04)  → agent04 → cortex12, cortex21
        │  sentinel-a05-in (←SNS02)   sentinel-a05-out (←SNS04)  → agent05 → cortex12
        └────────────────────────────────────────────────────────────────────────────┘
                cortex11 → cortex21
                cortex12 → cortex21
```

### O que cada agent monitora

| Agent | Observa | Correlação | Mede |
|-------|---------|------------|------|
| agent01 | APP02 (enricher): SNS01→SNS02 | `correlationId` | latência de enriquecimento, schema drift (value02 omitido, result errado) |
| agent02 | APP03 (batcher): SNS02→SNS03 | `correlationId` (grouping 20:1 — alta taxa de timeout esperada) | volume do batch, sequência de envios |
| agent03 | Segmento SNS02→SNS04 (batcher+validator) | `correlationId` | latência cross-segment; bridge entre cortex11 e cortex12 |
| agent04 | Pipeline completo SNS01→SNS04 | `correlationId` | latência end-to-end, anomalias no fluxo total |
| agent05 | APP04 (validator): SNS02→SNS04 | `correlationId` | latência e schema da validação isolados do batcher |

### Hierarquia dos Cortex

```
cortex11 (porta 50051): recebe agent01, agent02, agent03
cortex12 (porta 50052): recebe agent03, agent04, agent05
    ↓                        ↓
cortex21 (porta 50053): recebe agent04 + cortex11 + cortex12 (como super-agentes)
```

- **cortex11** — sectoral upstream: detecta cascatas entre enricher → batcher → segmento de validação
- **cortex12** — sectoral downstream: isola anomalias do validator e correlaciona com a visão end-to-end
- **agent03** reporta para ambos cortex11 e cortex12, servindo como bridge entre os dois setores
- **cortex21** sintetiza os dois sectoriais mais o sinal end-to-end direto (agent04)

---

## Pré-requisitos

```bash
# Verifique as instalações
docker --version          # Docker Desktop rodando
node --version            # Node.js 18+
python --version          # Python 3.11+
redis-cli ping            # deve retornar PONG (ou instale: ver abaixo)
```

### Instalar Redis localmente (se necessário)

**Windows (via scoop ou choco):**
```bash
choco install redis
redis-server
```

**WSL2:**
```bash
sudo apt install redis-server
sudo service redis-server start
```

---

## Passo 1: Subir a Infraestrutura

### Terminal 1 — LocalStack

```bash
cd /mnt/c/projects/Septimus/sentinel/usecase
docker compose up
```

Aguarde a mensagem de inicialização (`LocalStack startup complete`). O `setup.sh` é executado automaticamente via `init/ready.d` e criará **todas** as filas — incluindo as 8 filas do Sentinel.

**Verificar que as filas foram criadas:**
```bash
aws --endpoint-url=http://localhost:4566 sqs list-queues --output text | grep sentinel
```

Esperado: 10 linhas começando com `sentinel-a0{1,2,3,4,5}-{in,out}`.

---

## Passo 2: Instalar Dependências do Sentinel

> **WSL2 / Debian / Ubuntu:** o Python do sistema bloqueia `pip install` direto. Use um virtualenv:

```bash
cd /mnt/c/projects/Septimus/sentinel/sentinel
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

> **Importante:** o virtualenv precisa estar ativo em qualquer terminal que rode comandos `sentinel` ou `python -m sentinel.*`. Ative com:
> ```bash
> source /mnt/c/projects/Septimus/sentinel/sentinel/.venv/bin/activate
> ```

---

## Passo 3: Subir as Apps do Usecase

### Terminal 2 — Apps (APP01–04 + Dashboard)

```bash
cd /mnt/c/projects/Septimus/sentinel/usecase
bash start.sh
```

**Dashboard do usecase:** http://localhost:3000  
Aqui você controla os sliders de `intervalMs` e `errorRate` das apps.

---

## Passo 4: Iniciar o Sentinel

### Terminal 3 — Sentinel

```bash
source /mnt/c/projects/Septimus/sentinel/sentinel/.venv/bin/activate
cd /mnt/c/projects/Septimus/sentinel/usecase
python3 -m sentinel.runtime.cli start --config sentinel.json
```

Ou, se o `sentinel` CLI estiver no PATH (venv ativo):
```bash
sentinel start --config sentinel.json
```

**Dashboard do Sentinel:** http://localhost:8888  
Aqui você observa os agents e cortex em tempo real.

---

## Passo 5: Fase de Treinamento

Todos os agents iniciam em modo `TRAINING`. Nesta fase:
- O Isolation Forest **acumula amostras** (janela padrão: 200 mensagens)
- O Agent envia **todos** os eventos para o Cortex (para treinar o Autoencoder)
- Os eventos aparecem no dashboard com badge TRAINING

**O que observar no dashboard durante o treinamento:**

1. **Contador de mensagens** de cada agent sobe gradualmente
2. **Score de anomalia = 0.0** (modelo não treinado ainda)
3. **Heartbeats** aparecem no feed a cada 15s
4. **agent02** terá alta taxa de eventos TIMEOUT — isso é **normal** (batcher acumula 20 msgs antes de publicar, então 19 de cada 20 correlações expiram; o Isolation Forest aprende esse padrão como normal)

**Duração recomendada do treinamento:** deixe acumular pelo menos 200 mensagens em cada agent. Com `intervalMs=1000` no APP01, isso leva ~3-4 minutos (o batcher leva mais porque processa em lotes de 20).

> **Dica:** aumentar a velocidade do APP01 para `intervalMs=200` via dashboard do usecase acelera o treinamento.

---

## Passo 6: Transicionar para Inferência

Quando os agents tiverem mensagens suficientes, mude para INFERENCE via dashboard do Sentinel (http://localhost:8888) — botão "Switch to Inference" em cada card de agent — ou via CLI:

```bash
# (venv ativo)
sentinel set-phase agent01 INFERENCE
sentinel set-phase agent02 INFERENCE
sentinel set-phase agent03 INFERENCE
sentinel set-phase agent04 INFERENCE
sentinel set-phase agent05 INFERENCE
```

Em INFERENCE:
- O Isolation Forest **classifica** cada evento
- Scores abaixo de `-0.1` (configurável) são marcados como ANOMALIA
- Apenas anomalias e heartbeats são enviados ao Cortex (por padrão)

---

## Experimentos de Anomalia

### Experimento 1 — Anomalia de Schema (APP01)

**O que testar:** APP01 omite o campo `value02` (schema incompleto).

**Como induzir:**
1. Acesse http://localhost:3000 → tab APP01
2. Mova o slider `errorRate` para `0.5` (50% das mensagens ficarão sem `value02`)

**O que observar no Sentinel (http://localhost:8888):**
- **agent01**: score de anomalia cai abaixo de `-0.1`, badge ANOMALY aparece no feed. O schema hash muda (campos diferentes = hash diferente = feature diverge do treinamento).
- **agent04**: detecta o mesmo sinal na visão end-to-end.
- **cortex11**: pode emitir alerta CAUSAL se agent01 e outros mostrarem anomalias correlacionadas.

**Validação positiva:** agent01 detecta anomalia consistentemente quando `errorRate > 0`.  
**Validação negativa:** com `errorRate = 0`, score deve permanecer próximo de 0 ou positivo.

---

### Experimento 2 — Cálculo Errado (APP02)

**O que testar:** APP02 adiciona offset aleatório em `result` ou `stringSize`.

**Como induzir:**
1. Tab APP02, slider `errorRate` para `0.3`

**O que observar:**
- **agent01**: o campo `result` ou `stringSize` diverge do padrão. O agent não sabe a regra de negócio, mas o padrão estatístico (distribuição de latência, tamanho de output) muda.
- Score deve cair — o output enriquecido "parece diferente" do padrão treinado.

> **Nota:** Este é o teste mais interessante. O Sentinel não conhece a regra `result = value01 + value02`. Ele apenas aprendeu que, normalmente, as mensagens têm certo tamanho e latência. Um erro de cálculo muda o tamanho do JSON output e a correlação de features — e isso é detectável.

---

### Experimento 3 — Batch Incompleto (APP03)

**O que testar:** APP03 publica batch com menos de 20 strings.

**Como induzir:**
1. Tab APP03, slider `errorRate` para `0.5`

**O que observar:**
- **agent02**: o batch output tem tamanho menor → `output_size_bytes` diverge do padrão → score cai.
- O agente aprendeu que o batch normal tem ~20 strings. Batches menores são anomalia.

---

### Experimento 4 — Silêncio Total (APP02 parado)

**O que testar:** APP02 para de processar (simula travamento).

**Como induzir:**
```bash
# Pare o processo APP02 (Terminal 2 mostra os PIDs)
# Ou aumente o delay para 10000ms via API:
curl -X POST http://localhost:3002/config -H "Content-Type: application/json" -d '{"delayMs": 10000}'
```

**O que observar no Sentinel:**
- **agent01**: sem outputs do APP02 → correlações em agent01 começam a expirar com TIMEOUT
- **cortex11**: detecta `SILENCE` em agent01 após `silence_threshold_s = 45s` sem heartbeat
- **Alert** de tipo `SILENCE` aparece no feed e na aba Alerts do dashboard

**Validação:** restaurar APP02 (`delayMs: 0`) → agente volta a processar, silêncio desaparece.

---

### Experimento 5 — Cascata de Anomalias

**O que testar:** erro em APP02 causa anomalia em cascade nos agents downstream.

**Como induzir:**
1. Coloque APP01 em `errorRate: 0.8`
2. Coloque APP02 em `errorRate: 0.5` simultaneamente

**O que observar:**
- Múltiplos agents (agent01, agent03, agent04) devem mostrar anomalias ao mesmo tempo
- **cortex11** → Causal Chain: detecta que agent01 foi o primeiro a anomaliar → root cause
- **cortex21** → Autoencoder: vetor de estado sistêmico diverge → alerta SYSTEMIC
- Timeline dos alertas no dashboard deve mostrar agent01 antes de agent03/agent04

---

### Experimento 6 — Anomalia Discreta (valid invertido, APP04)

**O que testar:** APP04 inverte o campo `valid` (`stringSize % 2 !== 0` → true ao invés de false).

**Como induzir:**
1. Tab APP04, `errorRate: 0.5`

**O que observar:**
- **agent05**: monitor dedicado do APP04 (SNS02→SNS04). Mudança sutil — o payload tem os mesmos campos, mas a distribuição de `valid` muda. O schema hash é igual (mesmos campos), mas o tamanho do JSON pode diferir marginalmente. Este teste avalia o **limite de sensibilidade** do Sentinel.
- **agent03**: também cobre o segmento SNS02→SNS04 (cross-segment), mas dilui o sinal do validator com o ruído do batcher — agent05 é o canal mais limpo para este experimento.
- Se o Sentinel **não detectar**: isso é esperado e válido — o Sentinel aprendeu padrões estatísticos, não lógica de negócio. Para detectar inversão de `valid`, seria necessário incluir o valor do campo como feature, o que requereria um adapter específico.

> Este experimento documenta uma **limitação real** do Isolation Forest: anomalias semânticas invisíveis nas features numéricas não são detectadas. O Autoencoder do Cortex pode capturá-las se o vetor de estado incluir frequências de campos.

---

## Interpretando o Dashboard

### Cards de Agent

| Indicador | Normal | Suspeito | Anomalia |
|-----------|--------|----------|---------|
| Score | > 0 | -0.05 a -0.1 | < -0.1 |
| Anomaly rate | < 2% | 2-10% | > 10% |
| Heartbeat | Verde, < 20s | Amarelo, 20-45s | Vermelho, SILENT |
| Score sparkline | Oscila em torno de 0 | Descendo | Abaixo da linha vermelha |

### Aba Alerts (Cortex)

| Tipo | Significado |
|------|-------------|
| `CAUSAL` | Cascata de anomalias detectada — primeiro agente é o provável root cause |
| `SYSTEMIC` | Padrão global do pipeline diverge — pode não ter um único culpado |
| `SILENCE` | Um agente parou de publicar heartbeats |
| `TIMEOUT` | Correlação input/output expirou — output nunca chegou |

---

## Resetar o Ambiente

```bash
# Limpar modelos treinados (força retreinamento):
rm -rf ~/.sentinel/usecase/

# Reiniciar Sentinel (retorna para TRAINING):
# Ctrl+C no terminal do Sentinel, depois:
sentinel start --config sentinel.json

# Reiniciar todas as apps:
# Ctrl+C no start.sh, depois:
bash start.sh

# Zerar o LocalStack (recria filas do zero):
docker compose down -v && docker compose up
```

---

## Troubleshooting

| Problema | Causa provável | Solução |
|----------|----------------|---------|
| `externally-managed-environment` no pip | Python do sistema (Debian/Ubuntu) | Criar e ativar venv: ver Passo 2 |
| Sentinel não conecta ao Redis | Redis não está rodando | `redis-server` |
| "Queue not found" | setup.sh não rodou | `bash infra/init/setup.sh` |
| Score sempre 0.0 | Agent ainda em TRAINING ou janela não atingida | Aguardar 200 msgs ou mudar para INFERENCE |
| Dashboard vazio | SSE não conectado | Verificar console do browser (F12) |
| agent02 com 99% anomaly rate | Normal em TRAINING (timeouts) | Em INFERENCE, o IF aprende que timeout é normal |
| gRPC error | Cortex não subiu | Sentinel sobe cortex e agents juntos — verificar logs |

---

## Variáveis de Ambiente Úteis

```bash
# Log mais legível (ao invés de JSON):
export SENTINEL_LOG_FORMAT=pretty

# Mudar porta do dashboard:
export SENTINEL_DASHBOARD_PORT=9000

# Forçar endpoint LocalStack:
export SENTINEL_AWS_ENDPOINT_URL=http://localhost:4566
```
