# Sentinel — Use Case

Ambiente de simulação de uma pipeline de mensageria distribuída baseada em **AWS SNS/SQS**, rodando localmente via **LocalStack**. Serve como caso de uso para treinamento e validação do projeto **Sentinel** — uma rede neural de detecção de anomalias em arquiteturas orientadas a eventos.

---

## Visão Geral da Arquitetura

```
APP01 ──► SNS01 ──► SQS01 ──► APP02 ──► SNS02 ──┬──► SQS02 ──► APP03 ──► SNS03 ──► SQS04
                                                │
                                                └──► SQS03 ──► APP04 ──► SNS04 ──► SQS05
```

A pipeline possui dois caminhos independentes a partir do **SNS02**:

- **Caminho superior (agregação):** APP03 acumula 20 mensagens e publica um batch com todas as strings.
- **Caminho inferior (validação):** APP04 classifica cada mensagem individualmente com base no tamanho da string.

---

## Componentes

### Infraestrutura

| Recurso | Tipo | Papel |
|---------|------|-------|
| LocalStack | Docker | Emula os serviços AWS (SNS + SQS) localmente |
| SNS01 | Tópico | Recebe mensagens brutas do APP01 |
| SNS02 | Tópico | Distribui mensagens enriquecidas para dois caminhos |
| SNS03 | Tópico | Recebe batches do APP03 |
| SNS04 | Tópico | Recebe mensagens validadas do APP04 |
| SQS01 | Fila | Buffer entre SNS01 e APP02 |
| SQS02 | Fila | Subscriber do SNS02 → entrada do APP03 |
| SQS03 | Fila | Subscriber do SNS02 → entrada do APP04 |
| SQS04 | Fila | Subscriber do SNS03 → saída do APP03 |
| SQS05 | Fila | Subscriber do SNS04 → saída do APP04 |

---

### APP01 — Producer

**Porta de controle:** `3001`

Produz mensagens continuamente para o **SNS01** em intervalos configuráveis.

**Payload publicado (normal):**
```json
{
  "correlationId": "550e8400-e29b-41d4-a716-446655440000",
  "someString": "aBcXy12z...",
  "value01": 342,
  "value02": 187
}
```

**Campos:**
| Campo | Tipo | Descrição |
|-------|------|-----------|
| `correlationId` | UUID v4 | Identificador único gerado pelo APP01, propagado por toda a pipeline |
| `someString` | string | String aleatória com 10 a 50 caracteres |
| `value01` | inteiro | Número inteiro positivo (1–1000) |
| `value02` | inteiro | Número inteiro positivo (1–1000) |

**Configuração via API:**
```http
GET  /config          → retorna { intervalMs, errorRate }
POST /config          → body: { intervalMs?, errorRate? }
```

| Parâmetro | Padrão | Descrição |
|-----------|--------|-----------|
| `intervalMs` | 1000 | Intervalo entre publicações em milissegundos |
| `errorRate` | 0 | Taxa de erro (0.0 a 1.0) |

**Simulação de erro:**
- Omite o campo `value02` da mensagem (schema incompleto)

---

### APP02 — Enricher

**Porta de controle:** `3002`

Consome de **SQS01**, enriquece o payload e publica no **SNS02**.

**Payload publicado (normal):**
```json
{
  "correlationId": "550e8400-e29b-41d4-a716-446655440000",
  "someString": "aBcXy12z...",
  "value01": 342,
  "value02": 187,
  "stringSize": 10,
  "result": 529
}
```

**Campos adicionados:**
| Campo | Tipo | Descrição |
|-------|------|-----------|
| `stringSize` | inteiro | Comprimento exato de `someString` |
| `result` | inteiro | Soma de `value01 + value02` |

> `correlationId` é propagado sem alteração.

**Configuração via API:**
```http
GET  /config          → retorna { delayMs, errorRate }
POST /config          → body: { delayMs?, errorRate? }
```

| Parâmetro | Padrão | Descrição |
|-----------|--------|-----------|
| `delayMs` | 0 | Delay simulado de processamento em ms |
| `errorRate` | 0 | Taxa de erro (0.0 a 1.0) |

**Simulação de erro (escolha aleatória 50/50):**
| Tipo | Comportamento |
|------|---------------|
| Tipo 1 | `result` calculado com um offset aleatório (1–100) adicionado |
| Tipo 2 | `stringSize` calculado com um offset aleatório (1–100) adicionado |

---

### APP03 — Batcher

**Porta de controle:** `3003`

Consome de **SQS02**, acumula 20 mensagens em buffer e publica um único batch no **SNS03**.

**Payload publicado (normal):**
```json
{
  "records": [
    { "someString": "aBcXy12z...", "correlationId": "550e8400-e29b-41d4-a716-446655440000" },
    { "someString": "Zp9mNq...",   "correlationId": "661f9511-f30c-52e5-b827-557766551111" }
  ],
  "fileName": "batch_1712001234567.txt"
}
```

**Campos:**
| Campo | Tipo | Descrição |
|-------|------|-----------|
| `records` | array de objetos | Lista de 20 entradas, cada uma com `someString` e `correlationId` da mensagem original |
| `fileName` | string | Nome fictício do arquivo gerado para o batch |

**Configuração via API:**
```http
GET  /config          → retorna { delayMs, errorRate }
POST /config          → body: { delayMs?, errorRate? }
```

| Parâmetro | Padrão | Descrição |
|-----------|--------|-----------|
| `delayMs` | 0 | Delay aplicado antes de publicar o batch |
| `errorRate` | 0 | Taxa de erro por batch (0.0 a 1.0) |

**Simulação de erro (escolha aleatória 50/50):**
| Tipo | Comportamento |
|------|---------------|
| Tipo 1 | Publica batch com N registros (1–19), simulando perda de mensagens |
| Tipo 2 | Inclui um registro duplicado no array (20 itens, um `{ someString, correlationId }` repetido) |

---

### APP04 — Validator

**Porta de controle:** `3004`

Consome de **SQS03**, adiciona o campo `valid` ao payload e publica no **SNS04**.

**Payload publicado (normal):**
```json
{
  "correlationId": "550e8400-e29b-41d4-a716-446655440000",
  "someString": "aBcXy12z...",
  "value01": 342,
  "value02": 187,
  "stringSize": 10,
  "result": 529,
  "valid": true
}
```

**Campo adicionado:**
| Campo | Tipo | Descrição |
|-------|------|-----------|
| `valid` | boolean | `true` se `stringSize` for par, `false` se ímpar |

> `correlationId` é propagado sem alteração.

**Configuração via API:**
```http
GET  /config          → retorna { delayMs, errorRate }
POST /config          → body: { delayMs?, errorRate? }
```

| Parâmetro | Padrão | Descrição |
|-----------|--------|-----------|
| `delayMs` | 0 | Delay simulado de processamento em ms |
| `errorRate` | 0 | Taxa de erro (0.0 a 1.0) |

**Simulação de erro (escolha aleatória 50/50):**
| Tipo | Comportamento |
|------|---------------|
| Tipo 1 | Inverte o valor de `valid` (`stringSize % 2 !== 0`) |
| Tipo 2 | Omite o campo `valid` completamente |

---

### Dashboard

**Porta:** `3000`  
Acesse em: `http://localhost:3000`

Interface web em HTML/JS puro com duas seções:

#### Monitor de Filas
- Seletor dropdown para escolher qual fila monitorar (SQS01–SQS05)
- Stream em tempo real via **SSE (Server-Sent Events)** — as mensagens aparecem conforme chegam
- Botão "Clear" para limpar o log exibido

#### Controles das Apps
- Tabs para selecionar APP01, APP02, APP03 ou APP04
- **APP01:** slider de `intervalMs` (100ms–5000ms) + slider de `errorRate`
- **APP02–04:** slider de `delayMs` (0–3000ms) + slider de `errorRate`
- Alterações aplicadas ao soltar o slider (envia `POST /app/:id/config`)
- Valores atuais exibidos ao lado de cada slider

#### Botão "Reset Sentinel"

Limpa todo o estado acumulado do Sentinel em uma única operação:

1. Purga as filas do usecase (`sqs01–sqs05`) — remove o backlog de mensagens não processadas
2. Purga as filas de observação do Sentinel (`sentinel-a0{1-5}-in/out`, `sentinel-alarms`) — descarta mensagens que o Sentinel ainda não consumiu
3. Executa `FLUSHDB` no Redis — apaga todas as chaves de correlação pendentes dos agents

**Quando usar:** quando a taxa de produção do APP01 superar a capacidade de processamento do Sentinel, os timeouts gerados corrompem o treinamento. O fluxo correto é:
1. Parar o APP01 (botão ⏹ Stop na aba APP01)
2. Clicar em **Reset Sentinel**
3. Reiniciar o APP01 com um `intervalMs` maior (taxa menor)

> A lógica de reset está em `shared/reset.js` e pode ser reutilizada em scripts externos ou via `npm run reset`.

**API do servidor do dashboard:**
```http
GET  /queues              → lista de filas disponíveis
GET  /apps                → lista de apps com URLs de controle
GET  /stream?queue=sqs01  → SSE stream de mensagens da fila
GET  /app/:id/config      → proxy GET para a API de controle da app
POST /app/:id/config      → proxy POST para a API de controle da app
POST /reset               → purga todas as filas + FLUSHDB Redis (ver acima)
```

---

## Estrutura do Monorepo

```
usecase/
├── apps/
│   ├── app01/           # Producer
│   │   ├── index.js
│   │   └── package.json
│   ├── app02/           # Enricher
│   │   ├── index.js
│   │   └── package.json
│   ├── app03/           # Batcher
│   │   ├── index.js
│   │   └── package.json
│   └── app04/           # Validator
│       ├── index.js
│       └── package.json
├── dashboard/
│   ├── server.js
│   ├── public/
│   │   └── index.html
│   └── package.json
├── shared/
│   ├── aws.js           # Clientes SQS/SNS configurados para LocalStack
│   └── package.json
├── infra/
│   └── init/
│       └── setup.sh     # Cria tópicos, filas e subscriptions no LocalStack
├── docker-compose.yml
├── start.sh             # Script de inicialização com detecção de WSL
├── package.json         # Workspaces npm
└── README.md
```

---

## Como Executar

### Pré-requisitos

- [Docker Desktop](https://www.docker.com/products/docker-desktop/)
- Node.js 18+
- npm 9+
- AWS CLI (para execução manual do `setup.sh`)

> **WSL:** se estiver rodando no WSL2, use o `start.sh` — ele detecta automaticamente o IP do host Windows e configura o endpoint correto.

### Passo a passo

**1. Instale as dependências (apenas na primeira vez):**
```bash
cd /mnt/c/projects/Septimus/sentinel/usecase
npm install
```

**2. Suba o LocalStack (em um terminal dedicado):**
```bash
docker compose up
```
Aguarde a mensagem de inicialização do LocalStack. O `setup.sh` é executado automaticamente via `init/ready.d`.

**3. Suba todas as apps e o dashboard:**
```bash
bash start.sh
```
O script aguarda o LocalStack estar saudável antes de iniciar os serviços.

**4. Acesse o dashboard:**
```
http://localhost:3000
```

### Portas

| Serviço | Porta |
|---------|-------|
| Dashboard | 3000 |
| APP01 (controle) | 3001 |
| APP02 (controle) | 3002 |
| APP03 (controle) | 3003 |
| APP04 (controle) | 3004 |
| LocalStack | 4566 |

---

## Variáveis de Ambiente

| Variável | Padrão | Descrição |
|----------|--------|-----------|
| `LOCALSTACK_ENDPOINT` | `http://localhost:4566` | Endpoint do LocalStack. Ajustado automaticamente pelo `start.sh` no WSL |

---

## Simulação de Anomalias

Todas as apps expõem a configuração de `errorRate` via API REST. O valor vai de `0` (sem erros) a `1` (100% de erros).

**Resumo dos erros por app:**

| App | Erro Tipo 1 | Erro Tipo 2 |
|-----|-------------|-------------|
| APP01 | Omite `value02` (schema incompleto) | — |
| APP02 | `result` com offset aleatório (soma errada) | `stringSize` com offset aleatório |
| APP03 | Batch com menos de 20 strings (perda de mensagens) | Batch com string duplicada |
| APP04 | `valid` invertido | Campo `valid` omitido |

Os erros de Tipo 1 e Tipo 2 são escolhidos aleatoriamente com distribuição 50/50 quando o erro é acionado pelo `errorRate`.

---

## Contexto — Projeto Sentinel

Este use case serve como **ambiente de treinamento e validação** para o Sentinel, que tem como objetivo:

1. **Fase de treinamento:** observar os logs de entrada e saída de cada serviço durante o funcionamento normal, aprendendo os padrões esperados de cada etapa da pipeline.
2. **Fase de inferência:** detectar em tempo real desvios de comportamento — como somas incorretas, batches incompletos, campos ausentes ou inversões lógicas — sem que as regras sejam programadas explicitamente.

A combinação de `errorRate` configurável por app permite gerar datasets de treinamento com diferentes densidades e tipos de anomalia de forma controlada.
