const express = require('express');
const { SQSClient, SendMessageCommand } = require('@aws-sdk/client-sqs');

const app = express();
app.use(express.json());

const LOCALSTACK_ENDPOINT = process.env.LOCALSTACK_ENDPOINT;
const AWS_REGION = process.env.AWS_REGION || 'us-east-1';
const AWS_ACCOUNT_ID = process.env.AWS_ACCOUNT_ID || '000000000000';
const QUEUE_NAME = process.env.QUEUE_NAME || 'uc2-input';

const sqsClient = new SQSClient(
  LOCALSTACK_ENDPOINT
    ? { endpoint: LOCALSTACK_ENDPOINT, region: AWS_REGION, credentials: { accessKeyId: 'test', secretAccessKey: 'test' } }
    : { region: AWS_REGION }
);

function getQueueUrl() {
  if (LOCALSTACK_ENDPOINT) return `${LOCALSTACK_ENDPOINT}/${AWS_ACCOUNT_ID}/${QUEUE_NAME}`;
  return `https://sqs.${AWS_REGION}.amazonaws.com/${AWS_ACCOUNT_ID}/${QUEUE_NAME}`;
}

// ── Template evaluation ──────────────────────────────────────────────────────

function evalValue(raw) {
  if (typeof raw !== 'string') return raw;
  let m;

  m = raw.match(/^RandomString\((\d+)\)$/);
  if (m) return randomString(parseInt(m[1]));

  m = raw.match(/^RandomInteger\((\d+)\)$/);
  if (m) return randomInt(0, parseInt(m[1]));

  m = raw.match(/^RandomDouble\((\d+(?:\.\d+)?),(\d+)\)$/);
  if (m) return randomDouble(parseFloat(m[1]), parseInt(m[2]));

  m = raw.match(/^OneOf\((.+)\)$/);
  if (m) {
    const opts = m[1].split(';');
    return opts[Math.floor(Math.random() * opts.length)];
  }

  return raw;
}

function evalTemplate(node) {
  if (Array.isArray(node)) return node.map(evalTemplate);
  if (node !== null && typeof node === 'object') {
    const out = {};
    for (const [k, v] of Object.entries(node)) out[k] = evalTemplate(v);
    return out;
  }
  return evalValue(node);
}

function randomString(len) {
  const chars = 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789';
  let s = '';
  for (let i = 0; i < len; i++) s += chars[Math.floor(Math.random() * chars.length)];
  return s;
}

function randomInt(min, max) { return min + Math.floor(Math.random() * (max - min + 1)); }
function randomDouble(max, decimals) { return parseFloat((Math.random() * max).toFixed(decimals)); }

// ── Error injectors ──────────────────────────────────────────────────────────

function applyOmit(payload) {
  const keys = Object.keys(payload);
  if (!keys.length) return payload;
  const key = keys[Math.floor(Math.random() * keys.length)];
  const result = { ...payload };
  delete result[key];
  return result;
}

function applyCorrupt(payload) {
  const keys = Object.keys(payload);
  if (!keys.length) return payload;
  const key = keys[Math.floor(Math.random() * keys.length)];
  const original = payload[key];
  let corrupted;
  if (typeof original === 'number') corrupted = randomString(6);
  else if (typeof original === 'string') corrupted = randomInt(0, 9999);
  else if (Array.isArray(original)) corrupted = randomString(4);
  else corrupted = randomInt(0, 9999);
  return { ...payload, [key]: corrupted };
}

function applyInject(payload) {
  return { ...payload, [`__injected_${randomString(4)}`]: randomString(8) };
}

// ── Template selection ───────────────────────────────────────────────────────

function pickTemplate(templates) {
  const valid = templates.filter(t => t.weight > 0);
  if (!valid.length) return templates[0] || null;
  const total = valid.reduce((s, t) => s + t.weight, 0);
  let r = Math.random() * total;
  for (const t of valid) {
    r -= t.weight;
    if (r <= 0) return t;
  }
  return valid[valid.length - 1];
}

// ── Default config ───────────────────────────────────────────────────────────

const DEFAULT_TEMPLATES = [
  {
    id: 't1',
    name: 'Default',
    weight: 1,
    template: JSON.stringify({
      field1: 'RandomString(20)',
      field2: 'RandomInteger(13)',
      field3: 'RandomDouble(50,2)',
      list: [{ innerValue1: 'RandomString(5)', enum: 'OneOf(Teste01;Teste02;Teste03)' }],
    }, null, 2),
  },
];

let config = {
  intervalMs: 1000,
  templates: DEFAULT_TEMPLATES,
  errorRateOmit: 0,
  errorRateCorrupt: 0,
  errorRateInject: 0,
};

let running = false;
let timer = null;
let publishedCount = 0;

const MAX_LOG = 30;
const eventLog = [];

function addLog(entry) {
  eventLog.unshift(entry);
  if (eventLog.length > MAX_LOG) eventLog.pop();
}

// ── Produce ──────────────────────────────────────────────────────────────────

async function produce() {
  if (!config.templates.length) return;

  const tpl = pickTemplate(config.templates);
  if (!tpl) return;

  let templateObj;
  try {
    templateObj = JSON.parse(tpl.template);
  } catch {
    console.error(`[APP01] invalid JSON in template "${tpl.name}" — skipping`);
    return;
  }

  let payload = evalTemplate(templateObj);
  const errors = [];

  if (Math.random() < config.errorRateOmit)   { payload = applyOmit(payload);    errors.push('omit'); }
  if (Math.random() < config.errorRateCorrupt) { payload = applyCorrupt(payload); errors.push('corrupt'); }
  if (Math.random() < config.errorRateInject)  { payload = applyInject(payload);  errors.push('inject'); }

  try {
    await sqsClient.send(new SendMessageCommand({
      QueueUrl: getQueueUrl(),
      MessageBody: JSON.stringify(payload),
    }));
    publishedCount++;
    addLog({ timestamp: new Date().toISOString(), templateName: tpl.name, payload, errors });
    console.log(`[APP01] [${tpl.name}]${errors.length ? ` [${errors.join(',')}]` : ''}: ${JSON.stringify(payload)}`);
  } catch (err) {
    console.error(`[APP01] publish error: ${err?.message || String(err)}`);
  }
}

function restartTimer() {
  if (timer) clearInterval(timer);
  if (running) {
    timer = setInterval(produce, config.intervalMs);
  } else {
    publishedCount = 0;
  }
}

// ── Routes ───────────────────────────────────────────────────────────────────

app.get('/config', (req, res) => res.json({ ...config, running, publishedCount }));

app.post('/config', (req, res) => {
  const { intervalMs, templates, errorRateOmit, errorRateCorrupt, errorRateInject } = req.body;
  let timerChanged = false;

  if (intervalMs !== undefined) { config.intervalMs = Math.max(25, intervalMs); timerChanged = true; }
  if (templates !== undefined)  { config.templates = templates; }
  if (errorRateOmit !== undefined)   config.errorRateOmit = errorRateOmit;
  if (errorRateCorrupt !== undefined) config.errorRateCorrupt = errorRateCorrupt;
  if (errorRateInject !== undefined)  config.errorRateInject = errorRateInject;

  if (timerChanged) restartTimer();
  res.json(config);
});

app.post('/state', (req, res) => {
  if (req.body.running !== undefined) {
    running = !!req.body.running;
    restartTimer();
  }
  res.json({ running });
});

app.get('/log', (req, res) => res.json(eventLog));

const PORT = process.env.PORT || 3001;
app.listen(PORT, () => console.log(`[APP01] listening on port ${PORT} — queue: ${QUEUE_NAME}`));
