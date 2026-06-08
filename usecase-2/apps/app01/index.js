const express = require('express');
const { SNSClient, PublishCommand } = require('@aws-sdk/client-sns');
const { randomUUID } = require('crypto');

const app = express();
app.use(express.json());

const LOCALSTACK_ENDPOINT = process.env.LOCALSTACK_ENDPOINT;
const AWS_REGION = process.env.AWS_REGION || 'us-east-1';
const AWS_ACCOUNT_ID = process.env.AWS_ACCOUNT_ID || '000000000000';
const TOPIC_NAME = process.env.TOPIC_NAME || 'uc2-input';

const snsClient = new SNSClient(
  LOCALSTACK_ENDPOINT
    ? { endpoint: LOCALSTACK_ENDPOINT, region: AWS_REGION, credentials: { accessKeyId: 'test', secretAccessKey: 'test' } }
    : { region: AWS_REGION }
);

function getTopicArn() {
  if (LOCALSTACK_ENDPOINT) return `arn:aws:sns:${AWS_REGION}:${AWS_ACCOUNT_ID}:${TOPIC_NAME}`;
  return `arn:aws:sns:${AWS_REGION}:${AWS_ACCOUNT_ID}:${TOPIC_NAME}`;
}

// ── Template evaluation ──────────────────────────────────────────────────────

function parseVal(s) {
  if (s === 'true') return true;
  if (s === 'false') return false;
  if (s === 'null') return null;
  const n = Number(s);
  if (!isNaN(n) && s.trim() !== '') return n;
  return s;
}

function evalValue(raw) {
  if (typeof raw !== 'string') return raw;
  let m;

  if (raw === 'Now()') return new Date().toISOString();
  if (raw === 'UUID()') return randomUUID();

  m = raw.match(/^RandomString\((\d+)\)$/);
  if (m) return randomString(parseInt(m[1]));

  m = raw.match(/^RandomInteger\((\d+)\)$/);
  if (m) return randomInt(0, parseInt(m[1]));

  m = raw.match(/^RandomDouble\((\d+(?:\.\d+)?),(\d+)\)$/);
  if (m) return randomDouble(parseFloat(m[1]), parseInt(m[2]));

  m = raw.match(/^OneOf\((.+)\)$/);
  if (m) {
    const opts = m[1].split(';');
    return parseVal(opts[Math.floor(Math.random() * opts.length)]);
  }

  m = raw.match(/^WeightedOf\((.+)\)$/);
  if (m) {
    const entries = m[1].split(';').map(e => {
      const sep = e.lastIndexOf(':');
      return { val: parseVal(e.slice(0, sep)), w: parseFloat(e.slice(sep + 1)) || 0 };
    });
    const total = entries.reduce((s, e) => s + e.w, 0);
    let r = Math.random() * total;
    for (const e of entries) { r -= e.w; if (r <= 0) return e.val; }
    return entries[entries.length - 1].val;
  }

  // CorruptLinked(goodVal:goodRate;badVal)
  // effective good rate = goodRate * (1 - errorRateCorrupt)
  m = raw.match(/^CorruptLinked\((.+):([0-9.]+);(.+)\)$/);
  if (m) {
    const goodVal = parseVal(m[1]);
    const baseGoodRate = parseFloat(m[2]);
    const badVal = parseVal(m[3]);
    const effectiveGoodRate = baseGoodRate * (1 - config.errorRateCorrupt);
    return Math.random() < effectiveGoodRate ? goodVal : badVal;
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
    name: 'Automation Success',
    weight: 9,
    template: JSON.stringify({
      status: 'finished',
      finished_at: 'Now()',
      executed: 'CorruptLinked(true:0.8;false)',
      id: 'UUID()',
      automation_id: 3,
      card_id: 3,
      organization_id: 3,
      user_id: 10,
      action_name: 'move_single_card',
      deleted_at: null,
      created_at: 'Now()',
      updated_at: 'Now()',
      response: null,
      log_type: null,
      log_key: null,
      log_details: null,
      logs: null,
    }, null, 2),
  },
  {
    id: 't2',
    name: 'Automation Failure',
    weight: 1,
    template: JSON.stringify({
      status: 'failed',
      id: 'UUID()',
      automation_id: 4,
      card_id: 5,
      organization_id: 3,
      user_id: 10,
      action_name: 'move_single_card',
      finished_at: 'Now()',
      deleted_at: null,
      created_at: 'Now()',
      updated_at: 'Now()',
      response: null,
      executed: false,
      log_type: 'error',
      log_key: 'required_field',
      log_details: {
        label: 'Required field',
        field_name: 'Required field',
        repo_id: 16,
        repo_name: 'Automations',
        phase_name: 'Inbox',
      },
      logs: [
        {
          instance_class: 'Gatekeeper::AutomationMoveCard',
          id: null,
          key: '__automation_logs_error',
          args: [
            'required_field',
            { label: 'Required field', field_name: 'Required field', repo_id: 16, repo_name: 'Automations', phase_name: 'Inbox' },
          ],
          timestamp: 'Now()',
          type: 'error',
        },
        {
          instance_class: 'Gatekeeper::AutomationMoveCard',
          id: null,
          key: 'base',
          args: [
            'Field "Required field" is required! Please fill it and you\'ll be ready to go!',
            {},
          ],
          timestamp: 'Now()',
          type: 'error',
        },
      ],
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
    await snsClient.send(new PublishCommand({
      TopicArn: getTopicArn(),
      Message: JSON.stringify(payload),
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

const PORT = process.env.APP01_PORT || 3001;
app.listen(PORT, () => console.log(`[APP01] listening on port ${PORT} — topic: ${TOPIC_NAME}`));
