const express = require('express');
const path = require('path');
const { sqsClient, getQueueUrl } = require('@sentinel/shared');
const { resetSentinel, USECASE_QUEUES } = require('@sentinel/shared/reset');
const { deployWithRetry } = require('@sentinel/shared/sentinel-cp');
const { PurgeQueueCommand } = require('@aws-sdk/client-sqs');
const { ReceiveMessageCommand } = require('@aws-sdk/client-sqs');

const app = express();
app.use(express.json());
app.use(express.static(path.join(__dirname, 'public')));

const SENTINEL_CONFIG_PATH = process.env.SENTINEL_CONFIG_PATH
  || path.resolve(__dirname, '..', 'sentinel.json');

const QUEUES = ['sqs01', 'sqs02', 'sqs03', 'sqs04', 'sqs05'];

const APPS = [
  { id: 'app01', name: 'APP01', url: 'http://localhost:3001' },
  { id: 'app02', name: 'APP02', url: 'http://localhost:3002' },
  { id: 'app03', name: 'APP03', url: 'http://localhost:3003' },
  { id: 'app04', name: 'APP04', url: 'http://localhost:3004' },
];

const BUFFER_MAX = 25;
const appBuffers = {};
const appLastTimestamp = {};
APPS.forEach(a => { appBuffers[a.id] = []; appLastTimestamp[a.id] = null; });

async function pollAppLog(appDef) {
  try {
    const r = await fetch(`${appDef.url}/log`);
    if (!r.ok) return;
    const entries = await r.json();
    if (!Array.isArray(entries) || !entries.length) return;

    const buf = appBuffers[appDef.id];
    const lastTs = appLastTimestamp[appDef.id];
    const newEntries = lastTs ? entries.filter(e => e.timestamp > lastTs) : entries;
    if (!newEntries.length) return;

    buf.unshift(...newEntries);
    if (buf.length > BUFFER_MAX) buf.splice(BUFFER_MAX);
    appLastTimestamp[appDef.id] = buf[0].timestamp;
  } catch {}
}

setInterval(() => APPS.forEach(pollAppLog), 1000);

// ── Deploy topology to Sentinel on startup ──────────────────────────────────
deployWithRetry(SENTINEL_CONFIG_PATH)
  .then(result => {
    if (result.ok) {
      console.log('[sentinel-cp] topology deployed:', JSON.stringify(result.body));
    } else {
      console.error('[sentinel-cp] topology deploy failed:', JSON.stringify(result.body));
    }
  });

// ── Routes ──────────────────────────────────────────────────────────────────

app.get('/', (req, res) => {
  res.sendFile(path.join(__dirname, 'public', 'index.html'));
});

app.get('/queues', (req, res) => {
  res.json(QUEUES);
});

app.get('/apps', (req, res) => {
  res.json(APPS);
});

app.get('/peek', async (req, res) => {
  const queueName = req.query.queue;
  if (!QUEUES.includes(queueName)) {
    return res.status(400).json({ error: 'Invalid queue name' });
  }

  const queueUrl = getQueueUrl(queueName);
  try {
    const result = await sqsClient.send(new ReceiveMessageCommand({
      QueueUrl: queueUrl,
      MaxNumberOfMessages: 10,
      WaitTimeSeconds: 1,
    }));

    const messages = (result.Messages || []).map(msg => {
      let body;
      try {
        const parsed = JSON.parse(msg.Body);
        body = parsed.Message ? JSON.parse(parsed.Message) : parsed;
      } catch {
        body = msg.Body;
      }
      return {
        queue: queueName,
        messageId: msg.MessageId,
        body,
        timestamp: new Date().toISOString(),
      };
    });

    res.json(messages);
  } catch (err) {
    res.status(500).json({ error: err.message });
  }
});

app.get('/app/:id/config', async (req, res) => {
  const appDef = APPS.find(a => a.id === req.params.id);
  if (!appDef) return res.status(404).json({ error: 'App not found' });
  try {
    const r = await fetch(`${appDef.url}/config`);
    const data = await r.json();
    res.json(data);
  } catch (err) {
    res.status(502).json({ error: err.message });
  }
});

app.post('/app/:id/config', async (req, res) => {
  const appDef = APPS.find(a => a.id === req.params.id);
  if (!appDef) return res.status(404).json({ error: 'App not found' });
  try {
    const r = await fetch(`${appDef.url}/config`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(req.body),
    });
    const data = await r.json();
    res.json(data);
  } catch (err) {
    res.status(502).json({ error: err.message });
  }
});

app.post('/app/:id/state', async (req, res) => {
  const appDef = APPS.find(a => a.id === req.params.id);
  if (!appDef) return res.status(404).json({ error: 'App not found' });
  try {
    const r = await fetch(`${appDef.url}/state`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(req.body),
    });
    const data = await r.json();
    res.json(data);
  } catch (err) {
    res.status(502).json({ error: err.message });
  }
});

app.get('/app/:id/log', async (req, res) => {
  const appDef = APPS.find(a => a.id === req.params.id);
  if (!appDef) return res.status(404).json({ error: 'App not found' });
  try {
    const r = await fetch(`${appDef.url}/log`);
    const data = await r.json();
    res.json(data);
  } catch (err) {
    res.status(502).json({ error: err.message });
  }
});

app.get('/app/:id/buffer', (req, res) => {
  const appDef = APPS.find(a => a.id === req.params.id);
  if (!appDef) return res.status(404).json({ error: 'App not found' });
  res.json(appBuffers[appDef.id]);
});

app.post('/purge-usecase-queues', async (req, res) => {
  const errors = [];
  await Promise.all(USECASE_QUEUES.map(async (name) => {
    try {
      await sqsClient.send(new PurgeQueueCommand({ QueueUrl: getQueueUrl(name) }));
    } catch (err) {
      errors.push(`${name}: ${err.message}`);
    }
  }));
  res.status(errors.length ? 500 : 200).json({ ok: errors.length === 0, purged: USECASE_QUEUES, errors });
});

app.post('/reset', async (req, res) => {
  const result = await resetSentinel();
  res.status(result.ok ? 200 : 500).json(result);
});

app.post('/setup-topology', async (req, res) => {
  const result = await deployWithRetry(SENTINEL_CONFIG_PATH, { maxRetries: 3, retryMs: 1000 });
  res.status(result.ok ? 200 : 500).json(result);
});

app.listen(3000, () => console.log('[DASHBOARD] server listening on port 3000'));
