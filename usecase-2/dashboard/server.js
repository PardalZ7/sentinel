const express = require('express');
const path = require('path');
const { deployWithRetry } = require('./sentinel_cp');

const app = express();
app.use(express.json());
app.use(express.static(path.join(__dirname, 'public')));

const APP_URL = process.env.APP01_URL || 'http://localhost:3001';

const SENTINEL_CONFIG_PATH = process.env.SENTINEL_CONFIG_PATH
  || path.resolve(__dirname, '..', 'sentinel.json');

const BUFFER_MAX = 25;
let buffer = [];
let lastTimestamp = null;

async function pollLog() {
  try {
    const r = await fetch(`${APP_URL}/log`);
    if (!r.ok) return;
    const entries = await r.json();
    if (!Array.isArray(entries) || !entries.length) return;

    const newEntries = lastTimestamp ? entries.filter(e => e.timestamp > lastTimestamp) : entries;
    if (!newEntries.length) return;

    buffer.unshift(...newEntries);
    if (buffer.length > BUFFER_MAX) buffer.splice(BUFFER_MAX);
    lastTimestamp = buffer[0].timestamp;
  } catch {}
}

setInterval(pollLog, 1000);

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

app.get('/', (req, res) => res.sendFile(path.join(__dirname, 'public', 'index.html')));

app.get('/app/config', async (req, res) => {
  try {
    const r = await fetch(`${APP_URL}/config`);
    res.json(await r.json());
  } catch (err) {
    res.status(502).json({ error: err.message });
  }
});

app.post('/app/config', async (req, res) => {
  try {
    const r = await fetch(`${APP_URL}/config`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(req.body),
    });
    res.json(await r.json());
  } catch (err) {
    res.status(502).json({ error: err.message });
  }
});

app.post('/app/state', async (req, res) => {
  try {
    const r = await fetch(`${APP_URL}/state`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(req.body),
    });
    res.json(await r.json());
  } catch (err) {
    res.status(502).json({ error: err.message });
  }
});

app.get('/app/buffer', (req, res) => res.json(buffer));

app.post('/setup-topology', async (req, res) => {
  const result = await deployWithRetry(SENTINEL_CONFIG_PATH, { maxRetries: 3, retryMs: 1000 });
  res.status(result.ok ? 200 : 500).json(result);
});

const PORT = process.env.PORT || 3000;
app.listen(PORT, () => console.log(`[DASHBOARD] listening on port ${PORT}`));
