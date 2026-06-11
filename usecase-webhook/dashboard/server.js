const express = require('express');
const path = require('path');
const fs = require('fs');
const { v4: uuidv4 } = require('uuid');
const { deployWithRetry } = require('./sentinel_cp');

const app = express();
app.use(express.json());
app.use(express.static(path.join(__dirname, 'public')));

const WEBHOOK_URL = process.env.WEBHOOK_URL || 'http://localhost:3001';

const SENTINEL_CONFIG_PATH = process.env.SENTINEL_CONFIG_PATH
  || path.resolve(__dirname, '..', 'sentinel.json');

const HEADERS_FILE = path.resolve(__dirname, '..', '.webhook-headers.json');

// Load headers from file
function loadHeaders() {
  try {
    if (fs.existsSync(HEADERS_FILE)) {
      return JSON.parse(fs.readFileSync(HEADERS_FILE, 'utf-8'));
    }
  } catch (err) {
    console.warn('[DASHBOARD] failed to load headers:', err.message);
  }
  return [];
}

// Save headers to file
function saveHeaders(headers) {
  try {
    fs.writeFileSync(HEADERS_FILE, JSON.stringify(headers, null, 2), 'utf-8');
  } catch (err) {
    console.error('[DASHBOARD] failed to save headers:', err.message);
  }
}

let headers = loadHeaders();

const BUFFER_MAX = 25;
let buffer = [];
let lastReceivedAt = null;

async function pollLog() {
  try {
    const r = await fetch(`${WEBHOOK_URL}/log`);
    if (!r.ok) return;
    const entries = await r.json();
    if (!Array.isArray(entries) || !entries.length) return;

    const newEntries = lastReceivedAt
      ? entries.filter(e => e.receivedAt > lastReceivedAt)
      : entries;
    if (!newEntries.length) return;

    buffer.unshift(...newEntries);
    if (buffer.length > BUFFER_MAX) buffer.splice(BUFFER_MAX);
    lastReceivedAt = buffer[0].receivedAt;
  } catch {}
}

setInterval(pollLog, 1000);

deployWithRetry(SENTINEL_CONFIG_PATH)
  .then(result => {
    if (result.ok) {
      console.log('[sentinel-cp] topology deployed:', JSON.stringify(result.body));
    } else {
      console.error('[sentinel-cp] topology deploy failed:', JSON.stringify(result.body));
    }
  });

// ── Routes ───────────────────────────────────────────────────────────────────

app.get('/', (req, res) => res.sendFile(path.join(__dirname, 'public', 'index.html')));

app.get('/app/health', async (req, res) => {
  try {
    const r = await fetch(`${WEBHOOK_URL}/health`);
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

// ── Headers endpoints ─────────────────────────────────────────────────────────

app.get('/app/headers', (req, res) => {
  res.json(headers);
});

app.post('/app/headers', (req, res) => {
  const { name, path: headerPath } = req.body;

  if (!name || !headerPath) {
    return res.status(400).json({ error: 'name and path are required' });
  }

  const id = uuidv4();
  const newHeader = { id, name, path: headerPath };
  headers.push(newHeader);
  saveHeaders(headers);

  console.log('[DASHBOARD] added header:', newHeader);
  res.status(201).json(newHeader);
});

app.delete('/app/headers/:id', (req, res) => {
  const { id } = req.params;
  const before = headers.length;
  headers = headers.filter(h => h.id !== id);

  if (headers.length < before) {
    saveHeaders(headers);
    console.log('[DASHBOARD] deleted header:', id);
    res.json({ ok: true });
  } else {
    res.status(404).json({ error: 'header not found' });
  }
});

// Endpoint for webhook to fetch configured headers
app.get('/headers', (req, res) => {
  res.json(headers);
});

const DASHBOARD_PORT = process.env.DASHBOARD_PORT || 8890;
app.listen(DASHBOARD_PORT, () => console.log(`[DASHBOARD] listening on port ${DASHBOARD_PORT}`));
