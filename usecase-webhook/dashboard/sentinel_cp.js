const fs = require('fs');

const SENTINEL_URL  = process.env.SENTINEL_URL  || 'http://localhost:8890';
const TOPOLOGY_ID   = process.env.SENTINEL_TOPOLOGY_ID || 'usecase-webhook';

function loadConfig(configPath) {
  const raw = fs.readFileSync(configPath, 'utf8');
  return JSON.parse(raw);
}

async function deployTopology(configPath) {
  const config = loadConfig(configPath);
  const url = `${SENTINEL_URL}/cp/topologies/${encodeURIComponent(TOPOLOGY_ID)}`;
  let res;
  try {
    res = await fetch(url, {
      method: 'PUT',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ config: config.sentinel ?? config }),
    });
  } catch (err) {
    return { ok: false, status: 0, body: { error: err.message } };
  }

  let body;
  try { body = await res.json(); } catch { body = {}; }

  return { ok: res.status === 201 || res.status === 200, status: res.status, body };
}

async function deployWithRetry(configPath, { maxRetries = 10, retryMs = 3000 } = {}) {
  for (let i = 1; i <= maxRetries; i++) {
    const result = await deployTopology(configPath);
    if (result.ok) return result;
    console.warn(
      `[sentinel-cp] attempt ${i}/${maxRetries} failed (HTTP ${result.status}): ` +
      (result.body?.error || result.body?.detail || JSON.stringify(result.body))
    );
    if (i < maxRetries) await new Promise(r => setTimeout(r, retryMs));
  }
  return { ok: false, status: 0, body: { error: `Failed after ${maxRetries} attempts` } };
}

module.exports = { deployTopology, deployWithRetry, SENTINEL_URL, TOPOLOGY_ID };
