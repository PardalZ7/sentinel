const express = require('express');
const { v4: uuidv4 } = require('uuid');
const { snsClient, getTopicArn } = require('@sentinel/shared');
const { PublishCommand } = require('@aws-sdk/client-sns');

const app = express();
app.use(express.json());

let config = { intervalMs: 1000, errorRate: 0 };
let running = false;
let timer = null;
let publishedCount = 0;

const MAX_LOG = 30;
const eventLog = [];

function addLog(entry) {
  eventLog.unshift(entry);
  if (eventLog.length > MAX_LOG) eventLog.pop();
}

function randomString(min, max) {
  const len = min + Math.floor(Math.random() * (max - min + 1));
  const chars = 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789';
  let s = '';
  for (let i = 0; i < len; i++) s += chars[Math.floor(Math.random() * chars.length)];
  return s;
}

function randomInt(min, max) {
  return min + Math.floor(Math.random() * (max - min + 1));
}

async function produce() {
  const fireError = Math.random() < config.errorRate;
  const msg = {
    correlationId: uuidv4(),
    someString: randomString(10, 50),
    value01: randomInt(1, 1000),
  };
  if (!fireError) {
    msg.value02 = randomInt(1, 1000);
  }

  try {
    await snsClient.send(new PublishCommand({
      TopicArn: getTopicArn('sns01'),
      Message: JSON.stringify(msg),
    }));
    publishedCount++;
    addLog({ timestamp: new Date().toISOString(), output: msg });
    console.log(`[APP01] output: ${JSON.stringify(msg)}`);
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

app.get('/config', (req, res) => res.json({ ...config, running, publishedCount }));

app.post('/state', (req, res) => {
  if (req.body.running !== undefined) {
    running = !!req.body.running;
    restartTimer();
  }
  res.json({ running });
});

app.post('/config', (req, res) => {
  const { intervalMs, errorRate } = req.body;
  let changed = false;
  if (intervalMs !== undefined) { config.intervalMs = Math.max(25, intervalMs); changed = true; }
  if (errorRate !== undefined) { config.errorRate = errorRate; }
  if (changed) restartTimer();
  res.json(config);
});

app.get('/log', (req, res) => res.json(eventLog));

app.listen(3001);
