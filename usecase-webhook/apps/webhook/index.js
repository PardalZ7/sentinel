const express = require('express');
const { SNSClient, PublishCommand } = require('@aws-sdk/client-sns');
const { v4: uuidv4 } = require('uuid');
const chain = require('./transformers');
const { SkipPublish } = require('./transformers/chain');
const { getNestedValue } = require('./utils');

const app = express();
app.use(express.json());

const LOCALSTACK_ENDPOINT = process.env.LOCALSTACK_ENDPOINT;
const AWS_REGION = process.env.AWS_REGION || 'us-east-1';
const AWS_ACCOUNT_ID = process.env.AWS_ACCOUNT_ID || '000000000000';
const DEFAULT_OUTPUT_TOPIC = process.env.DEFAULT_OUTPUT_TOPIC || 'ucwh-output';
const DASHBOARD_URL = process.env.DASHBOARD_URL || 'http://localhost:8890';

let cachedHeaders = [];
let headersCacheTTL = 0;

async function getConfiguredHeaders() {
  const now = Date.now();
  // Refresh cache every 5 seconds
  if (cachedHeaders.length > 0 && now < headersCacheTTL) {
    return cachedHeaders;
  }

  try {
    const res = await fetch(`${DASHBOARD_URL}/headers`);
    if (!res.ok) return cachedHeaders;
    cachedHeaders = await res.json();
    headersCacheTTL = now + 5000;
    if (cachedHeaders.length > 0) {
      console.log(`[WEBHOOK] fetched header mappings from dashboard:`, JSON.stringify(cachedHeaders));
    }
  } catch (err) {
    console.warn(`[WEBHOOK] failed to fetch headers from dashboard:`, err.message);
  }

  return cachedHeaders;
}

// SNS topic name must match: [a-zA-Z0-9_-]{1,256}(.fifo)?
const TOPIC_NAME_RE = /^[a-zA-Z0-9_-]{1,256}(\.fifo)?$/;

const snsClient = new SNSClient(
  LOCALSTACK_ENDPOINT
    ? { endpoint: LOCALSTACK_ENDPOINT, region: AWS_REGION, credentials: { accessKeyId: 'test', secretAccessKey: 'test' } }
    : { region: AWS_REGION }
);

function buildTopicArn(topicName) {
  return `arn:aws:sns:${AWS_REGION}:${AWS_ACCOUNT_ID}:${topicName}`;
}

// ── Event log ────────────────────────────────────────────────────────────────

const MAX_LOG = 30;
const eventLog = [];
let receivedCount = 0;

function addLog(entry) {
  eventLog.unshift(entry);
  if (eventLog.length > MAX_LOG) eventLog.pop();
}

// ── Routes ───────────────────────────────────────────────────────────────────

app.post('/webhook', async (req, res) => {
  const receivedAt = new Date().toISOString();
  const rawPayload = req.body;
  console.log(`[WEBHOOK] received payload:`, JSON.stringify(rawPayload).substring(0, 200));

  // Resolve topic — pull from payload (and strip it) or use default.
  let topicName = DEFAULT_OUTPUT_TOPIC;
  let payload = { ...rawPayload };
  if (payload.output_topic !== undefined) {
    topicName = payload.output_topic;
    delete payload.output_topic;
  }

  // Generate id if missing
  if (payload.id === undefined) {
    payload.id = uuidv4();
    console.log(`[WEBHOOK] generated id: ${payload.id}`);
  }

  console.log(`[WEBHOOK] resolved topic: ${topicName}`);

  if (!TOPIC_NAME_RE.test(topicName)) {
    return res.status(400).json({ error: `invalid topic name: "${topicName}"` });
  }

  // Extract headers from payload using configured mappings
  const configuredHeaders = await getConfiguredHeaders();
  const extractedHeaders = {};

  configuredHeaders.forEach(mapping => {
    const headerValue = getNestedValue(payload, mapping.path);
    if (headerValue !== undefined && headerValue !== null) {
      extractedHeaders[mapping.name] = String(headerValue);
      console.log(`[WEBHOOK] extracted '${mapping.name}' from path '${mapping.path}': ${extractedHeaders[mapping.name]}`);
    }
  });

  // Prioritize request headers over extracted values
  const finalHeaders = {
    'content-type': req.headers['content-type'],
    'user-agent': req.headers['user-agent'],
    'x-webhook-source': req.headers['x-webhook-source'],
    ...extractedHeaders,
  };

  // Override with any request headers that match extraction mappings
  configuredHeaders.forEach(mapping => {
    if (req.headers[mapping.name]) {
      finalHeaders[mapping.name] = req.headers[mapping.name];
      console.log(`[WEBHOOK] header '${mapping.name}' from request overrides extracted value`);
    }
  });

  const context = {
    receivedAt,
    resolvedTopic: topicName,
    headers: finalHeaders,
  };

  try {
    payload = await chain.run(payload, context);
  } catch (err) {
    if (err instanceof SkipPublish) {
      console.log(`[WEBHOOK] skipped (${err.message})`);
      addLog({ receivedAt, topic: topicName, skipped: true, reason: err.message });
      return res.status(200).json({ skipped: true, reason: err.message });
    }
    console.error(`[WEBHOOK] transformer error: ${err.message}`);
    return res.status(500).json({ error: 'transformer error', detail: err.message });
  }

  try {
    const topicArn = buildTopicArn(topicName);
    console.log(`[WEBHOOK] publishing to SNS — TopicArn: ${topicArn}`);

    const messageAttributes = {
      source: { DataType: 'String', StringValue: 'webhook' },
    };

    // Add configured headers as message attributes
    Object.entries(finalHeaders).forEach(([key, value]) => {
      if (value && typeof value === 'string') {
        const attrName = key.replace(/^x-/i, '').replace(/-/g, '_').substring(0, 256);
        messageAttributes[attrName] = { DataType: 'String', StringValue: value.substring(0, 1024) };
        console.log(`[WEBHOOK] added header '${key}' to messageAttributes: ${value}`);
      }
    });

    // Add remaining X-* headers from request
    Object.entries(req.headers).forEach(([key, value]) => {
      if ((key.startsWith('x-') || key.startsWith('X-')) && !messageAttributes[key]) {
        const attrName = key.replace(/^x-/i, '').replace(/-/g, '_').substring(0, 256);
        if (value && typeof value === 'string') {
          messageAttributes[attrName] = { DataType: 'String', StringValue: value.substring(0, 1024) };
        }
      }
    });

    console.log(`[WEBHOOK] message attributes:`, JSON.stringify(messageAttributes));

    const result = await snsClient.send(new PublishCommand({
      TopicArn: topicArn,
      Message: JSON.stringify(payload),
      MessageAttributes: messageAttributes,
    }));

    receivedCount++;
    addLog({ receivedAt, topic: topicName, messageId: result.MessageId, payload });
    console.log(`[WEBHOOK] ✓ published to ${topicName} — MessageId: ${result.MessageId}`);
    return res.status(202).json({ topic: topicName, message_id: result.MessageId });
  } catch (err) {
    console.error(`[WEBHOOK] ✗ SNS publish FAILED: ${err?.message || String(err)}`);
    console.error(`[WEBHOOK] Error details:`, err);
    addLog({ receivedAt, topic: topicName, skipped: true, reason: `publish error: ${err?.message}` });
    return res.status(500).json({ error: 'publish failed', detail: err?.message });
  }
});

app.get('/health', (_req, res) => res.json({ status: 'ok', receivedCount }));

app.get('/log', (_req, res) => res.json(eventLog));

const PORT = process.env.WEBHOOK_PORT || 3001;
app.listen(PORT, () =>
  console.log(`[WEBHOOK] listening on port ${PORT} — default topic: ${DEFAULT_OUTPUT_TOPIC}`)
);
