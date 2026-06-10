const express = require('express');
const { SNSClient, PublishCommand } = require('@aws-sdk/client-sns');
const chain = require('./transformers');
const { SkipPublish } = require('./transformers/chain');

const app = express();
app.use(express.json());

const LOCALSTACK_ENDPOINT = process.env.LOCALSTACK_ENDPOINT;
const AWS_REGION = process.env.AWS_REGION || 'us-east-1';
const AWS_ACCOUNT_ID = process.env.AWS_ACCOUNT_ID || '000000000000';
const DEFAULT_OUTPUT_TOPIC = process.env.DEFAULT_OUTPUT_TOPIC || 'ucwh-output';

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

  console.log(`[WEBHOOK] resolved topic: ${topicName}`);

  if (!TOPIC_NAME_RE.test(topicName)) {
    return res.status(400).json({ error: `invalid topic name: "${topicName}"` });
  }

  const context = {
    receivedAt,
    resolvedTopic: topicName,
    headers: {
      'content-type': req.headers['content-type'],
      'user-agent': req.headers['user-agent'],
      'x-webhook-source': req.headers['x-webhook-source'],
    },
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

    const result = await snsClient.send(new PublishCommand({
      TopicArn: topicArn,
      Message: JSON.stringify(payload),
      MessageAttributes: {
        source: { DataType: 'String', StringValue: 'webhook' },
      },
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
