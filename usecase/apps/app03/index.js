const express = require('express');
const { sqsClient, snsClient, getQueueUrl, getTopicArn } = require('@sentinel/shared');
const { ReceiveMessageCommand, DeleteMessageCommand } = require('@aws-sdk/client-sqs');
const { PublishCommand } = require('@aws-sdk/client-sns');

const app = express();
app.use(express.json());

let config = { delayMs: 0, errorRate: 0 };

const BATCH_SIZE = 20;
const BATCH_FLUSH_TIMEOUT_MS = 30000; // flush incomplete batch after 30s of inactivity
let buffer = []; // { payload, receiptHandle }
let flushTimer = null;

const MAX_LOG = 30;
const eventLog = [];

function addLog(entry) {
  eventLog.unshift(entry);
  if (eventLog.length > MAX_LOG) eventLog.pop();
}

function unwrap(body) {
  try {
    const outer = JSON.parse(body);
    if (outer.Message) return JSON.parse(outer.Message);
    return outer;
  } catch {
    return {};
  }
}

async function publishBatch(items) {
  if (config.delayMs > 0) {
    await new Promise(r => setTimeout(r, config.delayMs));
  }

  const records = items.map(i => ({
    someString: i.payload.someString || '',
    correlationId: i.payload.correlationId || '',
  }));
  const fireError = Math.random() < config.errorRate;
  let outRecords = [...records];

  if (fireError) {
    const errType = Math.random() < 0.5 ? 1 : 2;
    if (errType === 1) {
      const count = 1 + Math.floor(Math.random() * 19);
      outRecords = outRecords.slice(0, count);
      console.error(`[APP03] error type1 - message loss: ${outRecords.length} records instead of 20`);
    } else {
      const dupIdx = Math.floor(Math.random() * outRecords.length);
      outRecords[dupIdx] = { ...outRecords[Math.floor(Math.random() * outRecords.length)] };
      console.error(`[APP03] error type2 - duplicate record in batch`);
    }
  }

  const batchPayload = {
    records: outRecords,
    fileName: `batch_${Date.now()}.txt`,
  };

  addLog({ timestamp: new Date().toISOString(), input: items.map(i => i.payload), output: batchPayload });
  console.log(`[APP03] input: ${JSON.stringify(items.map(i => i.payload))}`);
  console.log(`[APP03] output: ${JSON.stringify(batchPayload)}`);

  await snsClient.send(new PublishCommand({
    TopicArn: getTopicArn('sns03'),
    Message: JSON.stringify(batchPayload),
  }));
}

async function flushBuffer(queueUrl) {
  if (buffer.length === 0) return;
  const batch = buffer.splice(0, buffer.length);
  await publishBatch(batch);
  await Promise.all(batch.map(item =>
    sqsClient.send(new DeleteMessageCommand({
      QueueUrl: queueUrl,
      ReceiptHandle: item.receiptHandle,
    }))
  ));
}

function resetFlushTimer(queueUrl) {
  if (flushTimer) clearTimeout(flushTimer);
  flushTimer = setTimeout(async () => {
    if (buffer.length > 0) {
      console.warn(`[APP03] flush timer: publishing incomplete batch of ${buffer.length} records`);
      try { await flushBuffer(queueUrl); } catch (e) { console.error(`[APP03] flush error: ${e.message}`); }
    }
  }, BATCH_FLUSH_TIMEOUT_MS);
}

async function consumeLoop() {
  const queueUrl = getQueueUrl('sqs02');
  while (true) {
    try {
      const result = await sqsClient.send(new ReceiveMessageCommand({
        QueueUrl: queueUrl,
        MaxNumberOfMessages: 10,
        WaitTimeSeconds: 1,
      }));

      const messages = result.Messages || [];
      for (const sqsMsg of messages) {
        const payload = unwrap(sqsMsg.Body);
        buffer.push({ payload, receiptHandle: sqsMsg.ReceiptHandle });
        resetFlushTimer(queueUrl);

        if (buffer.length >= BATCH_SIZE) {
          if (flushTimer) { clearTimeout(flushTimer); flushTimer = null; }
          const batch = buffer.splice(0, BATCH_SIZE);
          await publishBatch(batch);
          await Promise.all(batch.map(item =>
            sqsClient.send(new DeleteMessageCommand({
              QueueUrl: queueUrl,
              ReceiptHandle: item.receiptHandle,
            }))
          ));
        }
      }
    } catch (err) {
      console.error(`[APP03] consumer error: ${err.message}`);
      await new Promise(r => setTimeout(r, 1000));
    }
  }
}

app.get('/config', (req, res) => res.json(config));
app.post('/config', (req, res) => {
  const { delayMs, errorRate } = req.body;
  if (delayMs !== undefined) config.delayMs = delayMs;
  if (errorRate !== undefined) config.errorRate = errorRate;
  res.json(config);
});

app.get('/log', (req, res) => res.json(eventLog));

consumeLoop();
app.listen(3003);
