const express = require('express');
const { sqsClient, snsClient, getQueueUrl, getTopicArn } = require('@sentinel/shared');
const { ReceiveMessageCommand, DeleteMessageCommand } = require('@aws-sdk/client-sqs');
const { PublishCommand } = require('@aws-sdk/client-sns');

const app = express();
app.use(express.json());

let config = { delayMs: 0, errorRate: 0 };

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

async function consumeLoop() {
  const queueUrl = getQueueUrl('sqs03');
  while (true) {
    try {
      const result = await sqsClient.send(new ReceiveMessageCommand({
        QueueUrl: queueUrl,
        MaxNumberOfMessages: 10,
        WaitTimeSeconds: 1,
      }));

      const messages = result.Messages || [];
      await Promise.all(messages.map(async (sqsMsg) => {
        const payload = unwrap(sqsMsg.Body);

        if (config.delayMs > 0) {
          await new Promise(r => setTimeout(r, config.delayMs));
        }

        const fireError = Math.random() < config.errorRate;
        let outPayload = { ...payload };

        if (fireError) {
          const errType = Math.random() < 0.5 ? 1 : 2;
          if (errType === 1) {
            outPayload.valid = (payload.stringSize % 2 !== 0);
            console.error(`[APP04] error type1 - inverted valid: ${JSON.stringify(outPayload)}`);
          } else {
            delete outPayload.valid;
            console.error(`[APP04] error type2 - omit valid: ${JSON.stringify(outPayload)}`);
          }
        } else {
          outPayload.valid = (payload.stringSize % 2 === 0);
        }

        addLog({ timestamp: new Date().toISOString(), input: payload, output: outPayload });
        console.log(`[APP04] input: ${JSON.stringify(payload)}`);
        console.log(`[APP04] output: ${JSON.stringify(outPayload)}`);

        await snsClient.send(new PublishCommand({
          TopicArn: getTopicArn('sns04'),
          Message: JSON.stringify(outPayload),
        }));

        await sqsClient.send(new DeleteMessageCommand({
          QueueUrl: queueUrl,
          ReceiptHandle: sqsMsg.ReceiptHandle,
        }));
      }));
    } catch (err) {
      console.error(`[APP04] consumer error: ${err?.message || String(err)}`);
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
app.listen(3004);
