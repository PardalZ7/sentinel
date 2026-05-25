const {
  CreateQueueCommand,
  GetQueueAttributesCommand,
} = require('@aws-sdk/client-sqs');
const {
  CreateTopicCommand,
  ListSubscriptionsByTopicCommand,
  SubscribeCommand,
} = require('@aws-sdk/client-sns');
const { sqsClient, snsClient, getQueueUrl, getTopicArn } = require('./aws');

const AWS_REGION = process.env.AWS_REGION || 'us-east-1';
const AWS_ACCOUNT_ID = process.env.AWS_ACCOUNT_ID || '000000000000';

function getQueueArn(name) {
  return `arn:aws:sqs:${AWS_REGION}:${AWS_ACCOUNT_ID}:${name}`;
}

async function ensureQueue(name) {
  await sqsClient.send(new CreateQueueCommand({ QueueName: name }));
}

async function ensureTopic(name) {
  const res = await snsClient.send(new CreateTopicCommand({ Name: name }));
  return res.TopicArn;
}

async function ensureSubscription(topicArn, queueName) {
  const queueArn = getQueueArn(queueName);

  // List existing subscriptions to avoid duplicates
  let nextToken;
  do {
    const res = await snsClient.send(new ListSubscriptionsByTopicCommand({
      TopicArn: topicArn,
      NextToken: nextToken,
    }));
    for (const sub of res.Subscriptions || []) {
      if (sub.Endpoint === queueArn) return; // already subscribed
    }
    nextToken = res.NextToken;
  } while (nextToken);

  await snsClient.send(new SubscribeCommand({
    TopicArn: topicArn,
    Protocol: 'sqs',
    Endpoint: queueArn,
  }));
}

/**
 * Creates all SQS queues, SNS topics, and subscriptions for the use case.
 * Idempotent: safe to call multiple times.
 */
async function setupTopology() {
  const log = [];

  // ── Queues ──────────────────────────────────────────────────────────────────
  const appQueues = ['sqs01', 'sqs02', 'sqs03', 'sqs04', 'sqs05'];
  const sentinelQueues = [
    'sentinel-a01-in', 'sentinel-a01-out',
    'sentinel-a02-in', 'sentinel-a02-out',
    'sentinel-a03-in', 'sentinel-a03-out',
    'sentinel-a04-in', 'sentinel-a04-out',
    'sentinel-alarms',
    'sentinel-a01-pairs', 'sentinel-a02-pairs',
    'sentinel-a03-pairs', 'sentinel-a04-pairs',
  ];

  for (const q of [...appQueues, ...sentinelQueues]) {
    try {
      await ensureQueue(q);
      log.push(`queue ok: ${q}`);
    } catch (e) {
      log.push(`queue error: ${q}: ${e.message}`);
    }
  }

  // ── Topics ──────────────────────────────────────────────────────────────────
  const topics = ['sns01', 'sns02', 'sns03', 'sns04'];
  for (const t of topics) {
    try {
      await ensureTopic(getTopicArn(t));
      log.push(`topic ok: ${t}`);
    } catch (e) {
      log.push(`topic error: ${t}: ${e.message}`);
    }
  }

  // ── Application subscriptions ────────────────────────────────────────────────
  const appSubs = [
    ['sns01', 'sqs01'],
    ['sns02', 'sqs02'],
    ['sns02', 'sqs03'],
    ['sns03', 'sqs04'],
    ['sns04', 'sqs05'],
  ];

  // ── Sentinel observation subscriptions ──────────────────────────────────────
  const sentinelSubs = [
    ['sns01', 'sentinel-a01-in'],
    ['sns02', 'sentinel-a01-out'],
    ['sns02', 'sentinel-a02-in'],
    ['sns03', 'sentinel-a02-out'],
    ['sns02', 'sentinel-a03-in'],
    ['sns04', 'sentinel-a03-out'],
    ['sns01', 'sentinel-a04-in'],
    ['sns04', 'sentinel-a04-out'],
  ];

  for (const [topic, queue] of [...appSubs, ...sentinelSubs]) {
    try {
      await ensureSubscription(getTopicArn(topic), queue);
      log.push(`sub ok: ${topic} → ${queue}`);
    } catch (e) {
      log.push(`sub error: ${topic} → ${queue}: ${e.message}`);
    }
  }

  const errors = log.filter(l => l.includes('error'));
  return { ok: errors.length === 0, log, errors };
}

module.exports = { setupTopology };
