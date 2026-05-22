const net = require('net');
const { sqsClient, getQueueUrl } = require('./aws');
const { PurgeQueueCommand } = require('@aws-sdk/client-sqs');

const USECASE_QUEUES = ['sqs01', 'sqs02', 'sqs03', 'sqs04', 'sqs05'];

const SENTINEL_QUEUES = [
  'sentinel-a01-in', 'sentinel-a01-out',
  'sentinel-a02-in', 'sentinel-a02-out',
  'sentinel-a03-in', 'sentinel-a03-out',
  'sentinel-a04-in', 'sentinel-a04-out',
  'sentinel-alarms',
];

function flushRedis(host = 'localhost', port = 6379) {
  return new Promise((resolve, reject) => {
    const socket = net.createConnection(port, host);
    socket.setTimeout(3000);
    let response = '';
    socket.on('connect', () => socket.write('*1\r\n$7\r\nFLUSHDB\r\n'));
    socket.on('data', d => { response += d.toString(); socket.destroy(); });
    socket.on('close', () => resolve(response.trim()));
    socket.on('error', reject);
    socket.on('timeout', () => { socket.destroy(); reject(new Error('Redis timeout')); });
  });
}

async function purgeQueues(queues) {
  const errors = [];
  await Promise.all(queues.map(async (name) => {
    try {
      await sqsClient.send(new PurgeQueueCommand({ QueueUrl: getQueueUrl(name) }));
    } catch (err) {
      errors.push(`purge ${name}: ${err.message}`);
    }
  }));
  return errors;
}

/**
 * Resets all Sentinel state:
 * - Purges usecase queues (sqs01-05) — clears pipeline backlog
 * - Purges Sentinel observation queues (sentinel-a0{1-4}-in/out, sentinel-alarms)
 * - Flushes Redis DB — removes all pending correlation keys
 *
 * Use this when the producer rate overwhelmed Sentinel and timeout-driven
 * false positives are corrupting the training data. Stop APP01 first,
 * reset, then restart with a lower intervalMs.
 *
 * @param {{ redis?: { host: string, port: number } }} [opts]
 * @returns {Promise<{ ok: boolean, errors: string[] }>}
 */
async function resetSentinel(opts = {}) {
  const redisOpts = opts.redis || { host: 'localhost', port: 6379 };
  const errors = [];

  const [queueErrors] = await Promise.all([
    purgeQueues([...USECASE_QUEUES, ...SENTINEL_QUEUES]),
    flushRedis(redisOpts.host, redisOpts.port).catch(e => errors.push(`redis: ${e.message}`)),
  ]);

  errors.push(...queueErrors);
  return { ok: errors.length === 0, errors };
}

module.exports = { resetSentinel, USECASE_QUEUES, SENTINEL_QUEUES };
