const { SQSClient } = require('@aws-sdk/client-sqs');
const { SNSClient } = require('@aws-sdk/client-sns');

const LOCALSTACK_ENDPOINT = process.env.LOCALSTACK_ENDPOINT || 'http://localhost:4566';

const config = {
  endpoint: LOCALSTACK_ENDPOINT,
  region: 'us-east-1',
  credentials: {
    accessKeyId: 'test',
    secretAccessKey: 'test',
  },
};

const sqsClient = new SQSClient(config);
const snsClient = new SNSClient(config);

function getQueueUrl(name) {
  return `${LOCALSTACK_ENDPOINT}/000000000000/${name}`;
}

function getTopicArn(name) {
  return `arn:aws:sns:us-east-1:000000000000:${name}`;
}

module.exports = { sqsClient, snsClient, getQueueUrl, getTopicArn };
