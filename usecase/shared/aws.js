const { SQSClient } = require('@aws-sdk/client-sqs');
const { SNSClient } = require('@aws-sdk/client-sns');

const LOCALSTACK_ENDPOINT = process.env.LOCALSTACK_ENDPOINT;
const AWS_REGION = process.env.AWS_REGION || 'us-east-1';
const AWS_ACCOUNT_ID = process.env.AWS_ACCOUNT_ID || '000000000000';

const config = LOCALSTACK_ENDPOINT
  ? {
      endpoint: LOCALSTACK_ENDPOINT,
      region: AWS_REGION,
      credentials: { accessKeyId: 'test', secretAccessKey: 'test' },
    }
  : { region: AWS_REGION };

const sqsClient = new SQSClient(config);
const snsClient = new SNSClient(config);

function getQueueUrl(name) {
  if (LOCALSTACK_ENDPOINT) {
    return `${LOCALSTACK_ENDPOINT}/${AWS_ACCOUNT_ID}/${name}`;
  }
  return `https://sqs.${AWS_REGION}.amazonaws.com/${AWS_ACCOUNT_ID}/${name}`;
}

function getTopicArn(name) {
  return `arn:aws:sns:${AWS_REGION}:${AWS_ACCOUNT_ID}:${name}`;
}

module.exports = { sqsClient, snsClient, getQueueUrl, getTopicArn };
