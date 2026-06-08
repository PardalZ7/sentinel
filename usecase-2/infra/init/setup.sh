#!/bin/bash

export AWS_ACCESS_KEY_ID=test
export AWS_SECRET_ACCESS_KEY=test
export AWS_DEFAULT_REGION=us-east-1

ENDPOINT="http://localhost:4566"
ACCOUNT="000000000000"
REGION="us-east-1"

echo "Creating SNS topic..."

# app01 publishes all Pipefy automation payloads here
INPUT_TOPIC_ARN=$(aws --endpoint-url=$ENDPOINT sns create-topic --topic-name uc2-input --query TopicArn --output text)

echo "Creating SQS queues..."

# One input queue per correlation engine (fan-out from uc2-input topic)
aws --endpoint-url=$ENDPOINT sqs create-queue --queue-name uc2-engine01-input
aws --endpoint-url=$ENDPOINT sqs create-queue --queue-name uc2-engine02-input

# Pairs queues — engine publishes correlated pairs here; agents consume from these
aws --endpoint-url=$ENDPOINT sqs create-queue --queue-name uc2-agent01-pairs
aws --endpoint-url=$ENDPOINT sqs create-queue --queue-name uc2-agent02-pairs

# SNS topics for pairs fan-out (one per agent)
AGENT01_TOPIC_ARN=$(aws --endpoint-url=$ENDPOINT sns create-topic --topic-name uc2-agent01-pairs-topic --query TopicArn --output text)
AGENT02_TOPIC_ARN=$(aws --endpoint-url=$ENDPOINT sns create-topic --topic-name uc2-agent02-pairs-topic --query TopicArn --output text)

# Alarm queue
aws --endpoint-url=$ENDPOINT sqs create-queue --queue-name sentinel-alarms

echo "Subscribing engine input queues to uc2-input topic..."

ENGINE01_QUEUE_ARN="arn:aws:sqs:${REGION}:${ACCOUNT}:uc2-engine01-input"
ENGINE02_QUEUE_ARN="arn:aws:sqs:${REGION}:${ACCOUNT}:uc2-engine02-input"

aws --endpoint-url=$ENDPOINT sns subscribe \
    --topic-arn "$INPUT_TOPIC_ARN" \
    --protocol sqs \
    --notification-endpoint "$ENGINE01_QUEUE_ARN"

aws --endpoint-url=$ENDPOINT sns subscribe \
    --topic-arn "$INPUT_TOPIC_ARN" \
    --protocol sqs \
    --notification-endpoint "$ENGINE02_QUEUE_ARN"

echo "Subscribing pairs queues to their respective topics..."

AGENT01_PAIRS_ARN="arn:aws:sqs:${REGION}:${ACCOUNT}:uc2-agent01-pairs"
AGENT02_PAIRS_ARN="arn:aws:sqs:${REGION}:${ACCOUNT}:uc2-agent02-pairs"

aws --endpoint-url=$ENDPOINT sns subscribe \
    --topic-arn "$AGENT01_TOPIC_ARN" \
    --protocol sqs \
    --notification-endpoint "$AGENT01_PAIRS_ARN"

aws --endpoint-url=$ENDPOINT sns subscribe \
    --topic-arn "$AGENT02_TOPIC_ARN" \
    --protocol sqs \
    --notification-endpoint "$AGENT02_PAIRS_ARN"

echo ""
echo "Setup complete!"
echo ""
echo "app01 publishes to  : arn:aws:sns:${REGION}:${ACCOUNT}:uc2-input"
echo "Engine01 input      : uc2-engine01-input"
echo "Engine02 input      : uc2-engine02-input"
echo "Agent01 pairs       : uc2-agent01-pairs  (via uc2-agent01-pairs-topic)"
echo "Agent02 pairs       : uc2-agent02-pairs  (via uc2-agent02-pairs-topic)"
echo "Alarm queue         : sentinel-alarms"
