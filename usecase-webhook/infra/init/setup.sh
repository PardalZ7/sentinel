#!/bin/bash

export AWS_ACCESS_KEY_ID=test
export AWS_SECRET_ACCESS_KEY=test
export AWS_DEFAULT_REGION=us-east-1

ENDPOINT="http://localhost:4566"
ACCOUNT="000000000000"
REGION="us-east-1"

echo "Creating SNS topic..."

# Default topic where the webhook publishes events
OUTPUT_TOPIC_ARN=$(aws --endpoint-url=$ENDPOINT sns create-topic --topic-name ucwh-output --query TopicArn --output text)

echo "Creating SQS queues..."

# Engine input queue — subscribed to ucwh-output for Sentinel observation
aws --endpoint-url=$ENDPOINT sqs create-queue --queue-name ucwh-engine01-input

# Pairs queue — engine publishes correlated pairs here; agent consumes from it
aws --endpoint-url=$ENDPOINT sqs create-queue --queue-name ucwh-agent01-pairs

# Alarm queue
aws --endpoint-url=$ENDPOINT sqs create-queue --queue-name sentinel-alarms

echo "Subscribing engine input queue to ucwh-output topic..."

ENGINE01_QUEUE_ARN="arn:aws:sqs:${REGION}:${ACCOUNT}:ucwh-engine01-input"

aws --endpoint-url=$ENDPOINT sns subscribe \
    --topic-arn "$OUTPUT_TOPIC_ARN" \
    --protocol sqs \
    --notification-endpoint "$ENGINE01_QUEUE_ARN"

echo ""
echo "Setup complete!"
echo ""
echo "Webhook publishes to : arn:aws:sns:${REGION}:${ACCOUNT}:ucwh-output"
echo "Engine01 input       : ucwh-engine01-input"
echo "Agent01 pairs        : ucwh-agent01-pairs"
echo "Alarm queue          : sentinel-alarms"
