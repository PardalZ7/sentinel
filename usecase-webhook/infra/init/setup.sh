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

# Engine01 input queue — receives SUCCESS messages (status=finished) from webhook
aws --endpoint-url=$ENDPOINT sqs create-queue --queue-name uc2-engine01-input

# Engine02 input queue — receives FAILURE messages (status != finished) from webhook
aws --endpoint-url=$ENDPOINT sqs create-queue --queue-name uc2-engine02-input

# Agent pairs queues — engines publish correlated pairs here; agents consume from these
aws --endpoint-url=$ENDPOINT sqs create-queue --queue-name uc2-agent01-pairs

aws --endpoint-url=$ENDPOINT sqs create-queue --queue-name uc2-agent02-pairs

# Alarm queue
aws --endpoint-url=$ENDPOINT sqs create-queue --queue-name sentinel-alarms

echo "Subscribing engine input queues to ucwh-output topic with filters..."

ENGINE01_QUEUE_ARN="arn:aws:sqs:${REGION}:${ACCOUNT}:uc2-engine01-input"
ENGINE02_QUEUE_ARN="arn:aws:sqs:${REGION}:${ACCOUNT}:uc2-engine02-input"

# Subscribe engine01 to SUCCESS messages only (status=finished)
aws --endpoint-url=$ENDPOINT sns subscribe \
    --topic-arn "$OUTPUT_TOPIC_ARN" \
    --protocol sqs \
    --notification-endpoint "$ENGINE01_QUEUE_ARN" \
    --attributes "FilterPolicy={\"status\":[\"finished\"]}"

# Subscribe engine02 to FAILURE messages (any status that is NOT "finished")
# Uses SNS FilterPolicy "anything-but" operator
aws --endpoint-url=$ENDPOINT sns subscribe \
    --topic-arn "$OUTPUT_TOPIC_ARN" \
    --protocol sqs \
    --notification-endpoint "$ENGINE02_QUEUE_ARN" \
    --attributes "FilterPolicy={\"status\":[{\"anything-but\":\"finished\"}]}"

echo ""
echo "Setup complete!"
echo ""
echo "Webhook publishes to       : arn:aws:sns:${REGION}:${ACCOUNT}:ucwh-output"
echo "Engine01 input (SUCCESS)   : uc2-engine01-input   (filters: status=finished)"
echo "Engine02 input (FAILURE)   : uc2-engine02-input   (filters: status!=finished)"
echo "Agent01 pairs              : uc2-agent01-pairs"
echo "Agent02 pairs              : uc2-agent02-pairs"
echo "Alarm queue                : sentinel-alarms"
