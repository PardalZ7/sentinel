#!/bin/bash

export AWS_ACCESS_KEY_ID=test
export AWS_SECRET_ACCESS_KEY=test
export AWS_DEFAULT_REGION=us-east-1

ENDPOINT="http://localhost:4566"
ACCOUNT="000000000000"
REGION="us-east-1"

echo "Creating SQS queues..."
aws --endpoint-url=$ENDPOINT sqs create-queue --queue-name sqs01
aws --endpoint-url=$ENDPOINT sqs create-queue --queue-name sqs02
aws --endpoint-url=$ENDPOINT sqs create-queue --queue-name sqs03
aws --endpoint-url=$ENDPOINT sqs create-queue --queue-name sqs04
aws --endpoint-url=$ENDPOINT sqs create-queue --queue-name sqs05

echo "Creating SNS topics..."
aws --endpoint-url=$ENDPOINT sns create-topic --name sns01
aws --endpoint-url=$ENDPOINT sns create-topic --name sns02
aws --endpoint-url=$ENDPOINT sns create-topic --name sns03
aws --endpoint-url=$ENDPOINT sns create-topic --name sns04

echo "Subscribing application queues to topics..."

SNS01_ARN="arn:aws:sns:$REGION:$ACCOUNT:sns01"
SNS02_ARN="arn:aws:sns:$REGION:$ACCOUNT:sns02"
SNS03_ARN="arn:aws:sns:$REGION:$ACCOUNT:sns03"
SNS04_ARN="arn:aws:sns:$REGION:$ACCOUNT:sns04"

# sns01 → sqs01
aws --endpoint-url=$ENDPOINT sns subscribe \
  --topic-arn "$SNS01_ARN" \
  --protocol sqs \
  --notification-endpoint "arn:aws:sqs:$REGION:$ACCOUNT:sqs01"

# sns02 → sqs02
aws --endpoint-url=$ENDPOINT sns subscribe \
  --topic-arn "$SNS02_ARN" \
  --protocol sqs \
  --notification-endpoint "arn:aws:sqs:$REGION:$ACCOUNT:sqs02"

# sns02 → sqs03
aws --endpoint-url=$ENDPOINT sns subscribe \
  --topic-arn "$SNS02_ARN" \
  --protocol sqs \
  --notification-endpoint "arn:aws:sqs:$REGION:$ACCOUNT:sqs03"

# sns03 → sqs04
aws --endpoint-url=$ENDPOINT sns subscribe \
  --topic-arn "$SNS03_ARN" \
  --protocol sqs \
  --notification-endpoint "arn:aws:sqs:$REGION:$ACCOUNT:sqs04"

# sns04 → sqs05
aws --endpoint-url=$ENDPOINT sns subscribe \
  --topic-arn "$SNS04_ARN" \
  --protocol sqs \
  --notification-endpoint "arn:aws:sqs:$REGION:$ACCOUNT:sqs05"

# ─── Sentinel parallel observation queues ─────────────────────────────────────
# These queues are used EXCLUSIVELY by Sentinel. They are additional SNS
# subscribers, so the application queues are never affected.
echo ""
echo "Creating Sentinel observation queues..."

# agent01 monitors APP02: observes SNS01 (input) and SNS02 (output)
aws --endpoint-url=$ENDPOINT sqs create-queue --queue-name sentinel-a01-in
aws --endpoint-url=$ENDPOINT sqs create-queue --queue-name sentinel-a01-out
aws --endpoint-url=$ENDPOINT sns subscribe \
  --topic-arn "$SNS01_ARN" \
  --protocol sqs \
  --notification-endpoint "arn:aws:sqs:$REGION:$ACCOUNT:sentinel-a01-in"
aws --endpoint-url=$ENDPOINT sns subscribe \
  --topic-arn "$SNS02_ARN" \
  --protocol sqs \
  --notification-endpoint "arn:aws:sqs:$REGION:$ACCOUNT:sentinel-a01-out"

# agent02 monitors APP03: observes SNS02 (input) and SNS03 (output/batch)
aws --endpoint-url=$ENDPOINT sqs create-queue --queue-name sentinel-a02-in
aws --endpoint-url=$ENDPOINT sqs create-queue --queue-name sentinel-a02-out
aws --endpoint-url=$ENDPOINT sns subscribe \
  --topic-arn "$SNS02_ARN" \
  --protocol sqs \
  --notification-endpoint "arn:aws:sqs:$REGION:$ACCOUNT:sentinel-a02-in"
aws --endpoint-url=$ENDPOINT sns subscribe \
  --topic-arn "$SNS03_ARN" \
  --protocol sqs \
  --notification-endpoint "arn:aws:sqs:$REGION:$ACCOUNT:sentinel-a02-out"

# agent03 monitors APP04 path from SNS02: observes SNS02 (input) and SNS04 (output)
aws --endpoint-url=$ENDPOINT sqs create-queue --queue-name sentinel-a03-in
aws --endpoint-url=$ENDPOINT sqs create-queue --queue-name sentinel-a03-out
aws --endpoint-url=$ENDPOINT sns subscribe \
  --topic-arn "$SNS02_ARN" \
  --protocol sqs \
  --notification-endpoint "arn:aws:sqs:$REGION:$ACCOUNT:sentinel-a03-in"
aws --endpoint-url=$ENDPOINT sns subscribe \
  --topic-arn "$SNS04_ARN" \
  --protocol sqs \
  --notification-endpoint "arn:aws:sqs:$REGION:$ACCOUNT:sentinel-a03-out"

# agent04 monitors full pipeline from APP01 to APP04: observes SNS01 (input) and SNS04 (output)
aws --endpoint-url=$ENDPOINT sqs create-queue --queue-name sentinel-a04-in
aws --endpoint-url=$ENDPOINT sqs create-queue --queue-name sentinel-a04-out
aws --endpoint-url=$ENDPOINT sns subscribe \
  --topic-arn "$SNS01_ARN" \
  --protocol sqs \
  --notification-endpoint "arn:aws:sqs:$REGION:$ACCOUNT:sentinel-a04-in"
aws --endpoint-url=$ENDPOINT sns subscribe \
  --topic-arn "$SNS04_ARN" \
  --protocol sqs \
  --notification-endpoint "arn:aws:sqs:$REGION:$ACCOUNT:sentinel-a04-out"

echo ""
echo "Setup complete!"
echo ""
echo "Application queues: sqs01-05, sns01-04"
echo "Sentinel queues: sentinel-a01-in/out, sentinel-a02-in/out, sentinel-a03-in/out, sentinel-a04-in/out"
