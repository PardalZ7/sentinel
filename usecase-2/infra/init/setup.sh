#!/bin/bash

export AWS_ACCESS_KEY_ID=test
export AWS_SECRET_ACCESS_KEY=test
export AWS_DEFAULT_REGION=us-east-1

ENDPOINT="http://localhost:4566"
ACCOUNT="000000000000"
REGION="us-east-1"

echo "Creating SQS queues..."

# Application queue — app01 publishes directly here
aws --endpoint-url=$ENDPOINT sqs create-queue --queue-name input_data_sample

# Sentinel queues
aws --endpoint-url=$ENDPOINT sqs create-queue --queue-name input_data_sample_pairs
aws --endpoint-url=$ENDPOINT sqs create-queue --queue-name sentinel-alarms

echo ""
echo "Setup complete!"
echo ""
echo "Application queue : input_data_sample"
echo "Sentinel queues   : input_data_sample_pairs, sentinel-alarms"
