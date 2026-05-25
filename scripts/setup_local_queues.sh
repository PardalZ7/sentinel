#!/usr/bin/env bash
# Creates all SQS queues and SNS topics in LocalStack.
# Works both as a LocalStack init script (/etc/localstack/init/ready.d/) and
# as a standalone helper run from the host (ENDPOINT defaults to localhost:4566).
set -e

ENDPOINT="${LOCALSTACK_ENDPOINT:-http://localhost:4566}"
REGION="us-east-1"
ACCOUNT="000000000000"
AWS="aws --endpoint-url=$ENDPOINT --region=$REGION"

echo "Waiting for LocalStack..."
until $AWS sqs list-queues &>/dev/null; do sleep 1; done
echo "LocalStack ready."

# --- Sentinel-internal queues ---
SENTINEL_QUEUES=(
  sentinel-a01-in
  sentinel-a01-out
  sentinel-a02-in
  sentinel-a02-out
  sentinel-a03-in
  sentinel-a03-out
  sentinel-a04-in
  sentinel-a04-out
  sentinel-a01-pairs
  sentinel-a02-pairs
  sentinel-a03-pairs
  sentinel-a04-pairs
  sentinel-alarms
)

for q in "${SENTINEL_QUEUES[@]}"; do
  $AWS sqs create-queue --queue-name "$q" --output text --query 'QueueUrl' \
    && echo "  created queue: $q" || echo "  already exists: $q"
done

# --- App-to-app queues (consumed by app02, app03, app04) ---
APP_QUEUES=(sqs01 sqs02 sqs03)

for q in "${APP_QUEUES[@]}"; do
  $AWS sqs create-queue --queue-name "$q" --output text --query 'QueueUrl' \
    && echo "  created queue: $q" || echo "  already exists: $q"
done

# --- SNS topics (published by app01-04) ---
SNS_TOPICS=(sns01 sns02 sns03 sns04)

for t in "${SNS_TOPICS[@]}"; do
  $AWS sns create-topic --name "$t" --output text --query 'TopicArn' \
    && echo "  created topic: $t" || echo "  already exists: $t"
done

# --- Subscribe SQS queues to SNS topics ---
# sns01 -> sqs01 (app01 publishes, app02 consumes)
# sns02 -> sqs02 (app02 publishes, app03 consumes)
# sns03 -> sqs03 (app03 publishes, app04 consumes)
subscribe() {
  local topic="$1"
  local queue="$2"
  local topic_arn="arn:aws:sns:$REGION:$ACCOUNT:$topic"
  local queue_arn="arn:aws:sqs:$REGION:$ACCOUNT:$queue"
  local queue_url="$ENDPOINT/$ACCOUNT/$queue"

  $AWS sns subscribe \
    --topic-arn "$topic_arn" \
    --protocol sqs \
    --notification-endpoint "$queue_arn" \
    --output text --query 'SubscriptionArn' \
    && echo "  subscribed $queue to $topic" || echo "  subscription exists: $topic -> $queue"

  # Allow SNS to send messages to the SQS queue
  $AWS sqs set-queue-attributes \
    --queue-url "$queue_url" \
    --attributes "{\"Policy\":\"{\\\"Version\\\":\\\"2012-10-17\\\",\\\"Statement\\\":[{\\\"Effect\\\":\\\"Allow\\\",\\\"Principal\\\":{\\\"Service\\\":\\\"sns.amazonaws.com\\\"},\\\"Action\\\":\\\"sqs:SendMessage\\\",\\\"Resource\\\":\\\"$queue_arn\\\"}]}\"}" \
    2>/dev/null || true
}

subscribe sns01 sqs01
subscribe sns02 sqs02
subscribe sns03 sqs03

echo "Done. All queues and topics ready."
