"""
Redis queue management
"""

import json
import logging
import time
from typing import Dict, Any, Optional

import redis

from config import REDIS_HOST, REDIS_PORT, INGESTION_QUEUE, RETRY_QUEUE, DLQ_QUEUE

logger = logging.getLogger(__name__)


class QueueManager:
    """
    Redis-based queue manager with retry and DLQ support
    """

    def __init__(self):
        """Initialize Redis connection"""
        logger.info(f"Connecting to Redis at {REDIS_HOST}:{REDIS_PORT}")
        self.client = redis.Redis(
            host=REDIS_HOST,
            port=REDIS_PORT,
            decode_responses=True,
            socket_timeout=5,
            socket_connect_timeout=5
        )

        # Test connection
        self.client.ping()
        logger.info("Successfully connected to Redis")

    def enqueue(self, queue_name: str, message: Dict[str, Any]):
        """
        Add message to queue

        Args:
            queue_name: Queue name
            message: Message data
        """
        message_json = json.dumps(message)
        self.client.lpush(queue_name, message_json)
        logger.debug(f"Enqueued message to {queue_name}: {message.get('doc_id', 'unknown')}")

    def dequeue(self, queue_name: str, timeout: int = 10) -> Optional[Dict[str, Any]]:
        """
        Blocking dequeue from queue

        Args:
            queue_name: Queue name
            timeout: Timeout in seconds

        Returns:
            Message data or None if timeout
        """
        result = self.client.brpop(queue_name, timeout=timeout)

        if result:
            _, message_json = result
            message = json.loads(message_json)
            logger.debug(f"Dequeued message from {queue_name}: {message.get('doc_id', 'unknown')}")
            return message

        return None

    def send_to_retry(self, message: Dict[str, Any], delay_seconds: int):
        """
        Schedule message for retry with delay

        Args:
            message: Message data
            delay_seconds: Delay before retry
        """
        score = time.time() + delay_seconds
        message_json = json.dumps(message)
        self.client.zadd(RETRY_QUEUE, {message_json: score})
        logger.info(f"Scheduled retry for {message.get('doc_id')} in {delay_seconds}s")

    def send_to_dlq(self, message: Dict[str, Any], error: str):
        """
        Send message to dead letter queue

        Args:
            message: Message data
            error: Error message
        """
        message['error'] = error
        message['dlq_timestamp'] = time.time()
        self.enqueue(DLQ_QUEUE, message)
        logger.warning(f"Sent message to DLQ: {message.get('doc_id')}")

    def process_retry_queue(self):
        """
        Process retry queue and move ready messages to main queue
        """
        current_time = time.time()

        # Get all messages ready for retry
        messages = self.client.zrangebyscore(
            RETRY_QUEUE,
            min=0,
            max=current_time,
            withscores=False
        )

        for message_json in messages:
            # Move to main queue
            message = json.loads(message_json)
            self.enqueue(INGESTION_QUEUE, message)

            # Remove from retry queue
            self.client.zrem(RETRY_QUEUE, message_json)

            logger.info(f"Moved message from retry queue: {message.get('doc_id')}")

    def get_queue_length(self, queue_name: str) -> int:
        """
        Get queue length

        Args:
            queue_name: Queue name

        Returns:
            Queue length
        """
        return self.client.llen(queue_name)

    def get_retry_queue_length(self) -> int:
        """
        Get retry queue length

        Returns:
            Retry queue length
        """
        return self.client.zcard(RETRY_QUEUE)

    def get_queue_stats(self) -> Dict[str, int]:
        """
        Get statistics for all queues

        Returns:
            Dictionary with queue lengths
        """
        return {
            "ingestion": self.get_queue_length(INGESTION_QUEUE),
            "retry": self.get_retry_queue_length(),
            "dlq": self.get_queue_length(DLQ_QUEUE)
        }
