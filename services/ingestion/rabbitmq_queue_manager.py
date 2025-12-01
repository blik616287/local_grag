"""
RabbitMQ queue management with guaranteed delivery
"""

import json
import logging
import time
from typing import Dict, Any, Optional, Callable

import pika
from pika.exceptions import AMQPConnectionError, AMQPChannelError

from config import (
    RABBITMQ_HOST, RABBITMQ_PORT, RABBITMQ_USER, RABBITMQ_PASSWORD,
    INGESTION_QUEUE, RETRY_QUEUE, DLQ_QUEUE
)

logger = logging.getLogger(__name__)


class RabbitMQManager:
    """
    RabbitMQ-based queue manager with:
    - Message acknowledgments (at-least-once delivery)
    - Dead letter exchanges for failed messages
    - Automatic retries with exponential backoff
    - Persistent messages
    """

    def __init__(self):
        """Initialize RabbitMQ connection"""
        logger.info(f"Connecting to RabbitMQ at {RABBITMQ_HOST}:{RABBITMQ_PORT}")

        credentials = pika.PlainCredentials(RABBITMQ_USER, RABBITMQ_PASSWORD)
        self.params = pika.ConnectionParameters(
            host=RABBITMQ_HOST,
            port=RABBITMQ_PORT,
            credentials=credentials,
            heartbeat=600,
            blocked_connection_timeout=300,
        )

        self.connection = None
        self.channel = None
        self._connect()
        self._setup_queues()

        logger.info("Successfully connected to RabbitMQ")

    def _connect(self):
        """Establish connection and channel"""
        try:
            self.connection = pika.BlockingConnection(self.params)
            self.channel = self.connection.channel()
        except AMQPConnectionError as e:
            logger.error(f"Failed to connect to RabbitMQ: {e}")
            raise

    def _setup_queues(self):
        """
        Set up queues and exchanges with dead letter handling
        """
        # Declare DLQ first (no DLX for DLQ itself)
        self.channel.queue_declare(
            queue=DLQ_QUEUE,
            durable=True,  # Survive broker restart
            arguments={}
        )

        # Declare main ingestion queue with DLX pointing to DLQ
        self.channel.queue_declare(
            queue=INGESTION_QUEUE,
            durable=True,
            arguments={
                'x-dead-letter-exchange': '',  # Default exchange
                'x-dead-letter-routing-key': DLQ_QUEUE
            }
        )

        # Declare retry queue with TTL and DLX back to main queue
        self.channel.queue_declare(
            queue=RETRY_QUEUE,
            durable=True,
            arguments={
                'x-message-ttl': 60000,  # 1 minute default TTL
                'x-dead-letter-exchange': '',
                'x-dead-letter-routing-key': INGESTION_QUEUE
            }
        )

        # Set QoS - prefetch only 1 message at a time per worker
        self.channel.basic_qos(prefetch_count=1)

        logger.info(f"Queues declared: {INGESTION_QUEUE}, {RETRY_QUEUE}, {DLQ_QUEUE}")

    def enqueue(self, queue_name: str, message: Dict[str, Any]):
        """
        Add message to queue with persistence

        Args:
            queue_name: Queue name
            message: Message data
        """
        try:
            message_json = json.dumps(message)

            self.channel.basic_publish(
                exchange='',
                routing_key=queue_name,
                body=message_json,
                properties=pika.BasicProperties(
                    delivery_mode=2,  # Persistent message
                    content_type='application/json',
                    timestamp=int(time.time())
                )
            )

            logger.debug(f"Enqueued message to {queue_name}: {message.get('doc_id', 'unknown')}")

        except (AMQPConnectionError, AMQPChannelError) as e:
            logger.error(f"Failed to enqueue message: {e}")
            self._reconnect()
            raise

    def consume(self, queue_name: str, callback: Callable, auto_ack: bool = False):
        """
        Start consuming messages from queue

        Args:
            queue_name: Queue name
            callback: Function to call for each message (message_dict, delivery_tag)
            auto_ack: Auto-acknowledge messages (not recommended)
        """
        def on_message(ch, method, properties, body):
            try:
                message = json.loads(body)
                logger.debug(f"Consumed message from {queue_name}: {message.get('doc_id', 'unknown')}")

                # Call user callback
                callback(message, method.delivery_tag)

            except json.JSONDecodeError as e:
                logger.error(f"Invalid JSON message: {e}")
                # Reject and don't requeue malformed messages
                ch.basic_nack(delivery_tag=method.delivery_tag, requeue=False)
            except Exception as e:
                logger.error(f"Error processing message: {e}")
                # Nack but requeue for retry
                ch.basic_nack(delivery_tag=method.delivery_tag, requeue=True)

        try:
            self.channel.basic_consume(
                queue=queue_name,
                on_message_callback=on_message,
                auto_ack=auto_ack
            )

            logger.info(f"Starting to consume from {queue_name}")
            self.channel.start_consuming()

        except KeyboardInterrupt:
            logger.info("Stopping consumer...")
            self.channel.stop_consuming()
        except (AMQPConnectionError, AMQPChannelError) as e:
            logger.error(f"Connection lost while consuming: {e}")
            self._reconnect()
            raise

    def ack(self, delivery_tag: int):
        """
        Acknowledge message processing

        Args:
            delivery_tag: Message delivery tag
        """
        try:
            self.channel.basic_ack(delivery_tag=delivery_tag)
        except (AMQPConnectionError, AMQPChannelError) as e:
            logger.error(f"Failed to ack message: {e}")
            self._reconnect()

    def nack(self, delivery_tag: int, requeue: bool = True):
        """
        Negative acknowledge (reject) message

        Args:
            delivery_tag: Message delivery tag
            requeue: Whether to requeue message
        """
        try:
            self.channel.basic_nack(delivery_tag=delivery_tag, requeue=requeue)
        except (AMQPConnectionError, AMQPChannelError) as e:
            logger.error(f"Failed to nack message: {e}")
            self._reconnect()

    def send_to_retry(self, message: Dict[str, Any], delay_seconds: int):
        """
        Send message to retry queue with delay

        Args:
            message: Message data
            delay_seconds: Delay before retry (will use queue TTL)
        """
        # Increment retry count
        message['retry_count'] = message.get('retry_count', 0) + 1
        message['retry_timestamp'] = time.time()

        # For RabbitMQ, we can set per-message TTL
        message_json = json.dumps(message)

        self.channel.basic_publish(
            exchange='',
            routing_key=RETRY_QUEUE,
            body=message_json,
            properties=pika.BasicProperties(
                delivery_mode=2,
                expiration=str(delay_seconds * 1000),  # milliseconds
                content_type='application/json'
            )
        )

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

    def get_queue_length(self, queue_name: str) -> int:
        """
        Get queue length

        Args:
            queue_name: Queue name

        Returns:
            Queue length
        """
        try:
            method = self.channel.queue_declare(queue=queue_name, durable=True, passive=True)
            return method.method.message_count
        except Exception as e:
            logger.error(f"Failed to get queue length: {e}")
            return 0

    def get_queue_stats(self) -> Dict[str, int]:
        """
        Get statistics for all queues

        Returns:
            Dictionary with queue lengths
        """
        return {
            "ingestion": self.get_queue_length(INGESTION_QUEUE),
            "retry": self.get_queue_length(RETRY_QUEUE),
            "dlq": self.get_queue_length(DLQ_QUEUE)
        }

    def _reconnect(self):
        """Reconnect to RabbitMQ"""
        logger.info("Attempting to reconnect to RabbitMQ...")
        try:
            if self.connection and not self.connection.is_closed:
                self.connection.close()
        except:
            pass

        self._connect()
        self._setup_queues()
        logger.info("Reconnected to RabbitMQ")

    def close(self):
        """Close connection"""
        try:
            if self.channel and self.channel.is_open:
                self.channel.close()
            if self.connection and self.connection.is_open:
                self.connection.close()
            logger.info("Closed RabbitMQ connection")
        except Exception as e:
            logger.error(f"Error closing connection: {e}")


# Backward compatibility alias
QueueManager = RabbitMQManager
