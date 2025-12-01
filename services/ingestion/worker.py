"""
Main ingestion worker
Processes documents from Redis queue and builds knowledge graph
"""

import logging
import time
import signal
import sys
from typing import Dict, Any
from datetime import datetime

import psycopg2
from psycopg2.extras import RealDictCursor

from config import (
    POSTGRES_URI, MAX_RETRIES, QUEUE_POLL_TIMEOUT,
    INGESTION_QUEUE, LOG_LEVEL
)
from rabbitmq_queue_manager import RabbitMQManager
from tenant_manager import TenantManager
from document_processor import DocumentProcessor
from entity_extractor import EntityExtractor
from graph_builder import GraphBuilder

# Configure logging
logging.basicConfig(
    level=getattr(logging, LOG_LEVEL),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class IngestionWorker:
    """
    Main worker for processing ingestion tasks
    """

    def __init__(self):
        """Initialize worker components"""
        logger.info("Initializing IngestionWorker")

        self.running = True
        self.db_conn = None

        # Initialize components
        try:
            self.queue_manager = RabbitMQManager()
            self.tenant_manager = TenantManager()
            self.document_processor = DocumentProcessor()

            # Initialize entity extractor (will be available when vLLM starts)
            try:
                self.entity_extractor = EntityExtractor()
                logger.info("Entity extractor initialized successfully")
            except Exception as e:
                logger.warning(f"Entity extractor initialization failed (vLLM may not be ready): {e}")
                self.entity_extractor = None

            self.graph_builder = GraphBuilder()
            self.db_conn = psycopg2.connect(POSTGRES_URI)

            logger.info("All components initialized successfully")

        except Exception as e:
            logger.error(f"Failed to initialize worker: {e}")
            raise

        # Set up signal handlers for graceful shutdown
        signal.signal(signal.SIGTERM, self._handle_shutdown)
        signal.signal(signal.SIGINT, self._handle_shutdown)

    def _handle_shutdown(self, signum, frame):
        """Handle graceful shutdown"""
        logger.info(f"Received shutdown signal: {signum}")
        self.running = False

    def close(self):
        """Clean up resources"""
        logger.info("Cleaning up worker resources")

        if self.graph_builder:
            self.graph_builder.close()

        if self.tenant_manager:
            self.tenant_manager.close()

        if self.db_conn:
            self.db_conn.close()

    def track_document_status(
        self,
        doc_id: str,
        tenant_id: str,
        status: str,
        error_message: str = None,
        processing_time: int = None
    ):
        """
        Update document status in database

        Args:
            doc_id: Document ID
            tenant_id: Tenant ID
            status: Status (pending, processing, completed, failed)
            error_message: Error message if failed
            processing_time: Processing time in seconds
        """
        with self.db_conn.cursor() as cur:
            try:
                if status == "processing":
                    cur.execute("""
                        INSERT INTO documents (document_id, tenant_id, url, status, created_at)
                        VALUES (%s, %s, %s, %s, %s)
                        ON CONFLICT (document_id) DO UPDATE
                        SET status = %s, updated_at = %s
                    """, (
                        doc_id, tenant_id, "", status, datetime.utcnow(),
                        status, datetime.utcnow()
                    ))
                elif status == "completed":
                    cur.execute("""
                        UPDATE documents
                        SET status = %s,
                            completed_at = %s,
                            processing_time_seconds = %s,
                            updated_at = %s
                        WHERE document_id = %s
                    """, (status, datetime.utcnow(), processing_time, datetime.utcnow(), doc_id))
                elif status == "failed":
                    cur.execute("""
                        UPDATE documents
                        SET status = %s,
                            error_message = %s,
                            updated_at = %s,
                            retry_count = retry_count + 1
                        WHERE document_id = %s
                    """, (status, error_message, datetime.utcnow(), doc_id))

                self.db_conn.commit()

            except Exception as e:
                logger.error(f"Failed to track document status: {e}")
                self.db_conn.rollback()

    def process_message(self, message: Dict[str, Any], delivery_tag: int = None) -> bool:
        """
        Process a single ingestion message

        Args:
            message: Message data
            delivery_tag: RabbitMQ delivery tag for acknowledgment

        Returns:
            True if successful, False otherwise
        """
        doc_id = message.get('doc_id', 'unknown')
        url = message.get('url')
        retry_count = message.get('retry_count', 0)

        logger.info(f"Processing document: {doc_id} (retry: {retry_count})")

        start_time = time.time()

        try:
            # Determine tenant ID
            tenant_id = self.tenant_manager.determine_tenant_id(message)
            logger.info(f"Document {doc_id} assigned to tenant {tenant_id}")

            # Update status to processing
            self.track_document_status(doc_id, tenant_id, "processing")

            # Load and chunk document
            logger.info(f"Loading document from {url}")
            chunks = self.document_processor.process_url(
                url=url,
                doc_id=doc_id,
                metadata=message.get('metadata', {})
            )

            if not chunks:
                raise ValueError("No chunks generated from document")

            logger.info(f"Generated {len(chunks)} chunks")

            # Extract entities from chunks (skip if no extractor)
            if self.entity_extractor:
                logger.info("Extracting entities...")
                texts = [chunk.page_content for chunk in chunks]
                entities_per_chunk = self.entity_extractor.extract_batch(texts)
                total_entities = sum(len(entities) for entities in entities_per_chunk)
                logger.info(f"Extracted {total_entities} entities")
            else:
                logger.info("Skipping entity extraction (not configured)")
                entities_per_chunk = [[] for _ in chunks]
                total_entities = 0

            # Build graph
            logger.info("Building knowledge graph...")
            self.graph_builder.process_document_chunks(
                tenant_id=tenant_id,
                doc_id=doc_id,
                url=url,
                chunks=chunks
            )

            # Link entities to chunks
            logger.info("Linking entities to chunks...")
            self.graph_builder.extract_and_link_entities(
                tenant_id=tenant_id,
                chunks=chunks,
                entities_per_chunk=entities_per_chunk
            )

            # Get final stats
            stats = self.graph_builder.get_tenant_stats(tenant_id)
            logger.info(f"Tenant stats: {stats}")

            # Update tenant stats
            self.tenant_manager.update_tenant_stats(tenant_id, stats)

            # Mark as completed
            processing_time = int(time.time() - start_time)
            self.track_document_status(
                doc_id, tenant_id, "completed",
                processing_time=processing_time
            )

            logger.info(
                f"Successfully processed document {doc_id} in {processing_time}s "
                f"({len(chunks)} chunks, {total_entities} entities)"
            )

            # Acknowledge message
            if delivery_tag is not None:
                self.queue_manager.ack(delivery_tag)

            return True

        except Exception as e:
            processing_time = int(time.time() - start_time)
            error_msg = str(e)
            logger.error(f"Failed to process document {doc_id}: {error_msg}")

            # Determine tenant ID for tracking (use cached or from message)
            try:
                tenant_id = self.tenant_manager.determine_tenant_id(message)
            except:
                tenant_id = "unknown"

            # Update status to failed
            self.track_document_status(doc_id, tenant_id, "failed", error_message=error_msg)

            # Handle retry
            if retry_count < MAX_RETRIES:
                # Exponential backoff: 2^retry_count minutes
                delay = (2 ** retry_count) * 60
                message['retry_count'] = retry_count + 1
                self.queue_manager.send_to_retry(message, delay)
                logger.info(f"Scheduled retry {retry_count + 1} for {doc_id} in {delay}s")
                # Acknowledge original message (it's now in retry queue)
                if delivery_tag is not None:
                    self.queue_manager.ack(delivery_tag)
            else:
                # Send to DLQ
                self.queue_manager.send_to_dlq(message, error_msg)
                logger.error(f"Max retries exceeded for {doc_id}, sent to DLQ")
                # Acknowledge original message (it's now in DLQ)
                if delivery_tag is not None:
                    self.queue_manager.ack(delivery_tag)

            return False

    def run(self):
        """
        Main worker loop - consumes from RabbitMQ
        """
        logger.info("Starting ingestion worker with RabbitMQ")
        logger.info(f"Consuming from queue: {INGESTION_QUEUE}")

        # Log initial queue stats
        stats = self.queue_manager.get_queue_stats()
        logger.info(f"Initial queue stats: {stats}")

        try:
            # Start consuming - this blocks until interrupted
            self.queue_manager.consume(
                queue_name=INGESTION_QUEUE,
                callback=self.process_message,
                auto_ack=False  # Manual acknowledgment for reliability
            )
        except KeyboardInterrupt:
            logger.info("Received keyboard interrupt")
        except Exception as e:
            logger.error(f"Worker error: {e}", exc_info=True)

        logger.info("Worker shutting down")
        self.close()


def main():
    """
    Main entry point
    """
    logger.info("Starting GraphRAG Ingestion Worker")
    logger.info("="*50)

    worker = IngestionWorker()

    try:
        worker.run()
    except Exception as e:
        logger.error(f"Worker failed: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
