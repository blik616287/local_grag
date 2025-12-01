"""
Configuration management for ingestion service
"""

import os

# Neo4j Configuration
NEO4J_URI = os.environ.get("NEO4J_URI", "bolt://neo4j:7687")
NEO4J_USER = os.environ.get("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.environ.get("NEO4J_PASSWORD", "graphrag123")

# PostgreSQL Configuration
POSTGRES_URI = os.environ.get("POSTGRES_URI", "postgresql://graphrag:graphrag123@postgres:5432/graphrag")

# Redis Configuration
REDIS_HOST = os.environ.get("REDIS_HOST", "redis")
REDIS_PORT = int(os.environ.get("REDIS_PORT", "6379"))

# RabbitMQ Configuration
RABBITMQ_HOST = os.environ.get("RABBITMQ_HOST", "rabbitmq")
RABBITMQ_PORT = int(os.environ.get("RABBITMQ_PORT", "5672"))
RABBITMQ_USER = os.environ.get("RABBITMQ_USER", "graphrag")
RABBITMQ_PASSWORD = os.environ.get("RABBITMQ_PASSWORD", "graphrag123")

# Inference Endpoints
VLLM_ENDPOINT = os.environ.get("VLLM_ENDPOINT", "http://vllm:8000/v1")
EMBEDDINGS_ENDPOINT = os.environ.get("EMBEDDINGS_ENDPOINT", "http://embeddings:8001")

# Model Configuration
LLM_MODEL = os.environ.get("LLM_MODEL", "Qwen/Qwen2.5-1.5B-Instruct")
EMBEDDING_MODEL = os.environ.get("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
EMBEDDING_DIMENSIONS = int(os.environ.get("EMBEDDING_DIMENSIONS", "384"))

# Worker Configuration
WORKER_CONCURRENCY = int(os.environ.get("WORKER_CONCURRENCY", "4"))
BATCH_SIZE = int(os.environ.get("BATCH_SIZE", "10"))
MAX_RETRIES = int(os.environ.get("MAX_RETRIES", "3"))
EXTRACTION_DIR = os.environ.get("EXTRACTION_DIR", "/tmp/extraction")

# Multi-tenant Configuration
MULTI_TENANT_MODE = os.environ.get("MULTI_TENANT_MODE", "true").lower() == "true"
TENANT_ISOLATION_MODE = os.environ.get("TENANT_ISOLATION_MODE", "property")
TENANT_ID_STRATEGY = os.environ.get("TENANT_ID_STRATEGY", "dynamic")
TENANT_ID_METADATA_FIELD = os.environ.get("TENANT_ID_METADATA_FIELD", "collection_name")

# Processing Configuration
MAX_DOCUMENT_SIZE_MB = int(os.environ.get("MAX_DOCUMENT_SIZE_MB", "100"))
CHUNK_SIZE = int(os.environ.get("CHUNK_SIZE", "1000"))
CHUNK_OVERLAP = int(os.environ.get("CHUNK_OVERLAP", "200"))

# Queue Names
INGESTION_QUEUE = "ingestion_queue"
RETRY_QUEUE = "ingestion_queue:retry"
DLQ_QUEUE = "ingestion_queue:dlq"

# Timeouts
PROCESSING_TIMEOUT = 300  # 5 minutes
QUEUE_POLL_TIMEOUT = 10  # 10 seconds

# Logging
LOG_LEVEL = os.environ.get("LOG_LEVEL", "INFO")
