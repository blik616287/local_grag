#!/usr/bin/env python3
"""
Backend API for GraphRAG Web UI
Aggregates calls to validation, query, and ingestion services
"""

import os
import logging
from typing import Optional
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import redis
import requests
import json
import pika

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="GraphRAG Backend API")

# CORS for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify exact origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configuration
QUERY_API = os.environ.get("QUERY_API_URL", "http://graphrag-api:8002")
VALIDATION_API = os.environ.get("VALIDATION_API_URL", "http://graphrag-validation:8003")
RABBITMQ_HOST = os.environ.get("RABBITMQ_HOST", "rabbitmq")
RABBITMQ_PORT = int(os.environ.get("RABBITMQ_PORT", "5672"))
RABBITMQ_USER = os.environ.get("RABBITMQ_USER", "graphrag")
RABBITMQ_PASSWORD = os.environ.get("RABBITMQ_PASSWORD", "graphrag123")

# Queue names
INGESTION_QUEUE = "ingestion_queue"
RETRY_QUEUE = "ingestion_queue:retry"
DLQ_QUEUE = "ingestion_queue:dlq"

# RabbitMQ connection
def get_rabbitmq_connection():
    """Create RabbitMQ connection"""
    credentials = pika.PlainCredentials(RABBITMQ_USER, RABBITMQ_PASSWORD)
    params = pika.ConnectionParameters(
        host=RABBITMQ_HOST,
        port=RABBITMQ_PORT,
        credentials=credentials
    )
    return pika.BlockingConnection(params)


class DocumentSubmission(BaseModel):
    doc_id: str
    url: str
    collection_name: str
    metadata: Optional[dict] = {}


class SearchRequest(BaseModel):
    query: str
    tenant_id: str
    top_k: int = 5


class QueryRequest(BaseModel):
    query: str
    tenant_id: str
    top_k: int = 5
    temperature: float = 0.7


@app.get("/health")
def health():
    """Health check"""
    return {"status": "healthy", "service": "backend"}


@app.post("/submit-document")
def submit_document(doc: DocumentSubmission):
    """Submit a document for ingestion"""
    try:
        message = {
            'doc_id': doc.doc_id,
            'url': doc.url,
            'collection_name': doc.collection_name,
            'metadata': doc.metadata
        }

        # Publish to RabbitMQ
        connection = get_rabbitmq_connection()
        channel = connection.channel()

        channel.basic_publish(
            exchange='',
            routing_key=INGESTION_QUEUE,
            body=json.dumps(message),
            properties=pika.BasicProperties(
                delivery_mode=2,  # Persistent
                content_type='application/json'
            )
        )

        # Get queue length
        method = channel.queue_declare(queue=INGESTION_QUEUE, durable=True, passive=True)
        queue_length = method.method.message_count

        connection.close()

        return {
            "status": "submitted",
            "doc_id": doc.doc_id,
            "queue_position": queue_length
        }
    except Exception as e:
        logger.error(f"Failed to submit document: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/queue-status")
def get_queue_status():
    """Get ingestion queue status"""
    try:
        connection = get_rabbitmq_connection()
        channel = connection.channel()

        # Get message counts from each queue
        ingestion_method = channel.queue_declare(queue=INGESTION_QUEUE, durable=True, passive=True)
        retry_method = channel.queue_declare(queue=RETRY_QUEUE, durable=True, passive=True)
        dlq_method = channel.queue_declare(queue=DLQ_QUEUE, durable=True, passive=True)

        ingestion_count = ingestion_method.method.message_count
        retry_count = retry_method.method.message_count
        dlq_count = dlq_method.method.message_count

        connection.close()

        return {
            "ingestion_queue": ingestion_count,
            "retry_queue": retry_count,
            "dead_letter_queue": dlq_count,
            "total_pending": ingestion_count + retry_count
        }
    except Exception as e:
        logger.error(f"Failed to get queue status: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/search")
def search(request: SearchRequest):
    """Semantic search without LLM"""
    try:
        response = requests.get(
            f"{QUERY_API}/search/{request.tenant_id}",
            params={"q": request.query, "top_k": request.top_k},
            timeout=30
        )
        response.raise_for_status()
        return response.json()
    except Exception as e:
        logger.error(f"Search failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/query")
def query(request: QueryRequest):
    """Full RAG query with LLM"""
    try:
        response = requests.post(
            f"{QUERY_API}/query",
            json={
                "query": request.query,
                "tenant_id": request.tenant_id,
                "top_k": request.top_k,
                "temperature": request.temperature
            },
            timeout=60
        )
        response.raise_for_status()
        return response.json()
    except Exception as e:
        logger.error(f"Query failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/validate")
def validate(tenant_id: Optional[str] = None):
    """Run validation checks"""
    try:
        if tenant_id:
            response = requests.get(f"{VALIDATION_API}/validate/{tenant_id}", timeout=30)
        else:
            response = requests.post(f"{VALIDATION_API}/validate", json={}, timeout=30)

        response.raise_for_status()
        return response.json()
    except Exception as e:
        logger.error(f"Validation failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/stats")
def get_stats(tenant_id: Optional[str] = None):
    """Get graph statistics"""
    try:
        if tenant_id:
            response = requests.get(f"{VALIDATION_API}/stats/{tenant_id}", timeout=30)
        else:
            response = requests.get(f"{VALIDATION_API}/stats", timeout=30)

        response.raise_for_status()
        return response.json()
    except Exception as e:
        logger.error(f"Failed to get stats: {e}")
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=5000)
