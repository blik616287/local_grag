#!/usr/bin/env python3
"""
GraphRAG Query API with LangChain
Provides RAG (Retrieval Augmented Generation) capabilities
"""

import os
import logging
from typing import List, Optional, Dict, Any
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import uvicorn
import numpy as np

from neo4j import GraphDatabase
from openai import OpenAI
import requests

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

app = FastAPI(title="GraphRAG Query API")

# Configuration
NEO4J_URI = os.environ.get("NEO4J_URI", "bolt://neo4j:7687")
NEO4J_USER = os.environ.get("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.environ.get("NEO4J_PASSWORD", "graphrag123")
VLLM_ENDPOINT = os.environ.get("VLLM_ENDPOINT", "http://vllm:8000/v1")
EMBEDDINGS_ENDPOINT = os.environ.get("EMBEDDINGS_ENDPOINT", "http://embeddings:8001")
LLM_MODEL = os.environ.get("LLM_MODEL", "microsoft/Phi-3-mini-4k-instruct")

# Initialize connections
neo4j_driver = None
llm_client = None


@app.on_event("startup")
async def startup_event():
    """Initialize connections on startup"""
    global neo4j_driver, llm_client

    logger.info(f"Connecting to Neo4j at {NEO4J_URI}")
    neo4j_driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
    neo4j_driver.verify_connectivity()
    logger.info("Connected to Neo4j")

    logger.info(f"Initializing LLM client for {VLLM_ENDPOINT}")
    llm_client = OpenAI(base_url=VLLM_ENDPOINT, api_key="EMPTY")
    logger.info("Query API ready")


@app.on_event("shutdown")
async def shutdown_event():
    """Clean up on shutdown"""
    global neo4j_driver
    if neo4j_driver:
        neo4j_driver.close()


class QueryRequest(BaseModel):
    query: str = Field(..., description="User's question")
    tenant_id: str = Field(..., description="Tenant ID for multi-tenant isolation")
    top_k: int = Field(default=5, description="Number of chunks to retrieve", ge=1, le=20)
    similarity_threshold: float = Field(default=0.3, description="Minimum similarity score", ge=0.0, le=1.0)
    include_entities: bool = Field(default=False, description="Include entity information")
    temperature: float = Field(default=0.7, description="LLM temperature", ge=0.0, le=2.0)


class QueryResponse(BaseModel):
    answer: str
    sources: List[Dict[str, Any]]
    entities: Optional[List[Dict[str, Any]]] = None
    query_embedding_time_ms: float
    retrieval_time_ms: float
    generation_time_ms: float
    total_time_ms: float


def get_query_embedding(query: str) -> List[float]:
    """Get embedding for query text"""
    import time
    start = time.time()

    response = requests.post(
        f"{EMBEDDINGS_ENDPOINT}/embed",
        json={"texts": query, "normalize": True},
        timeout=10
    )
    response.raise_for_status()
    embedding = response.json()['embeddings'][0]

    elapsed = (time.time() - start) * 1000
    logger.info(f"Query embedding generated in {elapsed:.2f}ms")

    return embedding, elapsed


def retrieve_similar_chunks(
    tenant_id: str,
    query_embedding: List[float],
    top_k: int,
    similarity_threshold: float
) -> tuple[List[Dict[str, Any]], float]:
    """
    Retrieve similar chunks from Neo4j using vector similarity

    Returns:
        (chunks, elapsed_ms)
    """
    import time
    start = time.time()

    with neo4j_driver.session() as session:
        result = session.run("""
            MATCH (c:Chunk)
            WHERE c.tenant_id = $tenant_id AND c.embedding IS NOT NULL
            RETURN c.id as chunk_id,
                   c.text as text,
                   c.embedding as embedding,
                   c.metadata as metadata
        """, {"tenant_id": tenant_id})

        # Calculate cosine similarity in Python
        def cosine_similarity(vec1, vec2):
            vec1 = np.array(vec1)
            vec2 = np.array(vec2)
            return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))

        chunks_with_scores = []
        query_vec = np.array(query_embedding)

        for record in result:
            chunk_vec = np.array(record['embedding'])
            score = cosine_similarity(query_vec, chunk_vec)

            if score >= similarity_threshold:
                chunks_with_scores.append({
                    'chunk_id': record['chunk_id'],
                    'text': record['text'],
                    'metadata': record['metadata'],
                    'similarity_score': float(score)
                })

        # Sort by score and take top_k
        chunks_with_scores.sort(key=lambda x: x['similarity_score'], reverse=True)
        chunks = chunks_with_scores[:top_k]

    elapsed = (time.time() - start) * 1000
    logger.info(f"Retrieved {len(chunks)} chunks in {elapsed:.2f}ms")

    return chunks, elapsed


def get_entities_for_chunks(tenant_id: str, chunk_ids: List[str]) -> List[Dict[str, Any]]:
    """Get entities mentioned in the retrieved chunks"""
    if not chunk_ids:
        return []

    with neo4j_driver.session() as session:
        result = session.run("""
            MATCH (e:Entity)-[:MENTIONED_IN]->(c:Chunk)
            WHERE c.id IN $chunk_ids AND e.tenant_id = $tenant_id
            RETURN DISTINCT e.name as name,
                            e.type as type,
                            e.description as description
            LIMIT 20
        """, {"chunk_ids": chunk_ids, "tenant_id": tenant_id})

        entities = [dict(record) for record in result]

    logger.info(f"Found {len(entities)} entities")
    return entities


def generate_answer(query: str, chunks: List[Dict[str, Any]], temperature: float) -> tuple[str, float]:
    """
    Generate answer using LLM with retrieved context

    Returns:
        (answer, elapsed_ms)
    """
    import time
    start = time.time()

    # Build context from chunks
    context_parts = []
    for i, chunk in enumerate(chunks, 1):
        context_parts.append(f"[Source {i}] {chunk['text']}")

    context = "\n\n".join(context_parts)

    # Build prompt
    system_prompt = """You are a helpful AI assistant that answers questions based on the provided context.
Always cite your sources by referring to [Source N] numbers.
If the context doesn't contain relevant information to answer the question, say so honestly."""

    user_prompt = f"""Context:
{context}

Question: {query}

Please provide a detailed answer based on the context above. Cite specific sources."""

    try:
        response = llm_client.chat.completions.create(
            model=LLM_MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=temperature,
            max_tokens=500,
        )

        answer = response.choices[0].message.content

    except Exception as e:
        logger.error(f"LLM generation failed: {e}")
        # Fallback: return context summary
        answer = f"I found {len(chunks)} relevant passages but couldn't generate a detailed answer. Here's what I found:\n\n{context[:500]}..."

    elapsed = (time.time() - start) * 1000
    logger.info(f"Answer generated in {elapsed:.2f}ms")

    return answer, elapsed


@app.get("/health")
async def health():
    """Health check endpoint"""
    return {"status": "healthy", "service": "query-api"}


@app.post("/query", response_model=QueryResponse)
async def query(request: QueryRequest):
    """
    Query the knowledge graph with RAG

    Request body:
    {
        "query": "What is Python?",
        "tenant_id": "fdsv1w6ymb",
        "top_k": 5,
        "similarity_threshold": 0.3,
        "include_entities": false,
        "temperature": 0.7
    }
    """
    import time
    total_start = time.time()

    try:
        # Step 1: Get query embedding
        query_embedding, embedding_time = get_query_embedding(request.query)

        # Step 2: Retrieve similar chunks
        chunks, retrieval_time = retrieve_similar_chunks(
            request.tenant_id,
            query_embedding,
            request.top_k,
            request.similarity_threshold
        )

        if not chunks:
            return QueryResponse(
                answer="I couldn't find any relevant information in the knowledge base to answer your question.",
                sources=[],
                entities=None,
                query_embedding_time_ms=embedding_time,
                retrieval_time_ms=retrieval_time,
                generation_time_ms=0,
                total_time_ms=(time.time() - total_start) * 1000
            )

        # Step 3: Get entities if requested
        entities = None
        if request.include_entities:
            chunk_ids = [c['chunk_id'] for c in chunks]
            entities = get_entities_for_chunks(request.tenant_id, chunk_ids)

        # Step 4: Generate answer
        answer, generation_time = generate_answer(request.query, chunks, request.temperature)

        total_time = (time.time() - total_start) * 1000

        return QueryResponse(
            answer=answer,
            sources=chunks,
            entities=entities,
            query_embedding_time_ms=embedding_time,
            retrieval_time_ms=retrieval_time,
            generation_time_ms=generation_time,
            total_time_ms=total_time
        )

    except Exception as e:
        logger.error(f"Query failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/search/{tenant_id}")
async def search(tenant_id: str, q: str, top_k: int = 10):
    """
    Simple semantic search without LLM generation

    Query params:
    - q: search query
    - top_k: number of results (default: 10)
    """
    try:
        query_embedding, _ = get_query_embedding(q)
        chunks, _ = retrieve_similar_chunks(tenant_id, query_embedding, top_k, 0.0)

        return {
            "query": q,
            "results": chunks,
            "count": len(chunks)
        }

    except Exception as e:
        logger.error(f"Search failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8002)
