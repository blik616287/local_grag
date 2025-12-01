# GraphRAG - Production-Ready Knowledge Graph RAG System

A complete, production-ready implementation of a Graph-based Retrieval Augmented Generation (GraphRAG) system with local LLM inference, vector search, entity extraction, and multi-tenant support.

## Overview

GraphRAG combines knowledge graphs with vector embeddings to provide semantic search and intelligent question-answering over your documents. The system processes documents, extracts entities and relationships, stores them in a graph database, and enables powerful semantic queries using both graph traversal and vector similarity.

## Key Features

- **Production-Ready**: Enterprise-grade message queuing (RabbitMQ), guaranteed delivery, automatic retries, and dead letter queues
- **Local LLM Inference**: Uses vLLM with GPU acceleration for fast entity extraction and text generation
- **Hybrid Search**: Combines vector similarity search with graph traversal for superior results
- **Knowledge Graph**: Neo4j-based graph database with APOC and GDS plugins
- **Multi-Tenant**: Complete tenant isolation with dynamic tenant ID generation
- **Entity Extraction**: Automatic NER using local LLMs (Qwen2.5-1.5B)
- **Vector Embeddings**: Sentence transformers for semantic similarity (384-dim)
- **Message Queue**: RabbitMQ with at-least-once delivery guarantees
- **Validation Service**: 7 comprehensive graph integrity checks
- **Web UI**: Clean, modern interface for search, document submission, and monitoring

## Prerequisites

- **Docker** and **Docker Compose** (v2.20+)
- **NVIDIA GPU** with CUDA support (for vLLM and embeddings)
- **nvidia-docker2** runtime
- At least **16GB RAM**
- At least **20GB disk space**

Verify your setup:
```bash
docker --version
docker-compose --version
nvidia-smi
docker run --rm --gpus all nvidia/cuda:12.0-base nvidia-smi
```

## Quick Start

### 1. Clone and Setup

```bash
git clone <your-repo-url>
cd composer

# Run the setup script (recommended)
./setup.sh

# Or manually start services
docker-compose up -d
```

**Note**: On first run, the system will automatically download required models (~3GB):
- Qwen2.5-1.5B-Instruct (LLM for entity extraction)
- all-MiniLM-L6-v2 (Embedding model)

This takes 5-10 minutes depending on your internet connection.

### 2. Wait for Services to Initialize

Services start in order with health checks. Full initialization takes ~3-5 minutes:

```bash
# Monitor vLLM model loading (takes longest)
docker logs -f graphrag-vllm

# Check all services are healthy
docker-compose ps | grep healthy
```

### 3. Access the Web UI

Open your browser to: **http://localhost:3000**

The UI provides 4 tabs:
- **Search**: Semantic search across your knowledge graph
- **Submit Document**: Add new documents for processing
- **Validation**: Run graph integrity checks
- **Statistics**: View graph statistics and metrics

### 4. Submit Your First Document

**Using the Web UI:**
1. Go to the "Submit Document" tab
2. Enter:
   - **Document ID**: `my_first_doc`
   - **URL**: `https://en.wikipedia.org/wiki/Python_(programming_language)`
   - **Collection Name**: `programming_docs`
3. Click "Submit Document"

**Using curl:**
```bash
curl -X POST http://localhost:5000/submit-document \
  -H "Content-Type: application/json" \
  -d '{
    "doc_id": "my_first_doc",
    "url": "https://en.wikipedia.org/wiki/Python_(programming_language)",
    "collection_name": "programming_docs",
    "metadata": {}
  }'
```

### 5. Monitor Processing

```bash
# Watch ingestion worker logs
docker logs -f graphrag-ingestion

# Check queue status
curl http://localhost:5000/queue-status | jq
```

Get the tenant ID from the logs (e.g., `abc123xyz`) - you'll need this for searching.

### 6. Search Your Knowledge Graph

Wait ~30 seconds for processing, then search:

**Using the Web UI:**
1. Go to the "Search" tab
2. Enter query: `What is Python used for?`
3. Enter tenant ID (from worker logs)
4. Click "Search"

**Using curl:**
```bash
curl -X POST http://localhost:5000/search \
  -H "Content-Type: application/json" \
  -d '{
    "query": "What is Python used for?",
    "tenant_id": "[tenant_id_from_logs]",
    "top_k": 5
  }' | jq
```

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                         Web UI (Port 3000)                      │
│                    Search | Submit | Validate | Stats           │
└────────────────────────────┬────────────────────────────────────┘
                             │
┌────────────────────────────▼────────────────────────────────────┐
│                     Backend API (Port 5000)                      │
│              Document Submission | Queue Status                 │
└──────┬──────────────────────┬───────────────────┬───────────────┘
       │                      │                   │
       │                      │                   │
┌──────▼──────────┐  ┌────────▼─────────┐  ┌─────▼──────────────┐
│  RabbitMQ       │  │  Query API       │  │  Validation API    │
│  Message Queue  │  │  (Port 8002)     │  │  (Port 8003)       │
│  - Ingestion    │  │  Vector Search   │  │  Graph Integrity   │
│  - Retry        │  │  + LLM Response  │  │  Checks            │
│  - DLQ          │  │                  │  │                    │
└──────┬──────────┘  └────────┬─────────┘  └─────┬──────────────┘
       │                      │                   │
┌──────▼──────────────────────▼───────────────────▼───────────────┐
│                    Ingestion Worker                              │
│  Document Processing | Entity Extraction | Graph Building       │
└──────┬────────────────────────┬─────────────────┬───────────────┘
       │                        │                 │
┌──────▼──────────┐  ┌──────────▼──────┐  ┌──────▼──────────────┐
│  Neo4j          │  │  vLLM Server    │  │  Embeddings Service │
│  Graph Database │  │  Local LLM      │  │  Sentence Transform │
│  (Port 7474)    │  │  (Port 8000)    │  │  (Port 8001)        │
└─────────────────┘  └─────────────────┘  └─────────────────────┘
       │
┌──────▼──────────┐
│  PostgreSQL     │
│  State Tracking │
│  (Port 5433)    │
└─────────────────┘
```

## Service Endpoints

| Service | Port | URL | Description |
|---------|------|-----|-------------|
| **Web UI** | 3000 | http://localhost:3000 | Main user interface |
| **Backend API** | 5000 | http://localhost:5000 | Document submission, search |
| **Neo4j Browser** | 7474 | http://localhost:7474 | Graph database UI (neo4j/graphrag123) |
| **RabbitMQ Management** | 15672 | http://localhost:15672 | Queue monitoring (graphrag/graphrag123) |
| **Query API** | 8002 | http://localhost:8002/docs | RAG query endpoints |
| **Validation API** | 8003 | http://localhost:8003/docs | Validation endpoints |
| **vLLM** | 8000 | http://localhost:8000/v1 | LLM inference server |
| **Embeddings** | 8001 | http://localhost:8001 | Embedding generation |

## Configuration

### Environment Variables

Key configuration in `docker-compose.yml`:

**LLM Model** (adjust based on your GPU):
```yaml
vllm:
  command: >
    --model Qwen/Qwen2.5-1.5B-Instruct
    --gpu-memory-utilization 0.5
    --max-model-len 2048
```

**Embedding Model**:
```yaml
embeddings:
  environment:
    - MODEL_NAME=sentence-transformers/all-MiniLM-L6-v2
```

**Worker Concurrency**:
```yaml
ingestion:
  environment:
    - WORKER_CONCURRENCY=4
```

### Adjusting for Different GPUs

**For GPUs with <8GB VRAM**: Already configured with Qwen2.5-1.5B

**For GPUs with 12GB+ VRAM**:
```yaml
--model microsoft/Phi-3-mini-4k-instruct
--gpu-memory-utilization 0.7
--max-model-len 4096
```

**For GPUs with 24GB+ VRAM**:
```yaml
--model meta-llama/Llama-2-7b-chat-hf
--gpu-memory-utilization 0.8
--max-model-len 8192
```

## API Reference

### Submit Document
```bash
POST /submit-document
{
  "doc_id": "unique_id",
  "url": "https://example.com/document",
  "collection_name": "my_collection",
  "metadata": {}
}
```

### Search (Vector Similarity)
```bash
POST /search
{
  "query": "search query",
  "tenant_id": "collection_tenant_id",
  "top_k": 5
}
```

### Query (RAG with LLM)
```bash
POST /query
{
  "query": "question to answer",
  "tenant_id": "collection_tenant_id",
  "top_k": 5,
  "temperature": 0.7
}
```

### Queue Status
```bash
GET /queue-status
```

### Validation
```bash
GET /validate?tenant_id=optional_tenant
```

### Graph Statistics
```bash
GET /stats?tenant_id=optional_tenant
```

## Data Model

### Neo4j Graph Structure

**Nodes**:
- `Document`: Source documents with URL and metadata
- `Chunk`: Text chunks with embeddings (384-dim vectors)
- `Entity`: Named entities extracted from text

**Relationships**:
- `(Chunk)-[:PART_OF]->(Document)`: Chunk belongs to document
- `(Entity)-[:MENTIONED_IN]->(Chunk)`: Entity appears in chunk

**Properties**:
- All nodes have `tenant_id` for multi-tenant isolation
- Chunks have `embedding` (vector), `text`, `metadata`
- Documents have `id`, `url`, `title`

### PostgreSQL State Tracking

Tracks document processing state:
- `documents`: Processing status, timestamps, retry counts
- `tenants`: Tenant statistics and metadata

## Example Use Cases

### Building a Technical Documentation Search

```bash
# Submit multiple documentation pages
for doc in "https://docs.python.org/3/tutorial/index.html" \
           "https://docs.python.org/3/library/index.html"; do
  DOC_ID=$(echo $doc | md5sum | cut -d' ' -f1)
  curl -X POST http://localhost:5000/submit-document \
    -H "Content-Type: application/json" \
    -d "{
      \"doc_id\": \"$DOC_ID\",
      \"url\": \"$doc\",
      \"collection_name\": \"python_docs\"
    }"
  sleep 2
done
```

### Creating a Research Paper Knowledge Base

```python
import requests
import time

papers = [
    {"id": "transformer_paper", "url": "https://arxiv.org/pdf/1706.03762.pdf"},
    {"id": "bert_paper", "url": "https://arxiv.org/pdf/1810.04805.pdf"}
]

for paper in papers:
    requests.post(
        'http://localhost:5000/submit-document',
        json={
            'doc_id': paper['id'],
            'url': paper['url'],
            'collection_name': 'ml_papers'
        }
    )
    time.sleep(2)
```

### Working with Neo4j Directly

```bash
# Access Neo4j Browser
open http://localhost:7474

# Or use cypher-shell
docker exec graphrag-neo4j cypher-shell -u neo4j -p graphrag123

# Count documents by tenant
MATCH (d:Document)
RETURN d.tenant_id, count(d) as doc_count
ORDER BY doc_count DESC;

# Find entities linked to a specific chunk
MATCH (e:Entity)-[:MENTIONED_IN]->(c:Chunk {id: 'chunk_123'})
RETURN e.name, e.type;
```

## Message Queue Architecture

### RabbitMQ Queues

1. **ingestion_queue**: Main processing queue
   - Durable, persistent messages
   - DLX configured for failed messages
   - QoS prefetch=1 for fair distribution

2. **ingestion_queue:retry**: Automatic retry queue
   - TTL: 60 seconds (configurable per message)
   - Auto-routes back to ingestion queue
   - Exponential backoff: 2^retry_count minutes

3. **ingestion_queue:dlq**: Dead letter queue
   - Stores permanently failed messages
   - Manual intervention required

### Delivery Guarantees

- **At-least-once delivery**: Messages acknowledged only after successful processing
- **Persistent messages**: Survive broker restarts
- **Automatic retries**: Up to 3 attempts with exponential backoff

## Validation

The system includes 7 comprehensive validation checks:

1. **Orphaned Chunks**: Chunks without parent documents
2. **Orphaned Entities**: Entities not linked to any chunks
3. **Missing Embeddings**: Chunks without vector embeddings
4. **Empty Chunks**: Chunks with no text content
5. **Duplicate Chunks**: Identical chunks from same document
6. **Missing Tenant IDs**: Nodes without tenant isolation
7. **Postgres-Neo4j Sync**: State consistency between databases

Run validation:
```bash
# Via UI: Go to Validation tab and click "Run Validation"
# Via API:
curl http://localhost:5000/validate | jq
```

## Monitoring

### RabbitMQ Management UI
- URL: http://localhost:15672
- Username: `graphrag`
- Password: `graphrag123`

### Neo4j Browser
- URL: http://localhost:7474
- Username: `neo4j`
- Password: `graphrag123`

### Service Logs
```bash
docker-compose logs -f           # All logs
docker logs -f graphrag-ingestion  # Specific service
docker logs -f graphrag-vllm
```

## Troubleshooting

### vLLM Out of Memory
```
ValueError: No available memory for the cache blocks
```
**Solution**: Use a smaller model or reduce `--gpu-memory-utilization`

### Entity Extractor Failed
```
Entity extractor initialization failed (vLLM may not be ready)
```
**Solution**: Wait for vLLM to finish loading the model (~2-3 minutes)

### RabbitMQ Connection Refused
```
AMQPConnectionError: Connection refused
```
**Solution**: Wait for RabbitMQ to start (`docker logs graphrag-rabbitmq`)

### Neo4j Not Ready
```
ServiceUnavailable: Unable to retrieve routing information
```
**Solution**: Wait for Neo4j initialization (`docker logs graphrag-neo4j`)

### Queue Stuck
```bash
curl http://localhost:5000/queue-status | jq
docker logs graphrag-ingestion --tail 50
docker-compose restart ingestion
```

### Poor Search Results
```bash
# Check chunk embeddings
docker exec graphrag-neo4j cypher-shell -u neo4j -p graphrag123 \
  "MATCH (c:Chunk) WHERE c.embedding IS NULL RETURN count(c)"

# Adjust chunk size in docker-compose.yml
CHUNK_SIZE=1500
CHUNK_OVERLAP=300
```

## Development

### Project Structure
```
composer/
├── docker-compose.yml          # Service orchestration
├── init-db.sql                 # PostgreSQL initialization
├── services/
│   ├── ingestion/              # Document processing worker
│   │   ├── worker.py           # Main worker loop
│   │   ├── rabbitmq_queue_manager.py  # RabbitMQ integration
│   │   ├── document_processor.py      # Document loading & chunking
│   │   ├── entity_extractor.py        # NER using LLM
│   │   ├── graph_builder.py           # Neo4j graph construction
│   │   └── tenant_manager.py          # Multi-tenant management
│   ├── embeddings/             # Embedding service
│   ├── query-api/              # RAG query API
│   ├── validation/             # Validation service
│   ├── backend/                # Backend API
│   └── frontend/               # Web UI
└── models/                     # Cached LLM models
```

### Adding New Document Sources

Edit `services/ingestion/document_processor.py` to support new formats:
1. Add loader in `_load_from_url()` method
2. Implement content extraction
3. Return list of LangChain Document objects

### Customizing Entity Extraction

Edit `services/ingestion/entity_extractor.py`:
- Modify the prompt in `_build_extraction_prompt()`
- Adjust entity types and extraction logic

## Performance

### Throughput
- **Document processing**: ~5-10 docs/minute (depends on size)
- **Chunk creation**: ~100-200 chunks/minute
- **Entity extraction**: ~50-100 entities/minute
- **Embedding generation**: ~200-300 embeddings/minute

### Resource Usage
- **vLLM**: ~4-6GB GPU VRAM (Qwen2.5-1.5B)
- **Embeddings**: ~2-3GB GPU VRAM
- **Neo4j**: ~2GB RAM
- **PostgreSQL**: ~500MB RAM
- **RabbitMQ**: ~200MB RAM

### Scaling
- **Horizontal**: Run multiple ingestion workers
- **Vertical**: Increase worker concurrency
- **Storage**: Neo4j supports billions of nodes

## Git Repository Notes

The `models/` directory contains ~13GB of downloaded AI models. **Exclude it from Git.**

### Add to .gitignore

```gitignore
# GraphRAG models (auto-download on setup)
composer/models/
**/composer/models/
composer/**/__pycache__/
```

### Or Clean Before Committing

```bash
./cleanup.sh  # Removes models directory
```

Models will auto-download when users run `./setup.sh`.

### What to Commit (~284KB)

| Path | Commit? |
|------|---------|
| `services/` | Yes |
| `docker-compose.yml` | Yes |
| `init-db.sql` | Yes |
| `README.md` | Yes |
| `setup.sh`, `cleanup.sh` | Yes |
| `.dockerignore` | Yes |
| `models/` | **No** (13GB) |

### Disk Space When Deployed

- Models: ~3GB
- Docker images: ~8GB
- Docker volumes: ~2GB
- **Total: ~13GB**

## License

MIT License

## Support

For issues and questions, please open an issue on GitHub.
