#!/bin/bash
set -e

echo "=========================================="
echo "  GraphRAG Setup Script"
echo "=========================================="
echo ""

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Check prerequisites
echo "Checking prerequisites..."

# Check Docker
if ! command -v docker &> /dev/null; then
    echo -e "${RED}❌ Docker is not installed${NC}"
    echo "Please install Docker: https://docs.docker.com/get-docker/"
    exit 1
fi
echo -e "${GREEN}✓${NC} Docker found: $(docker --version)"

# Check Docker Compose
if ! command -v docker-compose &> /dev/null; then
    echo -e "${RED}❌ Docker Compose is not installed${NC}"
    echo "Please install Docker Compose: https://docs.docker.com/compose/install/"
    exit 1
fi
echo -e "${GREEN}✓${NC} Docker Compose found: $(docker-compose --version)"

# Check NVIDIA Docker (for GPU support)
if ! docker run --rm --gpus all nvidia/cuda:12.0-base nvidia-smi &> /dev/null; then
    echo -e "${YELLOW}⚠${NC}  GPU support not detected"
    echo "For GPU acceleration, install NVIDIA Container Toolkit:"
    echo "https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html"
    echo ""
    read -p "Continue without GPU support? (y/N) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
else
    echo -e "${GREEN}✓${NC} NVIDIA GPU detected"
    nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | head -1
fi

echo ""
echo "=========================================="
echo "  Starting Services"
echo "=========================================="
echo ""

# Create models directory if it doesn't exist
if [ ! -d "models" ]; then
    echo "Creating models directory..."
    mkdir -p models
fi

# Start services
echo "Starting Docker services..."
echo "(This will download models on first run - may take 5-10 minutes)"
echo ""

docker-compose up -d

echo ""
echo "Waiting for services to initialize..."
sleep 5

# Monitor service health
echo ""
echo "Checking service status..."
docker-compose ps

echo ""
echo "=========================================="
echo "  Setup Complete!"
echo "=========================================="
echo ""
echo "Services are starting up. Initial model download may take 5-10 minutes."
echo ""
echo "🌐 Web UI:              http://localhost:3000"
echo "📊 Neo4j Browser:       http://localhost:7474 (neo4j/graphrag123)"
echo "🐰 RabbitMQ Management: http://localhost:15672 (graphrag/graphrag123)"
echo ""
echo "Monitor startup progress:"
echo "  docker logs -f graphrag-vllm       # LLM model loading (takes longest)"
echo "  docker logs -f graphrag-embeddings # Embedding model loading"
echo "  docker logs -f graphrag-ingestion  # Worker status"
echo ""
echo "Check all services are healthy:"
echo "  docker-compose ps"
echo ""
echo "View the quickstart guide:"
echo "  cat QUICKSTART.md"
echo ""
echo "To stop services:"
echo "  docker-compose down"
echo ""
echo "To remove all data (including downloaded models):"
echo "  docker-compose down -v"
echo "  rm -rf models/"
echo ""
