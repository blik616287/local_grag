#!/bin/bash

echo "=========================================="
echo "  GraphRAG Cleanup Script"
echo "=========================================="
echo ""

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

echo "This will remove:"
echo "  - Downloaded models (~13GB)"
echo "  - Docker volumes (database data)"
echo "  - Python cache files"
echo ""
read -p "Are you sure? (y/N) " -n 1 -r
echo

if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Cancelled."
    exit 0
fi

echo ""
echo "Stopping and removing containers..."
docker-compose down -v

if [ -d "models" ]; then
    echo "Removing models directory..."
    rm -rf models/
    echo -e "${GREEN}✓${NC} Models removed"
fi

echo "Removing Python cache..."
find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null
find . -type f -name "*.pyc" -delete 2>/dev/null
echo -e "${GREEN}✓${NC} Python cache removed"

echo ""
echo -e "${GREEN}Cleanup complete!${NC}"
echo ""
echo "To start fresh, run: ./setup.sh"
