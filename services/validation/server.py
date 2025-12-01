#!/usr/bin/env python3
"""
Validation service REST API
"""

import os
import logging
from typing import Optional
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn

from validator import GraphValidator

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

app = FastAPI(title="GraphRAG Validation Service")

# Configuration
NEO4J_URI = os.environ.get("NEO4J_URI", "bolt://neo4j:7687")
NEO4J_USER = os.environ.get("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.environ.get("NEO4J_PASSWORD", "graphrag123")
POSTGRES_URI = os.environ.get("POSTGRES_URI", "postgresql://graphrag:graphrag123@postgres:5432/graphrag")

# Initialize validator
validator = None


@app.on_event("startup")
async def startup_event():
    """Initialize validator on startup"""
    global validator
    logger.info("Initializing GraphValidator")
    validator = GraphValidator(NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD, POSTGRES_URI)
    logger.info("Validation service ready")


@app.on_event("shutdown")
async def shutdown_event():
    """Clean up on shutdown"""
    global validator
    if validator:
        validator.close()


class ValidationRequest(BaseModel):
    tenant_id: Optional[str] = None


class RepairRequest(BaseModel):
    tenant_id: Optional[str] = None
    delete: bool = False


@app.get("/health")
async def health():
    """Health check endpoint"""
    return {"status": "healthy", "service": "validation"}


@app.post("/validate")
async def validate(request: ValidationRequest):
    """
    Run validation checks

    Request body:
    {
        "tenant_id": "optional_tenant_id"
    }
    """
    try:
        results = validator.run_validation(request.tenant_id)
        return results
    except Exception as e:
        logger.error(f"Validation failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/validate/{tenant_id}")
async def validate_tenant(tenant_id: str):
    """Run validation for specific tenant"""
    try:
        results = validator.run_validation(tenant_id)
        return results
    except Exception as e:
        logger.error(f"Validation failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/stats")
async def get_stats():
    """Get overall graph statistics"""
    try:
        stats = validator.get_graph_stats(None)
        return stats
    except Exception as e:
        logger.error(f"Failed to get stats: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/stats/{tenant_id}")
async def get_tenant_stats(tenant_id: str):
    """Get statistics for specific tenant"""
    try:
        stats = validator.get_graph_stats(tenant_id)
        return stats
    except Exception as e:
        logger.error(f"Failed to get stats: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/repair/orphaned-chunks")
async def repair_orphaned_chunks(request: RepairRequest):
    """
    Repair orphaned chunks

    Request body:
    {
        "tenant_id": "optional_tenant_id",
        "delete": false  // set to true to actually delete
    }
    """
    try:
        count = validator.repair_orphaned_chunks(request.tenant_id, request.delete)
        return {
            "action": "deleted" if request.delete else "reported",
            "count": count
        }
    except Exception as e:
        logger.error(f"Repair failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8003)
