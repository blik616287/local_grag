"""
Embedding service using Sentence Transformers
Provides REST API for generating text embeddings
"""

import os
import logging
from typing import List, Union
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
import torch
from sentence_transformers import SentenceTransformer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Configuration
MODEL_NAME = os.environ.get("MODEL_NAME", "sentence-transformers/all-MiniLM-L6-v2")
DEVICE = os.environ.get("DEVICE", "cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = int(os.environ.get("BATCH_SIZE", "32"))
HF_HOME = os.environ.get("HF_HOME", "/models")

# Global model instance
model = None


class EmbedRequest(BaseModel):
    """Request model for embedding generation"""
    texts: Union[str, List[str]] = Field(..., description="Text or list of texts to embed")
    normalize: bool = Field(default=True, description="Whether to normalize embeddings")


class EmbedResponse(BaseModel):
    """Response model for embedding generation"""
    embeddings: List[List[float]] = Field(..., description="Generated embeddings")
    model: str = Field(..., description="Model used for embedding")
    dimensions: int = Field(..., description="Embedding dimensions")


class HealthResponse(BaseModel):
    """Response model for health check"""
    status: str
    model: str
    device: str
    dimensions: int


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Lifespan context manager for model loading/unloading
    """
    global model

    logger.info(f"Loading embedding model: {MODEL_NAME}")
    logger.info(f"Using device: {DEVICE}")
    logger.info(f"HuggingFace cache directory: {HF_HOME}")

    try:
        # Load the model
        model = SentenceTransformer(
            MODEL_NAME,
            device=DEVICE,
            cache_folder=HF_HOME
        )

        # Log model information
        embedding_dim = model.get_sentence_embedding_dimension()
        logger.info(f"Model loaded successfully. Embedding dimensions: {embedding_dim}")

        # Warm up the model
        logger.info("Warming up model with test embedding...")
        _ = model.encode("test", convert_to_tensor=True)
        logger.info("Model warm-up complete")

    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        raise

    yield

    # Cleanup
    logger.info("Shutting down embedding service")
    model = None


# Create FastAPI app
app = FastAPI(
    title="GraphRAG Embedding Service",
    description="Sentence Transformers embedding generation service",
    version="1.0.0",
    lifespan=lifespan
)


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """
    Health check endpoint
    """
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    return HealthResponse(
        status="healthy",
        model=MODEL_NAME,
        device=DEVICE,
        dimensions=model.get_sentence_embedding_dimension()
    )


@app.post("/embed", response_model=EmbedResponse)
async def generate_embeddings(request: EmbedRequest):
    """
    Generate embeddings for input text(s)

    Args:
        request: EmbedRequest containing text(s) to embed

    Returns:
        EmbedResponse with generated embeddings
    """
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    try:
        # Handle single string or list of strings
        texts = request.texts if isinstance(request.texts, list) else [request.texts]

        if not texts:
            raise HTTPException(status_code=400, detail="No texts provided")

        # Validate text length
        for i, text in enumerate(texts):
            if len(text) > 10000:
                logger.warning(f"Text {i} exceeds 10000 characters, truncating")
                texts[i] = text[:10000]

        logger.info(f"Generating embeddings for {len(texts)} text(s)")

        # Generate embeddings
        embeddings = model.encode(
            texts,
            batch_size=BATCH_SIZE,
            convert_to_tensor=True,
            normalize_embeddings=request.normalize,
            show_progress_bar=False
        )

        # Convert to list
        embeddings_list = embeddings.cpu().numpy().tolist()

        logger.info(f"Successfully generated {len(embeddings_list)} embeddings")

        return EmbedResponse(
            embeddings=embeddings_list,
            model=MODEL_NAME,
            dimensions=model.get_sentence_embedding_dimension()
        )

    except Exception as e:
        logger.error(f"Error generating embeddings: {e}")
        raise HTTPException(status_code=500, detail=f"Embedding generation failed: {str(e)}")


@app.post("/embed/query")
async def embed_query(request: EmbedRequest):
    """
    Generate embedding for a single query (alias for embed endpoint)
    Useful for semantic search queries
    """
    if not isinstance(request.texts, str):
        raise HTTPException(status_code=400, detail="Query must be a single string")

    result = await generate_embeddings(request)

    return {
        "embedding": result.embeddings[0],
        "model": result.model,
        "dimensions": result.dimensions
    }


@app.post("/embed/documents")
async def embed_documents(request: EmbedRequest):
    """
    Generate embeddings for multiple documents (alias for embed endpoint)
    Useful for batch processing
    """
    if not isinstance(request.texts, list):
        raise HTTPException(status_code=400, detail="Documents must be a list of strings")

    return await generate_embeddings(request)


@app.get("/")
async def root():
    """
    Root endpoint with service information
    """
    if model is None:
        return JSONResponse(
            status_code=503,
            content={"status": "starting", "message": "Model is loading..."}
        )

    return {
        "service": "GraphRAG Embedding Service",
        "model": MODEL_NAME,
        "device": DEVICE,
        "dimensions": model.get_sentence_embedding_dimension(),
        "endpoints": {
            "health": "/health",
            "embed": "/embed",
            "embed_query": "/embed/query",
            "embed_documents": "/embed/documents"
        }
    }


if __name__ == "__main__":
    import uvicorn

    port = int(os.environ.get("PORT", "8001"))
    host = os.environ.get("HOST", "0.0.0.0")

    logger.info(f"Starting embedding service on {host}:{port}")

    uvicorn.run(
        app,
        host=host,
        port=port,
        log_level="info"
    )
