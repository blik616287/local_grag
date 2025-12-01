"""
Neo4j graph construction for GraphRAG
"""

import logging
import hashlib
from typing import List, Dict, Any
import requests

from neo4j import GraphDatabase
from langchain.schema import Document

from config import (
    NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD,
    EMBEDDINGS_ENDPOINT, EMBEDDING_DIMENSIONS
)
from entity_extractor import Entity

logger = logging.getLogger(__name__)


class GraphBuilder:
    """
    Builds knowledge graph in Neo4j
    """

    def __init__(self):
        """Initialize Neo4j connection"""
        logger.info(f"Connecting to Neo4j at {NEO4J_URI}")
        self.driver = GraphDatabase.driver(
            NEO4J_URI,
            auth=(NEO4J_USER, NEO4J_PASSWORD)
        )

        # Verify connection
        self.driver.verify_connectivity()
        logger.info("Successfully connected to Neo4j")

        # Ensure vector index exists
        self._ensure_vector_index()

    def close(self):
        """Close Neo4j connection"""
        if self.driver:
            self.driver.close()
            logger.info("Closed Neo4j connection")

    def _ensure_vector_index(self):
        """Create vector index if it doesn't exist"""
        with self.driver.session() as session:
            # Check if index exists
            result = session.run("SHOW INDEXES")
            indexes = [record["name"] for record in result]

            if "chunk_embeddings" not in indexes:
                logger.info("Creating vector index for chunk embeddings")
                try:
                    session.run(f"""
                        CREATE VECTOR INDEX chunk_embeddings IF NOT EXISTS
                        FOR (c:Chunk)
                        ON c.embedding
                        OPTIONS {{
                            indexConfig: {{
                                `vector.dimensions`: {EMBEDDING_DIMENSIONS},
                                `vector.similarity_function`: 'cosine'
                            }}
                        }}
                    """)
                    logger.info("Vector index created successfully")
                except Exception as e:
                    logger.warning(f"Could not create vector index: {e}")

            # Create property indexes for performance
            try:
                session.run("""
                    CREATE INDEX tenant_index IF NOT EXISTS
                    FOR (n:Document) ON (n.tenant_id)
                """)
                session.run("""
                    CREATE INDEX entity_tenant_index IF NOT EXISTS
                    FOR (n:Entity) ON (n.tenant_id)
                """)
                session.run("""
                    CREATE INDEX chunk_tenant_index IF NOT EXISTS
                    FOR (n:Chunk) ON (n.tenant_id)
                """)
            except Exception as e:
                logger.warning(f"Could not create property indexes: {e}")

    def get_embeddings(self, texts: List[str]) -> List[List[float]]:
        """
        Get embeddings from embedding service

        Args:
            texts: List of texts to embed

        Returns:
            List of embedding vectors
        """
        if not texts:
            return []

        logger.info(f"Getting embeddings for {len(texts)} texts")

        try:
            response = requests.post(
                f"{EMBEDDINGS_ENDPOINT}/embed",
                json={"texts": texts, "normalize": True},
                timeout=60
            )
            response.raise_for_status()
            data = response.json()
            return data['embeddings']

        except Exception as e:
            logger.error(f"Failed to get embeddings: {e}")
            raise

    def add_document(self, tenant_id: str, doc_id: str, url: str, metadata: Dict[str, Any] = None):
        """
        Add document node to graph

        Args:
            tenant_id: Tenant ID
            doc_id: Document ID
            url: Document URL
            metadata: Additional metadata
        """
        with self.driver.session() as session:
            def create_doc(tx):
                import json
                return tx.run("""
                    MERGE (d:Document {id: $doc_id, tenant_id: $tenant_id})
                    SET d.url = $url,
                        d.metadata = $metadata,
                        d.created_at = datetime()
                    RETURN d
                """, {
                    "doc_id": doc_id,
                    "tenant_id": tenant_id,
                    "url": url,
                    "metadata": json.dumps(metadata or {})
                })
            session.execute_write(create_doc)

        logger.debug(f"Added document {doc_id} for tenant {tenant_id}")

    def add_chunk_with_embedding(
        self,
        tenant_id: str,
        chunk_id: str,
        text: str,
        embedding: List[float],
        doc_id: str,
        metadata: Dict[str, Any] = None
    ):
        """
        Add chunk node with embedding to graph

        Args:
            tenant_id: Tenant ID
            chunk_id: Chunk ID
            text: Chunk text
            embedding: Embedding vector
            doc_id: Parent document ID
            metadata: Additional metadata
        """
        with self.driver.session() as session:
            def create_chunk(tx):
                import json
                return tx.run("""
                    MERGE (c:Chunk {id: $chunk_id, tenant_id: $tenant_id})
                    SET c.text = $text,
                        c.embedding = $embedding,
                        c.metadata = $metadata,
                        c.created_at = datetime()
                    WITH c
                    MATCH (d:Document {id: $doc_id, tenant_id: $tenant_id})
                    MERGE (d)-[:CONTAINS_CHUNK]->(c)
                    RETURN c
                """, {
                    "chunk_id": chunk_id,
                    "tenant_id": tenant_id,
                    "text": text,
                    "embedding": embedding,
                    "doc_id": doc_id,
                    "metadata": json.dumps(metadata or {})
                })
            session.execute_write(create_chunk)

        logger.debug(f"Added chunk {chunk_id} for document {doc_id}")

    def add_entity(
        self,
        tenant_id: str,
        entity_id: str,
        name: str,
        entity_type: str,
        description: str = ""
    ):
        """
        Add entity node to graph

        Args:
            tenant_id: Tenant ID
            entity_id: Entity ID
            name: Entity name
            entity_type: Entity type
            description: Entity description
        """
        with self.driver.session() as session:
            session.run("""
                MERGE (e:Entity {id: $entity_id, tenant_id: $tenant_id})
                SET e.name = $name,
                    e.type = $entity_type,
                    e.description = $description,
                    e.created_at = datetime()
                RETURN e
            """, {
                "entity_id": entity_id,
                "tenant_id": tenant_id,
                "name": name,
                "entity_type": entity_type,
                "description": description
            })

        logger.debug(f"Added entity {name} ({entity_type})")

    def link_entity_to_chunk(self, tenant_id: str, entity_id: str, chunk_id: str):
        """
        Create relationship between entity and chunk

        Args:
            tenant_id: Tenant ID
            entity_id: Entity ID
            chunk_id: Chunk ID
        """
        with self.driver.session() as session:
            session.run("""
                MATCH (e:Entity {id: $entity_id, tenant_id: $tenant_id})
                MATCH (c:Chunk {id: $chunk_id, tenant_id: $tenant_id})
                MERGE (e)-[:MENTIONED_IN]->(c)
            """, {
                "entity_id": entity_id,
                "chunk_id": chunk_id,
                "tenant_id": tenant_id
            })

    def process_document_chunks(
        self,
        tenant_id: str,
        doc_id: str,
        url: str,
        chunks: List[Document]
    ):
        """
        Process all chunks for a document: embeddings, entities, graph construction

        Args:
            tenant_id: Tenant ID
            doc_id: Document ID
            url: Document URL
            chunks: List of document chunks
        """
        logger.info(f"Processing {len(chunks)} chunks for document {doc_id}")

        if not chunks:
            logger.warning(f"No chunks to process for document {doc_id}")
            return

        # Add document node
        self.add_document(tenant_id, doc_id, url)

        # Extract texts for embedding
        texts = [chunk.page_content for chunk in chunks]

        # Get embeddings
        logger.info("Generating embeddings...")
        embeddings = self.get_embeddings(texts)

        if len(embeddings) != len(chunks):
            raise ValueError(
                f"Embedding count mismatch: {len(embeddings)} vs {len(chunks)} chunks"
            )

        # Process each chunk
        for chunk, embedding in zip(chunks, embeddings):
            chunk_id = chunk.metadata['chunk_id']
            text = chunk.page_content

            # Add chunk to graph
            self.add_chunk_with_embedding(
                tenant_id=tenant_id,
                chunk_id=chunk_id,
                text=text,
                embedding=embedding,
                doc_id=doc_id,
                metadata=chunk.metadata
            )

        logger.info(f"Successfully processed all chunks for document {doc_id}")

    def extract_and_link_entities(
        self,
        tenant_id: str,
        chunks: List[Document],
        entities_per_chunk: List[List[Entity]]
    ):
        """
        Extract entities and create relationships with chunks

        Args:
            tenant_id: Tenant ID
            chunks: List of document chunks
            entities_per_chunk: Entities extracted from each chunk
        """
        logger.info(f"Linking entities to chunks")

        for chunk, entities in zip(chunks, entities_per_chunk):
            chunk_id = chunk.metadata['chunk_id']

            for entity in entities:
                # Generate entity ID (deterministic based on tenant + name + type)
                entity_id = self._generate_entity_id(tenant_id, entity.name, entity.entity_type)

                # Add entity
                self.add_entity(
                    tenant_id=tenant_id,
                    entity_id=entity_id,
                    name=entity.name,
                    entity_type=entity.entity_type,
                    description=entity.description
                )

                # Link to chunk
                self.link_entity_to_chunk(tenant_id, entity_id, chunk_id)

        logger.info("Successfully linked entities to chunks")

    def _generate_entity_id(self, tenant_id: str, name: str, entity_type: str) -> str:
        """
        Generate deterministic entity ID

        Args:
            tenant_id: Tenant ID
            name: Entity name
            entity_type: Entity type

        Returns:
            Entity ID
        """
        content = f"{tenant_id}_{name.lower()}_{entity_type.lower()}"
        return hashlib.md5(content.encode()).hexdigest()

    def get_tenant_stats(self, tenant_id: str) -> Dict[str, int]:
        """
        Get statistics for a tenant

        Args:
            tenant_id: Tenant ID

        Returns:
            Dictionary with counts
        """
        with self.driver.session() as session:
            result = session.run("""
                MATCH (d:Document {tenant_id: $tenant_id})
                OPTIONAL MATCH (d)-[:CONTAINS_CHUNK]->(c:Chunk)
                OPTIONAL MATCH (e:Entity {tenant_id: $tenant_id})
                RETURN
                    count(DISTINCT d) as doc_count,
                    count(DISTINCT c) as chunk_count,
                    count(DISTINCT e) as entity_count
            """, {"tenant_id": tenant_id})

            record = result.single()
            return {
                "documents": record["doc_count"],
                "chunks": record["chunk_count"],
                "entities": record["entity_count"]
            }
