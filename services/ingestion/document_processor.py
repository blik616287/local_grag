"""
Document loading and chunking processor
"""

import logging
import hashlib
from typing import List, Dict, Any
from urllib.parse import urlparse

import requests
from bs4 import BeautifulSoup
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document

from config import CHUNK_SIZE, CHUNK_OVERLAP, MAX_DOCUMENT_SIZE_MB

logger = logging.getLogger(__name__)


class DocumentProcessor:
    """
    Handles document loading, parsing, and chunking
    """

    def __init__(self):
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP,
            separators=["\n\n", "\n", ". ", " ", ""],
            length_function=len,
        )

    def load_from_url(self, url: str) -> str:
        """
        Load content from URL

        Args:
            url: URL to load

        Returns:
            Text content from URL

        Raises:
            Exception: If loading fails
        """
        logger.info(f"Loading document from URL: {url}")

        try:
            # Validate URL
            parsed = urlparse(url)
            if not parsed.scheme in ['http', 'https']:
                raise ValueError(f"Invalid URL scheme: {parsed.scheme}")

            # Fetch content
            headers = {
                'User-Agent': 'Mozilla/5.0 (compatible; GraphRAG/1.0)'
            }
            response = requests.get(url, headers=headers, timeout=30)
            response.raise_for_status()

            # Check content size
            content_length = len(response.content)
            max_size_bytes = MAX_DOCUMENT_SIZE_MB * 1024 * 1024
            if content_length > max_size_bytes:
                raise ValueError(
                    f"Document too large: {content_length / (1024 * 1024):.2f}MB "
                    f"(max: {MAX_DOCUMENT_SIZE_MB}MB)"
                )

            # Parse HTML content
            content_type = response.headers.get('content-type', '').lower()
            if 'html' in content_type:
                soup = BeautifulSoup(response.content, 'lxml')

                # Remove script and style elements
                for element in soup(['script', 'style', 'nav', 'footer', 'header']):
                    element.decompose()

                # Extract text
                text = soup.get_text(separator='\n', strip=True)
            else:
                # Plain text or other content
                text = response.text

            # Clean up text
            lines = [line.strip() for line in text.split('\n') if line.strip()]
            text = '\n'.join(lines)

            logger.info(f"Loaded {len(text)} characters from {url}")
            return text

        except requests.exceptions.RequestException as e:
            logger.error(f"Failed to load URL {url}: {e}")
            raise Exception(f"URL loading failed: {str(e)}")
        except Exception as e:
            logger.error(f"Error processing document from {url}: {e}")
            raise

    def chunk_document(
        self,
        text: str,
        doc_id: str,
        url: str,
        metadata: Dict[str, Any] = None
    ) -> List[Document]:
        """
        Split document into chunks

        Args:
            text: Document text
            doc_id: Document ID
            url: Source URL
            metadata: Additional metadata

        Returns:
            List of LangChain Document chunks
        """
        logger.info(f"Chunking document {doc_id} ({len(text)} characters)")

        if not text or len(text.strip()) == 0:
            logger.warning(f"Document {doc_id} has no content")
            return []

        # Create base metadata
        base_metadata = {
            "doc_id": doc_id,
            "url": url,
            **(metadata or {})
        }

        # Create a single document
        doc = Document(
            page_content=text,
            metadata=base_metadata
        )

        # Split into chunks
        chunks = self.text_splitter.split_documents([doc])

        # Add chunk-specific metadata
        for i, chunk in enumerate(chunks):
            chunk_id = self._generate_chunk_id(doc_id, i)
            chunk.metadata.update({
                "chunk_id": chunk_id,
                "chunk_index": i,
                "total_chunks": len(chunks)
            })

        logger.info(f"Created {len(chunks)} chunks for document {doc_id}")
        return chunks

    def _generate_chunk_id(self, doc_id: str, chunk_index: int) -> str:
        """
        Generate deterministic chunk ID

        Args:
            doc_id: Document ID
            chunk_index: Chunk index

        Returns:
            Chunk ID
        """
        content = f"{doc_id}_{chunk_index}"
        return hashlib.md5(content.encode()).hexdigest()[:16]

    def process_url(
        self,
        url: str,
        doc_id: str,
        metadata: Dict[str, Any] = None
    ) -> List[Document]:
        """
        Complete document processing pipeline: load + chunk

        Args:
            url: URL to process
            doc_id: Document ID
            metadata: Additional metadata

        Returns:
            List of document chunks
        """
        logger.info(f"Processing document: {doc_id} from {url}")

        try:
            # Load content
            text = self.load_from_url(url)

            # Chunk content
            chunks = self.chunk_document(text, doc_id, url, metadata)

            logger.info(f"Successfully processed document {doc_id}: {len(chunks)} chunks")
            return chunks

        except Exception as e:
            logger.error(f"Failed to process document {doc_id}: {e}")
            raise
