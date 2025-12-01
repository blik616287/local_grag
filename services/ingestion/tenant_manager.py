"""
Tenant management and ID generation
"""

import hashlib
import logging
import random
import string
from typing import Dict, Optional
from datetime import datetime

import psycopg2
from psycopg2.extras import RealDictCursor

from config import POSTGRES_URI, TENANT_ID_STRATEGY, TENANT_ID_METADATA_FIELD

logger = logging.getLogger(__name__)


class TenantManager:
    """
    Manages tenant IDs and metadata
    """

    def __init__(self):
        """Initialize database connection"""
        logger.info("Initializing TenantManager")
        self.conn = psycopg2.connect(POSTGRES_URI)
        self.tenant_id_cache = {}

    def close(self):
        """Close database connection"""
        if self.conn:
            self.conn.close()
            logger.info("Closed database connection")

    def generate_tenant_id_sha256_base36(self, text: str) -> str:
        """
        Generate deterministic tenant ID using SHA-256 with Base36 encoding

        Args:
            text: Input text (collection name)

        Returns:
            10-character tenant ID
        """
        # Limit input to 30 characters
        text = text[:30]

        # SHA-256 hash
        hash_bytes = hashlib.sha256(text.encode('utf-8')).digest()

        # Take first 8 bytes for better entropy (64 bits)
        hash_int = int.from_bytes(hash_bytes[:8], 'big')

        # Convert to base36 (0-9, a-z)
        def to_base36(num):
            chars = '0123456789abcdefghijklmnopqrstuvwxyz'
            result = []
            while num > 0 and len(result) < 10:
                result.append(chars[num % 36])
                num //= 36
            return ''.join(reversed(result))

        base36 = to_base36(hash_int)

        # Pad or truncate to exactly 10 characters
        return base36.ljust(10, '0')[:10]

    def generate_random_tenant_id(self) -> str:
        """
        Generate a random 10-character alphanumeric tenant ID

        Returns:
            Random tenant ID
        """
        chars = string.ascii_lowercase + string.digits
        return ''.join(random.choices(chars, k=10))

    def lookup_or_create_tenant_id(self, collection_name: str) -> str:
        """
        Look up existing tenant ID or create a new one

        Args:
            collection_name: Collection name

        Returns:
            Tenant ID
        """
        # Check local cache first
        if collection_name in self.tenant_id_cache:
            return self.tenant_id_cache[collection_name]

        # Generate the deterministic tenant ID
        tenant_id = self.generate_tenant_id_sha256_base36(collection_name)

        with self.conn.cursor(cursor_factory=RealDictCursor) as cur:
            try:
                # Try to get existing mapping
                cur.execute(
                    "SELECT tenant_id, collection_name FROM tenants WHERE tenant_id = %s",
                    (tenant_id,)
                )
                row = cur.fetchone()

                if row:
                    # Tenant exists, verify collection_name matches
                    stored_collection = row['collection_name']
                    if stored_collection != collection_name:
                        logger.warning(
                            f"Tenant ID collision detected: {tenant_id} maps to both "
                            f"'{stored_collection}' and '{collection_name}'. Using random ID."
                        )
                        # Generate a random ID for collision case
                        tenant_id = self.generate_random_tenant_id()
                else:
                    # New tenant, store the mapping
                    cur.execute("""
                        INSERT INTO tenants (tenant_id, collection_name, generation_method, created_at, first_ingestion)
                        VALUES (%s, %s, %s, %s, %s)
                    """, (
                        tenant_id,
                        collection_name,
                        'sha256_base36',
                        datetime.utcnow(),
                        datetime.utcnow()
                    ))
                    self.conn.commit()
                    logger.info(f"Created new tenant mapping: '{collection_name}' -> '{tenant_id}'")

            except Exception as e:
                logger.error(f"Failed to lookup/store tenant mapping: {e}")
                self.conn.rollback()

        # Cache the result
        self.tenant_id_cache[collection_name] = tenant_id

        return tenant_id

    def determine_tenant_id(self, message: Dict) -> str:
        """
        Determine tenant ID based on configured strategy

        Args:
            message: Message data

        Returns:
            Tenant ID
        """
        explicit_tenant_id = message.get('tenant_id')

        # If explicitly provided and not "auto", use it
        if explicit_tenant_id and explicit_tenant_id != "auto":
            logger.info(f"Using explicit tenant ID: {explicit_tenant_id}")
            return explicit_tenant_id

        # Extract from metadata field
        if TENANT_ID_STRATEGY == "dynamic":
            collection_name = message.get(TENANT_ID_METADATA_FIELD)
            if collection_name:
                tenant_id = self.lookup_or_create_tenant_id(collection_name)
                logger.info(f"Generated tenant ID for '{collection_name}': {tenant_id}")
                return tenant_id

        # Fallback to random
        tenant_id = self.generate_random_tenant_id()
        logger.info(f"Generated random tenant ID: {tenant_id}")
        return tenant_id

    def update_tenant_stats(self, tenant_id: str, stats: Dict):
        """
        Update tenant statistics

        Args:
            tenant_id: Tenant ID
            stats: Statistics to update
        """
        with self.conn.cursor() as cur:
            try:
                cur.execute("""
                    UPDATE tenants
                    SET last_ingestion = %s,
                        entity_count = COALESCE(%s, entity_count),
                        chunk_count = COALESCE(%s, chunk_count)
                    WHERE tenant_id = %s
                """, (
                    datetime.utcnow(),
                    stats.get('entities'),
                    stats.get('chunks'),
                    tenant_id
                ))
                self.conn.commit()
            except Exception as e:
                logger.error(f"Failed to update tenant stats: {e}")
                self.conn.rollback()
