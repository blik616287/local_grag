#!/usr/bin/env python3
"""
Graph validation service for GraphRAG
Checks graph integrity, orphaned nodes, embedding quality, etc.
"""

import logging
import time
from typing import Dict, List, Any, Optional
from datetime import datetime
import psycopg2
import psycopg2.extras

from neo4j import GraphDatabase

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class GraphValidator:
    """
    Validates Neo4j graph integrity
    """

    def __init__(self, neo4j_uri: str, neo4j_user: str, neo4j_password: str, postgres_uri: str):
        """Initialize validator"""
        logger.info(f"Connecting to Neo4j at {neo4j_uri}")
        self.driver = GraphDatabase.driver(neo4j_uri, auth=(neo4j_user, neo4j_password))
        self.driver.verify_connectivity()
        logger.info("Connected to Neo4j")

        logger.info("Connecting to PostgreSQL")
        self.db_conn = psycopg2.connect(postgres_uri)
        logger.info("Connected to PostgreSQL")

    def close(self):
        """Close connections"""
        if self.driver:
            self.driver.close()
        if self.db_conn:
            self.db_conn.close()

    def run_validation(self, tenant_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Run full validation suite

        Args:
            tenant_id: Optional tenant ID to validate (None for all)

        Returns:
            Validation results
        """
        logger.info(f"Starting validation for tenant: {tenant_id or 'ALL'}")
        start_time = time.time()

        results = {
            "timestamp": datetime.utcnow().isoformat(),
            "tenant_id": tenant_id,
            "checks": {},
            "issues": [],
            "stats": {}
        }

        # Run all validation checks
        results["checks"]["orphaned_chunks"] = self.check_orphaned_chunks(tenant_id)
        results["checks"]["orphaned_entities"] = self.check_orphaned_entities(tenant_id)
        results["checks"]["missing_embeddings"] = self.check_missing_embeddings(tenant_id)
        results["checks"]["empty_chunks"] = self.check_empty_chunks(tenant_id)
        results["checks"]["duplicate_chunks"] = self.check_duplicate_chunks(tenant_id)
        results["checks"]["missing_tenant_ids"] = self.check_missing_tenant_ids()
        results["checks"]["postgres_neo4j_sync"] = self.check_postgres_neo4j_sync(tenant_id)

        # Collect all issues
        for check_name, check_result in results["checks"].items():
            if check_result["issues"]:
                for issue in check_result["issues"]:
                    results["issues"].append({
                        "check": check_name,
                        "severity": check_result["severity"],
                        **issue
                    })

        # Get stats
        results["stats"] = self.get_graph_stats(tenant_id)

        results["duration_seconds"] = time.time() - start_time
        results["total_issues"] = len(results["issues"])
        results["status"] = "PASS" if len(results["issues"]) == 0 else "FAIL"

        logger.info(f"Validation complete: {results['status']} ({results['total_issues']} issues)")

        # Record in PostgreSQL
        self.record_validation_run(results)

        return results

    def check_orphaned_chunks(self, tenant_id: Optional[str] = None) -> Dict[str, Any]:
        """Check for chunks not connected to any document"""
        logger.info("Checking for orphaned chunks")

        with self.driver.session() as session:
            query = """
                MATCH (c:Chunk)
                WHERE NOT ((:Document)-[:CONTAINS_CHUNK]->(c))
            """
            if tenant_id:
                query += " AND c.tenant_id = $tenant_id"
            query += """
                RETURN c.id as chunk_id, c.tenant_id as tenant_id
                LIMIT 100
            """

            result = session.run(query, {"tenant_id": tenant_id} if tenant_id else {})
            orphaned = [dict(record) for record in result]

        return {
            "passed": len(orphaned) == 0,
            "severity": "high",
            "count": len(orphaned),
            "issues": orphaned
        }

    def check_orphaned_entities(self, tenant_id: Optional[str] = None) -> Dict[str, Any]:
        """Check for entities not connected to any chunk"""
        logger.info("Checking for orphaned entities")

        with self.driver.session() as session:
            query = """
                MATCH (e:Entity)
                WHERE NOT ((e)-[:MENTIONED_IN]->(:Chunk))
            """
            if tenant_id:
                query += " AND e.tenant_id = $tenant_id"
            query += """
                RETURN e.id as entity_id, e.name as name, e.tenant_id as tenant_id
                LIMIT 100
            """

            result = session.run(query, {"tenant_id": tenant_id} if tenant_id else {})
            orphaned = [dict(record) for record in result]

        return {
            "passed": len(orphaned) == 0,
            "severity": "medium",
            "count": len(orphaned),
            "issues": orphaned
        }

    def check_missing_embeddings(self, tenant_id: Optional[str] = None) -> Dict[str, Any]:
        """Check for chunks without embeddings"""
        logger.info("Checking for missing embeddings")

        with self.driver.session() as session:
            query = """
                MATCH (c:Chunk)
                WHERE c.embedding IS NULL
            """
            if tenant_id:
                query += " AND c.tenant_id = $tenant_id"
            query += """
                RETURN c.id as chunk_id, c.tenant_id as tenant_id
                LIMIT 100
            """

            result = session.run(query, {"tenant_id": tenant_id} if tenant_id else {})
            missing = [dict(record) for record in result]

        return {
            "passed": len(missing) == 0,
            "severity": "high",
            "count": len(missing),
            "issues": missing
        }

    def check_empty_chunks(self, tenant_id: Optional[str] = None) -> Dict[str, Any]:
        """Check for chunks with empty or very short text"""
        logger.info("Checking for empty chunks")

        with self.driver.session() as session:
            query = """
                MATCH (c:Chunk)
                WHERE c.text IS NULL OR size(c.text) < 10
            """
            if tenant_id:
                query += " AND c.tenant_id = $tenant_id"
            query += """
                RETURN c.id as chunk_id, c.text as text, c.tenant_id as tenant_id
                LIMIT 100
            """

            result = session.run(query, {"tenant_id": tenant_id} if tenant_id else {})
            empty = [dict(record) for record in result]

        return {
            "passed": len(empty) == 0,
            "severity": "medium",
            "count": len(empty),
            "issues": empty
        }

    def check_duplicate_chunks(self, tenant_id: Optional[str] = None) -> Dict[str, Any]:
        """Check for duplicate chunk IDs"""
        logger.info("Checking for duplicate chunks")

        with self.driver.session() as session:
            query = """
                MATCH (c:Chunk)
            """
            if tenant_id:
                query += " WHERE c.tenant_id = $tenant_id"
            query += """
                WITH c.id as chunk_id, c.tenant_id as tenant_id, count(*) as count
                WHERE count > 1
                RETURN chunk_id, tenant_id, count
                LIMIT 100
            """

            result = session.run(query, {"tenant_id": tenant_id} if tenant_id else {})
            duplicates = [dict(record) for record in result]

        return {
            "passed": len(duplicates) == 0,
            "severity": "high",
            "count": len(duplicates),
            "issues": duplicates
        }

    def check_missing_tenant_ids(self) -> Dict[str, Any]:
        """Check for nodes missing tenant_id property"""
        logger.info("Checking for missing tenant IDs")

        with self.driver.session() as session:
            result = session.run("""
                MATCH (n)
                WHERE n.tenant_id IS NULL
                RETURN labels(n)[0] as label, count(*) as count
            """)
            missing = [dict(record) for record in result]

        return {
            "passed": len(missing) == 0,
            "severity": "high",
            "count": sum(m["count"] for m in missing),
            "issues": missing
        }

    def check_postgres_neo4j_sync(self, tenant_id: Optional[str] = None) -> Dict[str, Any]:
        """Check if PostgreSQL and Neo4j are in sync"""
        logger.info("Checking PostgreSQL and Neo4j sync")

        # Get document count from PostgreSQL
        with self.db_conn.cursor() as cur:
            if tenant_id:
                cur.execute("""
                    SELECT COUNT(*) FROM documents d
                    JOIN tenants t ON d.tenant_id = t.tenant_id
                    WHERE t.tenant_id = %s AND d.status = 'completed'
                """, (tenant_id,))
            else:
                cur.execute("SELECT COUNT(*) FROM documents WHERE status = 'completed'")

            pg_count = cur.fetchone()[0]

        # Get document count from Neo4j
        with self.driver.session() as session:
            query = "MATCH (d:Document)"
            if tenant_id:
                query += " WHERE d.tenant_id = $tenant_id"
            query += " RETURN count(d) as count"

            result = session.run(query, {"tenant_id": tenant_id} if tenant_id else {})
            neo4j_count = result.single()["count"]

        diff = abs(pg_count - neo4j_count)
        issues = []
        if diff > 0:
            issues.append({
                "postgres_count": pg_count,
                "neo4j_count": neo4j_count,
                "difference": diff
            })

        return {
            "passed": diff == 0,
            "severity": "high",
            "count": diff,
            "issues": issues
        }

    def get_graph_stats(self, tenant_id: Optional[str] = None) -> Dict[str, int]:
        """Get graph statistics"""
        logger.info("Collecting graph statistics")

        with self.driver.session() as session:
            # Node counts
            query = "MATCH (n)"
            if tenant_id:
                query += " WHERE n.tenant_id = $tenant_id"
            query += " RETURN labels(n)[0] as label, count(*) as count"

            result = session.run(query, {"tenant_id": tenant_id} if tenant_id else {})
            node_counts = {record["label"]: record["count"] for record in result}

            # Relationship counts
            query = "MATCH ()-[r]->()"
            if tenant_id:
                query += " MATCH (n) WHERE n.tenant_id = $tenant_id WITH n MATCH (n)-[r]->()"
            query += " RETURN type(r) as rel_type, count(*) as count"

            result = session.run(query, {"tenant_id": tenant_id} if tenant_id else {})
            rel_counts = {record["rel_type"]: record["count"] for record in result}

        return {
            "nodes": node_counts,
            "relationships": rel_counts,
            "total_nodes": sum(node_counts.values()),
            "total_relationships": sum(rel_counts.values())
        }

    def record_validation_run(self, results: Dict[str, Any]):
        """Record validation run in PostgreSQL"""
        logger.info("Recording validation run in PostgreSQL")

        try:
            with self.db_conn.cursor() as cur:
                # Insert validation run
                cur.execute("""
                    INSERT INTO validation_runs
                    (tenant_id, status, total_checks, total_issues, duration_seconds, results)
                    VALUES (%s, %s, %s, %s, %s, %s)
                    RETURNING run_id
                """, (
                    results.get("tenant_id"),
                    results["status"],
                    len(results["checks"]),
                    results["total_issues"],
                    results["duration_seconds"],
                    psycopg2.extras.Json(results)
                ))

                run_id = cur.fetchone()[0]

                # Insert individual issues
                for issue in results["issues"]:
                    cur.execute("""
                        INSERT INTO validation_issues
                        (run_id, tenant_id, check_name, severity, issue_type, description, affected_nodes)
                        VALUES (%s, %s, %s, %s, %s, %s, %s)
                    """, (
                        run_id,
                        results.get("tenant_id"),
                        issue["check"],
                        issue["severity"],
                        issue["check"],
                        str(issue),
                        psycopg2.extras.Json([issue])
                    ))

                self.db_conn.commit()
                logger.info(f"Recorded validation run {run_id}")

        except Exception as e:
            logger.error(f"Failed to record validation run: {e}")
            self.db_conn.rollback()

    def repair_orphaned_chunks(self, tenant_id: Optional[str] = None, delete: bool = False) -> int:
        """
        Repair orphaned chunks

        Args:
            tenant_id: Optional tenant ID
            delete: If True, delete orphaned chunks; if False, just report

        Returns:
            Number of chunks repaired/deleted
        """
        logger.info(f"Repairing orphaned chunks (delete={delete})")

        if not delete:
            # Just report
            check_result = self.check_orphaned_chunks(tenant_id)
            return check_result["count"]

        # Delete orphaned chunks
        with self.driver.session() as session:
            def delete_orphaned(tx):
                query = """
                    MATCH (c:Chunk)
                    WHERE NOT ((:Document)-[:CONTAINS_CHUNK]->(c))
                """
                if tenant_id:
                    query += " AND c.tenant_id = $tenant_id"
                query += " DELETE c RETURN count(*) as deleted"

                result = tx.run(query, {"tenant_id": tenant_id} if tenant_id else {})
                return result.single()["deleted"]

            deleted = session.execute_write(delete_orphaned)
            logger.info(f"Deleted {deleted} orphaned chunks")
            return deleted


def main():
    """Main validation runner"""
    import os
    import json

    NEO4J_URI = os.environ.get("NEO4J_URI", "bolt://localhost:7687")
    NEO4J_USER = os.environ.get("NEO4J_USER", "neo4j")
    NEO4J_PASSWORD = os.environ.get("NEO4J_PASSWORD", "graphrag123")
    POSTGRES_URI = os.environ.get("POSTGRES_URI", "postgresql://graphrag:graphrag123@localhost:5433/graphrag")

    validator = GraphValidator(NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD, POSTGRES_URI)

    try:
        # Run validation
        results = validator.run_validation()

        # Print results
        print("\n" + "=" * 70)
        print(f"VALIDATION RESULTS: {results['status']}")
        print("=" * 70)
        print(f"Timestamp: {results['timestamp']}")
        print(f"Total Checks: {len(results['checks'])}")
        print(f"Total Issues: {results['total_issues']}")
        print(f"Duration: {results['duration_seconds']:.2f}s")
        print("\nChecks:")
        for check_name, check_result in results["checks"].items():
            status = "✅ PASS" if check_result["passed"] else f"❌ FAIL ({check_result['count']})"
            print(f"  {check_name:30s}: {status}")

        if results["issues"]:
            print("\nIssues:")
            for i, issue in enumerate(results["issues"][:10], 1):
                print(f"  {i}. [{issue['severity'].upper()}] {issue['check']}")
                print(f"     {issue}")

        print("\nGraph Stats:")
        print(f"  Total Nodes: {results['stats']['total_nodes']}")
        print(f"  Total Relationships: {results['stats']['total_relationships']}")
        for label, count in results['stats']['nodes'].items():
            print(f"    {label}: {count}")

        print("=" * 70)

    finally:
        validator.close()


if __name__ == "__main__":
    main()
