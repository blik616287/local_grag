-- GraphRAG Database Initialization Script
-- PostgreSQL + pgvector for state management

-- Enable pgvector extension
CREATE EXTENSION IF NOT EXISTS vector;

-- Tenants Table
CREATE TABLE IF NOT EXISTS tenants (
    tenant_id VARCHAR(50) PRIMARY KEY,
    collection_name VARCHAR(255) NOT NULL,
    generation_method VARCHAR(50),
    metadata JSONB,
    created_at TIMESTAMP DEFAULT NOW(),
    first_ingestion TIMESTAMP,
    last_ingestion TIMESTAMP,
    document_count INTEGER DEFAULT 0,
    entity_count INTEGER DEFAULT 0,
    chunk_count INTEGER DEFAULT 0
);

CREATE INDEX idx_tenants_collection ON tenants(collection_name);
CREATE INDEX idx_tenants_created ON tenants(created_at DESC);

-- Documents Table
CREATE TABLE IF NOT EXISTS documents (
    document_id VARCHAR(255) PRIMARY KEY,
    tenant_id VARCHAR(50) NOT NULL REFERENCES tenants(tenant_id),
    url TEXT NOT NULL,
    status VARCHAR(50) NOT NULL,  -- pending, processing, completed, failed
    metadata JSONB,
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW(),
    completed_at TIMESTAMP,
    error_message TEXT,
    retry_count INTEGER DEFAULT 0,
    processing_time_seconds INTEGER,
    chunk_count INTEGER DEFAULT 0,
    entity_count INTEGER DEFAULT 0
);

CREATE INDEX idx_documents_tenant ON documents(tenant_id);
CREATE INDEX idx_documents_status ON documents(status);
CREATE INDEX idx_documents_created ON documents(created_at DESC);
CREATE INDEX idx_documents_tenant_status ON documents(tenant_id, status);

-- Retry Tracking Table
CREATE TABLE IF NOT EXISTS retry_tracking (
    id SERIAL PRIMARY KEY,
    message_id VARCHAR(255) NOT NULL,
    tenant_id VARCHAR(50) NOT NULL,
    document_id VARCHAR(255),
    attempt_number INTEGER NOT NULL,
    status VARCHAR(50) NOT NULL,  -- processing, success, failed, dlq
    error_message TEXT,
    error_type VARCHAR(100),
    created_at TIMESTAMP DEFAULT NOW()
);

CREATE INDEX idx_retry_tracking_message ON retry_tracking(message_id);
CREATE INDEX idx_retry_tracking_tenant ON retry_tracking(tenant_id);
CREATE INDEX idx_retry_tracking_document ON retry_tracking(document_id);
CREATE INDEX idx_retry_tracking_status ON retry_tracking(status);

-- Validation Runs Table
CREATE TABLE IF NOT EXISTS validation_runs (
    id SERIAL PRIMARY KEY,
    tenant_id VARCHAR(50) NOT NULL,
    run_id VARCHAR(100) NOT NULL UNIQUE,
    status VARCHAR(50) NOT NULL,  -- running, completed, failed
    validation_type VARCHAR(100),  -- full, incremental, repair
    started_at TIMESTAMP DEFAULT NOW(),
    completed_at TIMESTAMP,
    duration_seconds INTEGER,
    issues_found INTEGER DEFAULT 0,
    issues_repaired INTEGER DEFAULT 0,
    metadata JSONB
);

CREATE INDEX idx_validation_runs_tenant ON validation_runs(tenant_id);
CREATE INDEX idx_validation_runs_status ON validation_runs(status);
CREATE INDEX idx_validation_runs_started ON validation_runs(started_at DESC);

-- Validation Issues Table
CREATE TABLE IF NOT EXISTS validation_issues (
    id SERIAL PRIMARY KEY,
    run_id VARCHAR(100) NOT NULL,
    tenant_id VARCHAR(50) NOT NULL,
    issue_type VARCHAR(100) NOT NULL,  -- orphaned_node, missing_edge, etc.
    severity VARCHAR(20) NOT NULL,  -- critical, warning, info
    entity_id VARCHAR(255),
    description TEXT,
    repaired BOOLEAN DEFAULT FALSE,
    repair_action TEXT,
    created_at TIMESTAMP DEFAULT NOW()
);

CREATE INDEX idx_validation_issues_run ON validation_issues(run_id);
CREATE INDEX idx_validation_issues_tenant ON validation_issues(tenant_id);
CREATE INDEX idx_validation_issues_type ON validation_issues(issue_type);
CREATE INDEX idx_validation_issues_severity ON validation_issues(severity);

-- Processing Queue State (for monitoring)
CREATE TABLE IF NOT EXISTS queue_state (
    id SERIAL PRIMARY KEY,
    queue_name VARCHAR(100) NOT NULL,
    message_id VARCHAR(255) NOT NULL,
    tenant_id VARCHAR(50),
    document_id VARCHAR(255),
    status VARCHAR(50) NOT NULL,  -- enqueued, processing, completed, failed
    enqueued_at TIMESTAMP DEFAULT NOW(),
    started_at TIMESTAMP,
    completed_at TIMESTAMP,
    worker_id VARCHAR(100),
    metadata JSONB
);

CREATE INDEX idx_queue_state_queue ON queue_state(queue_name);
CREATE INDEX idx_queue_state_status ON queue_state(status);
CREATE INDEX idx_queue_state_tenant ON queue_state(tenant_id);
CREATE INDEX idx_queue_state_message ON queue_state(message_id);

-- System Metrics Table
CREATE TABLE IF NOT EXISTS system_metrics (
    id SERIAL PRIMARY KEY,
    metric_name VARCHAR(100) NOT NULL,
    metric_value NUMERIC,
    labels JSONB,
    recorded_at TIMESTAMP DEFAULT NOW()
);

CREATE INDEX idx_system_metrics_name ON system_metrics(metric_name);
CREATE INDEX idx_system_metrics_recorded ON system_metrics(recorded_at DESC);

-- Function to update document updated_at timestamp
CREATE OR REPLACE FUNCTION update_modified_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- Trigger for documents table
CREATE TRIGGER update_documents_modtime
    BEFORE UPDATE ON documents
    FOR EACH ROW
    EXECUTE FUNCTION update_modified_column();

-- Function to increment tenant document count
CREATE OR REPLACE FUNCTION increment_tenant_document_count()
RETURNS TRIGGER AS $$
BEGIN
    IF NEW.status = 'completed' AND (OLD.status IS NULL OR OLD.status != 'completed') THEN
        UPDATE tenants
        SET document_count = document_count + 1,
            last_ingestion = NOW()
        WHERE tenant_id = NEW.tenant_id;
    END IF;
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- Trigger for document completion
CREATE TRIGGER increment_tenant_docs
    AFTER INSERT OR UPDATE ON documents
    FOR EACH ROW
    EXECUTE FUNCTION increment_tenant_document_count();

-- Create views for monitoring
CREATE OR REPLACE VIEW v_tenant_statistics AS
SELECT
    t.tenant_id,
    t.collection_name,
    t.document_count,
    t.entity_count,
    t.chunk_count,
    t.created_at,
    t.first_ingestion,
    t.last_ingestion,
    COUNT(DISTINCT d.document_id) FILTER (WHERE d.status = 'pending') as pending_docs,
    COUNT(DISTINCT d.document_id) FILTER (WHERE d.status = 'processing') as processing_docs,
    COUNT(DISTINCT d.document_id) FILTER (WHERE d.status = 'completed') as completed_docs,
    COUNT(DISTINCT d.document_id) FILTER (WHERE d.status = 'failed') as failed_docs,
    AVG(d.processing_time_seconds) FILTER (WHERE d.status = 'completed') as avg_processing_time
FROM tenants t
LEFT JOIN documents d ON t.tenant_id = d.tenant_id
GROUP BY t.tenant_id, t.collection_name, t.document_count, t.entity_count,
         t.chunk_count, t.created_at, t.first_ingestion, t.last_ingestion;

CREATE OR REPLACE VIEW v_recent_failures AS
SELECT
    d.document_id,
    d.tenant_id,
    t.collection_name,
    d.url,
    d.error_message,
    d.retry_count,
    d.updated_at
FROM documents d
JOIN tenants t ON d.tenant_id = t.tenant_id
WHERE d.status = 'failed'
ORDER BY d.updated_at DESC
LIMIT 100;

-- Insert default system metrics
INSERT INTO system_metrics (metric_name, metric_value, labels)
VALUES
    ('service_started', 1, '{"service": "postgres", "version": "16"}'),
    ('schema_version', 1, '{"migration": "initial"}');

-- Grant permissions (if needed for specific user)
-- GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA public TO graphrag;
-- GRANT ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA public TO graphrag;

COMMENT ON TABLE tenants IS 'Tenant metadata and statistics';
COMMENT ON TABLE documents IS 'Document processing status and metadata';
COMMENT ON TABLE retry_tracking IS 'Retry attempts for failed operations';
COMMENT ON TABLE validation_runs IS 'Graph validation execution history';
COMMENT ON TABLE validation_issues IS 'Issues found during validation';
COMMENT ON TABLE queue_state IS 'Message queue processing state';
COMMENT ON TABLE system_metrics IS 'System-level metrics and monitoring data';
