-- Enhanced Entity Extraction and Knowledge Graph Schema
\connect osint

-- Entity nodes table for knowledge graph
CREATE TABLE IF NOT EXISTS entity_nodes (
    id SERIAL PRIMARY KEY,
    name TEXT NOT NULL,
    normalized_name TEXT NOT NULL,
    type VARCHAR(50) NOT NULL,
    wikidata_id VARCHAR(50),
    description TEXT,
    aliases JSONB DEFAULT '[]',
    properties JSONB DEFAULT '{}',
    confidence FLOAT DEFAULT 1.0,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW(),
    UNIQUE(name, type)
);

-- Entity relationships table for knowledge graph
CREATE TABLE IF NOT EXISTS entity_relations (
    id SERIAL PRIMARY KEY,
    source_entity_id INTEGER REFERENCES entity_nodes(id) ON DELETE CASCADE,
    target_entity_id INTEGER REFERENCES entity_nodes(id) ON DELETE CASCADE,
    relation_type VARCHAR(100) NOT NULL,
    confidence FLOAT DEFAULT 1.0,
    context TEXT,
    properties JSONB DEFAULT '{}',
    created_at TIMESTAMPTZ DEFAULT NOW(),
    UNIQUE(source_entity_id, target_entity_id, relation_type)
);

-- Enhanced article-entity mentions table
CREATE TABLE IF NOT EXISTS article_entity_mentions (
    id SERIAL PRIMARY KEY,
    article_id BIGINT NOT NULL,
    entity_id INTEGER REFERENCES entity_nodes(id) ON DELETE CASCADE,
    start_pos INTEGER NOT NULL,
    end_pos INTEGER NOT NULL,
    confidence FLOAT DEFAULT 1.0,
    context TEXT,
    mention_text TEXT NOT NULL,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    UNIQUE(article_id, entity_id, start_pos)
);

-- Geotagging and place normalization
CREATE TABLE IF NOT EXISTS geographic_entities (
    id SERIAL PRIMARY KEY,
    name TEXT NOT NULL,
    normalized_name TEXT NOT NULL,
    country_code CHAR(2),
    latitude DECIMAL(10, 8),
    longitude DECIMAL(11, 8),
    geohash VARCHAR(12),
    aliases JSONB DEFAULT '[]',
    properties JSONB DEFAULT '{}',
    created_at TIMESTAMPTZ DEFAULT NOW(),
    UNIQUE(name, country_code)
);

-- Article geotagging
CREATE TABLE IF NOT EXISTS article_geotags (
    id SERIAL PRIMARY KEY,
    article_id BIGINT NOT NULL,
    geo_entity_id INTEGER REFERENCES geographic_entities(id) ON DELETE CASCADE,
    confidence FLOAT DEFAULT 1.0,
    context TEXT,
    mention_text TEXT NOT NULL,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    UNIQUE(article_id, geo_entity_id, mention_text)
);

-- Multimodal content support
CREATE TABLE IF NOT EXISTS media_files (
    id SERIAL PRIMARY KEY,
    article_id BIGINT NOT NULL,
    file_type VARCHAR(20) NOT NULL, -- 'image', 'video', 'audio'
    file_path TEXT NOT NULL,
    file_size BIGINT,
    duration_seconds INTEGER, -- for video/audio
    extracted_text TEXT, -- OCR/ASR results
    vector_embedding VECTOR(1024), -- CLIP embeddings
    metadata JSONB DEFAULT '{}',
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- Reranking results cache
CREATE TABLE IF NOT EXISTS reranking_cache (
    id SERIAL PRIMARY KEY,
    query_hash VARCHAR(64) NOT NULL,
    query_text TEXT NOT NULL,
    results JSONB NOT NULL,
    model_used VARCHAR(100),
    created_at TIMESTAMPTZ DEFAULT NOW(),
    expires_at TIMESTAMPTZ NOT NULL,
    UNIQUE(query_hash)
);

-- Topic clustering results
CREATE TABLE IF NOT EXISTS topic_clusters (
    id SERIAL PRIMARY KEY,
    cluster_name VARCHAR(200) NOT NULL,
    cluster_keywords JSONB NOT NULL,
    cluster_centroid VECTOR(1024),
    article_count INTEGER DEFAULT 0,
    coherence_score FLOAT,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

-- Article-topic cluster assignments
CREATE TABLE IF NOT EXISTS article_topic_clusters (
    article_id BIGINT NOT NULL,
    cluster_id INTEGER REFERENCES topic_clusters(id) ON DELETE CASCADE,
    similarity_score FLOAT NOT NULL,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    PRIMARY KEY (article_id, cluster_id)
);

-- Create indexes for performance
CREATE INDEX IF NOT EXISTS idx_entity_nodes_name ON entity_nodes (name);
CREATE INDEX IF NOT EXISTS idx_entity_nodes_type ON entity_nodes (type);
CREATE INDEX IF NOT EXISTS idx_entity_nodes_wikidata ON entity_nodes (wikidata_id);
CREATE INDEX IF NOT EXISTS idx_entity_nodes_normalized ON entity_nodes (normalized_name);

CREATE INDEX IF NOT EXISTS idx_entity_relations_source ON entity_relations (source_entity_id);
CREATE INDEX IF NOT EXISTS idx_entity_relations_target ON entity_relations (target_entity_id);
CREATE INDEX IF NOT EXISTS idx_entity_relations_type ON entity_relations (relation_type);

CREATE INDEX IF NOT EXISTS idx_article_entity_mentions_article ON article_entity_mentions (article_id);
CREATE INDEX IF NOT EXISTS idx_article_entity_mentions_entity ON article_entity_mentions (entity_id);
CREATE INDEX IF NOT EXISTS idx_article_entity_mentions_confidence ON article_entity_mentions (confidence);

CREATE INDEX IF NOT EXISTS idx_geographic_entities_name ON geographic_entities (name);
CREATE INDEX IF NOT EXISTS idx_geographic_entities_country ON geographic_entities (country_code);
CREATE INDEX IF NOT EXISTS idx_geographic_entities_geohash ON geographic_entities (geohash);

CREATE INDEX IF NOT EXISTS idx_article_geotags_article ON article_geotags (article_id);
CREATE INDEX IF NOT EXISTS idx_article_geotags_geo ON article_geotags (geo_entity_id);

CREATE INDEX IF NOT EXISTS idx_media_files_article ON media_files (article_id);
CREATE INDEX IF NOT EXISTS idx_media_files_type ON media_files (file_type);
CREATE INDEX IF NOT EXISTS idx_media_files_vector ON media_files 
USING ivfflat (vector_embedding vector_cosine_ops);

CREATE INDEX IF NOT EXISTS idx_reranking_cache_query_hash ON reranking_cache (query_hash);
CREATE INDEX IF NOT EXISTS idx_reranking_cache_expires ON reranking_cache (expires_at);

CREATE INDEX IF NOT EXISTS idx_topic_clusters_name ON topic_clusters (cluster_name);
CREATE INDEX IF NOT EXISTS idx_topic_clusters_centroid ON topic_clusters 
USING ivfflat (cluster_centroid vector_cosine_ops);

CREATE INDEX IF NOT EXISTS idx_article_topic_clusters_article ON article_topic_clusters (article_id);
CREATE INDEX IF NOT EXISTS idx_article_topic_clusters_cluster ON article_topic_clusters (cluster_id);

-- Create functions for entity management
CREATE OR REPLACE FUNCTION upsert_entity_node(
    p_name TEXT,
    p_type VARCHAR(50),
    p_wikidata_id VARCHAR(50) DEFAULT NULL,
    p_description TEXT DEFAULT NULL,
    p_aliases JSONB DEFAULT '[]',
    p_properties JSONB DEFAULT '{}'
) RETURNS INTEGER AS $$
DECLARE
    entity_id INTEGER;
BEGIN
    INSERT INTO entity_nodes (name, normalized_name, type, wikidata_id, description, aliases, properties)
    VALUES (p_name, LOWER(TRIM(p_name)), p_type, p_wikidata_id, p_description, p_aliases, p_properties)
    ON CONFLICT (name, type) 
    DO UPDATE SET
        normalized_name = EXCLUDED.normalized_name,
        wikidata_id = COALESCE(EXCLUDED.wikidata_id, entity_nodes.wikidata_id),
        description = COALESCE(EXCLUDED.description, entity_nodes.description),
        aliases = EXCLUDED.aliases,
        properties = EXCLUDED.properties,
        updated_at = NOW()
    RETURNING id INTO entity_id;
    
    RETURN entity_id;
END;
$$ LANGUAGE plpgsql;

-- Function to get entity statistics
CREATE OR REPLACE FUNCTION get_entity_statistics()
RETURNS JSON AS $$
DECLARE
    result JSON;
BEGIN
    SELECT json_build_object(
        'total_entities', (SELECT COUNT(*) FROM entity_nodes),
        'total_relations', (SELECT COUNT(*) FROM entity_relations),
        'total_mentions', (SELECT COUNT(*) FROM article_entity_mentions),
        'entity_types', (
            SELECT json_agg(json_build_object('type', entity_type, 'count', count))
            FROM (
                SELECT type as entity_type, COUNT(*) as count
                FROM entity_nodes
                GROUP BY type
                ORDER BY count DESC
            ) t
        ),
        'top_entities', (
            SELECT json_agg(json_build_object('name', name, 'type', type, 'mentions', mention_count))
            FROM (
                SELECT e.name, e.type, COUNT(aem.id) as mention_count
                FROM entity_nodes e
                LEFT JOIN article_entity_mentions aem ON e.id = aem.entity_id
                GROUP BY e.id, e.name, e.type
                ORDER BY mention_count DESC
                LIMIT 10
            ) t
        )
    ) INTO result;
    
    RETURN result;
END;
$$ LANGUAGE plpgsql;

-- Function to clean up expired reranking cache
CREATE OR REPLACE FUNCTION cleanup_reranking_cache()
RETURNS void AS $$
BEGIN
    DELETE FROM reranking_cache WHERE expires_at < NOW();
END;
$$ LANGUAGE plpgsql;

-- Create materialized view for entity co-occurrence
CREATE MATERIALIZED VIEW IF NOT EXISTS entity_cooccurrence AS
SELECT 
    e1.id as entity1_id,
    e1.name as entity1_name,
    e1.type as entity1_type,
    e2.id as entity2_id,
    e2.name as entity2_name,
    e2.type as entity2_type,
    COUNT(*) as cooccurrence_count
FROM article_entity_mentions aem1
JOIN article_entity_mentions aem2 ON aem1.article_id = aem2.article_id AND aem1.entity_id < aem2.entity_id
JOIN entity_nodes e1 ON aem1.entity_id = e1.id
JOIN entity_nodes e2 ON aem2.entity_id = e2.id
GROUP BY e1.id, e1.name, e1.type, e2.id, e2.name, e2.type
ORDER BY cooccurrence_count DESC;

CREATE INDEX IF NOT EXISTS idx_entity_cooccurrence_entity1 ON entity_cooccurrence (entity1_id);
CREATE INDEX IF NOT EXISTS idx_entity_cooccurrence_entity2 ON entity_cooccurrence (entity2_id);
CREATE INDEX IF NOT EXISTS idx_entity_cooccurrence_count ON entity_cooccurrence (cooccurrence_count);

-- Function to refresh entity co-occurrence view
CREATE OR REPLACE FUNCTION refresh_entity_cooccurrence()
RETURNS void AS $$
BEGIN
    REFRESH MATERIALIZED VIEW CONCURRENTLY entity_cooccurrence;
END;
$$ LANGUAGE plpgsql;
