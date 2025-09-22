-- Simple Reddit ID to UUID conversion fix
\connect osint

-- Ensure UUID extension exists
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- Create function to convert Reddit IDs to UUIDs
CREATE OR REPLACE FUNCTION reddit_id_to_uuid(reddit_id TEXT)
RETURNS UUID AS $$
DECLARE
    namespace_uuid UUID := '6ba7b810-9dad-11d1-80b4-00c04fd430c8'; -- DNS namespace
BEGIN
    -- Check if it's a valid Reddit ID format
    IF reddit_id ~ '^t[123]_[a-zA-Z0-9]+$' THEN
        -- Generate deterministic UUID from Reddit ID
        RETURN uuid_generate_v5(namespace_uuid, reddit_id);
    ELSE
        -- If not a Reddit ID, try to parse as UUID
        BEGIN
            RETURN reddit_id::UUID;
        EXCEPTION WHEN invalid_text_representation THEN
            -- If not a valid UUID either, generate a new one
            RETURN gen_random_uuid();
        END;
    END IF;
END;
$$ LANGUAGE plpgsql;

-- Create function to safely insert articles with Reddit ID handling
CREATE OR REPLACE FUNCTION safe_upsert_article(
    p_url TEXT,
    p_title TEXT DEFAULT NULL,
    p_content TEXT DEFAULT NULL,
    p_language VARCHAR(10) DEFAULT 'en',
    p_published_at TIMESTAMPTZ DEFAULT NULL,
    p_source_name TEXT DEFAULT NULL,
    p_metadata JSONB DEFAULT '{}'::JSONB,
    p_article_id TEXT DEFAULT NULL
)
RETURNS UUID AS $$
DECLARE
    article_id UUID;
    source_id INTEGER;
    final_article_id UUID;
BEGIN
    -- Get or create source
    IF p_source_name IS NOT NULL THEN
        INSERT INTO sources(name) VALUES(p_source_name) 
        ON CONFLICT (name) DO UPDATE SET name=EXCLUDED.name 
        RETURNING id INTO source_id;
    END IF;
    
    -- Convert article_id to UUID if provided
    IF p_article_id IS NOT NULL THEN
        final_article_id := reddit_id_to_uuid(p_article_id);
    ELSE
        final_article_id := gen_random_uuid();
    END IF;
    
    -- Insert or update article
    INSERT INTO articles (id, url, title, content, language, published_at, source_id, source_name, metadata)
    VALUES (final_article_id, p_url, p_title, p_content, p_language, p_published_at, source_id, p_source_name, p_metadata)
    ON CONFLICT (url) DO UPDATE SET
        title = COALESCE(EXCLUDED.title, articles.title),
        content = COALESCE(EXCLUDED.content, articles.content),
        language = COALESCE(EXCLUDED.language, articles.language),
        published_at = COALESCE(EXCLUDED.published_at, articles.published_at),
        source_name = COALESCE(EXCLUDED.source_name, articles.source_name),
        metadata = articles.metadata || EXCLUDED.metadata,
        fetched_at = NOW()
    RETURNING id INTO article_id;
    
    RETURN article_id;
END;
$$ LANGUAGE plpgsql;

-- Grant permissions
GRANT EXECUTE ON FUNCTION reddit_id_to_uuid TO osint_user;
GRANT EXECUTE ON FUNCTION safe_upsert_article TO osint_user;

-- Test the function
SELECT 'Reddit ID conversion functions created successfully!' as status;
SELECT reddit_id_to_uuid('t3_1nm0pvw') as test_reddit_uuid;
SELECT reddit_id_to_uuid('550e8400-e29b-41d4-a716-446655440000') as test_valid_uuid;
