-- Add upsert function for articles table
\connect osint

-- Create a function to upsert articles
CREATE OR REPLACE FUNCTION upsert_article(
    p_url TEXT,
    p_title TEXT DEFAULT NULL,
    p_content TEXT DEFAULT NULL,
    p_language VARCHAR(10) DEFAULT 'en',
    p_published_at TIMESTAMPTZ DEFAULT NULL,
    p_source_name TEXT DEFAULT NULL,
    p_source_id INTEGER DEFAULT NULL,
    p_metadata JSONB DEFAULT '{}'::JSONB
)
RETURNS TABLE(
    id UUID,
    url TEXT,
    title TEXT,
    content TEXT,
    language VARCHAR(10),
    published_at TIMESTAMPTZ,
    source_name TEXT,
    source_id INTEGER,
    metadata JSONB,
    fetched_at TIMESTAMPTZ,
    was_inserted BOOLEAN
) AS $$
DECLARE
    article_id UUID;
    was_new BOOLEAN := FALSE;
BEGIN
    -- Try to insert first
    INSERT INTO articles (url, title, content, language, published_at, source_name, source_id, metadata)
    VALUES (p_url, p_title, p_content, p_language, p_published_at, p_source_name, p_source_id, p_metadata)
    ON CONFLICT (url) 
    DO UPDATE SET
        title = COALESCE(EXCLUDED.title, articles.title),
        content = COALESCE(EXCLUDED.content, articles.content),
        language = COALESCE(EXCLUDED.language, articles.language),
        published_at = COALESCE(EXCLUDED.published_at, articles.published_at),
        source_name = COALESCE(EXCLUDED.source_name, articles.source_name),
        source_id = COALESCE(EXCLUDED.source_id, articles.source_id),
        metadata = articles.metadata || EXCLUDED.metadata,  -- Merge metadata
        fetched_at = NOW()
    RETURNING articles.id, FALSE INTO article_id, was_new;
    
    -- If no conflict occurred, it was a new insert
    IF article_id IS NULL THEN
        SELECT id, FALSE INTO article_id, was_new FROM articles WHERE url = p_url;
    ELSE
        was_new := TRUE;
    END IF;
    
    -- Return the article data
    RETURN QUERY
    SELECT 
        a.id,
        a.url,
        a.title,
        a.content,
        a.language,
        a.published_at,
        a.source_name,
        a.source_id,
        a.metadata,
        a.fetched_at,
        was_new
    FROM articles a
    WHERE a.id = article_id;
END;
$$ LANGUAGE plpgsql;

-- Create a simpler upsert function for n8n
CREATE OR REPLACE FUNCTION simple_upsert_article(
    p_url TEXT,
    p_title TEXT DEFAULT NULL,
    p_content TEXT DEFAULT NULL,
    p_language VARCHAR(10) DEFAULT 'en',
    p_published_at TIMESTAMPTZ DEFAULT NULL,
    p_source_name TEXT DEFAULT NULL,
    p_metadata JSONB DEFAULT '{}'::JSONB
)
RETURNS UUID AS $$
DECLARE
    article_id UUID;
BEGIN
    INSERT INTO articles (url, title, content, language, published_at, source_name, metadata)
    VALUES (p_url, p_title, p_content, p_language, p_published_at, p_source_name, p_metadata)
    ON CONFLICT (url) 
    DO UPDATE SET
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
GRANT EXECUTE ON FUNCTION upsert_article TO osint;
GRANT EXECUTE ON FUNCTION simple_upsert_article TO osint;

-- Test the function
SELECT 'Upsert function created successfully!' as status;
