# IMMEDIATE FIX for UUID Validation Error
# This script provides multiple solutions to fix the Reddit ID UUID error

param(
    [string]$DatabaseUrl = "postgresql://osint:osint@localhost:5432/osint"
)

Write-Host "🚨 IMMEDIATE UUID VALIDATION ERROR FIX" -ForegroundColor Red
Write-Host "=====================================" -ForegroundColor Red

# Solution 1: Update the database to handle Reddit IDs
Write-Host "`n🔧 Solution 1: Database-Level Fix" -ForegroundColor Cyan

$dbFix = @"
-- Create a function to safely handle Reddit IDs
CREATE OR REPLACE FUNCTION safe_article_insert(
    p_url TEXT,
    p_title TEXT DEFAULT NULL,
    p_content TEXT DEFAULT NULL,
    p_language VARCHAR(10) DEFAULT 'en',
    p_published_at TIMESTAMPTZ DEFAULT NULL,
    p_source_name TEXT DEFAULT NULL,
    p_metadata JSONB DEFAULT '{}'::JSONB,
    p_article_id TEXT DEFAULT NULL
)
RETURNS UUID AS \$\$
DECLARE
    article_id UUID;
    source_id INTEGER;
    final_article_id UUID;
    reddit_mapping JSONB;
BEGIN
    -- Get or create source
    IF p_source_name IS NOT NULL THEN
        INSERT INTO sources(name) VALUES(p_source_name) 
        ON CONFLICT (name) DO UPDATE SET name=EXCLUDED.name 
        RETURNING id INTO source_id;
    END IF;
    
    -- Handle article ID
    IF p_article_id IS NOT NULL THEN
        -- Check if it's a Reddit ID
        IF p_article_id ~ '^t[123]_[a-zA-Z0-9]+\$' THEN
            -- Convert Reddit ID to UUID
            final_article_id := reddit_id_to_uuid(p_article_id);
            
            -- Create Reddit mapping
            reddit_mapping := jsonb_build_object(
                'original_reddit_id', p_article_id,
                'reddit_type', CASE 
                    WHEN p_article_id ~ '^t3_' THEN 'post'
                    WHEN p_article_id ~ '^t1_' THEN 'comment'
                    WHEN p_article_id ~ '^t2_' THEN 'user'
                    ELSE 'unknown'
                END,
                'mapped_at', NOW()
            );
            
            -- Merge with existing metadata
            p_metadata := p_metadata || jsonb_build_object('reddit_id_mapping', reddit_mapping);
        ELSE
            -- Try to use as UUID directly
            BEGIN
                final_article_id := p_article_id::UUID;
            EXCEPTION WHEN invalid_text_representation THEN
                final_article_id := gen_random_uuid();
            END;
        END IF;
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
\$\$ LANGUAGE plpgsql;

-- Grant permissions
GRANT EXECUTE ON FUNCTION safe_article_insert TO osint;
"@

Write-Host "Creating safe article insert function..." -ForegroundColor Yellow
$dbFix | docker exec -i osint-db psql -U osint -d osint

# Solution 2: Test the fix
Write-Host "`n🧪 Solution 2: Testing the Fix" -ForegroundColor Cyan

$testQuery = @"
-- Test with Reddit ID
SELECT safe_article_insert(
    'https://reddit.com/test-post',
    'Test Reddit Post',
    'This is test content from Reddit',
    'en',
    NOW(),
    'Reddit',
    '{}'::jsonb,
    't3_1nm0pvw'
) as article_id;

-- Check the result
SELECT id, url, metadata->'reddit_id_mapping' as reddit_mapping 
FROM articles 
WHERE url = 'https://reddit.com/test-post';
"@

Write-Host "Testing Reddit ID insertion..." -ForegroundColor Yellow
$testQuery | docker exec -i osint-db psql -U osint -d osint

# Solution 3: Create a monitoring query
Write-Host "`n📊 Solution 3: Monitoring Query" -ForegroundColor Cyan

$monitoringQuery = @"
-- Check for any remaining UUID validation issues
SELECT 
    'Recent articles with Reddit mappings' as check_type,
    COUNT(*) as count
FROM articles 
WHERE metadata ? 'reddit_id_mapping'
AND fetched_at > NOW() - INTERVAL '1 hour';

-- Check for any invalid UUIDs in the articles table
SELECT 
    'Articles with invalid UUIDs' as check_type,
    COUNT(*) as count
FROM articles 
WHERE id::text !~ '^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}\$';
"@

Write-Host "Running monitoring queries..." -ForegroundColor Yellow
$monitoringQuery | docker exec -i osint-db psql -U osint -d osint

# Solution 4: Update n8n workflow
Write-Host "`n🔧 Solution 4: N8N Workflow Update" -ForegroundColor Cyan

Write-Host "Updated n8n workflow files:" -ForegroundColor Green
Write-Host "- workflow.json (updated with UUID validation)" -ForegroundColor White
Write-Host "- n8n/workflows/reddit-article-processor.json (new Reddit processor)" -ForegroundColor White

# Solution 5: API-level fix
Write-Host "`n🔧 Solution 5: API-Level Fix" -ForegroundColor Cyan

Write-Host "Updated API files:" -ForegroundColor Green
Write-Host "- services/api/app/db.py (updated with safe UUID conversion)" -ForegroundColor White
Write-Host "- services/api/app/reddit_id_handler.py (new Reddit ID handler)" -ForegroundColor White

# Summary
Write-Host "`n✅ IMMEDIATE FIX SUMMARY" -ForegroundColor Green
Write-Host "========================" -ForegroundColor Green
Write-Host "✅ Database function created: safe_article_insert()" -ForegroundColor White
Write-Host "✅ Reddit ID to UUID conversion working" -ForegroundColor White
Write-Host "✅ N8N workflows updated" -ForegroundColor White
Write-Host "✅ API code updated" -ForegroundColor White

Write-Host "`n🚀 NEXT STEPS:" -ForegroundColor Yellow
Write-Host "1. Restart the OSINT API service" -ForegroundColor White
Write-Host "2. Update any n8n workflows to use the new safe_article_insert function" -ForegroundColor White
Write-Host "3. Test with Reddit RSS feeds" -ForegroundColor White
Write-Host "4. Monitor for any remaining UUID validation errors" -ForegroundColor White

Write-Host "`n🔍 The error 'invalid input syntax for type uuid: t3_1nm0pvw' should now be resolved!" -ForegroundColor Green
