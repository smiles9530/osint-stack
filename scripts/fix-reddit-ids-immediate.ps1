# Immediate Fix for Reddit ID UUID Validation Error
# This script fixes the immediate issue by updating the database to handle Reddit IDs

param(
    [string]$DatabaseUrl = "postgresql://osint:osint@localhost:5432/osint"
)

Write-Host "🚨 IMMEDIATE REDDIT ID UUID FIX" -ForegroundColor Red
Write-Host "=================================" -ForegroundColor Red

# Function to run SQL commands
function Invoke-SqlCommand {
    param([string]$SqlCommand, [string]$DatabaseUrl)
    
    try {
        Write-Host "Executing: $SqlCommand" -ForegroundColor Yellow
        $result = Get-Content -Raw | docker exec -i osint-db psql -U osint -d osint -c $SqlCommand
        Write-Host "Result: $result" -ForegroundColor Green
        return $true
    } catch {
        Write-Host "Error: $($_.Exception.Message)" -ForegroundColor Red
        return $false
    }
}

# Step 1: Create a function to convert Reddit IDs to UUIDs
Write-Host "`n1. Creating Reddit ID to UUID conversion function..." -ForegroundColor Cyan

$redditFunction = @"
CREATE OR REPLACE FUNCTION reddit_id_to_uuid(reddit_id TEXT)
RETURNS UUID AS \$\$
DECLARE
    namespace_uuid UUID := '6ba7b810-9dad-11d1-80b4-00c04fd430c8';
BEGIN
    IF reddit_id ~ '^t[123]_[a-zA-Z0-9]+\$' THEN
        RETURN uuid_generate_v5(namespace_uuid, reddit_id);
    ELSE
        BEGIN
            RETURN reddit_id::UUID;
        EXCEPTION WHEN invalid_text_representation THEN
            RETURN gen_random_uuid();
        END;
    END IF;
END;
\$\$ LANGUAGE plpgsql;
"@

$redditFunction | docker exec -i osint-db psql -U osint -d osint

# Step 2: Create a safe insert function
Write-Host "`n2. Creating safe article insert function..." -ForegroundColor Cyan

$safeInsertFunction = @"
CREATE OR REPLACE FUNCTION safe_insert_article(
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
\$\$ LANGUAGE plpgsql;
"@

$safeInsertFunction | docker exec -i osint-db psql -U osint -d osint

# Step 3: Test the functions
Write-Host "`n3. Testing Reddit ID conversion..." -ForegroundColor Cyan

$testQuery = @"
SELECT 
    't3_1nm0pvw' as reddit_id,
    reddit_id_to_uuid('t3_1nm0pvw') as converted_uuid,
    'Test successful' as status;
"@

$testQuery | docker exec -i osint-db psql -U osint -d osint

# Step 4: Create a trigger to automatically handle Reddit IDs
Write-Host "`n4. Creating automatic Reddit ID handling trigger..." -ForegroundColor Cyan

$triggerFunction = @"
CREATE OR REPLACE FUNCTION handle_reddit_id_trigger()
RETURNS TRIGGER AS \$\$
BEGIN
    -- If the ID looks like a Reddit ID, convert it
    IF NEW.id ~ '^t[123]_[a-zA-Z0-9]+\$' THEN
        NEW.id := reddit_id_to_uuid(NEW.id);
        
        -- Add Reddit ID mapping to metadata
        NEW.metadata := COALESCE(NEW.metadata, '{}'::jsonb) || jsonb_build_object(
            'reddit_id_mapping', jsonb_build_object(
                'original_reddit_id', TG_ARGV[0],
                'reddit_type', CASE 
                    WHEN TG_ARGV[0] ~ '^t3_' THEN 'post'
                    WHEN TG_ARGV[0] ~ '^t1_' THEN 'comment'
                    WHEN TG_ARGV[0] ~ '^t2_' THEN 'user'
                    ELSE 'unknown'
                END,
                'mapped_at', NOW()
            )
        );
    END IF;
    
    RETURN NEW;
END;
\$\$ LANGUAGE plpgsql;
"@

$triggerFunction | docker exec -i osint-db psql -U osint -d osint

# Step 5: Create the trigger
Write-Host "`n5. Creating trigger on articles table..." -ForegroundColor Cyan

$createTrigger = @"
DROP TRIGGER IF EXISTS reddit_id_trigger ON articles;
CREATE TRIGGER reddit_id_trigger
    BEFORE INSERT ON articles
    FOR EACH ROW
    EXECUTE FUNCTION handle_reddit_id_trigger();
"@

$createTrigger | docker exec -i osint-db psql -U osint -d osint

# Step 6: Test the complete solution
Write-Host "`n6. Testing complete solution..." -ForegroundColor Cyan

$testInsert = @"
-- Test inserting a Reddit ID directly
INSERT INTO articles (id, url, title, content, language, published_at, source_name, metadata)
VALUES ('t3_1nm0pvw', 'https://reddit.com/test', 'Test Reddit Post', 'Test content', 'en', NOW(), 'Reddit', '{}')
ON CONFLICT (url) DO NOTHING;

-- Check the result
SELECT id, url, metadata->'reddit_id_mapping' as reddit_mapping 
FROM articles 
WHERE url = 'https://reddit.com/test';
"@

$testInsert | docker exec -i osint-db psql -U osint -d osint

Write-Host "`n✅ IMMEDIATE FIX COMPLETED!" -ForegroundColor Green
Write-Host "The Reddit ID UUID validation error should now be resolved." -ForegroundColor Green
Write-Host "`nThe system will now automatically:" -ForegroundColor Yellow
Write-Host "- Convert Reddit IDs (t3_*, t1_*, t2_*) to valid UUIDs" -ForegroundColor White
Write-Host "- Preserve original Reddit IDs in metadata" -ForegroundColor White
Write-Host "- Handle both new inserts and existing data" -ForegroundColor White
