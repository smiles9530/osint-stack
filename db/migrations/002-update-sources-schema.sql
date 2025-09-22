-- Migration: Update Sources Schema
-- This migration adds missing columns to the sources table to support the enhanced functionality

\connect osint

-- Start transaction
BEGIN;

-- Add missing columns to sources table
ALTER TABLE sources 
ADD COLUMN IF NOT EXISTS url TEXT,
ADD COLUMN IF NOT EXISTS category VARCHAR(50),
ADD COLUMN IF NOT EXISTS language VARCHAR(10) DEFAULT 'en',
ADD COLUMN IF NOT EXISTS country VARCHAR(100),
ADD COLUMN IF NOT EXISTS is_enabled BOOLEAN DEFAULT TRUE,
ADD COLUMN IF NOT EXISTS last_checked TIMESTAMPTZ,
ADD COLUMN IF NOT EXISTS success_rate FLOAT DEFAULT 0.0,
ADD COLUMN IF NOT EXISTS article_count INTEGER DEFAULT 0,
ADD COLUMN IF NOT EXISTS metadata JSONB DEFAULT '{}',
ADD COLUMN IF NOT EXISTS updated_at TIMESTAMPTZ DEFAULT NOW();

-- Create indexes for new columns
CREATE INDEX IF NOT EXISTS idx_sources_url ON sources (url);
CREATE INDEX IF NOT EXISTS idx_sources_category ON sources (category);
CREATE INDEX IF NOT EXISTS idx_sources_language ON sources (language);
CREATE INDEX IF NOT EXISTS idx_sources_country ON sources (country);
CREATE INDEX IF NOT EXISTS idx_sources_enabled ON sources (is_enabled);
CREATE INDEX IF NOT EXISTS idx_sources_last_checked ON sources (last_checked);
CREATE INDEX IF NOT EXISTS idx_sources_success_rate ON sources (success_rate);
CREATE INDEX IF NOT EXISTS idx_sources_article_count ON sources (article_count);
CREATE INDEX IF NOT EXISTS idx_sources_metadata ON sources USING GIN (metadata);
CREATE INDEX IF NOT EXISTS idx_sources_updated_at ON sources (updated_at);

-- Create trigger to update updated_at timestamp
CREATE OR REPLACE FUNCTION update_sources_updated_at()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER trigger_update_sources_updated_at
    BEFORE UPDATE ON sources
    FOR EACH ROW
    EXECUTE FUNCTION update_sources_updated_at();

-- Update existing sources with default values
UPDATE sources 
SET 
    is_enabled = TRUE,
    success_rate = 0.0,
    article_count = 0,
    metadata = '{}',
    updated_at = NOW()
WHERE is_enabled IS NULL;

-- Grant permissions (osint user already has permissions)

-- Commit transaction
COMMIT;

-- Verify the migration
DO $$
DECLARE
    column_count INTEGER;
BEGIN
    SELECT COUNT(*) INTO column_count 
    FROM information_schema.columns 
    WHERE table_name = 'sources' AND table_schema = 'public';
    
    IF column_count < 10 THEN
        RAISE EXCEPTION 'Migration failed: Expected at least 10 columns, found %', column_count;
    END IF;
    
    RAISE NOTICE 'Sources table migration successful: % columns found', column_count;
END $$;
