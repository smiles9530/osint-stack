\connect osint

-- Create users table
CREATE TABLE IF NOT EXISTS users (
    id SERIAL PRIMARY KEY,
    username VARCHAR(50) NOT NULL UNIQUE,
    email VARCHAR(255) NOT NULL UNIQUE,
    hashed_password VARCHAR(255) NOT NULL,
    is_active BOOLEAN DEFAULT TRUE,
    is_superuser BOOLEAN DEFAULT FALSE,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW(),
    last_login TIMESTAMPTZ
);

-- Create user sessions table for token management
CREATE TABLE IF NOT EXISTS user_sessions (
    id SERIAL PRIMARY KEY,
    user_id INTEGER REFERENCES users(id) ON DELETE CASCADE,
    token_hash VARCHAR(255) NOT NULL UNIQUE,
    expires_at TIMESTAMPTZ NOT NULL,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    is_active BOOLEAN DEFAULT TRUE
);

-- Create indexes for users table
CREATE INDEX IF NOT EXISTS idx_users_username ON users (username);
CREATE INDEX IF NOT EXISTS idx_users_email ON users (email);
CREATE INDEX IF NOT EXISTS idx_users_active ON users (is_active);
CREATE INDEX IF NOT EXISTS idx_user_sessions_user_id ON user_sessions (user_id);
CREATE INDEX IF NOT EXISTS idx_user_sessions_expires_at ON user_sessions (expires_at);

CREATE TABLE IF NOT EXISTS sources (
  id SERIAL PRIMARY KEY,
  name TEXT NOT NULL UNIQUE,
  domain TEXT UNIQUE,
  country_iso CHAR(2),
  bias_label TEXT,
  reliability_label TEXT,
  created_at TIMESTAMPTZ DEFAULT now()
);
-- Create articles table with partitioning by published_at
CREATE TABLE IF NOT EXISTS articles (
  id BIGSERIAL,
  url TEXT NOT NULL,
  source_id INT REFERENCES sources(id) ON DELETE SET NULL,
  title TEXT,
  text TEXT,
  lang TEXT,
  published_at TIMESTAMPTZ,
  fetched_at TIMESTAMPTZ DEFAULT now(),
  tone NUMERIC,
  dedupe_hash TEXT,
  country_iso CHAR(2),
  geom GEOGRAPHY(Point, 4326),
  PRIMARY KEY (id, published_at),
  UNIQUE (url, published_at)
) PARTITION BY RANGE (published_at);

-- Create monthly partitions for articles (last 2 years + future)
DO $$
DECLARE
    start_date DATE := '2023-01-01';
    end_date DATE := '2025-12-31';
    iter_date DATE := start_date;
    partition_name TEXT;
BEGIN
    WHILE iter_date <= end_date LOOP
        partition_name := 'articles_' || to_char(iter_date, 'YYYY_MM');
        
        EXECUTE format('CREATE TABLE IF NOT EXISTS %I PARTITION OF articles
                       FOR VALUES FROM (%L) TO (%L)',
                       partition_name,
                       iter_date,
                       iter_date + INTERVAL '1 month');
        
        iter_date := iter_date + INTERVAL '1 month';
    END LOOP;
END $$;
CREATE INDEX IF NOT EXISTS idx_articles_published_at ON articles (published_at);
CREATE INDEX IF NOT EXISTS idx_articles_country ON articles (country_iso);
CREATE TABLE IF NOT EXISTS embeddings (
  article_id BIGINT,
  article_published_at TIMESTAMPTZ,
  PRIMARY KEY (article_id, article_published_at),
  FOREIGN KEY (article_id, article_published_at) REFERENCES articles(id, published_at) ON DELETE CASCADE,
  vec VECTOR(1024),
  model TEXT NOT NULL,
  created_at TIMESTAMPTZ DEFAULT now()
);
CREATE TABLE IF NOT EXISTS topics (
  id BIGSERIAL PRIMARY KEY,
  label TEXT,
  created_at TIMESTAMPTZ DEFAULT now()
);
CREATE TABLE IF NOT EXISTS article_topics (
  article_id BIGINT,
  topic_id BIGINT REFERENCES topics(id) ON DELETE CASCADE,
  score REAL,
  PRIMARY KEY(article_id, topic_id)
);
-- Create topic_timeseries table with TimescaleDB hypertable
CREATE TABLE IF NOT EXISTS topic_timeseries (
  topic_id BIGINT REFERENCES topics(id) ON DELETE CASCADE,
  day DATE NOT NULL,
  count INT DEFAULT 0,
  mean_tone NUMERIC,
  burst_level REAL DEFAULT 0,
  cp_flag BOOLEAN DEFAULT FALSE,
  PRIMARY KEY (topic_id, day)
);

-- Create TimescaleDB hypertable for time-series data
SELECT create_hypertable('topic_timeseries', by_range('day'), if_not_exists => TRUE);

-- Create additional performance indexes
CREATE INDEX IF NOT EXISTS idx_articles_source_id ON articles (source_id);
CREATE INDEX IF NOT EXISTS idx_articles_lang ON articles (lang);
CREATE INDEX IF NOT EXISTS idx_articles_tone ON articles (tone);
CREATE INDEX IF NOT EXISTS idx_articles_fetched_at ON articles (fetched_at);
CREATE INDEX IF NOT EXISTS idx_articles_dedupe_hash ON articles (dedupe_hash);

-- Create partial indexes for better performance
CREATE INDEX IF NOT EXISTS idx_articles_recent ON articles (published_at DESC) 
WHERE published_at > '2024-01-01';

CREATE INDEX IF NOT EXISTS idx_articles_active_sources ON articles (source_id, published_at DESC)
WHERE source_id IS NOT NULL;

-- Create materialized view for article statistics
CREATE MATERIALIZED VIEW IF NOT EXISTS article_stats AS
SELECT 
    source_id,
    COUNT(*) as total_articles,
    COUNT(DISTINCT DATE(published_at)) as active_days,
    AVG(tone) as avg_tone,
    MIN(published_at) as first_article,
    MAX(published_at) as last_article
FROM articles 
WHERE published_at IS NOT NULL
GROUP BY source_id;

-- Create index on materialized view
CREATE INDEX IF NOT EXISTS idx_article_stats_source_id ON article_stats (source_id);

-- Create function to refresh materialized view
CREATE OR REPLACE FUNCTION refresh_article_stats()
RETURNS void AS $$
BEGIN
    REFRESH MATERIALIZED VIEW CONCURRENTLY article_stats;
END;
$$ LANGUAGE plpgsql;

-- ===== INTEL-AUTOMATOR INTEGRATION TABLES =====

-- Create UUID extension if not exists
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- seen URLs for deduplication
CREATE TABLE IF NOT EXISTS news_seen (
  url_hash TEXT PRIMARY KEY,
  first_seen TIMESTAMPTZ DEFAULT now()
);

-- Enhanced analysis table for bias/sentiment analysis
CREATE TABLE IF NOT EXISTS article_analysis (
  article_id BIGINT,
  article_published_at TIMESTAMPTZ,
  PRIMARY KEY (article_id, article_published_at),
  FOREIGN KEY (article_id, article_published_at) REFERENCES articles(id, published_at) ON DELETE CASCADE,
  subjectivity INT CHECK (subjectivity >= 0 AND subjectivity <= 100),
  sensationalism INT CHECK (sensationalism >= 0 AND sensationalism <= 100),
  loaded_language INT CHECK (loaded_language >= 0 AND loaded_language <= 100),
  bias_lr INT CHECK (bias_lr >= 0 AND bias_lr <= 100),
  stance TEXT CHECK (stance IN ('pro', 'neutral', 'anti', 'unclear')),
  evidence_density INT CHECK (evidence_density >= 0 AND evidence_density <= 100),
  sentiment TEXT,
  sentiment_confidence NUMERIC CHECK (sentiment_confidence >= 0 AND sentiment_confidence <= 1),
  agenda_signals JSONB,
  risk_flags JSONB,
  entities JSONB,
  tags JSONB,
  key_quotes JSONB,
  summary_bullets JSONB,
  confidence_score NUMERIC CHECK (confidence_score >= 0 AND confidence_score <= 1),
  model_agreement NUMERIC CHECK (model_agreement >= 0 AND model_agreement <= 1),
  bias_trend JSONB DEFAULT '{}',
  analysis_timestamp TIMESTAMPTZ,
  created_at TIMESTAMPTZ DEFAULT now()
);

-- User/system feedback for reinforcement learning
CREATE TABLE IF NOT EXISTS feedback (
  id BIGSERIAL PRIMARY KEY,
  article_id BIGINT,
  article_published_at TIMESTAMPTZ,
  FOREIGN KEY (article_id, article_published_at) REFERENCES articles(id, published_at) ON DELETE CASCADE,
  user_id INT REFERENCES users(id) ON DELETE SET NULL,
  clicked BOOLEAN,
  upvote BOOLEAN,
  correct_after_days BOOLEAN,
  feedback_score NUMERIC CHECK (feedback_score >= 0 AND feedback_score <= 1),
  created_at TIMESTAMPTZ DEFAULT now()
);

-- Bandit parameters for adaptive learning
CREATE TABLE IF NOT EXISTS bandit_state (
  key TEXT PRIMARY KEY,            -- e.g., 'source:reuters' or 'prompt:variantA'
  count BIGINT DEFAULT 0,
  success BIGINT DEFAULT 0,
  params JSONB,
  updated_at TIMESTAMPTZ DEFAULT now()
);

-- Daily digests
CREATE TABLE IF NOT EXISTS digests (
  id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
  date DATE NOT NULL,
  topic TEXT,
  content_md TEXT,
  created_at TIMESTAMPTZ DEFAULT now()
);

-- Dead-letter table for error handling
CREATE TABLE IF NOT EXISTS processing_errors (
  id BIGSERIAL PRIMARY KEY,
  url TEXT NOT NULL,
  error_message TEXT,
  stage TEXT,
  retry_count INT DEFAULT 0,
  created_at TIMESTAMPTZ DEFAULT now()
);

-- Source reputation and metadata
CREATE TABLE IF NOT EXISTS source_metadata (
  domain TEXT PRIMARY KEY,
  reputation_score NUMERIC DEFAULT 0.5 CHECK (reputation_score >= 0 AND reputation_score <= 1),
  paywall_detected BOOLEAN DEFAULT FALSE,
  last_checked TIMESTAMPTZ DEFAULT now(),
  reliability_label TEXT DEFAULT 'unknown',
  bias_score NUMERIC DEFAULT 0.5 CHECK (bias_score >= 0 AND bias_score <= 1)
);

-- Create indexes for new tables
CREATE INDEX IF NOT EXISTS idx_news_seen_first_seen ON news_seen (first_seen);
CREATE INDEX IF NOT EXISTS idx_article_analysis_subjectivity ON article_analysis (subjectivity);
CREATE INDEX IF NOT EXISTS idx_article_analysis_bias_lr ON article_analysis (bias_lr);
CREATE INDEX IF NOT EXISTS idx_article_analysis_stance ON article_analysis (stance);
CREATE INDEX IF NOT EXISTS idx_article_analysis_created_at ON article_analysis (created_at);
CREATE INDEX IF NOT EXISTS idx_feedback_article_id ON feedback (article_id);
CREATE INDEX IF NOT EXISTS idx_feedback_user_id ON feedback (user_id);
CREATE INDEX IF NOT EXISTS idx_feedback_created_at ON feedback (created_at);
CREATE INDEX IF NOT EXISTS idx_bandit_state_updated_at ON bandit_state (updated_at);
CREATE INDEX IF NOT EXISTS idx_digests_date ON digests (date);
CREATE INDEX IF NOT EXISTS idx_digests_topic ON digests (topic);
CREATE INDEX IF NOT EXISTS idx_processing_errors_stage ON processing_errors (stage);
CREATE INDEX IF NOT EXISTS idx_processing_errors_created_at ON processing_errors (created_at);
CREATE INDEX IF NOT EXISTS idx_source_metadata_reputation ON source_metadata (reputation_score);
CREATE INDEX IF NOT EXISTS idx_source_metadata_last_checked ON source_metadata (last_checked);
