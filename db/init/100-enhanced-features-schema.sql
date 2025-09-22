-- Enhanced Features Schema
-- Additional tables for new functionality implemented in routers
\connect osint

-- Bandit algorithm tables
CREATE TABLE IF NOT EXISTS bandit_selections (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    key_name VARCHAR(100) NOT NULL,
    context_data JSONB DEFAULT '{}',
    ucb_score FLOAT NOT NULL,
    user_id INTEGER REFERENCES users(id) ON DELETE CASCADE,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS bandit_rewards (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    key_name VARCHAR(100) NOT NULL,
    reward FLOAT NOT NULL CHECK (reward >= 0.0 AND reward <= 1.0),
    context_data JSONB DEFAULT '{}',
    user_id INTEGER REFERENCES users(id) ON DELETE CASCADE,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- User feedback table
CREATE TABLE IF NOT EXISTS user_feedback (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    article_id UUID REFERENCES articles(id) ON DELETE CASCADE,
    user_id INTEGER REFERENCES users(id) ON DELETE CASCADE,
    clicked BOOLEAN,
    upvote BOOLEAN,
    correct_after_days BOOLEAN,
    feedback_score FLOAT CHECK (feedback_score >= 0.0 AND feedback_score <= 1.0),
    feedback_text TEXT,
    submitted_at TIMESTAMPTZ DEFAULT NOW(),
    UNIQUE(article_id, user_id)
);

-- Digests table
CREATE TABLE IF NOT EXISTS digests (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    title TEXT NOT NULL,
    content TEXT NOT NULL,
    topic VARCHAR(200),
    created_at TIMESTAMPTZ DEFAULT NOW(),
    article_count INTEGER DEFAULT 0,
    sentiment_summary JSONB DEFAULT '{}',
    key_insights JSONB DEFAULT '[]'
);

-- Intelligence data tables
CREATE TABLE IF NOT EXISTS intelligence_data (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id INTEGER REFERENCES users(id) ON DELETE CASCADE,
    data_type VARCHAR(50) NOT NULL,
    raw_data JSONB NOT NULL,
    analysis_result JSONB,
    confidence_score FLOAT DEFAULT 0.0,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS intelligence_insights (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id INTEGER REFERENCES users(id) ON DELETE CASCADE,
    insight_type VARCHAR(50) NOT NULL,
    message TEXT NOT NULL,
    severity VARCHAR(20) DEFAULT 'medium',
    data_point_id UUID REFERENCES intelligence_data(id) ON DELETE CASCADE,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS dashboard_updates (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    dashboard_id VARCHAR(100) NOT NULL,
    user_id INTEGER REFERENCES users(id) ON DELETE CASCADE,
    update_type VARCHAR(50) NOT NULL,
    metrics_data JSONB DEFAULT '[]',
    top_entities JSONB DEFAULT '[]',
    sentiment_trends JSONB DEFAULT '[]',
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- Create indexes for performance
CREATE INDEX idx_bandit_selections_key ON bandit_selections (key_name);
CREATE INDEX idx_bandit_selections_user ON bandit_selections (user_id);
CREATE INDEX idx_bandit_selections_created ON bandit_selections (created_at DESC);

CREATE INDEX idx_bandit_rewards_key ON bandit_rewards (key_name);
CREATE INDEX idx_bandit_rewards_user ON bandit_rewards (user_id);
CREATE INDEX idx_bandit_rewards_created ON bandit_rewards (created_at DESC);

CREATE INDEX idx_user_feedback_article ON user_feedback (article_id);
CREATE INDEX idx_user_feedback_user ON user_feedback (user_id);
CREATE INDEX idx_user_feedback_submitted ON user_feedback (submitted_at DESC);

CREATE INDEX idx_digests_topic ON digests (topic);
CREATE INDEX idx_digests_created ON digests (created_at DESC);

CREATE INDEX idx_intelligence_data_user ON intelligence_data (user_id);
CREATE INDEX idx_intelligence_data_type ON intelligence_data (data_type);
CREATE INDEX idx_intelligence_data_created ON intelligence_data (created_at DESC);
CREATE INDEX idx_intelligence_data_analysis ON intelligence_data USING GIN (analysis_result);

CREATE INDEX idx_intelligence_insights_user ON intelligence_insights (user_id);
CREATE INDEX idx_intelligence_insights_type ON intelligence_insights (insight_type);
CREATE INDEX idx_intelligence_insights_severity ON intelligence_insights (severity);
CREATE INDEX idx_intelligence_insights_created ON intelligence_insights (created_at DESC);

CREATE INDEX idx_dashboard_updates_dashboard ON dashboard_updates (dashboard_id);
CREATE INDEX idx_dashboard_updates_user ON dashboard_updates (user_id);
CREATE INDEX idx_dashboard_updates_created ON dashboard_updates (created_at DESC);

-- Create materialized view for bandit statistics
CREATE MATERIALIZED VIEW bandit_statistics AS
SELECT 
    bs.key_name,
    COUNT(bs.id) as selection_count,
    AVG(bs.ucb_score) as avg_ucb_score,
    MAX(bs.ucb_score) as max_ucb_score,
    COUNT(br.id) as reward_count,
    AVG(br.reward) as avg_reward,
    MIN(br.reward) as min_reward,
    MAX(br.reward) as max_reward,
    STDDEV(br.reward) as reward_stddev
FROM bandit_selections bs
LEFT JOIN bandit_rewards br ON bs.key_name = br.key_name AND bs.user_id = br.user_id
GROUP BY bs.key_name;

-- Create materialized view for user engagement metrics
CREATE MATERIALIZED VIEW user_engagement_metrics AS
SELECT 
    uf.article_id,
    COUNT(uf.id) as feedback_count,
    AVG(uf.feedback_score) as avg_feedback_score,
    SUM(CASE WHEN uf.upvote = TRUE THEN 1 ELSE 0 END) as upvotes,
    SUM(CASE WHEN uf.clicked = TRUE THEN 1 ELSE 0 END) as clicks,
    MAX(uf.submitted_at) as last_feedback
FROM user_feedback uf
GROUP BY uf.article_id;

-- Refresh materialized views function
CREATE OR REPLACE FUNCTION refresh_enhanced_views()
RETURNS void AS $$
BEGIN
    REFRESH MATERIALIZED VIEW bandit_statistics;
    REFRESH MATERIALIZED VIEW user_engagement_metrics;
END;
$$ LANGUAGE plpgsql;

-- Grant permissions
GRANT SELECT, INSERT, UPDATE, DELETE ON bandit_selections TO osint_user;
GRANT SELECT, INSERT, UPDATE, DELETE ON bandit_rewards TO osint_user;
GRANT SELECT, INSERT, UPDATE, DELETE ON user_feedback TO osint_user;
GRANT SELECT, INSERT, UPDATE, DELETE ON digests TO osint_user;
GRANT SELECT, INSERT, UPDATE, DELETE ON intelligence_data TO osint_user;
GRANT SELECT, INSERT, UPDATE, DELETE ON intelligence_insights TO osint_user;
GRANT SELECT, INSERT, UPDATE, DELETE ON dashboard_updates TO osint_user;
GRANT SELECT ON bandit_statistics TO osint_user;
GRANT SELECT ON user_engagement_metrics TO osint_user;
GRANT EXECUTE ON FUNCTION refresh_enhanced_views() TO osint_user;
