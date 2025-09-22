-- Fix Sequences - Restore missing sequences for core tables
\connect osint

-- Recreate the users_id_seq sequence
CREATE SEQUENCE IF NOT EXISTS users_id_seq;
ALTER SEQUENCE users_id_seq OWNED BY users.id;
ALTER TABLE users ALTER COLUMN id SET DEFAULT nextval('users_id_seq'::regclass);

-- Recreate the sources_id_seq sequence
CREATE SEQUENCE IF NOT EXISTS sources_id_seq;
ALTER SEQUENCE sources_id_seq OWNED BY sources.id;
ALTER TABLE sources ALTER COLUMN id SET DEFAULT nextval('sources_id_seq'::regclass);

-- Recreate the user_sessions_id_seq sequence
CREATE SEQUENCE IF NOT EXISTS user_sessions_id_seq;
ALTER SEQUENCE user_sessions_id_seq OWNED BY user_sessions.id;
ALTER TABLE user_sessions ALTER COLUMN id SET DEFAULT nextval('user_sessions_id_seq'::regclass);

-- Recreate other core sequences
CREATE SEQUENCE IF NOT EXISTS processing_errors_id_seq;
ALTER SEQUENCE processing_errors_id_seq OWNED BY processing_errors.id;
ALTER TABLE processing_errors ALTER COLUMN id SET DEFAULT nextval('processing_errors_id_seq'::regclass);

CREATE SEQUENCE IF NOT EXISTS dashboard_configs_id_seq;
ALTER SEQUENCE dashboard_configs_id_seq OWNED BY dashboard_configs.id;
ALTER TABLE dashboard_configs ALTER COLUMN id SET DEFAULT nextval('dashboard_configs_id_seq'::regclass);

CREATE SEQUENCE IF NOT EXISTS real_time_alerts_id_seq;
ALTER SEQUENCE real_time_alerts_id_seq OWNED BY real_time_alerts.id;
ALTER TABLE real_time_alerts ALTER COLUMN id SET DEFAULT nextval('real_time_alerts_id_seq'::regclass);

CREATE SEQUENCE IF NOT EXISTS websocket_connections_id_seq;
ALTER SEQUENCE websocket_connections_id_seq OWNED BY websocket_connections.id;
ALTER TABLE websocket_connections ALTER COLUMN id SET DEFAULT nextval('websocket_connections_id_seq'::regclass);

CREATE SEQUENCE IF NOT EXISTS intelligence_processing_id_seq;
ALTER SEQUENCE intelligence_processing_id_seq OWNED BY intelligence_processing.id;
ALTER TABLE intelligence_processing ALTER COLUMN id SET DEFAULT nextval('intelligence_processing_id_seq'::regclass);

CREATE SEQUENCE IF NOT EXISTS intelligence_synthesis_id_seq;
ALTER SEQUENCE intelligence_synthesis_id_seq OWNED BY intelligence_synthesis.id;
ALTER TABLE intelligence_synthesis ALTER COLUMN id SET DEFAULT nextval('intelligence_synthesis_id_seq'::regclass);

CREATE SEQUENCE IF NOT EXISTS topic_analysis_id_seq;
ALTER SEQUENCE topic_analysis_id_seq OWNED BY topic_analysis.id;
ALTER TABLE topic_analysis ALTER COLUMN id SET DEFAULT nextval('topic_analysis_id_seq'::regclass);

CREATE SEQUENCE IF NOT EXISTS trend_analysis_id_seq;
ALTER SEQUENCE trend_analysis_id_seq OWNED BY trend_analysis.id;
ALTER TABLE trend_analysis ALTER COLUMN id SET DEFAULT nextval('trend_analysis_id_seq'::regclass);

CREATE SEQUENCE IF NOT EXISTS anomaly_detections_id_seq;
ALTER SEQUENCE anomaly_detections_id_seq OWNED BY anomaly_detections.id;
ALTER TABLE anomaly_detections ALTER COLUMN id SET DEFAULT nextval('anomaly_detections_id_seq'::regclass);

CREATE SEQUENCE IF NOT EXISTS ml_model_performance_id_seq;
ALTER SEQUENCE ml_model_performance_id_seq OWNED BY ml_model_performance.id;
ALTER TABLE ml_model_performance ALTER COLUMN id SET DEFAULT nextval('ml_model_performance_id_seq'::regclass);

CREATE SEQUENCE IF NOT EXISTS ml_insights_cache_id_seq;
ALTER SEQUENCE ml_insights_cache_id_seq OWNED BY ml_insights_cache.id;
ALTER TABLE ml_insights_cache ALTER COLUMN id SET DEFAULT nextval('ml_insights_cache_id_seq'::regclass);

-- Set the current values of sequences to match existing data
SELECT setval('users_id_seq', COALESCE(MAX(id), 1)) FROM users;
SELECT setval('sources_id_seq', COALESCE(MAX(id), 1)) FROM sources;
SELECT setval('user_sessions_id_seq', COALESCE(MAX(id), 1)) FROM user_sessions;
SELECT setval('processing_errors_id_seq', COALESCE(MAX(id), 1)) FROM processing_errors;
SELECT setval('dashboard_configs_id_seq', COALESCE(MAX(id), 1)) FROM dashboard_configs;
SELECT setval('real_time_alerts_id_seq', COALESCE(MAX(id), 1)) FROM real_time_alerts;
SELECT setval('websocket_connections_id_seq', COALESCE(MAX(id), 1)) FROM websocket_connections;
SELECT setval('intelligence_processing_id_seq', COALESCE(MAX(id), 1)) FROM intelligence_processing;
SELECT setval('intelligence_synthesis_id_seq', COALESCE(MAX(id), 1)) FROM intelligence_synthesis;
SELECT setval('topic_analysis_id_seq', COALESCE(MAX(id), 1)) FROM topic_analysis;
SELECT setval('trend_analysis_id_seq', COALESCE(MAX(id), 1)) FROM trend_analysis;
SELECT setval('anomaly_detections_id_seq', COALESCE(MAX(id), 1)) FROM anomaly_detections;
SELECT setval('ml_model_performance_id_seq', COALESCE(MAX(id), 1)) FROM ml_model_performance;
SELECT setval('ml_insights_cache_id_seq', COALESCE(MAX(id), 1)) FROM ml_insights_cache;

-- Verify the fix
SELECT 'users' as table_name, column_name, column_default 
FROM information_schema.columns 
WHERE table_name = 'users' AND column_name = 'id'
UNION ALL
SELECT 'sources' as table_name, column_name, column_default 
FROM information_schema.columns 
WHERE table_name = 'sources' AND column_name = 'id';

\echo 'Sequences restored successfully!'
