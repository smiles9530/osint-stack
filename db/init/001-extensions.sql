-- Create additional databases
CREATE DATABASE osint;
CREATE DATABASE n8n;
CREATE DATABASE superset;

-- Connect to osint database and create extensions
\c osint
CREATE EXTENSION IF NOT EXISTS postgis;
CREATE EXTENSION IF NOT EXISTS timescaledb;
CREATE EXTENSION IF NOT EXISTS vector;

-- Connect to superset database and create extensions
\c superset
CREATE EXTENSION IF NOT EXISTS postgis;

-- Connect to n8n database and create extensions
\c n8n
CREATE EXTENSION IF NOT EXISTS postgis;

-- Connect back to osint for schema creation
\c osint
