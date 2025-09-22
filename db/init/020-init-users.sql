-- Initialize default admin user
\connect osint

-- Create admin user if it doesn't exist
INSERT INTO users (username, email, hashed_password, is_active, is_superuser)
SELECT 'admin', 'admin@example.com', '$2b$12$LQv3c1yqBWVHxkd0LHAkCOYz6TtxMQJqhN8/LewdBPj4J/7.8.8.8', TRUE, TRUE
WHERE NOT EXISTS (SELECT 1 FROM users WHERE username = 'admin');

-- Create a regular user for testing
INSERT INTO users (username, email, hashed_password, is_active, is_superuser)
SELECT 'testuser', 'test@example.com', '$2b$12$LQv3c1yqBWVHxkd0LHAkCOYz6TtxMQJqhN8/LewdBPj4J/7.8.8.8', TRUE, FALSE
WHERE NOT EXISTS (SELECT 1 FROM users WHERE username = 'testuser');
