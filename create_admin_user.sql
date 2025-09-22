-- Create admin user for OSINT Stack
-- This creates a user with username 'admin' and a secure password

INSERT INTO users (username, email, hashed_password, is_superuser, is_active, created_at, updated_at)
VALUES (
    'admin',                                    -- username
    'admin@example.com',                       -- email
    '$2b$12$placeholder_hash_replace_with_secure_password_hash',  -- Replace with secure password hash
    true,                                      -- is_superuser
    true,                                      -- is_active
    NOW(),                                     -- created_at
    NOW()                                      -- updated_at
) ON CONFLICT (username) DO NOTHING;

-- Create a regular user
INSERT INTO users (username, email, hashed_password, is_superuser, is_active, created_at, updated_at)
VALUES (
    'user',                                    -- username
    'user@example.com',                        -- email
    '$2b$12$LQv3c1yqBWVHxkd0LHAkCOYz6TtxMQJqhN8/LewdBPj4J/8.8.8.8',  -- hashed password (password: 'user123')
    false,                                     -- is_superuser
    true,                                      -- is_active
    NOW(),                                     -- created_at
    NOW()                                      -- updated_at
) ON CONFLICT (username) DO NOTHING;

-- Check created users
SELECT id, username, email, is_active, is_superuser, created_at 
FROM users 
ORDER BY created_at DESC;
