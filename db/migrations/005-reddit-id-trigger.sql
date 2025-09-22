-- Create trigger to automatically handle Reddit IDs
\connect osint

-- Create trigger function to handle Reddit IDs
CREATE OR REPLACE FUNCTION handle_reddit_id_trigger()
RETURNS TRIGGER AS $$
BEGIN
    -- If the ID looks like a Reddit ID, convert it
    IF NEW.id ~ '^t[123]_[a-zA-Z0-9]+$' THEN
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
$$ LANGUAGE plpgsql;

-- Create the trigger
DROP TRIGGER IF EXISTS reddit_id_trigger ON articles;
CREATE TRIGGER reddit_id_trigger
    BEFORE INSERT ON articles
    FOR EACH ROW
    EXECUTE FUNCTION handle_reddit_id_trigger();

-- Test the trigger
SELECT 'Reddit ID trigger created successfully!' as status;
