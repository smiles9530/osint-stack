-- Database partition management and maintenance
\connect osint

-- Create function to automatically create new monthly partitions
CREATE OR REPLACE FUNCTION create_monthly_partition(table_name TEXT, start_date DATE)
RETURNS void AS $$
DECLARE
    partition_name TEXT;
    end_date DATE;
BEGIN
    partition_name := table_name || '_' || to_char(start_date, 'YYYY_MM');
    end_date := start_date + INTERVAL '1 month';
    
    EXECUTE format('CREATE TABLE IF NOT EXISTS %I PARTITION OF %I
                   FOR VALUES FROM (%L) TO (%L)',
                   partition_name, table_name, start_date, end_date);
END;
$$ LANGUAGE plpgsql;

-- Create function to drop old partitions (older than 2 years)
CREATE OR REPLACE FUNCTION drop_old_partitions(table_name TEXT, retention_months INTEGER DEFAULT 24)
RETURNS void AS $$
DECLARE
    partition_name TEXT;
    cutoff_date DATE;
    rec RECORD;
BEGIN
    cutoff_date := CURRENT_DATE - (retention_months || ' months')::INTERVAL;
    
    FOR rec IN
        SELECT schemaname, tablename
        FROM pg_tables
        WHERE tablename LIKE table_name || '_%'
        AND tablename ~ '^\d{4}_\d{2}$'
        AND to_date(substring(tablename from '(\d{4}_\d{2})$'), 'YYYY_MM') < cutoff_date
    LOOP
        EXECUTE format('DROP TABLE IF EXISTS %I.%I CASCADE',
                      rec.schemaname, rec.tablename);
    END LOOP;
END;
$$ LANGUAGE plpgsql;

-- Create function to get partition statistics
CREATE OR REPLACE FUNCTION get_partition_stats(table_name TEXT)
RETURNS TABLE(
    partition_name TEXT,
    row_count BIGINT,
    size_mb NUMERIC,
    last_updated TIMESTAMPTZ
) AS $$
BEGIN
    RETURN QUERY
    SELECT 
        t.tablename::TEXT,
        COALESCE(s.n_tup_ins - s.n_tup_del, 0)::BIGINT as row_count,
        ROUND(pg_total_relation_size(c.oid) / 1024.0 / 1024.0, 2) as size_mb,
        s.last_autovacuum
    FROM pg_tables t
    LEFT JOIN pg_class c ON c.relname = t.tablename
    LEFT JOIN pg_stat_user_tables s ON s.relname = t.tablename
    WHERE t.tablename LIKE table_name || '_%'
    AND t.tablename ~ '^\d{4}_\d{2}$'
    ORDER BY t.tablename;
END;
$$ LANGUAGE plpgsql;

-- Create function to optimize partitions
CREATE OR REPLACE FUNCTION optimize_partitions()
RETURNS void AS $$
DECLARE
    rec RECORD;
BEGIN
    -- Analyze all partitions
    FOR rec IN
        SELECT schemaname, tablename
        FROM pg_tables
        WHERE tablename LIKE 'articles_%'
        AND tablename ~ '^\d{4}_\d{2}$'
    LOOP
        EXECUTE format('ANALYZE %I.%I', rec.schemaname, rec.tablename);
    END LOOP;
    
    -- Refresh materialized views
    PERFORM refresh_article_stats();
END;
$$ LANGUAGE plpgsql;

-- Create scheduled job for partition maintenance (requires pg_cron extension)
-- This would be run by a cron job or scheduled task in production
CREATE OR REPLACE FUNCTION schedule_partition_maintenance()
RETURNS void AS $$
BEGIN
    -- Create next month's partition
    PERFORM create_monthly_partition('articles', 
        date_trunc('month', CURRENT_DATE + INTERVAL '1 month')::DATE);
    
    -- Drop old partitions (older than 24 months)
    PERFORM drop_old_partitions('articles', 24);
    
    -- Optimize partitions
    PERFORM optimize_partitions();
END;
$$ LANGUAGE plpgsql;
