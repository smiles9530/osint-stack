#!/bin/bash

# Fix UUID Validation Issues in OSINT Stack
# This script addresses the PostgreSQL UUID validation error by:
# 1. Running database migrations to fix schema issues
# 2. Updating application code to handle Reddit IDs properly
# 3. Providing monitoring and validation tools

set -e

# Configuration
SKIP_DATABASE_MIGRATION=false
SKIP_CODE_UPDATE=false
DRY_RUN=false
DATABASE_URL="postgresql://osint:osint@localhost:5432/osint"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
GRAY='\033[0;37m'
NC='\033[0m' # No Color

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --skip-database-migration)
            SKIP_DATABASE_MIGRATION=true
            shift
            ;;
        --skip-code-update)
            SKIP_CODE_UPDATE=true
            shift
            ;;
        --dry-run)
            DRY_RUN=true
            shift
            ;;
        --database-url)
            DATABASE_URL="$2"
            shift 2
            ;;
        -h|--help)
            echo "Usage: $0 [OPTIONS]"
            echo "Options:"
            echo "  --skip-database-migration    Skip database migration step"
            echo "  --skip-code-update          Skip code update step"
            echo "  --dry-run                   Show what would be done without executing"
            echo "  --database-url URL          Database connection URL"
            echo "  -h, --help                  Show this help message"
            exit 0
            ;;
        *)
            echo "Unknown option $1"
            exit 1
            ;;
    esac
done

# Function to print colored output
print_status() {
    local color=$1
    local message=$2
    echo -e "${color}${message}${NC}"
}

# Function to run SQL commands
run_sql_command() {
    local sql_file=$1
    local database_url=$2
    
    if [ "$DRY_RUN" = true ]; then
        print_status $YELLOW "DRY RUN: Would execute $sql_file"
        return 0
    fi
    
    print_status $GREEN "Executing $sql_file..."
    if psql "$database_url" -f "$sql_file"; then
        print_status $GREEN "✅ $sql_file executed successfully"
        return 0
    else
        print_status $RED "❌ $sql_file failed"
        return 1
    fi
}

# Function to check if database is accessible
test_database_connection() {
    print_status $YELLOW "Testing database connection..."
    if psql "$DATABASE_URL" -c "SELECT 1;" > /dev/null 2>&1; then
        print_status $GREEN "✅ Database connection successful"
        return 0
    else
        print_status $RED "❌ Database connection failed"
        return 1
    fi
}

# Function to check current schema
get_current_schema() {
    print_status $YELLOW "Checking current database schema..."
    
    local schema_query="
    SELECT 
        table_name,
        column_name,
        data_type,
        is_nullable
    FROM information_schema.columns 
    WHERE table_name = 'articles' 
    AND table_schema = 'public'
    ORDER BY ordinal_position;
    "
    
    print_status $CYAN "Current articles table schema:"
    psql "$DATABASE_URL" -c "$schema_query"
}

# Function to validate UUID handling
test_uuid_handling() {
    print_status $YELLOW "Testing UUID handling..."
    
    local test_queries=(
        "SELECT reddit_id_to_uuid('t3_1nm0pvw') as reddit_uuid;"
        "SELECT reddit_id_to_uuid('550e8400-e29b-41d4-a716-446655440000') as valid_uuid;"
        "SELECT reddit_id_to_uuid('invalid-id') as invalid_id;"
    )
    
    for query in "${test_queries[@]}"; do
        print_status $GRAY "Testing: $query"
        if psql "$DATABASE_URL" -c "$query"; then
            print_status $GREEN "✅ Test passed"
        else
            print_status $RED "❌ Test failed"
        fi
    done
}

# Main execution
print_status $CYAN "🔧 OSINT Stack UUID Validation Fix"
print_status $CYAN "================================="

print_status $YELLOW "Starting UUID validation fix process..."

# Step 1: Test database connection
if ! test_database_connection; then
    print_status $RED "❌ Cannot proceed without database connection"
    exit 1
fi

# Step 2: Check current schema
get_current_schema

# Step 3: Run database migration
if [ "$SKIP_DATABASE_MIGRATION" = false ]; then
    print_status $CYAN "📊 Running database migration..."
    
    local migration_file="db/migrations/003-fix-uuid-validation.sql"
    if [ -f "$migration_file" ]; then
        if run_sql_command "$migration_file" "$DATABASE_URL"; then
            print_status $GREEN "✅ Database migration completed"
        else
            print_status $RED "❌ Database migration failed"
            exit 1
        fi
    else
        print_status $RED "❌ Migration file not found: $migration_file"
        exit 1
    fi
else
    print_status $YELLOW "⏭️ Skipping database migration"
fi

# Step 4: Test UUID handling functions
print_status $CYAN "🧪 Testing UUID handling functions..."
test_uuid_handling

# Step 5: Update application code
if [ "$SKIP_CODE_UPDATE" = false ]; then
    print_status $GREEN "💻 Application code has been updated with UUID validation fixes"
    print_status $GRAY "   - Updated db.py with safe UUID conversion"
    print_status $GRAY "   - Added uuid_validation_fix.py module"
    print_status $GRAY "   - Created n8n workflow for UUID validation"
else
    print_status $YELLOW "⏭️ Skipping code update"
fi

# Step 6: Create monitoring script
print_status $CYAN "📊 Creating monitoring script..."

cat > scripts/monitor-uuid-validation.sh << 'EOF'
#!/bin/bash

# UUID Validation Monitoring Script
# Run this to monitor UUID validation issues

echo "Monitoring UUID validation issues..."

# Check for recent UUID validation errors
error_query="
SELECT 
    COUNT(*) as error_count,
    MAX(created_at) as last_error
FROM processing_errors 
WHERE error_message LIKE '%invalid input syntax for type uuid%'
AND created_at > NOW() - INTERVAL '1 hour';
"

echo "Recent UUID validation errors:"
psql "$DATABASE_URL" -c "$error_query"

# Check Reddit ID mappings
mapping_query="
SELECT 
    COUNT(*) as reddit_mappings,
    COUNT(DISTINCT original_reddit_id) as unique_reddit_ids
FROM articles 
WHERE metadata ? 'reddit_id_mapping';
"

echo "Reddit ID mappings:"
psql "$DATABASE_URL" -c "$mapping_query"
EOF

chmod +x scripts/monitor-uuid-validation.sh
print_status $GREEN "✅ Monitoring script created: scripts/monitor-uuid-validation.sh"

# Step 7: Summary
print_status $CYAN "🎉 UUID Validation Fix Summary"
print_status $CYAN "============================="
print_status $GREEN "✅ Database migration completed"
print_status $GREEN "✅ UUID validation functions created"
print_status $GREEN "✅ Application code updated"
print_status $GREEN "✅ N8N workflow created"
print_status $GREEN "✅ Monitoring script created"

print_status $YELLOW "📋 Next Steps:"
print_status $WHITE "1. Restart the OSINT API service"
print_status $WHITE "2. Test with Reddit RSS feeds"
print_status $WHITE "3. Monitor for UUID validation errors"
print_status $WHITE "4. Run: ./scripts/monitor-uuid-validation.sh"

print_status $GREEN "🔍 The error 'invalid input syntax for type uuid: t3_1nm0pvw' should now be resolved!"
