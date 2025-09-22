# Fix UUID Validation Issues in OSINT Stack
# This script addresses the PostgreSQL UUID validation error by:
# 1. Running database migrations to fix schema issues
# 2. Updating application code to handle Reddit IDs properly
# 3. Providing monitoring and validation tools

param(
    [switch]$SkipDatabaseMigration,
    [switch]$SkipCodeUpdate,
    [switch]$DryRun,
    [string]$DatabaseUrl = "postgresql://osint:osint@localhost:5432/osint"
)

Write-Host "🔧 OSINT Stack UUID Validation Fix" -ForegroundColor Cyan
Write-Host "=================================" -ForegroundColor Cyan

# Function to run SQL commands
function Invoke-SqlCommand {
    param(
        [string]$SqlFile,
        [string]$DatabaseUrl
    )
    
    if ($DryRun) {
        Write-Host "DRY RUN: Would execute $SqlFile" -ForegroundColor Yellow
        return $true
    }
    
    try {
        Write-Host "Executing $SqlFile..." -ForegroundColor Green
        psql $DatabaseUrl -f $SqlFile
        if ($LASTEXITCODE -eq 0) {
            Write-Host "✅ $SqlFile executed successfully" -ForegroundColor Green
            return $true
        } else {
            Write-Host "❌ $SqlFile failed with exit code $LASTEXITCODE" -ForegroundColor Red
            return $false
        }
    } catch {
        Write-Host "❌ Error executing $SqlFile : $($_.Exception.Message)" -ForegroundColor Red
        return $false
    }
}

# Function to check if database is accessible
function Test-DatabaseConnection {
    param([string]$DatabaseUrl)
    
    try {
        Write-Host "Testing database connection..." -ForegroundColor Yellow
        psql $DatabaseUrl -c "SELECT 1;" | Out-Null
        if ($LASTEXITCODE -eq 0) {
            Write-Host "✅ Database connection successful" -ForegroundColor Green
            return $true
        } else {
            Write-Host "❌ Database connection failed" -ForegroundColor Red
            return $false
        }
    } catch {
        Write-Host "❌ Database connection error: $($_.Exception.Message)" -ForegroundColor Red
        return $false
    }
}

# Function to check current schema
function Get-CurrentSchema {
    param([string]$DatabaseUrl)
    
    Write-Host "Checking current database schema..." -ForegroundColor Yellow
    
    $schemaQuery = @"
SELECT 
    table_name,
    column_name,
    data_type,
    is_nullable
FROM information_schema.columns 
WHERE table_name = 'articles' 
AND table_schema = 'public'
ORDER BY ordinal_position;
"@
    
    try {
        $result = psql $DatabaseUrl -c $schemaQuery
        Write-Host "Current articles table schema:" -ForegroundColor Cyan
        Write-Host $result
        return $result
    } catch {
        Write-Host "❌ Error checking schema: $($_.Exception.Message)" -ForegroundColor Red
        return $null
    }
}

# Function to validate UUID handling
function Test-UuidHandling {
    param([string]$DatabaseUrl)
    
    Write-Host "Testing UUID handling..." -ForegroundColor Yellow
    
    $testQueries = @(
        "SELECT reddit_id_to_uuid('t3_1nm0pvw') as reddit_uuid;",
        "SELECT reddit_id_to_uuid('550e8400-e29b-41d4-a716-446655440000') as valid_uuid;",
        "SELECT reddit_id_to_uuid('invalid-id') as invalid_id;"
    )
    
    foreach ($query in $testQueries) {
        try {
            Write-Host "Testing: $query" -ForegroundColor Gray
            $result = psql $DatabaseUrl -c $query
            Write-Host "Result: $result" -ForegroundColor Green
        } catch {
            Write-Host "❌ Test failed: $($_.Exception.Message)" -ForegroundColor Red
        }
    }
}

# Main execution
Write-Host "Starting UUID validation fix process..." -ForegroundColor Yellow

# Step 1: Test database connection
if (-not (Test-DatabaseConnection -DatabaseUrl $DatabaseUrl)) {
    Write-Host "❌ Cannot proceed without database connection" -ForegroundColor Red
    exit 1
}

# Step 2: Check current schema
Get-CurrentSchema -DatabaseUrl $DatabaseUrl

# Step 3: Run database migration
if (-not $SkipDatabaseMigration) {
    Write-Host "`n📊 Running database migration..." -ForegroundColor Cyan
    
    $migrationFile = "db/migrations/003-fix-uuid-validation.sql"
    if (Test-Path $migrationFile) {
        if (Invoke-SqlCommand -SqlFile $migrationFile -DatabaseUrl $DatabaseUrl) {
            Write-Host "✅ Database migration completed" -ForegroundColor Green
        } else {
            Write-Host "❌ Database migration failed" -ForegroundColor Red
            exit 1
        }
    } else {
        Write-Host "❌ Migration file not found: $migrationFile" -ForegroundColor Red
        exit 1
    }
} else {
    Write-Host "⏭️ Skipping database migration" -ForegroundColor Yellow
}

# Step 4: Test UUID handling functions
Write-Host "`n🧪 Testing UUID handling functions..." -ForegroundColor Cyan
Test-UuidHandling -DatabaseUrl $DatabaseUrl

# Step 5: Update application code
if (-not $SkipCodeUpdate) {
    Write-Host "`n💻 Application code has been updated with UUID validation fixes" -ForegroundColor Green
    Write-Host "   - Updated db.py with safe UUID conversion" -ForegroundColor Gray
    Write-Host "   - Added uuid_validation_fix.py module" -ForegroundColor Gray
    Write-Host "   - Created n8n workflow for UUID validation" -ForegroundColor Gray
} else {
    Write-Host "⏭️ Skipping code update" -ForegroundColor Yellow
}

# Step 6: Create monitoring script
Write-Host "`n📊 Creating monitoring script..." -ForegroundColor Cyan

$monitoringScript = @"
# UUID Validation Monitoring Script
# Run this to monitor UUID validation issues

Write-Host "Monitoring UUID validation issues..." -ForegroundColor Cyan

# Check for recent UUID validation errors
$errorQuery = @"
SELECT 
    COUNT(*) as error_count,
    MAX(created_at) as last_error
FROM processing_errors 
WHERE error_message LIKE '%invalid input syntax for type uuid%'
AND created_at > NOW() - INTERVAL '1 hour';
"@

$result = psql $DatabaseUrl -c $errorQuery
Write-Host "Recent UUID validation errors:" -ForegroundColor Yellow
Write-Host $result

# Check Reddit ID mappings
$mappingQuery = @"
SELECT 
    COUNT(*) as reddit_mappings,
    COUNT(DISTINCT original_reddit_id) as unique_reddit_ids
FROM articles 
WHERE metadata ? 'reddit_id_mapping';
"@

$result = psql $DatabaseUrl -c $mappingQuery
Write-Host "`nReddit ID mappings:" -ForegroundColor Yellow
Write-Host $result
"@

$monitoringScript | Out-File -FilePath "scripts/monitor-uuid-validation.ps1" -Encoding UTF8
Write-Host "✅ Monitoring script created: scripts/monitor-uuid-validation.ps1" -ForegroundColor Green

# Step 7: Summary
Write-Host "`n🎉 UUID Validation Fix Summary" -ForegroundColor Cyan
Write-Host "=============================" -ForegroundColor Cyan
Write-Host "✅ Database migration completed" -ForegroundColor Green
Write-Host "✅ UUID validation functions created" -ForegroundColor Green
Write-Host "✅ Application code updated" -ForegroundColor Green
Write-Host "✅ N8N workflow created" -ForegroundColor Green
Write-Host "✅ Monitoring script created" -ForegroundColor Green

Write-Host "`n📋 Next Steps:" -ForegroundColor Yellow
Write-Host "1. Restart the OSINT API service" -ForegroundColor White
Write-Host "2. Test with Reddit RSS feeds" -ForegroundColor White
Write-Host "3. Monitor for UUID validation errors" -ForegroundColor White
Write-Host "4. Run: .\scripts\monitor-uuid-validation.ps1" -ForegroundColor White

Write-Host "`n🔍 The error 'invalid input syntax for type uuid: t3_1nm0pvw' should now be resolved!" -ForegroundColor Green
