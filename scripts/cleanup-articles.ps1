# Articles Table Cleanup Script
# Provides multiple cleanup options for the articles table

param(
    [string]$DatabaseUrl = "postgresql://osint:osint@localhost:5432/osint",
    [string]$CleanupType = "all",
    [int]$KeepDays = 7,
    [switch]$DryRun
)

Write-Host "🧹 ARTICLES TABLE CLEANUP" -ForegroundColor Cyan
Write-Host "=========================" -ForegroundColor Cyan

# Function to run SQL commands
function Invoke-SqlCommand {
    param([string]$SqlCommand, [string]$DatabaseUrl)
    
    if ($DryRun) {
        Write-Host "DRY RUN: Would execute: $SqlCommand" -ForegroundColor Yellow
        return $true
    }
    
    try {
        Write-Host "Executing: $SqlCommand" -ForegroundColor Yellow
        $result = $SqlCommand | docker exec -i osint-db psql -U osint -d osint
        Write-Host "Result: $result" -ForegroundColor Green
        return $true
    } catch {
        Write-Host "Error: $($_.Exception.Message)" -ForegroundColor Red
        return $false
    }
}

# Show current state
Write-Host "`n📊 Current Articles Table State:" -ForegroundColor Yellow
$currentState = @"
SELECT 
    COUNT(*) as total_articles,
    MIN(fetched_at) as earliest,
    MAX(fetched_at) as latest,
    COUNT(DISTINCT source_name) as unique_sources
FROM articles;
"@
$currentState | docker exec -i osint-db psql -U osint -d osint

# Show source breakdown
Write-Host "`n📈 Articles by Source:" -ForegroundColor Yellow
$sourceBreakdown = @"
SELECT source_name, COUNT(*) as count 
FROM articles 
GROUP BY source_name 
ORDER BY count DESC;
"@
$sourceBreakdown | docker exec -i osint-db psql -U osint -d osint

# Cleanup options
Write-Host "`n🔧 Cleanup Options:" -ForegroundColor Cyan
Write-Host "1. Remove all articles (complete cleanup)" -ForegroundColor White
Write-Host "2. Remove articles older than X days" -ForegroundColor White
Write-Host "3. Remove articles from specific sources" -ForegroundColor White
Write-Host "4. Remove test articles only" -ForegroundColor White
Write-Host "5. Remove duplicate articles (by URL)" -ForegroundColor White

# Execute cleanup based on type
switch ($CleanupType.ToLower()) {
    "all" {
        Write-Host "`n🗑️ Option 1: Complete Cleanup (Remove All Articles)" -ForegroundColor Red
        $cleanupQuery = "DELETE FROM articles;"
        Invoke-SqlCommand -SqlCommand $cleanupQuery -DatabaseUrl $DatabaseUrl
    }
    
    "old" {
        Write-Host "`n🗑️ Option 2: Remove Articles Older Than $KeepDays Days" -ForegroundColor Yellow
        $cleanupQuery = @"
DELETE FROM articles 
WHERE fetched_at < NOW() - INTERVAL '$KeepDays days';
"@
        Invoke-SqlCommand -SqlCommand $cleanupQuery -DatabaseUrl $DatabaseUrl
    }
    
    "test" {
        Write-Host "`n🗑️ Option 4: Remove Test Articles Only" -ForegroundColor Yellow
        $cleanupQuery = @"
DELETE FROM articles 
WHERE source_name ILIKE '%test%' 
   OR title ILIKE '%test%' 
   OR url ILIKE '%test%';
"@
        Invoke-SqlCommand -SqlCommand $cleanupQuery -DatabaseUrl $DatabaseUrl
    }
    
    "duplicates" {
        Write-Host "`n🗑️ Option 5: Remove Duplicate Articles (Keep Latest)" -ForegroundColor Yellow
        $cleanupQuery = @"
DELETE FROM articles 
WHERE id NOT IN (
    SELECT DISTINCT ON (url) id 
    FROM articles 
    ORDER BY url, fetched_at DESC
);
"@
        Invoke-SqlCommand -SqlCommand $cleanupQuery -DatabaseUrl $DatabaseUrl
    }
    
    "bbc" {
        Write-Host "`n🗑️ Option 3: Remove BBC Articles Only" -ForegroundColor Yellow
        $cleanupQuery = @"
DELETE FROM articles 
WHERE source_name ILIKE '%BBC%';
"@
        Invoke-SqlCommand -SqlCommand $cleanupQuery -DatabaseUrl $DatabaseUrl
    }
    
    default {
        Write-Host "`n❌ Invalid cleanup type. Available options:" -ForegroundColor Red
        Write-Host "  -CleanupType all        (remove all articles)" -ForegroundColor White
        Write-Host "  -CleanupType old        (remove articles older than X days)" -ForegroundColor White
        Write-Host "  -CleanupType test       (remove test articles only)" -ForegroundColor White
        Write-Host "  -CleanupType duplicates (remove duplicate articles)" -ForegroundColor White
        Write-Host "  -CleanupType bbc        (remove BBC articles only)" -ForegroundColor White
        Write-Host "`nExample: .\cleanup-articles.ps1 -CleanupType all" -ForegroundColor Yellow
        Write-Host "Example: .\cleanup-articles.ps1 -CleanupType old -KeepDays 3" -ForegroundColor Yellow
        exit 1
    }
}

# Show final state
Write-Host "`n📊 Final Articles Table State:" -ForegroundColor Green
$finalState = @"
SELECT 
    COUNT(*) as total_articles,
    MIN(fetched_at) as earliest,
    MAX(fetched_at) as latest,
    COUNT(DISTINCT source_name) as unique_sources
FROM articles;
"@
$finalState | docker exec -i osint-db psql -U osint -d osint

# Clean up staging table as well
Write-Host "`n🧹 Cleaning up staging table..." -ForegroundColor Yellow
$stagingCleanup = "DELETE FROM articles_staging;"
Invoke-SqlCommand -SqlCommand $stagingCleanup -DatabaseUrl $DatabaseUrl

Write-Host "`n✅ Articles table cleanup completed!" -ForegroundColor Green
