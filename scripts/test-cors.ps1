# CORS Configuration Test Script
# This script helps validate CORS configuration for the OSINT Stack

param(
    [string]$ApiUrl = "http://localhost:8000",
    [string]$Origin = "http://localhost:3000"
)

Write-Host "🧪 Testing CORS configuration..." -ForegroundColor Green
Write-Host "API URL: $ApiUrl" -ForegroundColor Yellow
Write-Host "Origin: $Origin" -ForegroundColor Yellow
Write-Host ""

# Test health endpoint with CORS headers
try {
    Write-Host "Testing preflight request..." -ForegroundColor Cyan
    
    $headers = @{
        'Origin' = $Origin
        'Access-Control-Request-Method' = 'GET'
        'Access-Control-Request-Headers' = 'Content-Type,Authorization'
    }
    
    $response = Invoke-WebRequest -Uri "$ApiUrl/healthz" -Method OPTIONS -Headers $headers -UseBasicParsing
    
    Write-Host "✅ Preflight request successful" -ForegroundColor Green
    Write-Host "Status Code: $($response.StatusCode)" -ForegroundColor White
    
    # Check CORS headers
    $corsHeaders = @()
    foreach ($header in $response.Headers.GetEnumerator()) {
        if ($header.Key -like "*Access-Control*") {
            $corsHeaders += "$($header.Key): $($header.Value)"
        }
    }
    
    if ($corsHeaders.Count -gt 0) {
        Write-Host ""
        Write-Host "CORS Headers:" -ForegroundColor Cyan
        foreach ($header in $corsHeaders) {
            Write-Host "  $header" -ForegroundColor White
        }
    } else {
        Write-Host "⚠️  No CORS headers found in response" -ForegroundColor Yellow
    }
    
} catch {
    Write-Host "❌ CORS test failed" -ForegroundColor Red
    Write-Host "Error: $($_.Exception.Message)" -ForegroundColor Red
}

Write-Host ""
Write-Host "🔍 Current environment CORS configuration:" -ForegroundColor Cyan

# Check .env file for CORS configuration
if (Test-Path ".env") {
    $corsConfig = Select-String -Path ".env" -Pattern "CORS_ORIGINS" | Select-Object -First 1
    if ($corsConfig) {
        Write-Host "  $($corsConfig.Line)" -ForegroundColor White
        
        # Parse and display origins
        $origins = ($corsConfig.Line -split "=")[1] -split ","
        Write-Host ""
        Write-Host "Allowed origins:" -ForegroundColor Cyan
        foreach ($origin in $origins) {
            $trimmedOrigin = $origin.Trim()
            if ($trimmedOrigin -eq $Origin) {
                Write-Host "  ✅ $trimmedOrigin (matches test origin)" -ForegroundColor Green
            } else {
                Write-Host "  📋 $trimmedOrigin" -ForegroundColor White
            }
        }
    } else {
        Write-Host "  ⚠️  CORS_ORIGINS not found in .env file" -ForegroundColor Yellow
    }
} else {
    Write-Host "  ⚠️  .env file not found" -ForegroundColor Yellow
}

Write-Host ""
Write-Host "💡 Tips:" -ForegroundColor Blue
Write-Host "  • Ensure your origin is listed in CORS_ORIGINS" -ForegroundColor White
Write-Host "  • Use HTTPS origins in production" -ForegroundColor White
Write-Host "  • Restart services after changing CORS configuration" -ForegroundColor White
Write-Host ""
Write-Host "Usage examples:" -ForegroundColor Blue
Write-Host "  .\test-cors.ps1" -ForegroundColor Gray
Write-Host "  .\test-cors.ps1 -ApiUrl http://localhost:8000 -Origin http://localhost:3000" -ForegroundColor Gray
Write-Host "  .\test-cors.ps1 -ApiUrl https://api.yourdomain.com -Origin https://yourdomain.com" -ForegroundColor Gray
