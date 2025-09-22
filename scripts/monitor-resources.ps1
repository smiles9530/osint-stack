# Resource Monitoring Script for OSINT Stack (PowerShell)
# Monitor Docker container resource usage

Write-Host "📊 OSINT Stack Resource Monitoring" -ForegroundColor Cyan
Write-Host "=================================" -ForegroundColor Cyan

# Monitor Docker containers
Write-Host "`n🐳 Container Resource Usage:" -ForegroundColor Yellow
Write-Host "----------------------------" -ForegroundColor Yellow
docker stats --no-stream --format "table {{.Container}}\t{{.CPUPerc}}\t{{.MemUsage}}\t{{.MemPerc}}\t{{.NetIO}}\t{{.BlockIO}}"

Write-Host "`n💾 System Memory Usage:" -ForegroundColor Yellow
Write-Host "----------------------" -ForegroundColor Yellow
$memory = Get-WmiObject -Class Win32_OperatingSystem
$totalMemory = [math]::Round($memory.TotalVisibleMemorySize / 1MB, 2)
$freeMemory = [math]::Round($memory.FreePhysicalMemory / 1MB, 2)
$usedMemory = $totalMemory - $freeMemory
$memoryPercent = [math]::Round(($usedMemory / $totalMemory) * 100, 2)

Write-Host "Total Memory: $totalMemory GB" -ForegroundColor White
Write-Host "Used Memory: $usedMemory GB ($memoryPercent%)" -ForegroundColor White
Write-Host "Free Memory: $freeMemory GB" -ForegroundColor White

Write-Host "`n🖥️  CPU Usage:" -ForegroundColor Yellow
Write-Host "-------------" -ForegroundColor Yellow
$cpu = Get-WmiObject -Class Win32_Processor
Write-Host "CPU: $($cpu.Name)" -ForegroundColor White
Write-Host "Cores: $($cpu.NumberOfCores)" -ForegroundColor White
Write-Host "Logical Processors: $($cpu.NumberOfLogicalProcessors)" -ForegroundColor White

Write-Host "`n💿 Disk Usage:" -ForegroundColor Yellow
Write-Host "-------------" -ForegroundColor Yellow
Get-WmiObject -Class Win32_LogicalDisk | Where-Object {$_.DriveType -eq 3} | Select-Object DeviceID, @{Name="Size(GB)";Expression={[math]::Round($_.Size/1GB,2)}}, @{Name="FreeSpace(GB)";Expression={[math]::Round($_.FreeSpace/1GB,2)}}, @{Name="PercentFree";Expression={[math]::Round(($_.FreeSpace/$_.Size)*100,2)}} | Format-Table

Write-Host "`n🔍 Docker System Info:" -ForegroundColor Yellow
Write-Host "---------------------" -ForegroundColor Yellow
docker system df
