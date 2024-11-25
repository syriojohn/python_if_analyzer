# Run this script as administrator
Write-Host "Fixing Python PATH and Windows Store aliases..." -ForegroundColor Green

# Disable Windows Store Python aliases
Write-Host "Disabling Windows Store Python aliases..." -ForegroundColor Yellow
$registryPath = "HKCU:\Software\Microsoft\Windows\CurrentVersion\App Paths\python.exe"
if (Test-Path $registryPath) {
    Remove-Item $registryPath -Force
    Write-Host "Removed python.exe alias" -ForegroundColor Green
}

$registryPath = "HKCU:\Software\Microsoft\Windows\CurrentVersion\App Paths\python3.exe"
if (Test-Path $registryPath) {
    Remove-Item $registryPath -Force
    Write-Host "Removed python3.exe alias" -ForegroundColor Green
}

# Fix PATH order
Write-Host "Fixing PATH order..." -ForegroundColor Yellow

# Get current user PATH
$userPath = [Environment]::GetEnvironmentVariable("PATH", "User")
$paths = $userPath -split ";"

# Python paths we want to ensure are present and at the start
$pythonPaths = @(
    "C:\Users\syrio\AppData\Local\Programs\Python\Python313\",
    "C:\Users\syrio\AppData\Local\Programs\Python\Python313\Scripts\"
)

# Remove any existing Python paths (to avoid duplicates)
$paths = $paths | Where-Object { 
    $path = $_
    -not ($pythonPaths | Where-Object { $path -like "$_*" })
}

# Remove WindowsApps path
$paths = $paths | Where-Object { -not $_.Contains("WindowsApps") }

# Add Python paths at the start
$newPath = ($pythonPaths + $paths) -join ";"

# Set the new PATH
[Environment]::SetEnvironmentVariable("PATH", $newPath, "User")

Write-Host "PATH updated successfully!" -ForegroundColor Green

# Verify changes
Write-Host "`nVerifying changes..." -ForegroundColor Yellow
Write-Host "Current Python location:" -ForegroundColor Yellow
$pythonLocation = (Get-Command python -ErrorAction SilentlyContinue).Source
if ($pythonLocation) {
    Write-Host $pythonLocation -ForegroundColor Green
    $version = & python --version
    Write-Host "Python Version: $version" -ForegroundColor Green
} else {
    Write-Host "Python not found in PATH. Please restart your terminal." -ForegroundColor Red
}

Write-Host "`nNew PATH value:" -ForegroundColor Yellow
$newUserPath = [Environment]::GetEnvironmentVariable("PATH", "User")
$newUserPath -split ";" | ForEach-Object { Write-Host $_ }

Write-Host "`nPlease restart any open terminals for changes to take effect." -ForegroundColor Green
Write-Host "Press any key to continue..."
$null = $Host.UI.RawUI.ReadKey("NoEcho,IncludeKeyDown")
