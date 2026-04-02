param(
    [Parameter(Mandatory = $true)]
    [string]$Message,

    [string]$Branch = "main",

    [switch]$SkipPush
)

$root = Split-Path -Parent $PSScriptRoot
Set-Location $root

$gitCmd = Get-Command git -ErrorAction SilentlyContinue
if (-not $gitCmd -and (Test-Path "C:\Program Files\Git\cmd\git.exe")) {
    $gitCmd = @{ Source = "C:\Program Files\Git\cmd\git.exe" }
}

if (-not $gitCmd) {
    Write-Error "Git is not installed or not available in PATH."
    exit 1
}

$git = $gitCmd.Source
$env:GIT_SSH_COMMAND = "C:\Windows\System32\OpenSSH\ssh.exe"

& "$PSScriptRoot\update_changelog.ps1" -Summary $Message
if ($LASTEXITCODE -ne 0) {
    exit $LASTEXITCODE
}

& $git add .
if ($LASTEXITCODE -ne 0) {
    exit $LASTEXITCODE
}

& $git commit -m $Message
if ($LASTEXITCODE -ne 0) {
    Write-Host "No commit created. There may be no staged changes."
}

if (-not $SkipPush) {
    & $git push origin $Branch
    exit $LASTEXITCODE
}
