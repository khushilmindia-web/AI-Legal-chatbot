param(
    [Parameter(Mandatory = $true)]
    [string]$Summary
)

$root = Split-Path -Parent $PSScriptRoot
$changelog = Join-Path $root "changelog.md"

if (-not (Test-Path $changelog)) {
    "# Changelog`n" | Set-Content -Path $changelog -Encoding UTF8
}

$dateHeader = "## $(Get-Date -Format 'yyyy-MM-dd')"
$content = Get-Content -Path $changelog -Raw -Encoding UTF8

if ($content -notmatch [regex]::Escape($dateHeader)) {
    $updated = "# Changelog`n`n$dateHeader`n`n### Added`n- $Summary`n`n" + ($content -replace '^# Changelog\s*', '')
    Set-Content -Path $changelog -Value $updated -Encoding UTF8
    Write-Host "Added new date section to changelog."
    exit 0
}

$replacement = "$dateHeader`n`n### Added`n- $Summary"
$updatedContent = [regex]::Replace(
    $content,
    [regex]::Escape($dateHeader) + "(\r?\n)+### Added",
    $replacement,
    1
)

if ($updatedContent -eq $content) {
    $updatedContent = $content -replace [regex]::Escape($dateHeader), "$dateHeader`n`n### Added`n- $Summary"
}

Set-Content -Path $changelog -Value $updatedContent -Encoding UTF8
Write-Host "Updated changelog."
