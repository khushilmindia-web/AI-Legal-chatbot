# Local Git Automation

These helper scripts prepare the repo for lightweight local Git automation once Git is installed and authenticated.

## Files

- `scripts/update_changelog.ps1`
- `scripts/git_sync.ps1`

## What they do

`update_changelog.ps1`
- updates `changelog.md`
- adds a dated entry with a short summary

`git_sync.ps1`
- updates `changelog.md`
- stages all changes
- creates a commit
- pushes to `origin`

## Usage

```powershell
.\scripts\git_sync.ps1 -Message "Add MVP upload flow and chat history fixes"
```

If you only want the local commit:

```powershell
.\scripts\git_sync.ps1 -Message "Local checkpoint" -SkipPush
```

## Before this works

You still need:

1. Git installed
2. repository remote configured
3. GitHub authentication working for `git@github.com:khushilmindia-web/AI-Legal-chatbot.git`

The helper script now also falls back to:

- `C:\Program Files\Git\cmd\git.exe`

so it can work even before a fresh terminal picks up the new PATH.
