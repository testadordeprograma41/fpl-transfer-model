# Project Instructions — FPL Transfer Model

These instructions apply to all Cowork sessions working in this folder.

## Before every `git push`

1. Update `CHANGELOG.md` in the project root with an entry for this session's work, including:
   - Date
   - Files changed (paths)
   - What was changed (short summary)
   - Why it was changed (the reasoning / goal behind it)
2. Do not push until the changelog entry has been written and saved.
3. Always ask for explicit confirmation before running `git push`. Do not push automatically, even if the changelog is up to date.

## General working style

- Prefer small, meaningful commits over large, mixed-purpose ones (matches the existing workflow: develop → test → status → diff → add → commit → push).
- Do not modify files outside this project folder.
- If a change touches data leakage prevention logic (e.g. feature construction in `src/modeling/`), flag it explicitly in the changelog entry — this is a known area of concern for this project.
- If unsure whether a change is safe to make automatically (e.g. deleting data, rewriting existing models, changing the evaluation methodology), pause and ask before proceeding.
