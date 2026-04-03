# Legal Corpus Ingestion Guide

## Required folders
- `data/Docs`
- `data/StateLaws`
- `data/CaseLaw`

## Steps
1. Add raw legal source `.txt` files into `data/raw_sources`.
2. Update `data/manifests/corpus_manifest.json` with the real source metadata.
3. Run the corpus importer:
   - `python -m backend.corpus_importer`
4. Confirm normalized files were created in:
   - `data/Docs`
   - `data/StateLaws`
   - `data/CaseLaw`
5. Restart the backend if it is not running.
6. Run the ingestion endpoint:
   - `http://127.0.0.1:5000/debug/ingest`
7. Verify vector stores:
   - `http://127.0.0.1:5000/debug/status`

## Current blocker
This workspace currently does not contain real statute, state-rule, or case-law source files, so a meaningful ingestion run cannot be completed yet.

## Minimum recommended first corpus
- Constitution of India
- Information Technology Act, 2000
- Bharatiya Nyaya Sanhita, 2023 relevant sections
- Bharatiya Nagarik Suraksha Sanhita, 2023 procedural extracts
- Consumer Protection Act, 2019
- Transfer of Property Act, 1882
- Hindu Marriage Act, 1955
- State-specific land / rent / labour rule extracts
- A small set of leading Supreme Court and High Court cases
