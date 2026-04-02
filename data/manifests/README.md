# Corpus Manifest

`corpus_manifest.json` controls how raw legal source files are normalized into the project corpus folders.

Supported `target_type` values:
- `statute`
- `state_rule`
- `case_law`

Each entry should provide:
- `title`
- `target_type`
- `source_file`

Optional but recommended:
- `output_name`
- `source_url`
- `jurisdiction`
- `state`
- `citation`
- `court`
- `year`
- `tags`
