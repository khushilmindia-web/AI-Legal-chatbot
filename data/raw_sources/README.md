# Raw Legal Source Files

Place unprocessed `.txt` files here before running the corpus importer.

These raw files are referenced by `data/manifests/corpus_manifest.json`.

Examples:
- `Constitution_of_India_raw.txt`
- `Information_Technology_Act_2000_raw.txt`
- `Gujarat_State_Rule_raw.txt`
- `Sample_Supreme_Court_Case_raw.txt`

Suggested workflow:
1. Download or prepare source text from official legal sources.
2. Save the raw text files here.
3. Add or update entries in `data/manifests/corpus_manifest.json`.
4. Run:
   - `python -m backend.corpus_importer`
5. Then run:
   - `http://127.0.0.1:5000/debug/ingest`
