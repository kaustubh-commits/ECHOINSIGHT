# Legacy files written before PR #2

These JSON files were written by the code on `feature/stem-separation` at
commit `86c84ec`, before the honest-analysis cleanup (PR #2). They are real
outputs of the old code, not hand-edited, and `tests/test_legacy_files.py`
checks that the current code still loads them.

The source was a 24-second synthetic clip (clicks plus a tone, then clicks
plus noise). It was analysed by `extract_full_pipeline`, enriched by the old
`SpotifyEnricher` stub, and saved by the old `PackageStore`.

| File | Contents |
|---|---|
| `song_dna.json` | Bare SongDNA, schema 2.1.0: tempo confidence 1.0, boundary confidence 0.7 |
| `package_offline_stub.json` | Package with the stub's filename-search record (`track_id="offline_id"`, artist "Unknown Artist", popularity 50) |
| `package_mock_stub.json` | Package with the stub's `mock_` ID record ("Beyond / Daft Punk", `images.example.com` art) |
| `package_realid_stub.json` | Package where the stub echoed a real-looking ID with the same fake "Beyond" data |

`metadata.filename` holds the temporary path the clip was written to. No
audio is needed to load these files.
