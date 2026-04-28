# Download Required Files Here

Download these two files from the shared Google Drive link and place them in this folder (`data/processed/`):

| File | Size | Required for |
|---|---|---|
| `modeling_table.parquet` | 13 MB | Dashboard — station/route metadata, target labels |
| `X_features.parquet` | 24 MB | Dashboard — all 38 model input features |

**These files are already in this folder (from GitHub, no download needed):**
- `train_val_test_indices.json`
- `thresholds.json`
- `prediction_sets.parquet`

**Optional (only needed to re-run notebooks 01–04):**
- `transit_events.parquet` (79 MB) — raw synthetic events
- `transit_events_cleaned.parquet` (78 MB) — cleaned events
