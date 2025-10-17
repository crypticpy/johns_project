# Backup and Restore Operations

This document describes concrete, production-safe procedures for backing up and restoring the development SQLite database and FAISS vector indexes. Commands are designed for offline determinism and reproducibility. Adjust paths for your environment if needed.

## Prerequisites

- Project virtual environment (.venv) created and activated for any Python operations.
- Ensure application is not running during backup to avoid file locks.

## Directory Layout

- SQLite database: `./data/app.db`
- FAISS indexes: `./data/faiss/*.index`
- FAISS metadata: `./data/faiss/*.meta.json`
- Backups directory: `./backups/`

Create backups directory if missing:
```bash
mkdir -p ./backups
```

## SQLite Backup (Development)

SQLite backups can be performed with a filesystem copy. Prefer copying while the app is stopped.

- Snapshot backup with timestamp:
```bash
cp ./data/app.db ./backups/app_$(date +"%Y%m%d_%H%M%S").db
```

- Verify backup integrity (optional):
```bash
sqlite3 ./backups/app_$(ls -t ./backups | grep -E '^app_.*\.db$' | head -n1) "PRAGMA integrity_check;"
```

- Quick backup (named):
```bash
cp ./data/app.db ./backups/app_latest.db
```

### Restore SQLite

Stop the application, then restore the selected snapshot:

```bash
# Restore latest snapshot (example)
cp ./backups/app_latest.db ./data/app.db
```

Alternatively, restore a specific dated snapshot:
```bash
cp ./backups/app_20250101_120000.db ./data/app.db
```

After restore, if the application manages migrations separately, run them as appropriate. For this repository, dev tables are created idempotently on startup.

## FAISS Vector Index Backup

FAISS index data and metadata are stored under `./data/faiss/`. Back up both `.index` and `.meta.json` files together to ensure consistency.

- Create backup of FAISS indexes:
```bash
mkdir -p ./backups/faiss
cp ./data/faiss/*.index ./backups/faiss/ 2>/dev/null || true
cp ./data/faiss/*.meta.json ./backups/faiss/ 2>/dev/null || true
```

If there are multiple index shards or versions, preserve directory structure:
```bash
rsync -av --exclude=".DS_Store" ./data/faiss/ ./backups/faiss/
```

### Restore FAISS Indexes

Restore the indexes back to the data directory:

```bash
mkdir -p ./data/faiss
cp ./backups/faiss/*.index ./data/faiss/ 2>/dev/null || true
cp ./backups/faiss/*.meta.json ./data/faiss/ 2>/dev/null || true
# Or:
rsync -av --exclude=".DS_Store" ./backups/faiss/ ./data/faiss/
```

Ensure the application is restarted so that any in-memory index caches are rebuilt from the restored files.

## Retention Policies

Adopt a rotation policy suitable for your environment:

- Development:
  - Keep last 5 daily SQLite snapshots and last 3 FAISS snapshots.
  - Remove older files to conserve space.

Example cleanup (dangerous—review before running):
```bash
# Keep last 5 SQLite backups matching app_*.db
ls -t ./backups/app_*.db | tail -n +6 | xargs -I{} rm -f {}

# Keep last 3 FAISS backups if dated directories are used (e.g., faiss_YYYYMMDD/)
# Adjust globbing patterns as needed
```

## Sample Scripts

Create helper scripts under `./scripts/` to standardize operations.

### scripts/backup_dev.sh
```bash
#!/usr/bin/env bash
set -euo pipefail

mkdir -p ./backups ./backups/faiss

# SQLite snapshot with timestamp
STAMP="$(date +"%Y%m%d_%H%M%S")"
cp ./data/app.db "./backups/app_${STAMP}.db"

# FAISS indexes
rsync -av --exclude=".DS_Store" ./data/faiss/ ./backups/faiss/

echo "Backup completed: SQLite -> backups/app_${STAMP}.db; FAISS -> backups/faiss/"
```

### scripts/restore_dev.sh
```bash
#!/usr/bin/env bash
set -euo pipefail

if [ $# -lt 1 ]; then
  echo "Usage: $0 <sqlite_backup_file>"
  echo "Example: $0 ./backups/app_20250101_120000.db"
  exit 1
fi

SQLITE_SRC="$1"

# Restore SQLite
cp "$SQLITE_SRC" ./data/app.db

# Restore FAISS (latest backup directory)
rsync -av --exclude=".DS_Store" ./backups/faiss/ ./data/faiss/

echo "Restore completed: SQLite <- $SQLITE_SRC; FAISS <- backups/faiss/"
```

Make scripts executable:
```bash
chmod +x ./scripts/backup_dev.sh ./scripts/restore_dev.sh
```

## Notes

- Avoid storing PII in backups. This repo’s metrics and audit metadata should be PII-free by design.
- For production systems, use database-native backup tooling and consistent snapshotting across DB and vector storage.
- Always verify integrity and run smoke tests post-restore (e.g., API `/health`, list datasets, perform a sample search).
