"""
Build DuckDB database from raw scheme JSON files for fast search and metadata retrieval.

Reads all per-scheme JSON files from SCHEMES_DIR, extracts flattened search metadata,
stores the raw JSON payload alongside indexed fields, and constructs a Full-Text Search (FTS) index.
"""

import json
import time
import hashlib
import argparse
from pathlib import Path
import duckdb

from backend.config import SCHEMES_DIR, SCHEMES_DB_PATH


def _label(value):
    """myScheme often nests fields as {'value': ..., 'label': ...}."""
    if isinstance(value, dict):
        return value.get('label', '') or value.get('value', '')
    return value or ''


def _flatten(raw, slug):
    """Build a flat summary dict from raw per-slug scraper file."""
    basic = raw.get('en', {}).get('basicDetails', {})
    content = raw.get('en', {}).get('schemeContent', {})

    state_label = _label(basic.get('state'))
    beneficiary_state = [state_label] if state_label else []

    ministry = _label(basic.get('nodalMinistryName')) or _label(basic.get('nodalDepartmentName'))
    categories = [_label(c) for c in (basic.get('schemeCategory') or []) if c]
    tags = basic.get('tags') or []

    scheme_name = basic.get('schemeName', '') or ''
    short_title = basic.get('schemeShortTitle', '') or ''
    level = _label(basic.get('level')) or ''
    scheme_for = basic.get('schemeFor', '') or ''
    brief = content.get('briefDescription', '') or ''

    # Build rich text for FTS search
    search_parts = [
        f"Scheme: {scheme_name}",
        f"Short Title: {short_title}",
        f"Level: {level}",
        f"States: {', '.join(beneficiary_state)}",
        f"Categories: {', '.join(categories)}",
        f"Ministry: {ministry}",
        f"Tags: {', '.join(tags)}",
        f"For: {scheme_for}",
        f"Description: {brief}"
    ]
    search_text = "\n".join(search_parts)

    return {
        'slug': slug,
        'scheme_name': scheme_name,
        'short_title': short_title,
        'level': level,
        'states': json.dumps(beneficiary_state),
        'categories': json.dumps(categories),
        'tags': json.dumps(tags),
        'ministry': ministry,
        'scheme_for': scheme_for,
        'brief': brief,
        'search_text': search_text,
        'raw_json': json.dumps(raw)
    }


def build_db(schemes_dir=None, db_path=None, force=False):
    """
    Scan schemes_dir and incrementally update the DuckDB database.

    - New JSON files are inserted.
    - Modified JSON files (detected via SHA-256 hash) are upserted.
    - Slugs whose source file was removed are deleted.
    - Unchanged files are skipped entirely.
    - The FTS index is rebuilt only when at least one record changed.

    Pass force=True (or --force CLI flag) to delete the existing DB and do a full rebuild.
    """
    start_time = time.time()
    schemes_dir = Path(schemes_dir or SCHEMES_DIR)
    db_path = Path(db_path or SCHEMES_DB_PATH)

    if not schemes_dir.exists():
        print(f"Error: schemes directory not found at {schemes_dir}")
        return False

    # --force: delete existing DB so we start completely fresh
    if force and db_path.exists():
        try:
            db_path.unlink()
            print("--force: removed existing DB for full rebuild.")
        except Exception as e:
            print(f"Warning: could not remove existing DB file: {e}")

    conn = duckdb.connect(str(db_path))

    # Ensure tables exist (no-op if already present)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS schemes (
            slug VARCHAR PRIMARY KEY,
            scheme_name VARCHAR,
            short_title VARCHAR,
            level VARCHAR,
            states VARCHAR,
            categories VARCHAR,
            tags VARCHAR,
            ministry VARCHAR,
            scheme_for VARCHAR,
            brief VARCHAR,
            search_text VARCHAR,
            raw_json VARCHAR
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS _build_meta (
            slug VARCHAR PRIMARY KEY,
            hash VARCHAR NOT NULL
        )
    """)

    # Load hashes already stored in the DB (empty dict on first run)
    db_slugs = {
        row[0]: row[1]
        for row in conn.execute("SELECT slug, hash FROM _build_meta").fetchall()
    }

    json_files = {f.stem: f for f in schemes_dir.glob("*.json")}
    processed_slugs = set()

    scheme_rows = []   # tuples for INSERT OR REPLACE INTO schemes
    meta_rows = []     # (slug, hash) for INSERT OR REPLACE INTO _build_meta
    inserted = updated = skipped = parse_errors = 0

    for slug, filepath in sorted(json_files.items()):
        # Compute hash with a single binary read
        try:
            raw_bytes = filepath.read_bytes()
            file_hash = hashlib.sha256(raw_bytes).hexdigest()
        except Exception as e:
            print(f"Skipping {filepath.name} (read error): {e}")
            parse_errors += 1
            continue

        processed_slugs.add(slug)

        # Skip if content hasn't changed
        if db_slugs.get(slug) == file_hash:
            skipped += 1
            continue

        try:
            raw = json.loads(raw_bytes.decode('utf-8'))
        except Exception as e:
            print(f"Skipping {filepath.name} (JSON parse error): {e}")
            parse_errors += 1
            continue

        record = _flatten(raw, slug)
        if not record['scheme_name']:
            parse_errors += 1
            continue

        scheme_rows.append(tuple(record.values()))
        meta_rows.append((slug, file_hash))

        if slug in db_slugs:
            updated += 1
        else:
            inserted += 1

    # Delete slugs whose source JSON file was removed
    stale = set(db_slugs) - processed_slugs
    deleted = len(stale)
    if stale:
        placeholders = ','.join('?' * len(stale))
        stale_list = list(stale)
        conn.execute(f"DELETE FROM schemes WHERE slug IN ({placeholders})", stale_list)
        conn.execute(f"DELETE FROM _build_meta WHERE slug IN ({placeholders})", stale_list)

    # Bulk upsert changed/new records
    if scheme_rows:
        conn.executemany(
            "INSERT OR REPLACE INTO schemes VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
            scheme_rows
        )
        conn.executemany(
            "INSERT OR REPLACE INTO _build_meta (slug, hash) VALUES (?,?)",
            meta_rows
        )

    total_changed = inserted + updated + deleted

    # Rebuild FTS index only when something actually changed
    if total_changed > 0:
        print("Rebuilding Full-Text Search (FTS) index...")
        try:
            # Drop existing index (best-effort; may not exist)
            try:
                conn.execute("PRAGMA drop_fts_index('schemes')")
            except Exception:
                pass
            conn.execute("INSTALL fts")
            conn.execute("LOAD fts")
            conn.execute(
                "PRAGMA create_fts_index('schemes', 'slug', "
                "'scheme_name', 'short_title', 'brief', 'search_text');"
            )
            print("FTS index rebuilt successfully.")
        except Exception as e:
            print(f"Warning: FTS index creation failed (search will fall back to ILIKE): {e}")
    else:
        print("No changes detected — skipping FTS rebuild.")

    count = conn.execute("SELECT COUNT(*) FROM schemes").fetchone()[0]
    conn.close()

    elapsed = time.time() - start_time
    print(
        f"Done in {elapsed:.2f}s | Total: {count} schemes | "
        f"Inserted: {inserted} | Updated: {updated} | Deleted: {deleted} | "
        f"Skipped (unchanged): {skipped} | Errors: {parse_errors}"
    )
    return True


def main():
    parser = argparse.ArgumentParser(description="Build/update the schemes DuckDB database.")
    parser.add_argument(
        "--force", action="store_true",
        help="Delete existing DB and perform a full rebuild from scratch."
    )
    args = parser.parse_args()
    build_db(force=args.force)


if __name__ == '__main__':
    main()
