"""
Build DuckDB database from raw scheme JSON files for fast search and metadata retrieval.

Reads all per-scheme JSON files from SCHEMES_DIR, extracts flattened search metadata,
stores the raw JSON payload alongside indexed fields, and constructs a Full-Text Search (FTS) index.
"""

import json
import time
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


def build_db(schemes_dir=None, db_path=None):
    """Scan schemes_dir, build DuckDB database table and FTS index."""
    start_time = time.time()
    schemes_dir = Path(schemes_dir or SCHEMES_DIR)
    db_path = Path(db_path or SCHEMES_DB_PATH)

    if not schemes_dir.exists():
        print(f"Error: schemes directory not found at {schemes_dir}")
        return False

    # Remove existing DB file to rebuild cleanly
    if db_path.exists():
        try:
            db_path.unlink()
        except Exception as e:
            print(f"Warning: could not remove existing DB file: {e}")

    conn = duckdb.connect(str(db_path))

    # Create schemes table
    conn.execute("""
        CREATE TABLE schemes (
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

    json_files = sorted(list(schemes_dir.glob("*.json")))
    print(f"Found {len(json_files)} scheme JSON files in {schemes_dir}. Ingesting into DuckDB...")

    records = []
    skipped = 0
    for filepath in json_files:
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                raw = json.load(f)
            slug = raw.get('slug') or filepath.stem
            record = _flatten(raw, slug)
            if not record['scheme_name']:
                skipped += 1
                continue
            records.append((
                record['slug'],
                record['scheme_name'],
                record['short_title'],
                record['level'],
                record['states'],
                record['categories'],
                record['tags'],
                record['ministry'],
                record['scheme_for'],
                record['brief'],
                record['search_text'],
                record['raw_json']
            ))
        except Exception as e:
            skipped += 1
            print(f"Skipping {filepath.name}: {e}")

    # Bulk insert records
    conn.executemany("""
        INSERT INTO schemes VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, records)

    count = conn.execute("SELECT COUNT(*) FROM schemes").fetchone()[0]

    # Initialize FTS Index
    print("Building Full-Text Search (FTS) index...")
    try:
        conn.execute("INSTALL fts;")
        conn.execute("LOAD fts;")
        conn.execute("PRAGMA create_fts_index('schemes', 'slug', 'scheme_name', 'short_title', 'brief', 'search_text');")
        print("FTS index built successfully.")
    except Exception as e:
        print(f"Warning: FTS index creation failed (search will fall back to ILIKE): {e}")

    conn.close()

    elapsed = time.time() - start_time
    print(f"DuckDB database created successfully at {db_path} in {elapsed:.2f}s!")
    print(f"Total schemes indexed: {count} ({skipped} skipped).")
    return True


def main():
    build_db()


if __name__ == '__main__':
    main()
