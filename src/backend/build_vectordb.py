"""
One-time script to build the ChromaDB vector database.
Embeds all ~4,720 schemes using the bge-m3 embedding API
and stores them in a persistent ChromaDB collection.
"""

import os
import sys
import time

from backend.scheme_loader import scheme_loader
from backend.vector_store import vector_store
from backend.embedding_service import embedding_service
from backend.config import CHROMA_DB_PATH


def main():
    print("=" * 55)
    print("  SchemeSathi — Vector Database Builder")
    print("=" * 55)

    # 1. Load scheme data
    print("\n[1/4] Loading scheme data...")
    scheme_loader.load_data()
    print(f"  Index: {len(scheme_loader.all_schemes)} schemes")
    print(f"  Detailed: {len(scheme_loader.detailed_schemes)} schemes")

    # 2. Initialize vector store
    print("\n[2/4] Initializing ChromaDB...")
    vector_store.init()

    # 3. Prepare data
    print("\n[3/4] Preparing scheme texts for embedding...")
    scheme_data = scheme_loader.get_all_for_embedding()
    total_loaded = len(scheme_data)
    print(f"  Total schemes loaded from disk: {total_loaded}")

    # Check for existing data
    existing = vector_store.collection.count()
    incremental = False

    if existing > 0:
        print(f"\n  Vector DB already has {existing} entries.")
        print("  1. Update changes (Incremental: add new, update modified, delete removed) [Default]")
        print("  2. Rebuild completely (Slow: clears collection and embeds everything)")
        print("  3. Keep existing and exit")
        ans = input("  Select option (1/2/3): ").strip()
        if ans == '2':
            print("  Clearing existing collection...")
            vector_store.clear()
            incremental = False
        elif ans == '3':
            print("  Keeping existing database. Done.")
            return
        else:
            print("  Performing incremental update...")
            incremental = True

    # 4. Process updates/adds/deletes
    import hashlib
    process_items = []
    failed = 0

    if incremental:
        print("\nChecking for database changes...")
        # Get existing collection entries (retrieve all IDs and metadatas)
        # Note: ChromaDB get() without a limit retrieves all items.
        existing_data = vector_store.collection.get(include=['metadatas'])
        existing_ids = existing_data.get('ids', []) or []
        existing_metadatas = existing_data.get('metadatas', []) or []
        existing_map = {
            id_: meta for id_, meta in zip(existing_ids, existing_metadatas)
            if meta is not None
        }

        to_add = []
        to_update = []
        loaded_slugs = set()

        for slug, text, scheme_raw in scheme_data:
            loaded_slugs.add(slug)
            text_hash = hashlib.md5(text.encode('utf-8')).hexdigest()

            if slug not in existing_map:
                to_add.append((slug, text, scheme_raw, text_hash))
            else:
                stored_hash = existing_map[slug].get('text_hash')
                if stored_hash != text_hash:
                    to_update.append((slug, text, scheme_raw, text_hash))

        to_delete = [slug for slug in existing_map if slug not in loaded_slugs]

        print(f"  To add: {len(to_add)}")
        print(f"  To update: {len(to_update)}")
        print(f"  To delete: {len(to_delete)}")

        if not to_add and not to_update and not to_delete:
            print("\nNo changes detected. Database is up to date.")
            return

        if to_delete:
            print(f"  Deleting {len(to_delete)} obsolete schemes...")
            vector_store.delete_schemes(to_delete)

        # Combine items to process
        process_items = [((slug, text, scheme_raw, text_hash), 'add') for slug, text, scheme_raw, text_hash in to_add] + \
                        [((slug, text, scheme_raw, text_hash), 'update') for slug, text, scheme_raw, text_hash in to_update]
    else:
        # Full rebuild
        for slug, text, scheme_raw in scheme_data:
            text_hash = hashlib.md5(text.encode('utf-8')).hexdigest()
            process_items.append(((slug, text, scheme_raw, text_hash), 'add'))

    total_to_process = len(process_items)
    if total_to_process > 0:
        print(f"\n[4/4] Processing {total_to_process} schemes (embedding & storing)...")
        print(f"  Embedding API: bge-m3")
        print(f"  Target: {CHROMA_DB_PATH}\n")

        add_ids, add_embeddings, add_documents, add_metadatas = [], [], [], []
        up_ids, up_embeddings, up_documents, up_metadatas = [], [], [], []
        batch_size = 50
        start_time = time.time()

        for i, ((slug, text, scheme_raw, text_hash), action) in enumerate(process_items):
            try:
                emb = embedding_service.embed_text_sync(text)
                meta = {
                    'name': scheme_raw.get('schemeName', ''),
                    'level': scheme_raw.get('level', ''),
                    'states': ', '.join(scheme_raw.get('beneficiaryState', [])),
                    'categories': ', '.join(scheme_raw.get('schemeCategory', [])),
                    'ministry': scheme_raw.get('nodalMinistryName', '') or '',
                    'tags': ', '.join(scheme_raw.get('tags', [])),
                    'has_details': str(slug in scheme_loader.detailed_schemes),
                    'text_hash': text_hash,
                }

                if action == 'add':
                    add_ids.append(slug)
                    add_embeddings.append(emb)
                    add_documents.append(text)
                    add_metadatas.append(meta)
                else:
                    up_ids.append(slug)
                    up_embeddings.append(emb)
                    up_documents.append(text)
                    up_metadatas.append(meta)

                # Store in batches
                if len(add_ids) >= batch_size:
                    vector_store.add_schemes(add_ids, add_embeddings, add_documents, add_metadatas)
                    add_ids, add_embeddings, add_documents, add_metadatas = [], [], [], []
                if len(up_ids) >= batch_size:
                    vector_store.update_schemes(up_ids, up_embeddings, up_documents, up_metadatas)
                    up_ids, up_embeddings, up_documents, up_metadatas = [], [], [], []

                # Progress display
                if (i + 1) % 10 == 0 or (i + 1) == total_to_process:
                    pct = (i + 1) / total_to_process * 100
                    elapsed = time.time() - start_time
                    rate = (i + 1) / elapsed if elapsed > 0 else 0
                    eta = (total_to_process - i - 1) / rate if rate > 0 else 0
                    name = scheme_raw.get('schemeName', '')[:45]
                    print(f"  [{i+1}/{total_to_process}] {pct:5.1f}%  ETA: {eta/60:.0f}m  ({action}) {name}")

                time.sleep(0.05)  # Rate limiting

            except Exception as e:
                failed += 1
                print(f"  ERROR [{slug}]: {e}")
                continue

        # Store remaining batches
        if add_ids:
            vector_store.add_schemes(add_ids, add_embeddings, add_documents, add_metadatas)
        if up_ids:
            vector_store.update_schemes(up_ids, up_embeddings, up_documents, up_metadatas)

    # Summary
    final_count = vector_store.collection.count()
    print(f"\n{'=' * 55}")
    print(f"  DONE!")
    print(f"  Vectors stored: {final_count}")
    print(f"  Failed embeddings: {failed}")
    print(f"  Database: {CHROMA_DB_PATH}")
    print(f"{'=' * 55}")


if __name__ == '__main__':
    main()

