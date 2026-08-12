"""
One-time script to build the ChromaDB vector database.
Embeds all ~4,720 schemes using the bge-m3 embedding API
and stores them in a persistent ChromaDB collection.

Usage:
    cd d:\vit\ty-sem1\edi\scraper\backend
    python build_vectordb.py
"""

import os
import sys
import time

# Ensure imports work when running from backend/ directory
sys.path.insert(0, os.path.dirname(__file__))

from scheme_loader import scheme_loader
from vector_store import vector_store
from embedding_service import embedding_service
from config import CHROMA_DB_PATH


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

    # Check for existing data
    existing = vector_store.collection.count()
    if existing > 0:
        ans = input(f"\n  Vector DB already has {existing} entries. Rebuild? (y/N): ")
        if ans.lower() != 'y':
            print("  Keeping existing database. Done.")
            return
        print("  Clearing existing collection...")
        vector_store.clear()

    # 3. Prepare data
    print("\n[3/4] Preparing scheme texts for embedding...")
    scheme_data = scheme_loader.get_all_for_embedding()
    total = len(scheme_data)
    print(f"  Total schemes to embed: {total}")

    # 4. Embed and store
    print(f"\n[4/4] Embedding schemes (this will take ~15-30 minutes)...")
    print(f"  Embedding API: bge-m3")
    print(f"  Target: {CHROMA_DB_PATH}\n")

    ids = []
    embeddings = []
    documents = []
    metadatas = []
    batch_size = 50
    failed = 0
    start_time = time.time()

    for i, (slug, text, scheme_raw) in enumerate(scheme_data):
        try:
            emb = embedding_service.embed_text_sync(text)

            ids.append(slug)
            embeddings.append(emb)
            documents.append(text)
            metadatas.append({
                'name': scheme_raw.get('schemeName', ''),
                'level': scheme_raw.get('level', ''),
                'states': ', '.join(scheme_raw.get('beneficiaryState', [])),
                'categories': ', '.join(scheme_raw.get('schemeCategory', [])),
                'ministry': scheme_raw.get('nodalMinistryName', '') or '',
                'tags': ', '.join(scheme_raw.get('tags', [])),
                'has_details': str(slug in scheme_loader.detailed_schemes),
            })

            # Store in batches
            if len(ids) >= batch_size:
                vector_store.add_schemes(ids, embeddings, documents, metadatas)
                ids, embeddings, documents, metadatas = [], [], [], []

            # Progress every 10 schemes
            if (i + 1) % 10 == 0 or (i + 1) == total:
                pct = (i + 1) / total * 100
                elapsed = time.time() - start_time
                rate = (i + 1) / elapsed if elapsed > 0 else 0
                eta = (total - i - 1) / rate if rate > 0 else 0
                name = scheme_raw.get('schemeName', '')[:45]
                print(f"  [{i+1}/{total}] {pct:5.1f}%  ETA: {eta/60:.0f}m  {name}")

            time.sleep(0.05)  # Rate limiting

        except Exception as e:
            failed += 1
            print(f"  ERROR [{slug}]: {e}")
            continue

    # Store remaining batch
    if ids:
        vector_store.add_schemes(ids, embeddings, documents, metadatas)

    # Summary
    elapsed = time.time() - start_time
    final_count = vector_store.collection.count()
    print(f"\n{'=' * 55}")
    print(f"  DONE!")
    print(f"  Vectors stored: {final_count}")
    print(f"  Failed: {failed}")
    print(f"  Time: {elapsed/60:.1f} minutes")
    print(f"  Database: {CHROMA_DB_PATH}")
    print(f"{'=' * 55}")


if __name__ == '__main__':
    main()
