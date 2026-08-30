"""
Scheme data loader for SchemeSathi.

Loads scheme data straight from DuckDB (schemes.duckdb) if available, or falls
back to loading individual per-scheme JSON files saved as schemes/<slug>.json.

Provides fast database query filtering, BM25 full-text search, and formatted
context for vector embedding and RAG augmentation.
"""

import json
from pathlib import Path

from backend.config import SCHEMES_DIR, SCHEMES_DB_PATH


class SchemeLoader:
    """Loads, indexes, and provides access to scheme data via DuckDB or JSON fallback."""

    def __init__(self):
        self.all_schemes = []       # Flattened summary entries (search/embedding)
        self.slug_to_index = {}     # slug -> flattened summary entry
        self.detailed_schemes = {}  # slug -> raw scraped detail JSON

    def load_data(self):
        """Load scheme data from schemes.duckdb if available, else load JSON files."""
        db_path = Path(SCHEMES_DB_PATH)

        if db_path.exists():
            try:
                import duckdb
                conn = duckdb.connect(str(db_path), read_only=True)
                rows = conn.execute("""
                    SELECT slug, scheme_name, short_title, level, states, categories, tags, ministry, scheme_for, brief, raw_json
                    FROM schemes
                """).fetchall()
                conn.close()

                self.all_schemes = []
                self.slug_to_index = {}
                self.detailed_schemes = {}

                for slug, name, short_title, level, states_str, categories_str, tags_str, ministry, scheme_for, brief, raw_json_str in rows:
                    states = json.loads(states_str) if states_str else []
                    categories = json.loads(categories_str) if categories_str else []
                    tags = json.loads(tags_str) if tags_str else []
                    raw = json.loads(raw_json_str) if raw_json_str else {}

                    summary = {
                        'slug': slug,
                        'schemeName': name or '',
                        'schemeShortTitle': short_title or '',
                        'level': level or '',
                        'beneficiaryState': states,
                        'schemeCategory': categories,
                        'nodalMinistryName': ministry or '',
                        'tags': tags,
                        'schemeFor': scheme_for or '',
                        'briefDescription': brief or '',
                    }
                    self.all_schemes.append(summary)
                    self.slug_to_index[slug] = summary
                    self.detailed_schemes[slug] = raw

                print(f"Loaded {len(rows)} schemes instantly from DuckDB ({db_path.name}).")
                return
            except Exception as e:
                print(f"Notice: Failed loading from DuckDB ({e}). Falling back to JSON files...")

        schemes_dir = Path(SCHEMES_DIR)
        if not schemes_dir.exists():
            print(f"Error: schemes directory not found at {schemes_dir}")
            return

        loaded, skipped = 0, 0
        for filepath in sorted(schemes_dir.glob("*.json")):
            try:
                with open(filepath, 'r', encoding='utf-8') as f:
                    raw = json.load(f)
            except Exception as e:
                print(f"Skipping {filepath.name}: {e}")
                skipped += 1
                continue

            slug = raw.get('slug') or filepath.stem
            summary = self._flatten(raw, slug)
            if not summary.get('schemeName'):
                skipped += 1
                continue

            self.all_schemes.append(summary)
            self.slug_to_index[slug] = summary
            self.detailed_schemes[slug] = raw
            loaded += 1

        print(f"Loaded {loaded} schemes from JSON files in {schemes_dir} ({skipped} skipped).")

    @staticmethod
    def _label(value):
        """myScheme often nests fields as {'value': ..., 'label': ...}."""
        if isinstance(value, dict):
            return value.get('label', '') or value.get('value', '')
        return value or ''

    def _flatten(self, raw, slug):
        """Build a flat summary dict from a raw per-slug scraper file."""
        basic = raw.get('en', {}).get('basicDetails', {})
        content = raw.get('en', {}).get('schemeContent', {})

        state_label = self._label(basic.get('state'))
        beneficiary_state = [state_label] if state_label else []

        ministry = self._label(basic.get('nodalMinistryName')) or self._label(basic.get('nodalDepartmentName'))

        return {
            'slug': slug,
            'schemeName': basic.get('schemeName', ''),
            'schemeShortTitle': basic.get('schemeShortTitle', ''),
            'level': self._label(basic.get('level')),
            'beneficiaryState': beneficiary_state,
            'schemeCategory': [self._label(c) for c in (basic.get('schemeCategory') or []) if c],
            'nodalMinistryName': ministry,
            'tags': basic.get('tags') or [],
            'schemeFor': basic.get('schemeFor', ''),
            'briefDescription': content.get('briefDescription', ''),
        }

    def search_schemes(self, query=None, state=None, category=None, limit=20):
        """Execute fast search using DuckDB FTS / ILIKE if database exists,
        else perform in-memory search."""
        db_path = Path(SCHEMES_DB_PATH)
        if db_path.exists():
            try:
                import duckdb
                conn = duckdb.connect(str(db_path), read_only=True)

                conditions = []
                params = []

                use_fts = False
                if query and query.strip():
                    try:
                        conn.execute("LOAD fts;")
                        use_fts = True
                    except Exception:
                        use_fts = False

                sql = "SELECT slug"

                if use_fts and query.strip():
                    words = [w for w in query.strip().split() if w.isalnum()]
                    clean_q = " ".join(words)
                    if clean_q:
                        sql += ", fts_main_schemes.match_bm25(slug, ?) AS score FROM schemes"
                        params.append(clean_q)
                        conditions.append("score IS NOT NULL")
                    else:
                        sql += " FROM schemes"
                else:
                    sql += " FROM schemes"
                    if query and query.strip():
                        conditions.append("(scheme_name ILIKE ? OR short_title ILIKE ? OR brief ILIKE ? OR search_text ILIKE ?)")
                        q_param = f"%{query.strip()}%"
                        params.extend([q_param, q_param, q_param, q_param])

                if state:
                    conditions.append("(states ILIKE ? OR level = 'Central')")
                    params.append(f"%{state}%")

                if category:
                    conditions.append("categories ILIKE ?")
                    params.append(f"%{category}%")

                if conditions:
                    sql += " WHERE " + " AND ".join(conditions)

                if use_fts and query and query.strip() and 'score' in sql:
                    sql += " ORDER BY score DESC"
                else:
                    sql += " ORDER BY scheme_name ASC"

                sql += f" LIMIT {int(limit)}"

                rows = conn.execute(sql, params).fetchall()
                conn.close()

                results = []
                for r in rows:
                    slug = r[0]
                    summary = self.slug_to_index.get(slug)
                    if summary:
                        results.append(summary)
                return results
            except Exception as e:
                print(f"DuckDB search warning ({e}), falling back to memory...")

        # In-memory fallback
        results = []
        q_lower = query.lower() if query else None
        for s in self.all_schemes:
            if state and s.get('level') != 'Central':
                states = s.get('beneficiaryState', [])
                if states and not any(state.lower() in st.lower() for st in states):
                    continue
            if category:
                cats = s.get('schemeCategory', [])
                if not any(category.lower() in c.lower() for c in cats):
                    continue
            if q_lower:
                text = f"{s.get('schemeName')} {s.get('schemeShortTitle')} {s.get('briefDescription')} {' '.join(s.get('tags', []))}".lower()
                if q_lower not in text:
                    continue
            results.append(s)
            if len(results) >= limit:
                break
        return results

    def search_with_filters(self, filters: dict, limit: int = 8) -> list[dict]:
        """Search schemes using LLM-extracted structured filters via DuckDB SQL.

        Builds a targeted WHERE clause from filters (state, category, level) and
        optionally layers BM25 keyword scoring on top. Falls back to search_schemes()
        if DuckDB is unavailable or the query fails.

        Args:
            filters: Dict with optional keys: state, category, keywords, level.
            limit: Maximum number of results to return.

        Returns:
            List of scheme summary dicts (same shape as slug_to_index values).
        """
        db_path = Path(SCHEMES_DB_PATH)
        keywords = (filters.get('keywords') or '').strip()
        state = (filters.get('state') or '').strip()
        category = (filters.get('category') or '').strip()
        level = (filters.get('level') or '').strip()

        if db_path.exists():
            try:
                import duckdb
                conn = duckdb.connect(str(db_path), read_only=True)

                params = []
                conditions = []
                use_fts = False

                # Try BM25 full-text search for keywords (SELECT clause param goes first)
                if keywords:
                    try:
                        conn.execute("LOAD fts;")
                        words = [w for w in keywords.split() if w.isalnum()]
                        if words:
                            clean_q = " ".join(words)
                            params.append(clean_q)  # positional param for BM25
                            select_clause = "SELECT slug, fts_main_schemes.match_bm25(slug, ?) AS score FROM schemes"
                            conditions.append("score IS NOT NULL")
                            use_fts = True
                        else:
                            select_clause = "SELECT slug FROM schemes"
                    except Exception:
                        select_clause = "SELECT slug FROM schemes"
                else:
                    select_clause = "SELECT slug FROM schemes"

                # Structured filter conditions (params appended after BM25 param)
                # NOTE: only apply level as a hard filter when there is NO state filter.
                # When state IS present, the state condition already includes Central schemes
                # via "OR level = 'Central'", so adding "AND level = 'State'" would
                # incorrectly exclude all Central government weaver/farmer/etc. schemes.
                if level and not state:
                    conditions.append("level = ?")
                    params.append(level)

                if state:
                    conditions.append("(states ILIKE ? OR level = 'Central')")
                    params.append(f"%{state}%")

                if category:
                    cp = f"%{category}%"
                    conditions.append(
                        "(categories ILIKE ? OR tags ILIKE ? OR scheme_for ILIKE ? OR search_text ILIKE ?)"
                    )
                    params.extend([cp, cp, cp, cp])

                # ILIKE keyword fallback when BM25 is unavailable
                if not use_fts and keywords:
                    conditions.append("search_text ILIKE ?")
                    params.append(f"%{keywords}%")

                sql = select_clause
                if conditions:
                    sql += " WHERE " + " AND ".join(conditions)
                sql += " ORDER BY score DESC" if use_fts else " ORDER BY scheme_name ASC"
                sql += f" LIMIT {int(limit)}"

                rows = conn.execute(sql, params).fetchall()
                conn.close()

                results = []
                for r in rows:
                    slug = r[0]
                    summary = self.slug_to_index.get(slug)
                    if summary:
                        results.append(summary)

                # Post-filter: remove Central schemes whose names mention a specific
                # state/region that does NOT match the user's requested state.
                # This catches schemes like "...for Jammu & Kashmir and Ladakh" leaking
                # into Maharashtra results because they are stored as level=Central, states=[].
                if state and results:
                    results = self._filter_geo_restricted(results, state)

                print(f"[SchemeLoader] search_with_filters → {len(results)} results "
                      f"(state={state or '-'}, category={category or '-'}, "
                      f"keywords={keywords or '-'}, level={level or '-'})")
                return results
            except Exception as e:
                print(f"[SchemeLoader] search_with_filters DuckDB error ({e}), falling back to search_schemes()")

        # Graceful fallback to basic keyword search
        return self.search_schemes(
            query=keywords or None,
            state=state or None,
            category=category or None,
            limit=limit,
        )

    # Known Indian state/UT name fragments used for geo-restriction detection.
    # Ordered longest-first to avoid partial matches (e.g. "Goa" inside "Meghalaya" won't match).
    _GEO_NAMES = [
        'andhra pradesh', 'arunachal pradesh', 'himachal pradesh', 'madhya pradesh',
        'uttar pradesh', 'west bengal', 'tamil nadu', 'jammu', 'kashmir', 'ladakh',
        'andaman', 'nicobar', 'lakshadweep', 'puducherry', 'chandigarh',
        'chhattisgarh', 'jharkhand', 'uttarakhand', 'meghalaya', 'mizoram',
        'nagaland', 'manipur', 'tripura', 'sikkim', 'telangana', 'karnataka',
        'maharashtra', 'rajasthan', 'gujarat', 'haryana', 'punjab', 'kerala',
        'assam', 'bihar', 'odisha', 'goa', 'delhi',
    ]

    def _filter_geo_restricted(self, results: list[dict], user_state: str) -> list[dict]:
        """Remove Central schemes whose names mention a specific state other than user_state.

        Targets schemes stored as level=Central, states=[] that are actually
        region-specific (e.g. J&K scholarships appearing in Maharashtra results).
        Schemes with explicit state lists in the DB are already handled by SQL.
        """
        user_state_lower = user_state.lower()
        filtered = []
        for scheme in results:
            # Only apply to schemes with no explicit state list (empty states field)
            if scheme.get('beneficiaryState'):
                filtered.append(scheme)
                continue

            name_lower = scheme.get('schemeName', '').lower()
            # Check if any geo name appears in the scheme name
            found_other_state = False
            for geo in self._GEO_NAMES:
                if geo in name_lower and geo not in user_state_lower and user_state_lower not in geo:
                    print(f"[SchemeLoader] Geo-filtering '{scheme.get('slug')}': "
                          f"name mentions '{geo}' but user state is '{user_state}'")
                    found_other_state = True
                    break
            if not found_other_state:
                filtered.append(scheme)
        return filtered

    def get_scheme_summary_context(self, slug: str) -> str:
        """Return a compact scheme context for LLM injection.

        Produces ~300-500 characters per scheme (vs 2000-5000 for get_scheme_context),
        which is sufficient for the LLM to recommend schemes and ask follow-up questions
        without token budget explosion. Full detail is only needed when the user
        explicitly asks about a specific scheme.
        """
        scheme = self.slug_to_index.get(slug, {})
        if not scheme:
            return ""
        # _build_embedding_text already produces a compact, informative representation:
        # Scheme name, short title, level, states, categories, ministry, tags, for, brief (~300 chars)
        text = self._build_embedding_text(scheme)
        text += f"\nLink: https://www.myscheme.gov.in/schemes/{slug}"
        return text

    def find_scheme_slug_by_name(self, name: str) -> str | None:
        """Resolve a (possibly partial/fuzzy) scheme name to its slug.

        Uses DuckDB ILIKE search on the scheme_name column for the best match.
        Falls back to in-memory substring search if DuckDB is unavailable.

        Args:
            name: Scheme name string as returned by classify_detail_query().
                  May be partial (e.g. "Research Grant") rather than the full name.

        Returns:
            The matching slug string, or None if no match found.
        """
        if not name or not name.strip():
            return None

        db_path = Path(SCHEMES_DB_PATH)
        if db_path.exists():
            try:
                import duckdb
                conn = duckdb.connect(str(db_path), read_only=True)
                # Try progressively shorter word prefixes until we get a match
                words = name.strip().split()
                for n_words in range(len(words), 0, -1):
                    partial = " ".join(words[:n_words])
                    rows = conn.execute(
                        "SELECT slug FROM schemes WHERE scheme_name ILIKE ? "
                        "ORDER BY LENGTH(scheme_name) ASC LIMIT 1",
                        [f"%{partial}%"]
                    ).fetchall()
                    if rows:
                        conn.close()
                        slug = rows[0][0]
                        print(f"[SchemeLoader] Resolved name '{name}' → slug '{slug}'")
                        return slug
                conn.close()
            except Exception as e:
                print(f"[SchemeLoader] find_scheme_slug_by_name error ({e}), trying in-memory")

        # In-memory fallback: substring match on schemeName
        name_lower = name.strip().lower()
        best_slug = None
        best_len = float('inf')
        for slug, scheme in self.slug_to_index.items():
            scheme_name = scheme.get('schemeName', '').lower()
            if name_lower in scheme_name and len(scheme_name) < best_len:
                best_slug = slug
                best_len = len(scheme_name)
        if best_slug:
            print(f"[SchemeLoader] Resolved name '{name}' → slug '{best_slug}' (in-memory)")
        return best_slug

    def get_all_for_embedding(self):
        """Return list of (slug, text, raw_scheme) tuples for vector DB building."""
        results = []
        for scheme in self.all_schemes:
            slug = scheme.get('slug')
            if not slug:
                continue
            text = self._build_embedding_text(scheme)
            results.append((slug, text, scheme))
        return results

    def _build_embedding_text(self, scheme):
        """Build rich text representation for embedding."""
        parts = []
        parts.append(f"Scheme: {scheme.get('schemeName', '')}")
        parts.append(f"Short Title: {scheme.get('schemeShortTitle', '')}")
        parts.append(f"Level: {scheme.get('level', '')}")

        states = scheme.get('beneficiaryState', [])
        if states:
            parts.append(f"States: {', '.join(states)}")

        categories = scheme.get('schemeCategory', [])
        if categories:
            parts.append(f"Categories: {', '.join(categories)}")

        ministry = scheme.get('nodalMinistryName', '')
        if ministry:
            parts.append(f"Ministry: {ministry}")

        tags = scheme.get('tags', [])
        if tags:
            parts.append(f"Tags: {', '.join(tags)}")

        scheme_for = scheme.get('schemeFor', '')
        if scheme_for and scheme_for != '#N/A':
            parts.append(f"For: {scheme_for}")

        brief = scheme.get('briefDescription', '')
        if brief:
            parts.append(f"Description: {brief}")

        return '\n'.join(parts)

    def get_scheme_context(self, slug):
        """Get formatted context for RAG prompt augmentation."""
        scheme = self.slug_to_index.get(slug, {})
        context = self._build_embedding_text(scheme)

        detailed = self.detailed_schemes.get(slug)
        if detailed:
            en = detailed.get('en', {})
            content = en.get('schemeContent', {})

            benefits = content.get('benefits_md', '')
            if benefits and benefits.strip():
                context += f"\n\nBenefits:\n{benefits}"

            elig = en.get('eligibilityCriteria', {})
            if isinstance(elig, dict):
                elig_md = elig.get('eligibilityDescription_md', '')
                if elig_md and elig_md.strip():
                    context += f"\n\nEligibility:\n{elig_md}"

            desc = content.get('detailedDescription_md', '')
            if desc and desc.strip():
                context += f"\n\nDetailed Description:\n{desc}"

            app_proc = en.get('applicationProcess', [])
            if app_proc:
                context += "\n\nApplication Process:"
                for proc in app_proc:
                    if not proc or not isinstance(proc, dict):
                        continue
                    mode = proc.get('mode', '')
                    url = proc.get('url', '')
                    proc_md = proc.get('process_md', '')
                    context += f"\n  Mode: {mode}"
                    if url:
                        context += f" | URL: {url}"
                    if proc_md:
                        context += f"\n  {proc_md}"

        context += f"\n\nOfficial Link: https://www.myscheme.gov.in/schemes/{slug}"
        return context

    def get_scheme_detail(self, slug):
        """Get full, comprehensive scheme detail for the /api/schemes/{slug} endpoint."""
        index_data = self.slug_to_index.get(slug, {})
        if not index_data:
            return None

        detailed = self.detailed_schemes.get(slug)

        name = index_data.get('schemeName', '')
        brief = index_data.get('briefDescription', '')
        level = index_data.get('level', '')
        states = index_data.get('beneficiaryState', [])
        categories = index_data.get('schemeCategory', [])
        tags = index_data.get('tags', [])
        ministry = index_data.get('nodalMinistryName', '')
        scheme_for = index_data.get('schemeFor', '')
        official_url = f"https://www.myscheme.gov.in/schemes/{slug}"

        result = {
            'slug': slug,
            'name': name,
            'brief': brief,
            'level': level,
            'states': states,
            'categories': categories,
            'tags': tags,
            'ministry': ministry,
            'scheme_for': scheme_for,
            'url': official_url,
            'has_details': True,
        }

        if detailed:
            en = detailed.get('en', {})
            content = en.get('schemeContent', {})

            result['benefits'] = content.get('benefits_md', '') or brief
            result['detailed_description'] = content.get('detailedDescription_md', '') or brief
            result['references'] = content.get('references', [])

            elig = en.get('eligibilityCriteria', {})
            if isinstance(elig, dict):
                result['eligibility'] = elig.get('eligibilityDescription_md', '')
            else:
                result['eligibility'] = str(elig) if elig else ''

            result['application_process'] = en.get('applicationProcess', []) or []
            result['documents'] = detailed.get('documents') or []
            result['faqs'] = detailed.get('faqs') or []

            defs = en.get('schemeDefinitions', [])
            if defs:
                result['definitions'] = [
                    {'name': d.get('name', ''), 'definition': d.get('definitions_md', '')}
                    for d in defs if isinstance(d, dict)
                ]
        else:
            result['detailed_description'] = brief
            result['benefits'] = f"Provides official government support, financial assistance, or services for {name}."

            elig_parts = []
            if scheme_for and scheme_for != '#N/A':
                elig_parts.append(f"**Target Beneficiaries:** {scheme_for}")
            if states:
                elig_parts.append(f"**Eligible States/UTs:** {', '.join(states)}")
            if categories:
                elig_parts.append(f"**Categories:** {', '.join(categories)}")
            if level:
                elig_parts.append(f"**Scheme Level:** {level}")

            result['eligibility'] = "\n\n".join(elig_parts) if elig_parts else "Open to eligible citizens meeting government criteria."

            result['application_process'] = [
                {
                    'mode': 'Online Application Portal',
                    'url': official_url,
                    'process_md': f'Visit the official myScheme portal at {official_url} to check detailed guidelines and submit your application online.'
                }
            ]
            result['documents'] = [
                "Identity Proof (Aadhaar Card / Voter ID / Passport)",
                "Proof of Residence / State Domicile Certificate",
                "Income Certificate / Caste Certificate (if applicable)",
                "Academic / Course Enrollment documents (for student schemes)"
            ]
            result['faqs'] = [
                {
                    'question': 'How can I apply for this scheme?',
                    'answer': 'Click on the "View Official Page on myScheme.gov.in" button below to access the direct application portal and submit your application online.'
                }
            ]
            result['definitions'] = []

        return result


scheme_loader = SchemeLoader()
