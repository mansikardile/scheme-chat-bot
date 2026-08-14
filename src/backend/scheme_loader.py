"""
Scheme data loader for SchemeSathi.

Loads scheme data straight from the scraper's output: one JSON file per
scheme, saved as schemes/<slug>.json (see scraper/scrape_all.py ->
save_all_schemes()). Each file is the raw myScheme "detail" API response,
so this loader builds its own lightweight summary index from those files
instead of relying on a separate all_schemes_index.json / schemes_data.json
pair (those came from an earlier prototype scraper and are no longer
produced by the current one).

State-level schemes carry a single `state` field in basicDetails (e.g.
{'value': 24, 'label': 'Gujarat'}); central schemes have none. This loader
normalizes that into `beneficiaryState` as a list, so downstream code that
does ', '.join(states) works the same regardless of scheme type.
"""

import json
from pathlib import Path

from backend.config import SCHEMES_DIR


class SchemeLoader:
    """Loads, indexes, and provides access to scheme data."""

    def __init__(self):
        self.all_schemes = []       # Flattened summary entries (search/embedding)
        self.slug_to_index = {}     # slug -> flattened summary entry
        self.detailed_schemes = {}  # slug -> raw scraped detail JSON

    def load_data(self):
        """Load every scheme JSON file from SCHEMES_DIR."""
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

        print(f"Loaded {loaded} schemes from {schemes_dir} ({skipped} skipped).")

    @staticmethod
    def _label(value):
        """myScheme often nests fields as {'value': ..., 'label': ...}."""
        if isinstance(value, dict):
            return value.get('label', '') or value.get('value', '')
        return value or ''

    def _flatten(self, raw, slug):
        """Build a flat summary dict (old all_schemes_index.json shape) from
        a raw per-slug scraper file."""
        basic = raw.get('en', {}).get('basicDetails', {})
        content = raw.get('en', {}).get('schemeContent', {})

        # State-level schemes carry a single 'state' field; central schemes
        # have none. Normalize to a list either way so downstream code
        # (which does ', '.join(states)) doesn't care which kind it is.
        state_label = self._label(basic.get('state'))
        beneficiary_state = [state_label] if state_label else []

        # State schemes use nodalDepartmentName; central schemes use
        # nodalMinistryName. Fall back between the two so 'ministry' isn't
        # blank just because a scheme is state-level.
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
