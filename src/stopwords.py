"""
Unified stopwords module for IMO project.
==========================================
Merges all stopwords from:
  - NLTK English stopwords
  - additional_stopwords (formerly in stopword.py)
  - IMO_STOPWORDS (formerly in bertopic_model.py)

Usage:
    from stopwords import STOPWORDS
    from stopwords import get_stopwords
"""

from nltk.corpus import stopwords as _nltk_stopwords

# ---------- NLTK English stopwords ----------
_nltk_english = set(_nltk_stopwords.words('english'))

# ---------- Additional stopwords (formerly stopword.py) ----------
_additional_stopwords = {
    # Number words
    'ii', 'iii', 'iv', 'v', 'vi', 'vii', 'viii', 'ix', 'x',
    'year', 'month', 'week', 'day', 'date',
    'first', 'second', 'third', 'fourth', 'fifth', 'last',
    # Common document/IMO terms
    'regulation', 'resolution', 'committee', 'guideline', 'ship', 'report',
    'imo', 'mepc', 'marpol',
    'session', 'water', 'information', 'test', 'secretariat', 'data', 'issue',
    'environment',
    'meeting', 'member', 'country', 'state', 'party', 'agreement', 'activity',
    'plan', 'group',
    'round', 'task', 'assessment', 'programme', 'reference', 'effort',
    'element',
    'fuel', 'emission', 'reduction', 'oil', 'gas', 'carbon', 'intensity',
    'energy', 'shipping',
    # Structural words
    'paragraph', 'section', 'page', 'annex', 'figure', 'table', 'chapter',
    'part', 'item', 'document',
    'amendment', 'appendix', 'convention', 'code', 'sector', 'operation',
    'system', 'project', 'strategy',
    # Common verbs and adjectives
    'adopt', 'approve', 'propose', 'consider', 'develop', 'include', 'require',
    'initial', 'new', 'final', 'interim', 'also', 'such', 'various',
    'specific', 'particular',
    # Miscellaneous
    'etc', 'e.g', 'i.e', 'may', 'shall', 'will', 'would', 'could', 'should',
    # Others
    'action', 'accordance', 'account', 'addition', 'administration', 'analysis',
    'application', 'approach', 'basis', 'case', 'comment', 'consideration',
    'decision', 'development', 'discussion', 'draft', 'example', 'framework',
    'guidance',
    'implementation', 'management', 'measure', 'measurement', 'method', 'model',
    'need',
    'number', 'option', 'order', 'performance', 'period', 'procedure',
    'process', 'proposal',
    'regard', 'requirement', 'result', 'review', 'standard', 'study',
    'submission',
    'term', 'time', 'type', 'use', 'value', 'view', 'work',
    'organization', 'area',
}

# ---------- IMO_STOPWORDS (formerly in bertopic_model.py) ----------
_imo_stopwords = {
    'document', 'committee', 'annex', 'paragraph', 'meeting', 'session',
    'agenda', 'item', 'note', 'secretariat', 'invited', 'approval',
    'consideration', 'report', 'information', 'page', 'resolution',
    'regulation', 'amendment', 'proposal', 'guidelines', 'draft',
    'organization', 'sub', 'ref', 'attached', 'related', 'submitted',
    'following', 'accordance', 'regard', 'relevant', 'associated',
    'concerning', 'assembly', 'recognized', 'appropriate', 'general',
    'particular', 'provisions', 'request', 'action',
    'may', 'shall', 'also', 'would', 'could', 'should', 'one',
    'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine', 'ten',
    'mepc', 'msc', 'ccc', 'sse', 'iswg', 'ghg', 'inf', 'wp',
}

# ---------- Merged constant ----------
STOPWORDS: set[str] = _nltk_english | _additional_stopwords | _imo_stopwords

# Legacy aliases (for backward compatibility)
additional_stopwords = _additional_stopwords
IMO_STOPWORDS = _imo_stopwords


def get_stopwords() -> set[str]:
    """Return the unified stopwords set."""
    return STOPWORDS.copy()
