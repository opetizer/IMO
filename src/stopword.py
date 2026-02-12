additional_stopwords = set([
    # Number words
	'ii', 'iii', 'iv', 'v', 'vi', 'vii', 'viii', 'ix', 'x',
    'year', 'month', 'week', 'day', 'date',
	'first', 'second', 'third', 'fourth', 'fifth', 'last',
	# Common document/IMO terms
    'regulation', 'resolution', 'committee', 'guideline', 'ship', 'report', 'imo', 'mepc', 'marpol', 
    'session', 'water', 'information', 'test', 'secretariat', 'data', 'issue', 'environment',
    'meeting', 'member', 'country', 'state', 'party', 'agreement', 'activity', 'plan','group',
	'round', 'task', 'assessment', 'programme', 'reference', 'effort', 'element', 'date',
	'fuel', 'emission', 'reduction', 'oil', 'gas', 'carbon', 'intensity', 'energy', 
	'shipping',
    # Structural words
    'paragraph', 'section', 'page', 'annex', 'figure', 'table', 'chapter', 'part', 'item', 'document', 
    'amendment', 'appendix', 'convention', 'code', 'sector', 'operation', 'system', 'project', 'strategy',
    # Common verbs and adjectives
    'adopt', 'approve', 'propose', 'consider', 'develop', 'include', 'require',
    'initial', 'new', 'final', 'interim', 'also', 'such', 'various', 'specific', 'particular',
    # Miscellaneous
    'etc', 'e.g', 'i.e', 'may', 'shall', 'will', 'would', 'could', 'should',
	# others
    'action', 'accordance', 'account', 'addition', 'administration', 'analysis',
    'application', 'approach', 'basis', 'case', 'comment', 'consideration', 
    'decision', 'development', 'discussion', 'draft', 'example', 'framework', 'guidance', 
    'implementation', 'management', 'measure', 'measurement', 'method', 'model', 'need', 
    'number', 'option', 'order', 'performance', 'period', 'procedure', 'process', 'proposal', 
    'regard', 'requirement', 'result', 'review', 'standard', 'study', 'submission', 
    'term', 'time', 'type', 'use', 'value', 'view', 'work',

    'organization', 'area', 

])
