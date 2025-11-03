additional_stopwords = set([
    # Number words
    'zero', 'one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine', 'ten',
    'twenty', 'thirty', 'forty', 'fifty', 'sixty', 'seventy', 'eighty', 'ninety',
    'eleven', 'twelve', 'thirteen', 'fourteen', 'fifteen', 'sixteen', 'seventeen', 'eighteen', 'nineteen',
    'hundred', 'thousand', 'ii', 'iii', 'iv', 'vi', 'vii', 'viii', 'ix',
    'year', 'years', 'first', 'second', 'third', 'fourth', 'fifth', 'last',
    # Common document/IMO terms
    'regulation', 'resolution', 'committee', 'guideline', 'ship', 'report', 'imo', 'mepc', 'marpol', 
    'session', 'water', 'information', 'test', 'secretariat', 'data', 'issue', 'environment',
    'meeting', 'member', 'country', 'state', 'party', 'agreement', 'activity', 'plan','group',
	'round', 'task', 'assessment', 'programme', 'reference', 'effort', 
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
    'term', 'time', 'type', 'use', 'value', 'view', 'work'
])
