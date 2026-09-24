"""Read-only corpus inventory; writes only its JSON report beside this script."""
from collections import Counter
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re


ROOT = Path(__file__).resolve().parents[2]
COMMITTEES = ['MEPC', 'MSC', 'CCC', 'SSE', 'ISWG-GHG']


def read_csv(path):
    with path.open(encoding='utf-8-sig', newline='') as stream:
        return list(csv.DictReader(stream))


def main():
    report = {
        'generated_at_utc': datetime.now(timezone.utc).isoformat(),
        'root': str(ROOT),
        'scope': 'File inventory and saved-output checks, not a model rerun or PDF content validation',
        'files': {}, 'committees': {}, 'source_sha256': {},
    }
    for folder in ['data', 'output', 'src', 'paper']:
        files = [p for p in (ROOT / folder).rglob('*')
                 if p.is_file() and '__pycache__' not in p.parts and 'planning' not in p.parts]
        report['files'][folder] = dict(Counter(p.suffix.lower() for p in files))
    metadata_path = ROOT / 'output/all_proposals_metadata.csv'
    metadata = read_csv(metadata_path)
    report['metadata'] = {
        'rows': len(metadata), 'columns': list(metadata[0]) if metadata else [],
        'rows_by_council': dict(Counter(r['Council'] for r in metadata)),
        'warning': 'Rows are not independently verified unique documents; no symbol column in this export',
    }
    sources = [metadata_path, ROOT / 'readme.md', ROOT / 'paper/draft_v1.md']
    for committee in COMMITTEES:
        path = ROOT / f'output/{committee}/bertopic/bertopic_assignments_{committee}.csv'
        sources.append(path)
        rows = read_csv(path)
        topics = Counter(r['topic_id'] for r in rows)
        mismatches = []
        for row in rows:
            match = re.match(r'^([A-Z-]+)\s+(\d+)[/\s]', row['symbol'].upper())
            if match:
                derived = f'{match[1]} {match[2]}'
                if derived != row['meeting'].strip().upper():
                    mismatches.append({'symbol': row['symbol'], 'meeting': row['meeting'],
                                       'symbol_meeting': derived})
        pdfs = list((ROOT / 'data' / committee).rglob('*.pdf'))
        report['committees'][committee] = {
            'raw_pdf_files': len(pdfs), 'assignment_rows': len(rows),
            'unique_assignment_symbols': len({r['symbol'] for r in rows}),
            'non_outlier_topic_clusters': len(set(topics) - {'-1'}),
            'outliers': topics['-1'], 'outlier_rate': topics['-1'] / len(rows),
            'assignment_meetings': dict(Counter(r['meeting'] for r in rows)),
            'raw_pdf_folders': dict(Counter(p.parent.name for p in pdfs)),
            'symbol_meeting_mismatches': mismatches,
        }
    for path in sources:
        report['source_sha256'][path.relative_to(ROOT).as_posix()] = hashlib.sha256(path.read_bytes()).hexdigest()
    report['totals'] = {
        key: sum(c[key] for c in report['committees'].values())
        for key in ['raw_pdf_files', 'assignment_rows', 'non_outlier_topic_clusters', 'outliers']
    }
    destination = Path(__file__).with_name('inventory_snapshot.json')
    destination.write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({'report': str(destination), 'totals': report['totals'],
                      'meeting_mismatches': {k: len(v['symbol_meeting_mismatches'])
                                             for k, v in report['committees'].items()}}, ensure_ascii=False))


if __name__ == '__main__':
    main()
