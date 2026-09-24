"""Build review bibliography and CSV from checked metadata; no network or corpus mutation."""
import csv
import html
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
source = ROOT / 'literature_sources_2026-09-20.json'
records = json.loads(source.read_text(encoding='utf-8'))
date_notes = {
    'L04': '在线 2024-08-30；卷期 2026',
    'L06': '在线 2025-12-11；卷期 2026',
    'L07': '卷期 2026；DOI 含 2025',
    'L11': '在线 2025-09-19；卷期 2026',
    'L17': '在线 2016-08-03；手册章节 2017',
    'L21': '在线 2025-10-15',
}
arxiv = {
    'L16': {
        'authors': 'Alexander Thomas; Hubert P. H. Shum; Darren Nellis; Manli Zhu; Phatpicha Yochum; William Bartle; Daniel Wrightson',
        'date': '2026-08-21，v1', 'id': '2608.21036',
    },
    'L19': {
        'authors': 'Darren Edge; Ha Trinh; Newman Cheng; Joshua Bradley; Alex Chao; Apurva Mody; Steven Truitt; Dasha Metropolitansky; Robert Osazuwa Ness; Jonathan Larson',
        'date': '初稿 2024-04-24；v2 2025-02-19', 'id': '2404.16130',
    },
    'L20': {
        'authors': 'Farhad Moghimifar; Yuan-Fang Li; Robert Thomson; Gholamreza Haffari',
        'date': '2024-02-18，v1', 'id': '2402.11712',
    },
}
rows, footnotes = [], []
for item in records:
    ident = item['id']
    metadata = item.get('crossref') or {}
    if metadata:
        authors = '; '.join(' '.join(filter(None, (a.get('given'), a.get('family')))) for a in metadata.get('author', []))
        title = html.unescape(metadata['title'][0])
        venue = html.unescape(metadata['container-title'][0])
        parts = metadata['published']['date-parts'][0]
        date = date_notes.get(ident, '-'.join(str(p) if i == 0 else f'{p:02d}' for i, p in enumerate(parts)))
    else:
        manual = arxiv[ident]
        authors, title = manual['authors'], item['title']
        venue, date = 'arXiv:' + manual['id'], manual['date']
        item['arxiv_metadata'] = manual
        item['status'] = 'arXiv version consulted; separate peer-reviewed publication status not established'
    item['publication_display'] = date
    item['authors_display'] = authors
    item['venue_display'] = venue
    item['title_verified'] = title
    rows.append({
        'id': ident, 'authors': authors, 'publication_date': date,
        'title': title, 'venue': venue, 'doi': item.get('doi') or '',
        'url': item['url'], 'status': item['status'],
        'reading_level': item['reading_level'], 'relevance': item['relevance'],
        'accessed': item['accessed'],
    })
    extra = '本次引用 arXiv 版本，未据此确认正式同行评审发表。' if ident in arxiv else ''
    footnotes.append(f"[^{ident}]: {authors}. {date}. [{title}]({item['url']}). *{venue}*. 读取层级：{item['reading_level']}。{extra}")
source.write_text(json.dumps(records, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
with (ROOT / 'literature_sources_2026-09-20.csv').open('w', encoding='utf-8-sig', newline='') as handle:
    writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
report = ROOT.parent / 'journal_literature_innovation_review.md'
content = report.read_text(encoding='utf-8')
marker = '\n### 文献参考资料\n'
content = content.split(marker)[0].rstrip()
report.write_text(content + '\n' + marker + '\n' + '\n\n'.join(footnotes) + '\n', encoding='utf-8')
print(f'Updated {len(rows)} records; {sum(bool(r.get("crossref")) for r in records)} Crossref metadata records; {len(footnotes)} footnotes.')
