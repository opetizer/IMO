"""
Topic Visualization - based on BERTopic results
=================================================
1. Topic x Session heatmap (per committee)
2. Key country proposal trends
3. Topic keyword word cloud
4. Committee proposal volume comparison

Usage:
    python src/topic_visualization.py --base-dir output --committees MEPC MSC CCC SSE ISWG-GHG
"""

import os, re, json, argparse, warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter

try:
    from wordcloud import WordCloud
    HAS_WC = True
except ImportError:
    HAS_WC = False


def load_assignments(committee, base_dir="output"):
    csv_path = os.path.join(base_dir, committee, "bertopic",
                            f"bertopic_assignments_{committee}.csv")
    if not os.path.exists(csv_path):
        return None
    return pd.read_csv(csv_path, encoding='utf-8-sig')


def load_topic_info(committee, base_dir="output"):
    json_path = os.path.join(base_dir, committee, "bertopic",
                             f"bertopic_analysis_{committee}.json")
    if not os.path.exists(json_path):
        return {}, {}
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    topic_map, topic_words = {}, {}
    for t in data.get('topics', []):
        tid = t['id']
        words = [w['word'] for w in t['top_words']]
        topic_map[tid] = ' / '.join(words[:3])
        topic_words[tid] = words
    return topic_map, topic_words


def _top_topic_ids(topic_words, limit=6):
    topic_ids = sorted(tid for tid in topic_words.keys() if isinstance(tid, int) and tid >= 0)
    return topic_ids[:limit]


def _session_num(meeting_str):
    m = re.search(r'(\d+)', str(meeting_str))
    return int(m.group(1)) if m else 0


def _split_orig(orig_str):
    if pd.isna(orig_str) or not str(orig_str).strip():
        return []
    parts = re.split(r'[;,/|]|\band\b', str(orig_str))
    return [p.strip() for p in parts
            if p.strip() and p.strip() not in ('Secretariat', 'Unknown', '')]


# ---------- 1. Heatmap ----------

def plot_topic_heatmap(df, topic_map, committee, out_dir):
    df2 = df[df['topic_id'] != -1].copy()
    if df2.empty:
        return
    df2['sn'] = df2['meeting'].apply(_session_num)
    meeting_order = (df2.drop_duplicates('meeting')
                     .sort_values('sn')['meeting'].tolist())

    counts = df2.groupby(['topic_id', 'meeting']).size().reset_index(name='n')
    top_tids = df2['topic_id'].value_counts().nlargest(15).index.tolist()
    counts = counts[counts['topic_id'].isin(top_tids)]

    pivot = counts.pivot_table(index='topic_id', columns='meeting',
                               values='n', fill_value=0)
    pivot = pivot.reindex(columns=[m for m in meeting_order if m in pivot.columns])
    pivot = pivot.reindex(top_tids)

    ylabels = [f"T{t}: {topic_map.get(t,'?')[:35]}" for t in pivot.index]
    matrix = pivot.values.astype(int)

    fig, ax = plt.subplots(figsize=(max(10, len(meeting_order)*1.4),
                                    max(6, len(top_tids)*0.55)))
    sns.heatmap(matrix, xticklabels=pivot.columns, yticklabels=ylabels,
                cmap='YlOrRd', annot=True, fmt='d', linewidths=.5, ax=ax,
                cbar_kws={'label': 'Document Count'})
    ax.set_title(f'{committee} Topic x Session Heatmap', fontsize=14, fontweight='bold')
    ax.set_xlabel('Session'); ax.set_ylabel('Topic')
    plt.xticks(rotation=30, ha='right')
    plt.tight_layout()
    p = os.path.join(out_dir, f'{committee}_topic_session_heatmap.png')
    fig.savefig(p, dpi=300, bbox_inches='tight'); plt.close(fig)
    print(f'  Saved: {p}')

    # CSV
    csv_df = pivot.copy(); csv_df.index = ylabels
    csv_df.to_csv(os.path.join(out_dir, f'{committee}_topic_session_heatmap.csv'),
                  encoding='utf-8-sig')


# ---------- 2. Country trends ----------

def plot_country_trends(df, committee, out_dir, top_n=10):
    df2 = df.copy()
    df2['sn'] = df2['meeting'].apply(_session_num)
    meeting_order = (df2.drop_duplicates('meeting')
                     .sort_values('sn')['meeting'].tolist())

    recs = []
    for _, row in df2.iterrows():
        for o in _split_orig(row.get('originator', '')):
            recs.append({'country': o, 'meeting': row['meeting']})
    if not recs:
        return
    rec_df = pd.DataFrame(recs)

    top_c = rec_df['country'].value_counts().nlargest(top_n).index.tolist()
    rec_sub = rec_df[rec_df['country'].isin(top_c)]
    counts = rec_sub.groupby(['country', 'meeting']).size().reset_index(name='n')

    fig, ax = plt.subplots(figsize=(14, 8))
    cmap = plt.cm.tab10(np.linspace(0, 1, len(top_c)))
    for i, c in enumerate(top_c):
        vals = [counts[(counts['country']==c)&(counts['meeting']==m)]['n'].sum()
                for m in meeting_order]
        ax.plot(meeting_order, vals, marker='o', label=c, lw=2, ms=5, color=cmap[i])
    ax.set_title(f'{committee} Key Country Proposal Trends (Top {top_n})',
                 fontsize=14, fontweight='bold')
    ax.set_xlabel('Session'); ax.set_ylabel('Proposals')
    ax.legend(loc='best', fontsize=8, ncol=2)
    ax.grid(True, alpha=.3, ls='--')
    plt.xticks(rotation=30, ha='right'); plt.tight_layout()
    p = os.path.join(out_dir, f'{committee}_country_trends.png')
    fig.savefig(p, dpi=300, bbox_inches='tight'); plt.close(fig)
    print(f'  Saved: {p}')

    # CSV
    piv = counts.pivot_table(index='country', columns='meeting', values='n', fill_value=0)
    piv = piv.reindex(columns=[m for m in meeting_order if m in piv.columns])
    piv['total'] = piv.sum(axis=1)
    piv.sort_values('total', ascending=False).to_csv(
        os.path.join(out_dir, f'{committee}_country_trends.csv'), encoding='utf-8-sig')


# ---------- 3. Word cloud / keyword frequency ----------

def plot_wordcloud(topic_words, topic_map, committee, out_dir, top_n_topics=6):
    if not HAS_WC or not topic_words:
        return
    topic_ids = _top_topic_ids(topic_words, limit=top_n_topics)
    if not topic_ids:
        return

    ncols = 2
    nrows = 3

    def _make_fig():
        fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 9, nrows * 5.5))
        axes = np.atleast_1d(axes).flatten()
        for ax, tid in zip(axes, topic_ids):
            words = topic_words[tid][:15]
            weights = {word: max(len(words) - idx, 1) for idx, word in enumerate(words)}
            wc = WordCloud(
                width=900, height=600,
                background_color='white',
                max_words=15,
                colormap='viridis',
                prefer_horizontal=1.0,
                min_font_size=14,
                max_font_size=200,
                relative_scaling=0.3,
            ).generate_from_frequencies(weights)
            ax.imshow(wc, interpolation='bilinear')
            ax.axis('off')
            ax.set_title(f"T{tid}: {topic_map.get(tid, '?')[:42]}",
                         fontsize=13, fontweight='bold', pad=10)
        for ax in axes[len(topic_ids):]:
            ax.axis('off')
        fig.suptitle(f'{committee} Topic Word Clouds (Top 6)',
                     fontsize=18, fontweight='bold', y=1.01)
        plt.tight_layout()
        return fig

    fig = _make_fig()
    p = os.path.join(out_dir, f'{committee}_topic_wordcloud_t0_t5.png')
    fig.savefig(p, dpi=300, bbox_inches='tight'); plt.close(fig)
    print(f'  Saved: {p}')

    fig2 = _make_fig()
    legacy_p = os.path.join(out_dir, f'topic_wordcloud_per_topic_{committee}.png')
    fig2.savefig(legacy_p, dpi=300, bbox_inches='tight'); plt.close(fig2)
    print(f'  Saved: {legacy_p}')


def plot_topic_keyword_frequency(topic_words, topic_map, committee, out_dir, top_n_topics=6):
    if not topic_words:
        return
    topic_ids = _top_topic_ids(topic_words, limit=top_n_topics)
    if not topic_ids:
        return

    ncols = 3
    nrows = int(np.ceil(len(topic_ids) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(18, 5.8 * nrows))
    axes = np.atleast_1d(axes).flatten()
    palette = sns.color_palette('YlGnBu', 10)

    for ax, tid in zip(axes, topic_ids):
        words = topic_words[tid][:10]
        scores = list(range(len(words), 0, -1))
        y = np.arange(len(words))
        ax.barh(y, scores, color=palette[:len(words)])
        ax.set_yticks(y)
        ax.set_yticklabels(words, fontsize=10)
        ax.invert_yaxis()
        ax.set_xlabel('Keyword Weight')
        ax.set_title(f"T{tid}: {topic_map.get(tid, '?')[:42]}", fontsize=12, fontweight='bold')
        ax.grid(axis='x', alpha=.25, ls='--')

    for ax in axes[len(topic_ids):]:
        ax.axis('off')

    fig.suptitle(f'{committee} Topic Keyword Frequency (T0-T5)', fontsize=18, fontweight='bold', y=0.98)
    plt.tight_layout()
    p = os.path.join(out_dir, f'{committee}_topic_keyword_frequency_t0_t5.png')
    fig.savefig(p, dpi=300, bbox_inches='tight'); plt.close(fig)
    print(f'  Saved: {p}')

    rows = []
    for tid in topic_ids:
        for rank, word in enumerate(topic_words[tid][:10], start=1):
            rows.append({
                'topic_id': tid,
                'topic_label': topic_map.get(tid, '?'),
                'rank': rank,
                'word': word,
                'weight': 11 - rank,
            })
    pd.DataFrame(rows).to_csv(
        os.path.join(out_dir, f'{committee}_topic_keyword_frequency_t0_t5.csv'),
        index=False,
        encoding='utf-8-sig',
    )


# ---------- 4. Committee comparison ----------

def plot_committee_comparison(all_stats, out_dir):
    if not all_stats:
        return
    comms = list(all_stats.keys())
    totals = [all_stats[c]['total_docs'] for c in comms]
    n_meet = [all_stats[c]['n_meetings'] for c in comms]
    avg = [t/n if n else 0 for t, n in zip(totals, n_meet)]
    colors = ['#1f77b4','#ff7f0e','#2ca02c','#d62728','#9467bd'][:len(comms)]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))
    bars = ax1.bar(comms, totals, color=colors, edgecolor='white', lw=1.2)
    for b, v in zip(bars, totals):
        ax1.text(b.get_x()+b.get_width()/2, b.get_height()+max(totals)*.01,
                 str(v), ha='center', va='bottom', fontsize=11, fontweight='bold')
    ax1.set_title('Total Proposals per Committee', fontsize=14, fontweight='bold')
    ax1.set_ylabel('Proposals'); ax1.grid(axis='y', alpha=.3, ls='--')

    bars2 = ax2.bar(comms, avg, color=colors, edgecolor='white', lw=1.2)
    for b, v in zip(bars2, avg):
        ax2.text(b.get_x()+b.get_width()/2, b.get_height()+max(avg)*.01,
                 f'{v:.1f}', ha='center', va='bottom', fontsize=11, fontweight='bold')
    ax2.set_title('Average Proposals per Session', fontsize=14, fontweight='bold')
    ax2.set_ylabel('Avg / Session'); ax2.grid(axis='y', alpha=.3, ls='--')

    plt.tight_layout()
    p = os.path.join(out_dir, 'committee_comparison.png')
    fig.savefig(p, dpi=300, bbox_inches='tight'); plt.close(fig)
    print(f'  Saved: {p}')

    pd.DataFrame({'committee': comms, 'total': totals,
                  'sessions': n_meet, 'avg': [round(a,1) for a in avg]}).to_csv(
        os.path.join(out_dir, 'committee_comparison.csv'), index=False, encoding='utf-8-sig')


# ---------- Interpretation ----------

def save_interpretation(committee, stats, out_dir):
    lines = [f"=== {committee} Topic Visualization Results ===\n"]
    lines.append("1. Basic Statistics")
    lines.append(f"   {stats['n_meetings']} sessions, {stats['n_topics']} topics, "
                 f"{stats['n_countries']} entities, {stats['total_docs']} proposals.\n")
    lines.append("2. Top Topics")
    for tid, name, cnt in stats.get('top5_topics', []):
        lines.append(f"   T{tid}: {name} ({cnt} docs)")
    lines.append("\n3. Most Active Countries")
    for c, cnt in stats.get('top5_countries', []):
        lines.append(f"   {c} ({cnt} proposals)")
    lines.append("\n4. China")
    ci = stats.get('china_info', {})
    if ci.get('docs', 0) > 0:
        lines.append(f"   {ci['docs']} proposals, rank #{ci.get('rank','?')}")
    else:
        lines.append("   No China data found in this committee.")
    with open(os.path.join(out_dir, f'{committee}_visualization_interpretation.txt'),
              'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))


# ---------- Main ----------

def process_committee(base_dir, committee):
    print(f"\n{'='*50}\nProcessing {committee}...\n{'='*50}")
    df = load_assignments(committee, base_dir)
    if df is None:
        print(f"  No BERTopic assignments for {committee}"); return None
    topic_map, tw = load_topic_info(committee, base_dir)
    out = os.path.join(base_dir, committee, 'bertopic')
    os.makedirs(out, exist_ok=True)
    print(f"  {len(df)} docs, {len(topic_map)} topics")

    plot_topic_heatmap(df, topic_map, committee, out)
    plot_country_trends(df, committee, out)
    plot_wordcloud(tw, topic_map, committee, out)
    plot_topic_keyword_frequency(tw, topic_map, committee, out)

    df_v = df[df['topic_id'] != -1]
    tc = df_v['topic_id'].value_counts()
    top5t = [(int(t), topic_map.get(t,'?'), int(c)) for t, c in tc.head(5).items()]
    origs = []
    for _, r in df.iterrows():
        origs.extend(_split_orig(r.get('originator', '')))
    cc = Counter(origs)
    china_d = cc.get('China', 0)
    china_r = 0
    if china_d:
        for i, (c, _) in enumerate(cc.most_common()):
            if c == 'China':
                china_r = i+1; break

    stats = dict(total_docs=len(df), n_topics=len(topic_map),
                 n_meetings=len(df['meeting'].unique()), n_countries=len(cc),
                 top5_topics=top5t, top5_countries=cc.most_common(5),
                 china_info=dict(docs=china_d, rank=china_r))
    save_interpretation(committee, stats, out)
    return stats


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--base-dir', default='output')
    p.add_argument('--committees', nargs='+',
                   default=['MEPC','MSC','CCC','SSE','ISWG-GHG'])
    a = p.parse_args()

    all_s = {}
    for c in a.committees:
        s = process_committee(a.base_dir, c)
        if s: all_s[c] = s

    od = os.path.join(a.base_dir, 'visualization')
    os.makedirs(od, exist_ok=True)
    plot_committee_comparison(all_s, od)
    print("\nAll visualizations complete!")

if __name__ == '__main__':
    main()
