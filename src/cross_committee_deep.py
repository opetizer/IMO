"""
Cross-Committee Deep Analysis (跨委员会深度关联分析)
====================================================
1. Topic semantic similarity across committees (SBERT embedding comparison)
2. Country participation overlap analysis
3. Policy diffusion pathway detection
4. Comprehensive integrated dashboard

Outputs:
  cross_topic_similarity_matrix.html  — inter-committee topic similarity heatmap
  country_committee_sankey.html       — country participation flow across committees
  policy_nexus_network.html           — integrated policy network
  collaboration_network.html          — who co-submits with whom
  deep_analysis_report.json           — full statistics
"""

import os
import json
import re
import argparse
from collections import defaultdict, Counter
from itertools import combinations

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.spatial.distance import cosine
from sklearn.metrics.pairwise import cosine_similarity

import matplotlib.pyplot as plt

from stopwords import STOPWORDS


# ──────────── Data Loading ────────────

def load_all_data(committees, base_dir="output"):
    """Load assignments and topic info for all committees."""
    all_data = {}
    for c in committees:
        csv_path = os.path.join(base_dir, c, "bertopic", f"bertopic_assignments_{c}.csv")
        json_path = os.path.join(base_dir, c, "bertopic", f"bertopic_analysis_{c}.json")
        
        if not os.path.exists(csv_path):
            continue
        
        df = pd.read_csv(csv_path, encoding='utf-8-sig')
        df['committee'] = c
        
        topic_map = {}
        topic_words = {}  # topic_id -> list of top words
        if os.path.exists(json_path):
            with open(json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            for t in data.get('topics', []):
                words = [w['word'] for w in t['top_words'][:5]]
                topic_map[t['id']] = ' / '.join(words[:3])
                topic_words[t['id']] = words
        
        all_data[c] = {
            'df': df,
            'topic_map': topic_map,
            'topic_words': topic_words,
            'n_topics': len(topic_map),
            'n_docs': len(df)
        }
    
    return all_data


def normalize_originator(orig):
    """Normalize originator name."""
    if pd.isna(orig) or not str(orig).strip():
        return None
    orig = str(orig).strip()
    aliases = {
        'Republic of Korea': 'Republic of Korea',
        'Korea': 'Republic of Korea',
        'United Kingdom': 'United Kingdom',
        'UK': 'United Kingdom',
        'Russian Federation': 'Russian Federation',
        'Russia': 'Russian Federation',
    }
    return aliases.get(orig, orig)


def split_originators(orig_str):
    """Split co-sponsors into individual entities."""
    if pd.isna(orig_str) or not str(orig_str).strip():
        return []
    orig_str = str(orig_str)
    parts = re.split(r',\s*(?:and\s+)?|\s+and\s+', orig_str)
    result = []
    for p in parts:
        p = p.strip().strip('.')
        if p and len(p) > 1:
            n = normalize_originator(p)
            if n:
                result.append(n)
    return result


# ──────────── 1. Cross-Topic Similarity ────────────

def split_keyword_terms(words):
    """Split keyword phrases into unique tokens and drop committee-like tokens."""
    terms = set()
    for w in words:
        raw = str(w).lower()
        # Drop the entire phrase if it is committee-related, e.g. maritime_safety_committee.
        if 'committee' in raw:
            continue
        for t in re.split(r'[_\W]+', raw):
            if t and 'committee' not in t:
                terms.add(t)
    return terms

def compute_topic_similarity_keywords(all_data):
    """Compute topic similarity across committees using keyword overlap (Jaccard)."""
    all_topics = []
    for c, d in all_data.items():
        for tid, words in d['topic_words'].items():
            terms = split_keyword_terms(words)
            all_topics.append({
                'committee': c,
                'topic_id': tid,
                'label': f"{c}:T{tid}",
                'full_label': f"{c}:T{tid} ({d['topic_map'].get(tid, '')})",
                'terms': terms,
                'terms_nostop': terms - STOPWORDS,
            })
    
    n = len(all_topics)
    sim_matrix = np.zeros((n, n))
    
    for i in range(n):
        for j in range(n):
            if i == j:
                sim_matrix[i][j] = 1.0
            else:
                # Word overlap after splitting keyword phrases into unique terms.
                w1 = all_topics[i]['terms']
                w2 = all_topics[j]['terms']
                
                # Jaccard similarity
                intersection = len(w1 & w2)
                union = len(w1 | w2)
                jaccard = intersection / union if union > 0 else 0
                
                # Count shared significant terms with stopwords removed.
                terms1 = all_topics[i]['terms_nostop']
                terms2 = all_topics[j]['terms_nostop']
                common_terms = terms1 & terms2
                term_overlap = len(common_terms) / max(len(terms1 | terms2), 1)
                
                sim_matrix[i][j] = max(jaccard, term_overlap)
    
    labels = [t['full_label'] for t in all_topics]
    short_labels = [t['label'] for t in all_topics]
    committees = [t['committee'] for t in all_topics]
    
    return sim_matrix, labels, short_labels, committees, all_topics


def plot_cross_topic_heatmap(sim_matrix, labels, short_labels, committees, all_data, out_dir):
    """Heatmap of topic similarity across committees, filtered to show cross-committee only."""
    
    # Filter to only show meaningful cross-committee similarities
    n = len(labels)
    cross_pairs = []
    
    for i in range(n):
        for j in range(i+1, n):
            if committees[i] != committees[j] and sim_matrix[i][j] > 0.1:
                cross_pairs.append({
                    'topic_a': labels[i],
                    'topic_b': labels[j],
                    'committee_a': committees[i],
                    'committee_b': committees[j],
                    'similarity': round(sim_matrix[i][j], 3)
                })
    
    cross_pairs.sort(key=lambda x: x['similarity'], reverse=True)
    
    # Show top cross-committee links
    if not cross_pairs:
        print("  No significant cross-committee topic links found")
        return cross_pairs
    
    # Build a focused heatmap with only inter-committee blocks
    fig = px.imshow(
        sim_matrix,
        x=short_labels,
        y=short_labels,
        color_continuous_scale='Viridis',
        aspect='auto',
        title='Cross-Committee Topic Similarity (Keyword Overlap)',
        labels=dict(color='Similarity')
    )
    
    # Add committee boundary lines
    boundaries = []
    cum = 0
    for c in all_data:
        n_topics = all_data[c]['n_topics']
        cum += n_topics
        boundaries.append(cum - 0.5)
    
    for b in boundaries[:-1]:
        fig.add_hline(y=b, line_dash="dash", line_color="white", line_width=2)
        fig.add_vline(x=b, line_dash="dash", line_color="white", line_width=2)
    
    fig.update_layout(
        width=1400, height=1200,
        xaxis_tickangle=-90,
        xaxis_tickfont_size=7,
        yaxis_tickfont_size=7,
        margin=dict(b=200, l=200)
    )
    
    path = os.path.join(out_dir, 'cross_topic_similarity_matrix.html')
    fig.write_html(path)
    print(f"  Saved: {path}")
    
    return cross_pairs


# ──────────── 2. Country Cross-Committee Analysis ────────────

def build_country_committee_data(all_data):
    """Build country participation data across committees."""
    records = []
    
    for c, d in all_data.items():
        df = d['df']
        for _, row in df.iterrows():
            if row.get('topic_id', -1) == -1:
                continue
            origs = split_originators(row.get('originator', ''))
            for orig in origs:
                if orig and orig != 'Secretariat':
                    records.append({
                        'country': orig,
                        'committee': c,
                        'topic_id': row['topic_id'],
                        'meeting': row.get('meeting', '')
                    })
    
    return pd.DataFrame(records)


def plot_country_committee_sankey(country_df, out_dir, top_n=20):
    """Sankey diagram: Countries -> Committees -> Topics."""
    if country_df.empty:
        return
    
    # Top active countries
    top_countries = country_df['country'].value_counts().nlargest(top_n).index
    sub = country_df[country_df['country'].isin(top_countries)]
    
    # Build sankey: Country -> Committee
    cc_flow = sub.groupby(['country', 'committee']).size().reset_index(name='count')
    
    # Node labels
    countries = sorted(top_countries)
    committees = sorted(sub['committee'].unique())
    
    labels = list(countries) + list(committees)
    label_idx = {l: i for i, l in enumerate(labels)}
    
    sources = [label_idx[r['country']] for _, r in cc_flow.iterrows()]
    targets = [label_idx[r['committee']] for _, r in cc_flow.iterrows()]
    values = cc_flow['count'].tolist()
    
    # Colors
    country_colors = ['rgba(31,119,180,0.5)'] * len(countries)
    committee_colors = {
        'MEPC': 'rgba(44,160,44,0.8)',
        'MSC': 'rgba(214,39,40,0.8)',
        'CCC': 'rgba(255,127,14,0.8)',
        'SSE': 'rgba(148,103,189,0.8)',
        'ISWG-GHG': 'rgba(140,86,75,0.8)'
    }
    node_colors = country_colors + [committee_colors.get(c, 'rgba(128,128,128,0.8)') for c in committees]
    
    fig = go.Figure(go.Sankey(
        node=dict(
            pad=15,
            thickness=20,
            line=dict(color='black', width=0.5),
            label=labels,
            color=node_colors
        ),
        link=dict(
            source=sources,
            target=targets,
            value=values,
            color=['rgba(128,128,128,0.2)'] * len(sources)
        )
    ))
    
    fig.update_layout(
        title=f'Country Participation Across IMO Committees (Top {top_n})',
        width=1200, height=800,
        font_size=10
    )
    
    path = os.path.join(out_dir, 'country_committee_sankey.html')
    fig.write_html(path)
    print(f"  Saved: {path}")


# ──────────── 3. Co-Sponsorship Network ────────────

def build_cosponsor_network(all_data, out_dir):
    """Network of co-sponsorship relationships."""
    co_sponsor_pairs = Counter()
    entity_docs = Counter()
    entity_committees = defaultdict(set)
    
    for c, d in all_data.items():
        df = d['df']
        for _, row in df.iterrows():
            origs = split_originators(row.get('originator', ''))
            origs = [o for o in origs if o and o != 'Secretariat']
            
            for o in origs:
                entity_docs[o] += 1
                entity_committees[o].add(c)
            
            # Co-sponsorship pairs
            for i in range(len(origs)):
                for j in range(i+1, len(origs)):
                    pair = tuple(sorted([origs[i], origs[j]]))
                    co_sponsor_pairs[pair] += 1
    
    # Build network
    import networkx as nx
    G = nx.Graph()
    
    # Only include entities with >= 5 docs
    active = {e for e, c in entity_docs.items() if c >= 5}
    
    for (a, b), count in co_sponsor_pairs.items():
        if a in active and b in active and count >= 2:
            G.add_edge(a, b, weight=count)
    
    # Add isolated active nodes
    for e in active:
        if e not in G:
            G.add_node(e)
    
    if len(G.nodes()) == 0:
        print("  No co-sponsorship network to build")
        return
    
    # Layout
    pos = nx.spring_layout(G, k=3, seed=42, weight='weight', iterations=50)
    
    # Edges
    edge_x, edge_y = [], []
    for u, v, d in G.edges(data=True):
        x0, y0 = pos[u]
        x1, y1 = pos[v]
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])
    
    edge_trace = go.Scatter(
        x=edge_x, y=edge_y,
        line=dict(width=0.5, color='#aaa'),
        hoverinfo='none',
        mode='lines'
    )
    
    # Nodes - color by type (country vs NGO vs industry)
    industry_orgs = {'ICS', 'INTERTANKO', 'INTERCARGO', 'BIMCO', 'IACS', 'SIGTTO', 
                     'ISO', 'OCIMF', 'WSC', 'CLIA', 'IFSMA', 'ITF', 'InterManager'}
    ngo_orgs = {'WWF', 'FOEI', 'Pacific Environment', 'CSC', 'BIC', 'ICMA', 'IMarEST',
                'Pacific Environment and CSC', 'ReCAAP-ISC'}
    
    node_x = []
    node_y = []
    node_text = []
    node_size = []
    node_color = []
    
    for n in G.nodes():
        x, y = pos[n]
        node_x.append(x)
        node_y.append(y)
        
        degree = G.degree(n, weight='weight')
        ndocs = entity_docs.get(n, 0)
        ncomms = len(entity_committees.get(n, set()))
        
        node_text.append(f"{n}<br>Docs: {ndocs}<br>Committees: {ncomms}<br>Co-sponsor links: {G.degree(n)}")
        node_size.append(max(8, min(40, ndocs * 0.5)))
        
        if n in industry_orgs:
            node_color.append('orange')
        elif n in ngo_orgs:
            node_color.append('green')
        else:
            node_color.append('steelblue')
    
    node_trace = go.Scatter(
        x=node_x, y=node_y,
        mode='markers+text',
        text=list(G.nodes()),
        textposition='top center',
        textfont=dict(size=7),
        marker=dict(
            size=node_size,
            color=node_color,
            line=dict(width=1, color='white')
        ),
        hoverinfo='text',
        hovertext=node_text
    )
    
    # Legend traces
    legend_traces = [
        go.Scatter(x=[None], y=[None], mode='markers',
                   marker=dict(size=10, color='steelblue'),
                   name='Countries/Flag States'),
        go.Scatter(x=[None], y=[None], mode='markers',
                   marker=dict(size=10, color='orange'),
                   name='Industry Organizations'),
        go.Scatter(x=[None], y=[None], mode='markers',
                   marker=dict(size=10, color='green'),
                   name='NGOs / Civil Society'),
    ]
    
    fig = go.Figure(data=[edge_trace, node_trace] + legend_traces)
    fig.update_layout(
        title='IMO Co-Sponsorship Network (entities with >= 5 docs, co-sponsoring >= 2 times)',
        showlegend=True,
        width=1200, height=900,
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        legend=dict(x=0, y=1)
    )
    
    path = os.path.join(out_dir, 'collaboration_network.html')
    fig.write_html(path)
    print(f"  Saved: {path}")
    
    # Return top pairs for report
    top_pairs = co_sponsor_pairs.most_common(30)
    return top_pairs


# ──────────── 4. Policy Nexus Network ────────────

def build_policy_nexus(all_data, cross_pairs, country_df, out_dir):
    """Build integrated policy nexus network showing committee-topic-country relationships."""
    
    import networkx as nx
    G = nx.Graph()
    
    # Add committee nodes
    for c in all_data:
        G.add_node(c, type='committee', size=30)
    
    # Add key topic nodes (top 5 per committee)
    for c, d in all_data.items():
        topic_counts = d['df'][d['df']['topic_id'] != -1].groupby('topic_id').size()
        top_topics = topic_counts.nlargest(5).index
        
        for tid in top_topics:
            label = d['topic_map'].get(tid, f'T{tid}')[:30]
            node_id = f"{c}:T{tid}"
            G.add_node(node_id, type='topic', label=label, size=15)
            G.add_edge(c, node_id, weight=int(topic_counts[tid]))
    
    # Add cross-committee topic links
    for pair in cross_pairs[:20]:  # Top 20 cross-links
        a = f"{pair['committee_a']}:T{pair['topic_a'].split(':T')[1].split(' ')[0]}"
        b = f"{pair['committee_b']}:T{pair['topic_b'].split(':T')[1].split(' ')[0]}"
        if a in G and b in G:
            G.add_edge(a, b, weight=pair['similarity'] * 10, cross=True)
    
    # Add top country nodes
    if not country_df.empty:
        top_countries = country_df['country'].value_counts().nlargest(10).index
        for country in top_countries:
            G.add_node(country, type='country', size=20)
            # Link to most active committee
            country_comms = country_df[country_df['country'] == country]['committee'].value_counts()
            for comm, count in country_comms.head(3).items():
                G.add_edge(country, comm, weight=count)
    
    # Layout
    pos = nx.spring_layout(G, k=2.5, seed=42, weight='weight', iterations=80)
    
    # Draw
    edge_x, edge_y = [], []
    for u, v, d in G.edges(data=True):
        x0, y0 = pos[u]
        x1, y1 = pos[v]
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])
    
    edge_trace = go.Scatter(
        x=edge_x, y=edge_y,
        line=dict(width=0.8, color='#ccc'),
        hoverinfo='none', mode='lines'
    )
    
    # Nodes by type
    type_config = {
        'committee': {'color': '#e74c3c', 'symbol': 'square', 'size_mult': 2},
        'topic': {'color': '#3498db', 'symbol': 'circle', 'size_mult': 1},
        'country': {'color': '#2ecc71', 'symbol': 'diamond', 'size_mult': 1.5}
    }
    
    traces = [edge_trace]
    for ntype, config in type_config.items():
        nodes = [n for n in G.nodes() if G.nodes[n].get('type') == ntype]
        if not nodes:
            continue
        
        nx_pos = [pos[n][0] for n in nodes]
        ny_pos = [pos[n][1] for n in nodes]
        sizes = [G.nodes[n].get('size', 10) * config['size_mult'] for n in nodes]
        
        display_labels = []
        for n in nodes:
            if ntype == 'topic':
                display_labels.append(G.nodes[n].get('label', n))
            else:
                display_labels.append(n)
        
        traces.append(go.Scatter(
            x=nx_pos, y=ny_pos,
            mode='markers+text',
            text=display_labels,
            textposition='top center',
            textfont=dict(size=8 if ntype == 'topic' else 11),
            marker=dict(
                size=sizes,
                color=config['color'],
                symbol=config['symbol'],
                line=dict(width=1, color='white')
            ),
            name=ntype.capitalize(),
            hoverinfo='text',
            hovertext=[f"{n}<br>Type: {ntype}" for n in nodes]
        ))
    
    fig = go.Figure(data=traces)
    fig.update_layout(
        title='IMO Policy Nexus Network (Committees - Topics - Countries)',
        width=1300, height=900,
        showlegend=True,
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False)
    )
    
    path = os.path.join(out_dir, 'policy_nexus_network.html')
    fig.write_html(path)
    print(f"  Saved: {path}")


# ──────────── 5. Committee Overlap Chord ────────────

def plot_committee_overlap(country_df, out_dir):
    """Which countries bridge multiple committees? Heatmap of overlap."""
    if country_df.empty:
        return
    
    # For each country, count docs per committee
    pivot = country_df.groupby(['country', 'committee']).size().reset_index(name='count')
    
    # Countries active in 3+ committees
    multi = pivot.groupby('country')['committee'].nunique()
    multi_countries = multi[multi >= 3].index
    
    sub = pivot[pivot['country'].isin(multi_countries)]
    matrix = sub.pivot_table(index='country', columns='committee', values='count', fill_value=0)
    
    # Sort by total
    matrix['total'] = matrix.sum(axis=1)
    matrix = matrix.sort_values('total', ascending=False).drop('total', axis=1)
    
    fig = px.imshow(
        matrix.values,
        x=matrix.columns.tolist(),
        y=matrix.index.tolist(),
        color_continuous_scale='Blues',
        aspect='auto',
        title='Country Activity Across IMO Committees (active in 3+ committees)',
        labels=dict(color='Proposals'),
        text_auto=True
    )
    fig.update_layout(
        width=900, height=max(400, len(matrix) * 22),
        margin=dict(l=200)
    )
    
    path = os.path.join(out_dir, 'committee_overlap_heatmap.html')
    fig.write_html(path)
    print(f"  Saved: {path}")


def save_deep_csv_and_txt(cross_pairs, top_pairs, report, out_dir):
    """Save CSV tables and interpretation TXT for cross-committee analysis."""
    import pandas as pd

    # 1. Cross topic similarity CSV (top 20)
    if cross_pairs:
        sim_df = pd.DataFrame(cross_pairs[:20])
        csv_path = os.path.join(out_dir, 'cross_topic_similarity_top.csv')
        sim_df.to_csv(csv_path, index=False, encoding='utf-8-sig')
        print(f"  Saved: {csv_path}")

    # 2. Bridge countries CSV
    bridge = report.get('bridge_countries', {})
    if bridge:
        bridge_rows = [{'country': k, 'committees_active': v} for k, v in
                       sorted(bridge.items(), key=lambda x: x[1], reverse=True)]
        bridge_df = pd.DataFrame(bridge_rows)
        csv_path = os.path.join(out_dir, 'bridge_countries.csv')
        bridge_df.to_csv(csv_path, index=False, encoding='utf-8-sig')
        print(f"  Saved: {csv_path}")

    # 3. Interpretation TXT
    lines = ["=== 跨委员会深度关联分析结果 ===\n"]

    lines.append("一、基本统计")
    lines.append(f"分析委员会: {', '.join(report.get('committees', []))}")
    lines.append(f"总文档数: {report.get('total_docs', 0)}，总主题数: {report.get('total_topics', 0)}\n")

    lines.append("二、跨委员会主题关联")
    if cross_pairs:
        lines.append(f"发现 {len(cross_pairs)} 对跨委员会相似主题，前5名:")
        for p in cross_pairs[:5]:
            lines.append(f"  {p['topic_a'][:35]} ↔ {p['topic_b'][:35]} (相似度: {p['similarity']:.3f})")
    else:
        lines.append("  未发现显著跨委员会主题关联。")
    lines.append("")

    lines.append("三、跨委员会活跃国家")
    if bridge:
        top_bridge = sorted(bridge.items(), key=lambda x: x[1], reverse=True)[:10]
        for country, n_comms in top_bridge:
            lines.append(f"  {country}: 活跃于 {n_comms} 个委员会")
    lines.append("")

    lines.append("四、联合提案网络")
    top_cs = report.get('top_cosponsor_pairs', [])
    if top_cs:
        lines.append("最频繁的跨委员会联合提案关系:")
        for pair_info in top_cs[:5]:
            if isinstance(pair_info, (list, tuple)) and len(pair_info) == 2:
                pair, count = pair_info
                if isinstance(pair, (list, tuple)) and len(pair) == 2:
                    lines.append(f"  {pair[0]} + {pair[1]}: {count} 次")

    txt_path = os.path.join(out_dir, 'cross_committee_interpretation.txt')
    with open(txt_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))
    print(f"  Saved: {txt_path}")


# ──────────── Main ────────────

def main():
    parser = argparse.ArgumentParser(description='Cross-Committee Deep Analysis')
    parser.add_argument('--committees', nargs='+', default=['MEPC', 'MSC', 'CCC', 'SSE', 'ISWG-GHG'])
    parser.add_argument('--base-dir', default='output')
    parser.add_argument('--cosine-threshold', type=float, default=0.1,
                        help='Threshold for cross-topic cosine similarity pairs')
    args = parser.parse_args()
    
    out_dir = os.path.join(args.base_dir, 'deep_analysis')
    os.makedirs(out_dir, exist_ok=True)
    
    print("Loading data from all committees...")
    all_data = load_all_data(args.committees, args.base_dir)
    print(f"Loaded {len(all_data)} committees\n")
    
    # 1. Cross-topic similarity
    print("=" * 60)
    print("1. Cross-Committee Topic Similarity...")
    print("=" * 60)
    sim_matrix, labels, short_labels, committees_list, all_topics = compute_topic_similarity_keywords(all_data)
    cross_pairs = plot_cross_topic_heatmap(sim_matrix, labels, short_labels, committees_list, all_data, out_dir)

    # 1b. Cross-topic cosine similarity (keywords)
    print("\n" + "=" * 60)
    print("1b. Cross-Committee Topic Cosine Similarity (keywords)...")
    print("=" * 60)
    print(f"  Cosine threshold: {args.cosine_threshold:.3f}")
    # 取所有主题的关键词全集
    all_keywords = set()
    for t in all_topics:
        all_keywords.update(t['terms'])
    all_keywords = sorted(all_keywords)
    kw_index = {w: i for i, w in enumerate(all_keywords)}
    # 构建主题-关键词0/1向量
    topic_vecs = np.zeros((len(all_topics), len(all_keywords)), dtype=int)
    for idx, t in enumerate(all_topics):
        for w in t['terms']:
            if w in kw_index:
                topic_vecs[idx, kw_index[w]] = 1
    # 计算余弦相似度
    cos_sim = cosine_similarity(topic_vecs)
    # 只保留跨委员会且不重复的高相似对
    cosine_pairs = []
    for i in range(len(all_topics)):
        for j in range(i+1, len(all_topics)):
            if all_topics[i]['committee'] != all_topics[j]['committee']:
                score = float(cos_sim[i, j])
                if score > args.cosine_threshold:
                    cosine_pairs.append({
                        'topic_a': all_topics[i]['full_label'],
                        'topic_b': all_topics[j]['full_label'],
                        'committee_a': all_topics[i]['committee'],
                        'committee_b': all_topics[j]['committee'],
                        'cosine_similarity': round(score, 4)
                    })
    cosine_pairs.sort(key=lambda x: x['cosine_similarity'], reverse=True)
    print(f"  Pairs above threshold: {len(cosine_pairs)}")
    # 输出top对到CSV
    out_csv = os.path.join(out_dir, 'cross_topic_cosine_similarity_top.csv')
    pd.DataFrame(cosine_pairs[:50]).to_csv(out_csv, index=False, encoding='utf-8-sig')
    print(f'  Saved: {out_csv}')
    # 可选：输出热力图
    # 只展示top 30对
    if cosine_pairs:
        import matplotlib.pyplot as plt
        topN = min(30, len(cosine_pairs))
        fig, ax = plt.subplots(figsize=(10, max(6, topN*0.4)), dpi=180)
        sim_vals = [p['cosine_similarity'] for p in cosine_pairs[:topN]]
        labels_a = [p['topic_a'][:40] for p in cosine_pairs[:topN]]
        labels_b = [p['topic_b'][:40] for p in cosine_pairs[:topN]]
        y = np.arange(topN)
        ax.barh(y, sim_vals, color='teal')
        ax.set_yticks(y, [f"{a}\n{b}" for a, b in zip(labels_a, labels_b)])
        ax.set_xlabel('Cosine Similarity')
        ax.set_title('Top Cross-Committee Topic Pairs (Cosine Similarity)')
        for i, v in enumerate(sim_vals):
            ax.text(v+0.01, i, f'{v:.2f}', va='center', fontsize=8)
        fig.tight_layout()
        out_png = os.path.join(out_dir, 'cross_topic_cosine_similarity_top.png')
        fig.savefig(out_png, bbox_inches='tight')
        plt.close(fig)
        print(f'  Saved: {out_png}')
    
    print(f"\n  Top 10 cross-committee topic links:")
    for p in cross_pairs[:10]:
        print(f"    {p['topic_a'][:40]} <-> {p['topic_b'][:40]}: {p['similarity']:.3f}")
    
    # 2. Country cross-committee participation
    print("\n" + "=" * 60)
    print("2. Country Cross-Committee Participation...")
    print("=" * 60)
    country_df = build_country_committee_data(all_data)
    print(f"  Total records: {len(country_df)}")
    plot_country_committee_sankey(country_df, out_dir, top_n=20)
    plot_committee_overlap(country_df, out_dir)

    # 2b. Committee-level Jaccard based on active country/entity overlap
    print("\n" + "=" * 60)
    print("2b. Committee-level Jaccard (active country/entity overlap)...")
    print("=" * 60)
    committees = list(all_data.keys())
    active_sets = {}
    for c in committees:
        df = all_data[c]['df']
        # 只统计有有效topic_id的提案
        actives = set()
        for _, row in df.iterrows():
            if row.get('topic_id', -1) == -1:
                continue
            origs = split_originators(row.get('originator', ''))
            for o in origs:
                if o and o != 'Secretariat':
                    actives.add(o)
        active_sets[c] = actives

    n = len(committees)
    mat = np.zeros((n, n), dtype=float)
    for i, ci in enumerate(committees):
        for j, cj in enumerate(committees):
            aset = active_sets[ci]
            bset = active_sets[cj]
            if not aset and not bset:
                mat[i, j] = 0.0
            elif i == j:
                mat[i, j] = 1.0
            else:
                inter = len(aset & bset)
                union = len(aset | bset)
                mat[i, j] = inter / union if union else 0.0

    # 输出CSV
    out_csv = os.path.join(out_dir, 'committee_active_entity_jaccard.csv')
    with open(out_csv, 'w', encoding='utf-8-sig') as f:
        f.write(',' + ','.join(committees) + '\n')
        for i, c in enumerate(committees):
            row = ','.join(f'{mat[i,j]:.4f}' for j in range(n))
            f.write(f'{c},{row}\n')

    # 输出PNG
    out_png = os.path.join(out_dir, 'committee_active_entity_jaccard.png')
    fig, ax = plt.subplots(figsize=(7.5, 6.2), dpi=180)
    im = ax.imshow(mat, cmap='YlGnBu', vmin=0, vmax=1)
    ax.set_xticks(range(n), committees, rotation=35, ha='right')
    ax.set_yticks(range(n), committees)
    ax.set_title('Committee Jaccard Similarity (Active Entities)')
    for i in range(n):
        for j in range(n):
            v = mat[i, j]
            color = 'white' if v > 0.55 else 'black'
            ax.text(j, i, f'{v:.2f}', ha='center', va='center', color=color, fontsize=8)
    cbar = fig.colorbar(im, ax=ax, shrink=0.85)
    cbar.set_label('Jaccard similarity')
    fig.tight_layout()
    fig.savefig(out_png, bbox_inches='tight')
    plt.close(fig)
    print(f'  Saved: {out_png}')
    print(f'  Saved: {out_csv}')
    
    # 3. Co-sponsorship network
    print("\n" + "=" * 60)
    print("3. Co-Sponsorship Network...")
    print("=" * 60)
    top_pairs = build_cosponsor_network(all_data, out_dir)
    if top_pairs:
        print(f"\n  Top 10 co-sponsorship pairs:")
        for (a, b), count in top_pairs[:10]:
            print(f"    {a} + {b}: {count} joint submissions")
    
    # 4. Policy nexus
    print("\n" + "=" * 60)
    print("4. Policy Nexus Network...")
    print("=" * 60)
    build_policy_nexus(all_data, cross_pairs, country_df, out_dir)
    
    # 5. Summary report
    report = {
        'committees': list(all_data.keys()),
        'total_docs': sum(d['n_docs'] for d in all_data.values()),
        'total_topics': sum(d['n_topics'] for d in all_data.values()),
        'top_cross_links': cross_pairs[:20],
        'top_cosponsor_pairs': [(list(pair), count) for pair, count in (top_pairs or [])[:20]],
        'multi_committee_countries': country_df.groupby('country')['committee'].nunique().to_dict() 
            if not country_df.empty else {}
    }
    
    # Filter to countries active in 3+ committees
    report['bridge_countries'] = {
        k: v for k, v in report['multi_committee_countries'].items() if v >= 3
    }
    del report['multi_committee_countries']  # too large
    
    path = os.path.join(out_dir, 'deep_analysis_report.json')
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(f"\n  Saved: {path}")
    
    # Save CSV and interpretation
    save_deep_csv_and_txt(cross_pairs, top_pairs, report, out_dir)
    
    print("\nDeep analysis complete!")


if __name__ == '__main__':
    main()
