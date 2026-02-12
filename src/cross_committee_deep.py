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

def compute_topic_similarity_keywords(all_data):
    """Compute topic similarity across committees using keyword overlap (Jaccard)."""
    all_topics = []
    for c, d in all_data.items():
        for tid, words in d['topic_words'].items():
            all_topics.append({
                'committee': c,
                'topic_id': tid,
                'label': f"{c}:T{tid}",
                'full_label': f"{c}:T{tid} ({d['topic_map'].get(tid, '')})",
                'words': set(w.lower() for w in words),
                'words_flat': ' '.join(words).lower()
            })
    
    n = len(all_topics)
    sim_matrix = np.zeros((n, n))
    
    for i in range(n):
        for j in range(n):
            if i == j:
                sim_matrix[i][j] = 1.0
            else:
                # Word overlap (both exact and partial)
                w1 = all_topics[i]['words']
                w2 = all_topics[j]['words']
                
                # Jaccard similarity
                intersection = len(w1 & w2)
                union = len(w1 | w2)
                jaccard = intersection / union if union > 0 else 0
                
                # Also check partial word overlap (e.g., "emissions" in "reduction emissions ships")
                flat1 = all_topics[i]['words_flat']
                flat2 = all_topics[j]['words_flat']
                
                # Count shared significant terms
                terms1 = set(flat1.split())
                terms2 = set(flat2.split())
                common_terms = terms1 & terms2
                # Remove stopwords
                stopwords = {'the', 'of', 'and', 'in', 'for', 'to', 'a', 'on', 'by', 'is', 'at', 'with'}
                common_terms -= stopwords
                term_overlap = len(common_terms) / max(len(terms1 | terms2 - stopwords), 1)
                
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
    edge_x, edge_y, edge_text = [], [], []
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
    edge_colors = []
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


# ──────────── Main ────────────

def main():
    parser = argparse.ArgumentParser(description='Cross-Committee Deep Analysis')
    parser.add_argument('--committees', nargs='+', default=['MEPC', 'MSC', 'CCC', 'SSE', 'ISWG-GHG'])
    parser.add_argument('--base-dir', default='output')
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
    
    print("\nDeep analysis complete!")


if __name__ == '__main__':
    main()
