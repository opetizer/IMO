"""
Country/Organization Stance Analysis (国家/组织立场对比分析)
===========================================================
基于 BERTopic 主题分配结果，分析各国/组织在不同主题上的参与度、
偏好差异、联盟关系，以及时间维度上的立场演变。

Outputs:
  1. country_topic_heatmap.html     — 国家-主题参与度热力图
  2. country_similarity_network.html — 国家议题相似性网络
  3. country_focus_radar.html       — 重点国家雷达图对比
  4. country_temporal_shift.html    — 国家议题关注度时间演变
  5. country_stance_report.csv      — 完整数据表
  6. country_stance_summary.json    — 摘要JSON

Usage:
    python src/country_stance.py --committees MEPC MSC CCC SSE ISWG-GHG
    python src/country_stance.py --committees MEPC --top-n 20
"""

import os
import re
import json
import argparse
from collections import defaultdict, Counter

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.spatial.distance import cosine
from scipy.cluster.hierarchy import linkage, fcluster
import networkx as nx


# ─────────────────── Data Loading ───────────────────

def load_assignments(committee, base_dir="output"):
    """Load bertopic_assignments CSV for a committee."""
    csv_path = os.path.join(base_dir, committee, "bertopic", f"bertopic_assignments_{committee}.csv")
    if not os.path.exists(csv_path):
        print(f"  Warning: {csv_path} not found, skipping {committee}")
        return None
    df = pd.read_csv(csv_path, encoding='utf-8-sig')
    df['committee'] = committee
    return df


def load_topic_info(committee, base_dir="output"):
    """Load topic names from analysis JSON."""
    json_path = os.path.join(base_dir, committee, "bertopic", f"bertopic_analysis_{committee}.json")
    if not os.path.exists(json_path):
        return {}
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    topic_map = {}
    for t in data.get('topics', []):
        tid = t['id']
        words = [w['word'] for w in t['top_words'][:3]]
        topic_map[tid] = ' / '.join(words)
    return topic_map


def split_originators(originator_str):
    """Split multi-originator strings into individual entities."""
    if pd.isna(originator_str):
        return []
    originator_str = str(originator_str).strip()
    if not originator_str:
        return []
    # Split by semicolons first, then by commas and 'and' within each part
    parts = re.split(r'\s*;\s*', originator_str)
    origs = []
    for part in parts:
        # Further split by ', and ', ' and ', ','
        subparts = re.split(r',\s*(?:and\s+)?|\s+and\s+', part)
        for p in subparts:
            p = p.strip().strip('.')
            if p and p.lower() not in ['et al', '']:
                origs.append(p)
    return origs


def normalize_country(name):
    """Normalize country/org names for consistency."""
    name = name.strip()
    # Common aliases
    aliases = {
        'Republic of Korea': 'Republic of Korea',
        'Korea': 'Republic of Korea',
        'ROK': 'Republic of Korea',
        'United Kingdom': 'United Kingdom',
        'UK': 'United Kingdom',
        'United States': 'United States',
        'USA': 'United States',
        'Russian Federation': 'Russian Federation',
        'Russia': 'Russian Federation',
        'Marshall Islands': 'Marshall Islands',
        'RMI': 'Marshall Islands',
        'Islamic Republic of Iran': 'Iran',
        'Iran': 'Iran',
    }
    return aliases.get(name, name)


# ─────────────────── Analysis Core ───────────────────

def build_country_topic_matrix(df, min_docs=3):
    """Build country x topic frequency matrix."""
    records = []
    for _, row in df.iterrows():
        topic_id = row.get('topic_id', -1)
        if topic_id == -1:  # skip outliers
            continue
        originators = split_originators(row.get('originator', ''))
        committee = row.get('committee', '')
        meeting = row.get('meeting', '')
        for orig in originators:
            orig = normalize_country(orig)
            if orig and orig != 'Secretariat':  # Secretariat is not a country stance
                records.append({
                    'country': orig,
                    'topic_id': topic_id,
                    'committee': committee,
                    'meeting': meeting
                })
    
    rec_df = pd.DataFrame(records)
    if rec_df.empty:
        return pd.DataFrame(), rec_df
    
    # Filter countries with at least min_docs submissions
    country_counts = rec_df['country'].value_counts()
    active_countries = country_counts[country_counts >= min_docs].index
    rec_df = rec_df[rec_df['country'].isin(active_countries)]
    
    # Pivot to matrix
    matrix = pd.crosstab(rec_df['country'], rec_df['topic_id'])
    
    return matrix, rec_df


def compute_topic_preference(matrix):
    """Compute normalized topic preference (TF-IDF style).
    
    For each country, compute the proportion of their docs in each topic,
    weighted by inverse document frequency across countries.
    """
    if matrix.empty:
        return pd.DataFrame()
    
    # TF: proportion within country
    tf = matrix.div(matrix.sum(axis=1), axis=0)
    
    # IDF: log(N / n_countries_in_topic)
    n_countries = len(matrix)
    idf = np.log(n_countries / (matrix > 0).sum(axis=0) + 1)
    
    tfidf = tf.multiply(idf, axis=1)
    return tfidf


def compute_country_similarity(matrix):
    """Compute cosine similarity between countries based on topic distribution."""
    if matrix.empty or len(matrix) < 2:
        return pd.DataFrame()
    
    # Normalize rows
    norm = matrix.div(matrix.sum(axis=1), axis=0).fillna(0)
    
    countries = norm.index.tolist()
    n = len(countries)
    sim_matrix = np.zeros((n, n))
    
    for i in range(n):
        for j in range(n):
            if i == j:
                sim_matrix[i][j] = 1.0
            else:
                v1 = norm.iloc[i].values
                v2 = norm.iloc[j].values
                if np.sum(v1) == 0 or np.sum(v2) == 0:
                    sim_matrix[i][j] = 0
                else:
                    sim_matrix[i][j] = 1 - cosine(v1, v2)
    
    return pd.DataFrame(sim_matrix, index=countries, columns=countries)


# ─────────────────── Visualization ───────────────────

def plot_country_topic_heatmap(matrix, topic_map, committee, out_dir, top_n=25):
    """Heatmap of top countries vs topics."""
    if matrix.empty:
        return
    
    # Select top N countries by total submissions
    top_countries = matrix.sum(axis=1).nlargest(top_n).index
    sub = matrix.loc[top_countries].copy()
    
    # Rename columns with topic labels
    col_labels = []
    for c in sub.columns:
        label = topic_map.get(c, f"Topic {c}")
        col_labels.append(f"T{c}: {label[:40]}")
    sub.columns = col_labels
    
    # Normalize by row (each country's focus distribution)
    sub_norm = sub.div(sub.sum(axis=1), axis=0).fillna(0)
    
    fig = px.imshow(
        sub_norm.values,
        x=sub_norm.columns.tolist(),
        y=sub_norm.index.tolist(),
        color_continuous_scale='YlOrRd',
        aspect='auto',
        title=f'{committee} — Country Topic Focus (Normalized)',
        labels=dict(color='Topic Share')
    )
    fig.update_layout(
        width=1200, height=max(500, top_n * 25),
        xaxis_tickangle=-45,
        margin=dict(b=200)
    )
    
    path = os.path.join(out_dir, f'country_topic_heatmap_{committee}.html')
    fig.write_html(path)
    print(f"  Saved: {path}")


def plot_country_similarity_network(sim_df, committee, out_dir, threshold=0.3, top_n=25, matrix=None):
    """Network graph of country similarity."""
    if sim_df.empty:
        return
    
    # Use only top N countries — rank by total docs from original matrix if available
    if matrix is not None and not matrix.empty:
        top = matrix.sum(axis=1).nlargest(top_n).index
        # Only keep countries that exist in sim_df
        top = [c for c in top if c in sim_df.index]
    else:
        # Fallback: use weighted degree (sum of similarities) as proxy
        total_sim = sim_df.sum(axis=1)
        top = total_sim.nlargest(top_n).index.tolist()
    sim_sub = sim_df.loc[top, top]
    
    G = nx.Graph()
    countries = sim_sub.index.tolist()
    
    for c in countries:
        G.add_node(c)
    
    for i, c1 in enumerate(countries):
        for j, c2 in enumerate(countries):
            if i < j:
                sim = sim_sub.loc[c1, c2]
                if sim > threshold:
                    G.add_edge(c1, c2, weight=sim)
    
    if len(G.edges()) == 0:
        print(f"  Warning: No edges above threshold {threshold}, lowering to 0.15")
        threshold = 0.15
        for i, c1 in enumerate(countries):
            for j, c2 in enumerate(countries):
                if i < j:
                    sim = sim_sub.loc[c1, c2]
                    if sim > threshold:
                        G.add_edge(c1, c2, weight=sim)
    
    # Layout
    pos = nx.spring_layout(G, k=2, seed=42, weight='weight')
    
    # Edges
    edge_x, edge_y = [], []
    edge_weights = []
    for u, v, d in G.edges(data=True):
        x0, y0 = pos[u]
        x1, y1 = pos[v]
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])
        edge_weights.append(d['weight'])
    
    edge_trace = go.Scatter(
        x=edge_x, y=edge_y,
        line=dict(width=0.5, color='#888'),
        hoverinfo='none',
        mode='lines'
    )
    
    # Nodes
    node_x = [pos[n][0] for n in G.nodes()]
    node_y = [pos[n][1] for n in G.nodes()]
    node_degree = [G.degree(n, weight='weight') for n in G.nodes()]
    
    node_trace = go.Scatter(
        x=node_x, y=node_y,
        mode='markers+text',
        text=list(G.nodes()),
        textposition='top center',
        textfont=dict(size=9),
        marker=dict(
            size=[max(8, d * 3) for d in node_degree],
            color=node_degree,
            colorscale='Viridis',
            colorbar=dict(title='Weighted Degree'),
            line=dict(width=1, color='white')
        ),
        hoverinfo='text',
        hovertext=[f"{n}<br>Connections: {G.degree(n)}<br>Weighted: {G.degree(n, weight='weight'):.2f}" 
                   for n in G.nodes()]
    )
    
    fig = go.Figure(data=[edge_trace, node_trace])
    fig.update_layout(
        title=f'{committee} — Country Similarity Network (cosine > {threshold})',
        showlegend=False,
        width=1000, height=800,
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False)
    )
    
    path = os.path.join(out_dir, f'country_similarity_network_{committee}.html')
    fig.write_html(path)
    print(f"  Saved: {path}")


def plot_country_radar(matrix, topic_map, committee, out_dir, focus_countries=None):
    """Radar chart comparing key countries' topic focus."""
    if matrix.empty:
        return
    
    if focus_countries is None:
        # Default: top 6 most active
        focus_countries = matrix.sum(axis=1).nlargest(6).index.tolist()
    else:
        focus_countries = [c for c in focus_countries if c in matrix.index]
    
    if not focus_countries:
        return
    
    # Normalize
    norm = matrix.div(matrix.sum(axis=1), axis=0).fillna(0)
    
    # Topic labels
    categories = []
    for c in norm.columns:
        label = topic_map.get(c, f"T{c}")
        categories.append(f"T{c}: {label[:25]}")
    
    fig = go.Figure()
    colors = px.colors.qualitative.Set2
    
    for i, country in enumerate(focus_countries):
        values = norm.loc[country].values.tolist()
        values.append(values[0])  # close the radar
        cats = categories + [categories[0]]
        
        fig.add_trace(go.Scatterpolar(
            r=values,
            theta=cats,
            fill='toself',
            name=country,
            opacity=0.6,
            line=dict(color=colors[i % len(colors)])
        ))
    
    fig.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
        title=f'{committee} — Country Topic Focus Comparison',
        width=900, height=700,
        showlegend=True
    )
    
    path = os.path.join(out_dir, f'country_focus_radar_{committee}.html')
    fig.write_html(path)
    print(f"  Saved: {path}")


def plot_temporal_shift(rec_df, topic_map, committee, out_dir, focus_countries=None):
    """Track how countries' topic focus shifts over meetings."""
    if rec_df.empty:
        return
    
    if focus_countries is None:
        top = rec_df['country'].value_counts().nlargest(8).index.tolist()
        focus_countries = top
    else:
        focus_countries = [c for c in focus_countries if c in rec_df['country'].unique()]
    
    sub = rec_df[rec_df['country'].isin(focus_countries)].copy()
    if sub.empty:
        return
    
    # For each country and meeting, compute dominant topic
    grouped = sub.groupby(['country', 'meeting', 'topic_id']).size().reset_index(name='count')
    
    # Pivot for area chart: meeting x topic, per country
    figs = []
    for country in focus_countries:
        cdata = grouped[grouped['country'] == country]
        if cdata.empty:
            continue
        pivot = cdata.pivot_table(index='meeting', columns='topic_id', values='count', fill_value=0)
        # Normalize per meeting
        pivot_norm = pivot.div(pivot.sum(axis=1), axis=0)
        
        # Sort meetings
        pivot_norm = pivot_norm.sort_index()
        figs.append((country, pivot_norm))
    
    if not figs:
        return
    
    n_countries = len(figs)
    fig = make_subplots(
        rows=n_countries, cols=1,
        subplot_titles=[f[0] for f in figs],
        shared_xaxes=True,
        vertical_spacing=0.03
    )
    
    colors = px.colors.qualitative.Set3
    all_topics = set()
    for _, pn in figs:
        all_topics.update(pn.columns)
    all_topics = sorted(all_topics)
    
    for row_idx, (country, pivot_norm) in enumerate(figs):
        for t_idx, topic in enumerate(all_topics):
            if topic in pivot_norm.columns:
                vals = pivot_norm[topic].values
            else:
                vals = [0] * len(pivot_norm)
            
            label = topic_map.get(topic, f"T{topic}")
            fig.add_trace(
                go.Bar(
                    x=pivot_norm.index.tolist(),
                    y=vals,
                    name=f"T{topic}: {label[:20]}",
                    marker_color=colors[t_idx % len(colors)],
                    showlegend=(row_idx == 0),
                    legendgroup=f"t{topic}"
                ),
                row=row_idx + 1, col=1
            )
    
    fig.update_layout(
        barmode='stack',
        title=f'{committee} — Country Topic Focus Over Time',
        width=1200, height=300 * n_countries,
        showlegend=True
    )
    
    path = os.path.join(out_dir, f'country_temporal_shift_{committee}.html')
    fig.write_html(path)
    print(f"  Saved: {path}")


def generate_stance_report(matrix, topic_map, rec_df, committee, out_dir):
    """Generate CSV report with country stance details."""
    if matrix.empty:
        return
    
    rows = []
    norm = matrix.div(matrix.sum(axis=1), axis=0).fillna(0)
    
    for country in matrix.index:
        total = matrix.loc[country].sum()
        dominant_topic = matrix.loc[country].idxmax()
        dominant_share = norm.loc[country].max()
        focus_label = topic_map.get(dominant_topic, f"Topic {dominant_topic}")
        
        # Diversity: Shannon entropy
        probs = norm.loc[country].values
        probs = probs[probs > 0]
        entropy = -np.sum(probs * np.log2(probs)) if len(probs) > 1 else 0
        max_entropy = np.log2(len(matrix.columns)) if len(matrix.columns) > 1 else 1
        diversity = entropy / max_entropy if max_entropy > 0 else 0
        
        rows.append({
            'committee': committee,
            'country': country,
            'total_proposals': int(total),
            'dominant_topic_id': int(dominant_topic),
            'dominant_topic': focus_label,
            'dominant_share': round(dominant_share, 3),
            'topic_diversity': round(diversity, 3)
        })
    
    report_df = pd.DataFrame(rows)
    report_df = report_df.sort_values('total_proposals', ascending=False)
    
    path = os.path.join(out_dir, f'country_stance_report_{committee}.csv')
    report_df.to_csv(path, index=False, encoding='utf-8-sig')
    print(f"  Saved: {path}")
    
    return report_df


# ─────────────────── Cross-Committee ───────────────────

def cross_committee_analysis(all_reports, all_matrices, out_dir):
    """Analyze countries across multiple committees."""
    if not all_reports:
        return
    
    combined = pd.concat(all_reports, ignore_index=True)
    
    # Country activity across committees
    activity = combined.groupby('country').agg(
        total_proposals=('total_proposals', 'sum'),
        committees_active=('committee', 'nunique'),
        avg_diversity=('topic_diversity', 'mean')
    ).sort_values('total_proposals', ascending=False)
    
    top30 = activity.head(30)
    
    fig = px.scatter(
        top30.reset_index(),
        x='committees_active',
        y='avg_diversity',
        size='total_proposals',
        text='country',
        title='Cross-Committee Country Activity Profile',
        labels={
            'committees_active': 'Committees Active In',
            'avg_diversity': 'Avg Topic Diversity',
            'total_proposals': 'Total Proposals'
        }
    )
    fig.update_traces(textposition='top center', textfont_size=8)
    fig.update_layout(width=900, height=600)
    
    path = os.path.join(out_dir, 'cross_committee_country_profile.html')
    fig.write_html(path)
    print(f"  Saved: {path}")
    
    # Save cross-committee summary
    path = os.path.join(out_dir, 'cross_committee_summary.csv')
    activity.to_csv(path, encoding='utf-8-sig')
    print(f"  Saved: {path}")


# ─────────────────── Main ───────────────────

def main():
    parser = argparse.ArgumentParser(description='Country/Organization Stance Analysis')
    parser.add_argument('--committees', nargs='+', default=['MEPC', 'MSC', 'CCC', 'SSE', 'ISWG-GHG'])
    parser.add_argument('--top-n', type=int, default=25, help='Top N countries to show')
    parser.add_argument('--min-docs', type=int, default=3, help='Min docs for a country to be included')
    parser.add_argument('--base-dir', default='output', help='Base output directory')
    parser.add_argument('--focus-countries', nargs='+', default=None,
                        help='Specific countries to highlight (e.g., China Japan Norway)')
    args = parser.parse_args()
    
    out_dir = os.path.join(args.base_dir, 'stance_analysis')
    os.makedirs(out_dir, exist_ok=True)
    
    all_reports = []
    all_matrices = {}
    
    # Key countries for maritime policy
    default_focus = ['China', 'Japan', 'Republic of Korea', 'Norway', 
                     'United States', 'United Kingdom', 'Germany', 'India',
                     'Marshall Islands', 'Brazil']
    focus = args.focus_countries or default_focus
    
    for committee in args.committees:
        print(f"\n{'='*60}")
        print(f"Analyzing {committee}...")
        print(f"{'='*60}")
        
        df = load_assignments(committee, args.base_dir)
        if df is None:
            continue
        
        topic_map = load_topic_info(committee, args.base_dir)
        
        print(f"  Loaded {len(df)} document assignments")
        print(f"  Topics: {len(topic_map)}")
        
        # Build matrix
        matrix, rec_df = build_country_topic_matrix(df, min_docs=args.min_docs)
        if matrix.empty:
            print(f"  Warning: empty matrix, skipping")
            continue
        
        print(f"  Active countries/orgs: {len(matrix)}")
        
        # Compute similarity
        sim_df = compute_country_similarity(matrix)
        
        # Committee-specific output dir
        comm_out = os.path.join(out_dir, committee)
        os.makedirs(comm_out, exist_ok=True)
        
        # Generate all visualizations
        print(f"\n  [1/5] Country-Topic Heatmap...")
        plot_country_topic_heatmap(matrix, topic_map, committee, comm_out, top_n=args.top_n)
        
        print(f"  [2/5] Country Similarity Network...")
        plot_country_similarity_network(sim_df, committee, comm_out, top_n=args.top_n, matrix=matrix)
        
        print(f"  [3/5] Country Focus Radar...")
        plot_country_radar(matrix, topic_map, committee, comm_out, focus_countries=focus)
        
        print(f"  [4/5] Temporal Shift Analysis...")
        plot_temporal_shift(rec_df, topic_map, committee, comm_out, focus_countries=focus)
        
        print(f"  [5/5] Stance Report...")
        report = generate_stance_report(matrix, topic_map, rec_df, committee, comm_out)
        if report is not None:
            all_reports.append(report)
        
        all_matrices[committee] = matrix
    
    # Cross-committee analysis
    if len(all_reports) > 1:
        print(f"\n{'='*60}")
        print(f"Cross-Committee Analysis...")
        print(f"{'='*60}")
        cross_committee_analysis(all_reports, all_matrices, out_dir)
    
    # Summary JSON
    summary = {
        'committees_analyzed': args.committees,
        'focus_countries': focus,
        'top_n': args.top_n,
        'min_docs': args.min_docs,
        'results': {}
    }
    for r in all_reports:
        comm = r['committee'].iloc[0]
        top5 = r.head(5)[['country', 'total_proposals', 'dominant_topic', 'topic_diversity']].to_dict('records')
        summary['results'][comm] = {
            'total_countries': len(r),
            'top_5_active': top5
        }
    
    path = os.path.join(out_dir, 'stance_analysis_summary.json')
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"\n  Saved summary: {path}")
    
    print("\nCountry Stance Analysis complete!")


if __name__ == '__main__':
    main()
