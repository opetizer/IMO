"""
文本相似度 & 引用网络分析 (Citation Network & Similarity Analysis)
================================================================
分析 IMO 会议文档间的引用关系和文本相似度，追踪政策提案的演化路径。

Features:
  1. 正则提取文档间引用 (MEPC xx/x/x, MSC xx/x/x, etc.)
  2. 构建有向引用网络 & 识别 hub documents
  3. TF-IDF 向量化 + 余弦相似度 → 文档聚类
  4. 追踪核心提案在会议间的演化链
  5. 交互式 Plotly 可视化

Usage:
    python src/citation_network.py --meeting_folder output/MEPC [--top_n_hubs 20] [--similarity_threshold 0.3]
"""

import os
import re
import json
import argparse
from collections import defaultdict, Counter

import numpy as np
import pandas as pd
import networkx as nx
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import AgglomerativeClustering

try:
    import plotly.graph_objects as go
    HAS_PLOTLY = True
except ImportError:
    HAS_PLOTLY = False

# Note: we read JSON directly instead of using json_read.load_data
# because load_data rebuilds full_text from content field, which
# doesn't exist in data_processed.json


# ───────────────────── helpers ─────────────────────

def natural_sort_key(name):
    """Sort strings with embedded numbers naturally."""
    m = re.search(r"(\d+)", str(name))
    return (0, int(m.group(1))) if m else (1, str(name))


def find_meeting_dirs(base_folder):
    """Find meeting subdirectories that contain processed data."""
    items = []
    for entry in os.listdir(base_folder):
        path = os.path.join(base_folder, entry)
        if os.path.isdir(path):
            for fname in ('data_processed.json', 'data_parsed.json', 'data.json'):
                if os.path.exists(os.path.join(path, fname)):
                    items.append(entry)
                    break
    return sorted(items, key=natural_sort_key)


def load_meeting_records(base_folder, meeting_dir):
    """Load raw JSON records from meeting directory (bypasses load_data to preserve full_text)."""
    for fname in ('data_processed.json', 'data_parsed.json', 'data.json'):
        path = os.path.join(base_folder, meeting_dir, fname)
        if os.path.exists(path):
            with open(path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            if isinstance(data, list):
                return data
            return []
    return []


def normalize_symbol(sym):
    """Normalize document symbol for matching: strip whitespace, uppercase."""
    if not sym:
        return ''
    return re.sub(r'\s+', ' ', sym.strip().upper())


def extract_meeting_session(symbol):
    """Extract meeting type and session number from symbol.
    e.g. 'MEPC 77/7/1' -> ('MEPC', 77)
    """
    m = re.match(r'(MEPC|MSC|CCC|SSE|ISWG[- ]?GHG)\s*(\d+)', symbol, re.IGNORECASE)
    if m:
        return m.group(1).upper().replace(' ', '-'), int(m.group(2))
    return None, None


# ───────────────────── citation extraction ─────────────────────

# Pattern to match document references like MEPC 77/7/1, MSC 102/3, ISWG-GHG 14/4/2
CITATION_PATTERN = re.compile(
    r'(?:MEPC|MSC|CCC|SSE|ISWG[- ]?GHG)\s+\d+/\d+(?:/\d+)?',
    re.IGNORECASE
)


def extract_citations(text):
    """Extract all document reference symbols from text."""
    if not text:
        return []
    matches = CITATION_PATTERN.findall(text)
    # Normalize
    return list(set(normalize_symbol(m) for m in matches))


# ───────────────────── data loading ─────────────────────

def load_all_documents(base_folder, meetings):
    """
    Load all documents from all meetings.
    Returns list of dicts with keys: symbol, title, meeting, full_text, originator, date
    """
    docs = []
    symbol_set = set()
    for m in meetings:
        records = load_meeting_records(base_folder, m)
        if not records:
            continue
        for row in records:
            symbol = normalize_symbol(row.get('Symbol', ''))
            if not symbol or symbol in symbol_set:
                continue
            symbol_set.add(symbol)
            
            full_text = row.get('full_text', '') or ''
            # Combine summary/introduction/annex if full_text is short
            if len(full_text) < 100:
                parts = []
                for field in ['Summary', 'Introduction', 'Annex_Content', 'Subject']:
                    v = row.get(field, '')
                    if v and isinstance(v, str):
                        parts.append(v)
                if parts:
                    full_text = ' '.join(parts)

            docs.append({
                'symbol': symbol,
                'title': row.get('Title', ''),
                'meeting': m,
                'full_text': full_text,
                'originator': row.get('Originator', ''),
                'originator_split': row.get('Originator_split', []),
                'date': row.get('Date', ''),
            })
    return docs


# ───────────────────── citation network ─────────────────────

def build_citation_network(docs):
    """
    Build a directed citation graph.
    Nodes = document symbols
    Edges = citing_doc -> cited_doc
    """
    # Build symbol lookup
    known_symbols = set(d['symbol'] for d in docs)

    G = nx.DiGraph()

    # Add all docs as nodes
    for d in docs:
        G.add_node(d['symbol'],
                   title=d['title'],
                   meeting=d['meeting'],
                   originator=d['originator'],
                   date=d['date'])

    # Extract citations and add edges
    citation_count = 0
    for d in docs:
        citing = d['symbol']
        text = d['full_text']
        refs = extract_citations(text)
        for ref in refs:
            if ref != citing:  # No self-citations
                # Check if the cited document exists in our corpus
                if ref in known_symbols:
                    G.add_edge(citing, ref)
                    citation_count += 1
                else:
                    # Still add as node (external reference)
                    if ref not in G:
                        G.add_node(ref, title='(external)', meeting='unknown',
                                   originator='', date='')
                    G.add_edge(citing, ref)
                    citation_count += 1

    return G, citation_count


def find_hub_documents(G, top_n=20):
    """Find top cited documents (highest in-degree)."""
    in_deg = sorted(G.in_degree(), key=lambda x: x[1], reverse=True)
    hubs = []
    for node, deg in in_deg[:top_n]:
        data = G.nodes[node]
        hubs.append({
            'symbol': node,
            'title': data.get('title', ''),
            'meeting': data.get('meeting', ''),
            'in_degree': deg,
            'out_degree': G.out_degree(node),
        })
    return hubs


def find_evolution_chains(G, docs, min_chain_length=3):
    """
    Find proposal evolution chains: sequences of documents on the same topic
    that cite each other across sessions.
    
    Strategy: For each document, follow its citation trail backwards and forwards
    to find chains of related documents across sessions.
    """
    # Group docs by title (agenda item)
    title_groups = defaultdict(list)
    for d in docs:
        title = d['title'].strip().upper()
        if title and title != 'LIST OF PARTICIPANTS':
            title_groups[title].append(d['symbol'])

    chains = []
    visited_chains = set()

    for title, symbols in title_groups.items():
        if len(symbols) < min_chain_length:
            continue

        # Sort by meeting session
        def sort_key(sym):
            _, sess = extract_meeting_session(sym)
            return sess or 0

        sorted_syms = sorted(symbols, key=sort_key)

        # Find citation links within this topic group
        sub_G = nx.DiGraph()
        for s in sorted_syms:
            sub_G.add_node(s)
        for s in sorted_syms:
            for cited in G.successors(s):
                if cited in sub_G:
                    sub_G.add_edge(s, cited)

        # Find connected components (weakly)
        for comp in nx.weakly_connected_components(sub_G):
            if len(comp) >= min_chain_length:
                chain_syms = sorted(comp, key=sort_key)
                chain_key = tuple(chain_syms)
                if chain_key not in visited_chains:
                    visited_chains.add(chain_key)
                    chains.append({
                        'title': title,
                        'length': len(chain_syms),
                        'documents': chain_syms,
                        'sessions': [G.nodes[s].get('meeting', '') for s in chain_syms],
                    })

    # Also find chains via direct citation paths (not limited by title)
    # Limit search to avoid O(N²) explosion on large graphs
    roots = [n for n in G.nodes() if G.in_degree(n) == 0 and G.out_degree(n) > 0]
    leaves = set(n for n in G.nodes() if G.in_degree(n) > 0 and G.out_degree(n) == 0)
    
    # Only search from a sample of roots to avoid excessive computation
    max_roots = 100
    for node in roots[:max_roots]:
        try:
            # Use cutoff to limit path length and search only reachable leaves
            for path in nx.all_simple_paths(G, node, leaves & nx.descendants(G, node), cutoff=8):
                if len(path) >= min_chain_length:
                    chain_key = tuple(path)
                    if chain_key not in visited_chains:
                        visited_chains.add(chain_key)
                        chains.append({
                            'title': '(citation chain)',
                            'length': len(path),
                            'documents': path,
                            'sessions': [G.nodes[s].get('meeting', '') for s in path],
                        })
        except nx.NetworkXError:
            pass

    chains.sort(key=lambda c: c['length'], reverse=True)
    return chains[:50]  # Top 50 chains


# ───────────────────── similarity analysis ─────────────────────

def compute_similarity(docs, threshold=0.3, max_features=5000):
    """
    Compute TF-IDF cosine similarity between documents.
    Returns similarity matrix and clustering labels.
    """
    texts = [d['full_text'] for d in docs]
    symbols = [d['symbol'] for d in docs]

    # Filter out empty texts
    valid_idx = [i for i, t in enumerate(texts) if len(t.strip()) > 50]
    valid_texts = [texts[i] for i in valid_idx]
    valid_symbols = [symbols[i] for i in valid_idx]

    if len(valid_texts) < 5:
        return None, None, valid_symbols, valid_idx

    print(f'  Computing TF-IDF for {len(valid_texts)} documents...')
    vectorizer = TfidfVectorizer(
        max_features=max_features,
        stop_words='english',
        min_df=2,
        max_df=0.95,
        ngram_range=(1, 2),
    )
    tfidf_matrix = vectorizer.fit_transform(valid_texts)

    print(f'  Computing cosine similarity...')
    sim_matrix = cosine_similarity(tfidf_matrix)

    # Clustering
    print(f'  Running hierarchical clustering...')
    n_clusters = min(max(5, len(valid_texts) // 20), 30)
    clustering = AgglomerativeClustering(
        n_clusters=n_clusters,
        metric='cosine',
        linkage='average',
    )
    # Convert sparse to dense for clustering
    labels = clustering.fit_predict(tfidf_matrix.toarray())

    return sim_matrix, labels, valid_symbols, valid_idx


def find_similar_pairs(sim_matrix, symbols, threshold=0.5, top_n=50):
    """Find top similar document pairs above threshold."""
    pairs = []
    n = len(symbols)
    for i in range(n):
        for j in range(i + 1, n):
            if sim_matrix[i, j] >= threshold:
                pairs.append({
                    'doc_a': symbols[i],
                    'doc_b': symbols[j],
                    'similarity': round(float(sim_matrix[i, j]), 4),
                })
    pairs.sort(key=lambda x: x['similarity'], reverse=True)
    return pairs[:top_n]


# ───────────────────── visualization ─────────────────────

def draw_citation_network(G, hubs, out_path, title='Citation Network', top_n=50):
    """Draw static citation network (top cited docs)."""
    # Subgraph: only include top hubs and their neighbors
    hub_symbols = set(h['symbol'] for h in hubs[:top_n])
    relevant_nodes = set()
    for h in hub_symbols:
        relevant_nodes.add(h)
        for pred in G.predecessors(h):
            relevant_nodes.add(pred)
    
    sub_G = G.subgraph(relevant_nodes).copy()

    fig, ax = plt.subplots(1, 1, figsize=(20, 16))
    fig.patch.set_facecolor('white')
    ax.set_facecolor('white')

    if len(sub_G) == 0:
        ax.text(0.5, 0.5, 'No data', ha='center', va='center', fontsize=20)
        fig.savefig(out_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        return

    # Layout
    pos = nx.spring_layout(sub_G, k=1.5 / (len(sub_G) ** 0.5 + 1),
                           iterations=60, seed=42)

    # Color by meeting
    meetings = list(set(sub_G.nodes[n].get('meeting', 'unknown') for n in sub_G.nodes()))
    meetings.sort(key=natural_sort_key)
    cmap = plt.colormaps.get_cmap('tab20').resampled(max(len(meetings), 2))
    meeting_color = {m: cmap(i) for i, m in enumerate(meetings)}

    node_colors = [meeting_color.get(sub_G.nodes[n].get('meeting', 'unknown'), 'grey')
                   for n in sub_G.nodes()]

    # Node sizes by in-degree
    in_degrees = dict(sub_G.in_degree())
    sizes = np.array([in_degrees.get(n, 0) for n in sub_G.nodes()], dtype=float)
    sizes = 50 + 500 * (sizes / (sizes.max() + 1e-9))

    # Draw
    nx.draw_networkx_edges(sub_G, pos, alpha=0.15, arrows=True,
                           arrowsize=8, edge_color='grey', ax=ax)
    nx.draw_networkx_nodes(sub_G, pos, node_color=node_colors, node_size=sizes,
                           edgecolors='#333', linewidths=0.5, alpha=0.85, ax=ax)

    # Label only hubs
    hub_labels = {n: n for n in sub_G.nodes() if n in hub_symbols}
    nx.draw_networkx_labels(sub_G, pos, labels=hub_labels, font_size=6,
                            font_weight='bold', ax=ax)

    # Legend
    patches = [mpatches.Patch(color=meeting_color[m], label=m) for m in meetings if m != 'unknown']
    if patches:
        ax.legend(handles=patches, loc='lower left', fontsize=7, framealpha=0.8, ncol=2)

    ax.set_title(title, fontsize=14, fontweight='bold', pad=15)
    ax.axis('off')

    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'  Saved: {out_path}')


def draw_interactive_citation(G, hubs, out_path, title='Citation Network', top_n=30):
    """Generate interactive Plotly citation network."""
    if not HAS_PLOTLY:
        print('  Plotly not available, skipping interactive visualization.')
        return

    hub_symbols = set(h['symbol'] for h in hubs[:top_n])
    relevant_nodes = set()
    for h in hub_symbols:
        relevant_nodes.add(h)
        for pred in G.predecessors(h):
            relevant_nodes.add(pred)
        for succ in G.successors(h):
            relevant_nodes.add(succ)

    sub_G = G.subgraph(relevant_nodes).copy()
    if len(sub_G) == 0:
        return

    pos = nx.spring_layout(sub_G, k=1.2 / (len(sub_G) ** 0.5 + 1),
                           iterations=80, seed=42)

    # Edges
    edge_x, edge_y = [], []
    for u, v in sub_G.edges():
        x0, y0 = pos[u]
        x1, y1 = pos[v]
        edge_x += [x0, x1, None]
        edge_y += [y0, y1, None]

    edge_trace = go.Scatter(
        x=edge_x, y=edge_y, mode='lines',
        line=dict(width=0.5, color='#888'),
        hoverinfo='none',
    )

    # Nodes
    node_x = [pos[n][0] for n in sub_G.nodes()]
    node_y = [pos[n][1] for n in sub_G.nodes()]
    in_degrees = dict(sub_G.in_degree())
    node_sizes = [5 + 3 * in_degrees.get(n, 0) for n in sub_G.nodes()]
    node_text = []
    for n in sub_G.nodes():
        d = sub_G.nodes[n]
        t = d.get('title', '')[:60]
        node_text.append(f"{n}<br>{t}<br>In-degree: {in_degrees.get(n,0)}")

    # Color by meeting
    meetings_list = sorted(set(sub_G.nodes[n].get('meeting', 'unknown') for n in sub_G.nodes()),
                           key=natural_sort_key)
    meeting_idx = {m: i for i, m in enumerate(meetings_list)}
    node_color = [meeting_idx.get(sub_G.nodes[n].get('meeting', 'unknown'), 0)
                  for n in sub_G.nodes()]

    node_trace = go.Scatter(
        x=node_x, y=node_y, mode='markers+text',
        text=[n if n in hub_symbols else '' for n in sub_G.nodes()],
        textposition='top center',
        textfont=dict(size=7),
        hovertext=node_text,
        hoverinfo='text',
        marker=dict(
            size=node_sizes,
            color=node_color,
            colorscale='Viridis',
            showscale=True,
            colorbar=dict(title='Meeting Session'),
            line=dict(width=0.5, color='#333'),
        ),
    )

    fig = go.Figure(data=[edge_trace, node_trace],
                    layout=go.Layout(
                        title=dict(text=title, font=dict(size=16)),
                        showlegend=False,
                        hovermode='closest',
                        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                        width=1200, height=900,
                        template='plotly_white',
                    ))
    fig.write_html(out_path)
    print(f'  Saved: {out_path}')


def draw_similarity_heatmap(sim_matrix, symbols, labels, out_path, title='Document Similarity'):
    """Draw similarity heatmap grouped by cluster."""
    if sim_matrix is None:
        return

    # Sort by cluster label
    order = np.argsort(labels)
    sorted_matrix = sim_matrix[order][:, order]
    sorted_symbols = [symbols[i] for i in order]

    n = len(sorted_symbols)
    fig_size = max(10, min(30, n // 10))

    fig, ax = plt.subplots(1, 1, figsize=(fig_size, fig_size))
    im = ax.imshow(sorted_matrix, cmap='YlOrRd', vmin=0, vmax=1, aspect='auto')
    plt.colorbar(im, ax=ax, shrink=0.6, label='Cosine Similarity')

    # Only show labels if not too many
    if n <= 80:
        ax.set_xticks(range(n))
        ax.set_yticks(range(n))
        ax.set_xticklabels(sorted_symbols, rotation=90, fontsize=4)
        ax.set_yticklabels(sorted_symbols, fontsize=4)
    else:
        ax.set_xticks([])
        ax.set_yticks([])

    ax.set_title(title, fontsize=14, fontweight='bold')
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'  Saved: {out_path}')


# ───────────────────── statistics ─────────────────────

def save_analysis(G, hubs, chains, similar_pairs, cluster_info, out_path):
    """Save comprehensive analysis to JSON."""
    analysis = {
        'network_summary': {
            'total_nodes': G.number_of_nodes(),
            'total_edges': G.number_of_edges(),
            'internal_nodes': sum(1 for n in G.nodes() if G.nodes[n].get('meeting', 'unknown') != 'unknown'),
            'external_references': sum(1 for n in G.nodes() if G.nodes[n].get('meeting', 'unknown') == 'unknown'),
            'avg_in_degree': round(sum(d for _, d in G.in_degree()) / max(G.number_of_nodes(), 1), 2),
            'avg_out_degree': round(sum(d for _, d in G.out_degree()) / max(G.number_of_nodes(), 1), 2),
            'density': round(nx.density(G), 6),
        },
        'hub_documents': hubs,
        'evolution_chains': chains[:20],
        'similar_document_pairs': similar_pairs[:30],
        'cluster_info': cluster_info,
    }

    # Degree distribution
    in_deg_dist = Counter(d for _, d in G.in_degree())
    analysis['in_degree_distribution'] = dict(sorted(in_deg_dist.items()))

    # Meeting-level citation stats
    meeting_stats = defaultdict(lambda: {'docs': 0, 'internal_citations': 0, 'external_citations': 0})
    for node in G.nodes():
        meeting = G.nodes[node].get('meeting', 'unknown')
        if meeting != 'unknown':
            meeting_stats[meeting]['docs'] += 1
            for cited in G.successors(node):
                if G.nodes[cited].get('meeting', 'unknown') == meeting:
                    meeting_stats[meeting]['internal_citations'] += 1
                else:
                    meeting_stats[meeting]['external_citations'] += 1
    analysis['meeting_citation_stats'] = dict(meeting_stats)

    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(analysis, f, ensure_ascii=False, indent=2)
    print(f'  Saved: {out_path}')


# ───────────────────── main ─────────────────────

def main():
    parser = argparse.ArgumentParser(description='Citation network & similarity analysis for IMO documents')
    parser.add_argument('--meeting_folder', type=str, required=True,
                        help='Path to meeting output folder (e.g. output/MEPC)')
    parser.add_argument('--top_n_hubs', type=int, default=20,
                        help='Number of top hub documents to highlight')
    parser.add_argument('--similarity_threshold', type=float, default=0.4,
                        help='Cosine similarity threshold for similar pairs')
    parser.add_argument('--min_chain_length', type=int, default=3,
                        help='Minimum chain length for evolution tracking')
    parser.add_argument('--out_folder', type=str, default=None)
    args = parser.parse_args()

    base = args.meeting_folder
    meetings = find_meeting_dirs(base)
    if not meetings:
        print(f'No meeting dirs found in {base}')
        return

    out_folder = args.out_folder or base
    os.makedirs(out_folder, exist_ok=True)
    prefix = os.path.basename(base).upper()

    print(f'Meetings: {meetings}')
    print(f'Output: {out_folder}')

    # ── Step 1: Load all documents ──
    print('\n[1/5] Loading documents...')
    docs = load_all_documents(base, meetings)
    print(f'  Loaded {len(docs)} documents')

    # ── Step 2: Build citation network ──
    print('\n[2/5] Building citation network...')
    G, citation_count = build_citation_network(docs)
    print(f'  Nodes: {G.number_of_nodes()}, Edges: {G.number_of_edges()} ({citation_count} citations)')

    hubs = find_hub_documents(G, top_n=args.top_n_hubs)
    print(f'  Top hub: {hubs[0]["symbol"]} (cited {hubs[0]["in_degree"]} times)' if hubs else '  No hubs found')

    # ── Step 3: Find evolution chains ──
    print('\n[3/5] Finding evolution chains...')
    chains = find_evolution_chains(G, docs, min_chain_length=args.min_chain_length)
    print(f'  Found {len(chains)} evolution chains')
    if chains:
        print(f'  Longest: {chains[0]["length"]} docs on "{chains[0]["title"][:60]}"')

    # ── Step 4: Similarity analysis ──
    print('\n[4/5] Computing document similarity...')
    sim_matrix, labels, valid_symbols, valid_idx = compute_similarity(
        docs, threshold=args.similarity_threshold
    )

    similar_pairs = []
    cluster_info = {}
    if sim_matrix is not None:
        similar_pairs = find_similar_pairs(sim_matrix, valid_symbols,
                                           threshold=args.similarity_threshold)
        print(f'  Found {len(similar_pairs)} similar pairs (threshold={args.similarity_threshold})')

        # Cluster statistics
        n_clusters = len(set(labels))
        cluster_sizes = Counter(labels)
        cluster_info = {
            'n_clusters': n_clusters,
            'cluster_sizes': {f'cluster_{k}': v for k, v in sorted(cluster_sizes.items())},
            'cluster_samples': {},
        }
        for cid in sorted(set(labels)):
            members = [valid_symbols[i] for i in range(len(labels)) if labels[i] == cid]
            cluster_info['cluster_samples'][f'cluster_{cid}'] = members[:5]
        print(f'  {n_clusters} clusters found')

    # ── Step 5: Visualization & output ──
    print('\n[5/5] Generating visualizations...')

    # Static citation network
    draw_citation_network(
        G, hubs,
        os.path.join(out_folder, f'citation_network_{prefix}.png'),
        title=f'{prefix} Document Citation Network',
        top_n=args.top_n_hubs,
    )

    # Interactive citation network
    draw_interactive_citation(
        G, hubs,
        os.path.join(out_folder, f'citation_network_{prefix}.html'),
        title=f'{prefix} Interactive Citation Network',
        top_n=args.top_n_hubs,
    )

    # Similarity heatmap
    if sim_matrix is not None:
        draw_similarity_heatmap(
            sim_matrix, valid_symbols, labels,
            os.path.join(out_folder, f'similarity_clusters_{prefix}.png'),
            title=f'{prefix} Document Similarity Clusters',
        )

    # Save analysis JSON
    save_analysis(
        G, hubs, chains, similar_pairs, cluster_info,
        os.path.join(out_folder, f'citation_analysis_{prefix}.json'),
    )

    print('\nDone!')


if __name__ == '__main__':
    main()
