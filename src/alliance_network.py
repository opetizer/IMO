"""
国家立场网络分析 (Alliance / Co-sponsorship Network)
=====================================================
利用提案的 Originator_split 共同提交关系，构建国家/组织间的合作网络图。

Usage:
    python src/alliance_network.py --meeting_folder output/MEPC [--top_n 25] [--min_weight 2]
"""

import os
import re
import json
import argparse
from collections import defaultdict, Counter
from itertools import combinations

import pandas as pd
import networkx as nx
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

from json_read import load_data


# ───────────────────── helpers ─────────────────────

def natural_sort_key(name):
    m = re.search(r"(\d+)", name)
    return int(m.group(1)) if m else name


def find_meeting_dirs(base_folder):
    items = []
    for entry in os.listdir(base_folder):
        path = os.path.join(base_folder, entry)
        if os.path.isdir(path):
            for fname in ('data_processed.json', 'data_parsed.json', 'data.json'):
                if os.path.exists(os.path.join(path, fname)):
                    items.append(entry)
                    break
    return sorted(items, key=natural_sort_key)


def load_meeting_df(base_folder, meeting_dir):
    for fname in ('data_processed.json', 'data_parsed.json', 'data.json'):
        path = os.path.join(base_folder, meeting_dir, fname)
        if os.path.exists(path):
            return load_data(path)
    return pd.DataFrame()


EXCLUDE = {'Secretariat', 'Unknown', 'Other', ''}


def extract_originators(row):
    """Extract list of originators from a row, handling both list and string fields."""
    # Prefer Originator_split if it exists
    osplit = row.get('Originator_split', None)
    if isinstance(osplit, list) and len(osplit) > 0:
        return [o.strip() for o in osplit if o and o.strip() and o.strip() not in EXCLUDE]

    raw = row.get('Originator', '')
    if not raw or not isinstance(raw, str):
        return []
    parts = re.split(r'[;,/|]|\band\b', raw)
    return [p.strip() for p in parts if p and p.strip() and p.strip() not in EXCLUDE]


# ───────────────────── core ─────────────────────

def build_cooccurrence(meetings, base_folder, title_filter=None):
    """
    Build co-sponsorship edges.
    Returns:
        edges: Counter of (a, b) -> weight  (alphabetically sorted tuples)
        solo:  Counter of country -> solo submission count
        docs_per_country: Counter of country -> total docs
    """
    edges = Counter()
    solo = Counter()
    docs_per_country = Counter()
    title_per_country = defaultdict(Counter)  # country -> {title -> count}

    for m in meetings:
        df = load_meeting_df(base_folder, m)
        if df.empty:
            continue
        for _, row in df.iterrows():
            title = row.get('Title', '')
            if title_filter and title_filter.lower() not in str(title).lower():
                continue
            authors = extract_originators(row)
            if not authors:
                continue

            for a in authors:
                docs_per_country[a] += 1
                title_per_country[a][title] += 1

            if len(authors) == 1:
                solo[authors[0]] += 1
            else:
                for a, b in combinations(sorted(set(authors)), 2):
                    edges[(a, b)] += 1

    return edges, solo, docs_per_country, title_per_country


def build_graph(edges, docs_per_country, top_n=30, min_weight=1):
    """Build networkx graph from edges, keeping only top_n countries by doc count."""
    top_countries = set(c for c, _ in docs_per_country.most_common(top_n))

    G = nx.Graph()
    for (a, b), w in edges.items():
        if a in top_countries and b in top_countries and w >= min_weight:
            G.add_edge(a, b, weight=w)

    # Add isolated top countries that had no qualifying edges
    for c in top_countries:
        if c not in G:
            G.add_node(c)

    # Node attributes
    for n in G.nodes():
        G.nodes[n]['doc_count'] = docs_per_country.get(n, 0)

    return G


def detect_communities(G):
    """Louvain community detection, returns dict node -> community_id."""
    try:
        from community import community_louvain
        partition = community_louvain.best_partition(G, weight='weight', random_state=42)
    except ImportError:
        # fallback: greedy modularity
        communities = nx.community.greedy_modularity_communities(G, weight='weight')
        partition = {}
        for cid, members in enumerate(communities):
            for m in members:
                partition[m] = cid
    return partition


def draw_network(G, partition, out_path, title='Co-sponsorship Network'):
    """Draw and save network visualization."""
    fig, ax = plt.subplots(1, 1, figsize=(16, 12))
    fig.patch.set_facecolor('white')
    ax.set_facecolor('white')

    if len(G) == 0:
        ax.text(0.5, 0.5, 'No data', ha='center', va='center', fontsize=20)
        fig.savefig(out_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        return

    # Layout
    pos = nx.spring_layout(G, k=2.0 / (len(G) ** 0.5 + 1), iterations=80, seed=42, weight='weight')

    # Community colors
    n_comm = max(partition.values()) + 1 if partition else 1
    cmap = plt.cm.get_cmap('tab20', max(n_comm, 2))
    node_colors = [cmap(partition.get(n, 0)) for n in G.nodes()]

    # Node sizes proportional to doc count
    sizes = np.array([G.nodes[n].get('doc_count', 1) for n in G.nodes()], dtype=float)
    sizes = 200 + 1500 * (sizes / (sizes.max() + 1e-9))

    # Edge widths proportional to weight
    edge_weights = [G[u][v]['weight'] for u, v in G.edges()]
    max_w = max(edge_weights) if edge_weights else 1
    edge_widths = [0.5 + 4.0 * (w / max_w) for w in edge_weights]
    edge_alphas = [0.2 + 0.6 * (w / max_w) for w in edge_weights]

    # Draw edges
    for (u, v), width, alpha in zip(G.edges(), edge_widths, edge_alphas):
        x = [pos[u][0], pos[v][0]]
        y = [pos[u][1], pos[v][1]]
        ax.plot(x, y, '-', color='grey', linewidth=width, alpha=alpha, zorder=1)

    # Draw nodes
    nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=sizes,
                           edgecolors='#333', linewidths=0.8, alpha=0.9, ax=ax)

    # Labels
    nx.draw_networkx_labels(G, pos, font_size=8, font_weight='bold', ax=ax)

    # Edge weight labels (only for strong edges)
    strong_edges = {(u, v): d['weight'] for u, v, d in G.edges(data=True) if d['weight'] >= max_w * 0.3}
    if strong_edges:
        nx.draw_networkx_edge_labels(G, pos, edge_labels=strong_edges, font_size=7, font_color='#555', ax=ax)

    # Legend for communities
    comm_labels = defaultdict(list)
    for n, cid in partition.items():
        comm_labels[cid].append(n)
    patches = []
    for cid in sorted(comm_labels.keys()):
        members = sorted(comm_labels[cid])
        label = f"Group {cid + 1}: {', '.join(members[:4])}" + ('...' if len(members) > 4 else '')
        patches.append(mpatches.Patch(color=cmap(cid), label=label))
    if patches:
        ax.legend(handles=patches, loc='lower left', fontsize=7, framealpha=0.8)

    ax.set_title(title, fontsize=14, fontweight='bold', pad=15)
    ax.axis('off')

    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'  Saved: {out_path}')


def save_stats(G, partition, docs_per_country, title_per_country, edges, out_path):
    """Save analysis statistics to JSON."""
    stats = {
        'summary': {
            'total_countries': len(G),
            'total_edges': G.number_of_edges(),
            'total_co_sponsorships': sum(d['weight'] for _, _, d in G.edges(data=True)),
            'n_communities': len(set(partition.values())) if partition else 0,
        },
        'communities': {},
        'top_pairs': [],
        'country_stats': [],
    }

    # Communities
    comm_members = defaultdict(list)
    for n, cid in partition.items():
        comm_members[cid].append(n)
    for cid in sorted(comm_members.keys()):
        stats['communities'][f'group_{cid + 1}'] = sorted(comm_members[cid])

    # Top co-sponsorship pairs
    for (a, b), w in edges.most_common(30):
        if a in G and b in G:
            stats['top_pairs'].append({'pair': [a, b], 'weight': w})

    # Country stats
    for n in sorted(G.nodes(), key=lambda x: docs_per_country.get(x, 0), reverse=True):
        degree = G.degree(n, weight='weight')
        top_titles = title_per_country[n].most_common(5)
        stats['country_stats'].append({
            'country': n,
            'total_docs': docs_per_country.get(n, 0),
            'weighted_degree': degree,
            'community': partition.get(n, -1) + 1,
            'top_topics': [{'title': t, 'count': c} for t, c in top_titles],
        })

    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)
    print(f'  Saved: {out_path}')


# ───────────────────── per-topic networks ─────────────────────

TOPIC_KEYWORDS = {
    'GHG': 'ghg',
    'BALLAST_WATER': 'ballast',
    'ENERGY_EFFICIENCY': 'energy efficiency',
    'AIR_POLLUTION': 'air pollution',
    'MARINE_PLASTIC': 'plastic',
}


# ───────────────────── main ─────────────────────

def main():
    parser = argparse.ArgumentParser(description='Build co-sponsorship / alliance network from IMO meeting data')
    parser.add_argument('--meeting_folder', type=str, required=True)
    parser.add_argument('--top_n', type=int, default=25, help='Number of top countries to include')
    parser.add_argument('--min_weight', type=int, default=1, help='Minimum co-sponsorship count for an edge')
    parser.add_argument('--out_folder', type=str, default=None)
    parser.add_argument('--per_topic', action='store_true', help='Also generate per-topic networks')
    args = parser.parse_args()

    base = args.meeting_folder
    meetings = find_meeting_dirs(base)
    if not meetings:
        print(f'No meeting dirs found in {base}')
        return

    out_folder = args.out_folder or base
    os.makedirs(out_folder, exist_ok=True)

    print(f'Meetings: {meetings}')
    print(f'Building overall co-sponsorship network...')

    # Overall network
    edges, solo, docs_per_country, title_per_country = build_cooccurrence(meetings, base)
    G = build_graph(edges, docs_per_country, top_n=args.top_n, min_weight=args.min_weight)
    partition = detect_communities(G)

    prefix = os.path.basename(base).upper()
    draw_network(G, partition, os.path.join(out_folder, f'alliance_network_{prefix}.png'),
                 title=f'{prefix} Co-sponsorship Network (MEPC sessions)')
    save_stats(G, partition, docs_per_country, title_per_country, edges,
               os.path.join(out_folder, f'alliance_stats_{prefix}.json'))

    # Per-topic networks
    if args.per_topic:
        print(f'\nBuilding per-topic networks...')
        for label, keyword in TOPIC_KEYWORDS.items():
            print(f'  Topic: {label} (keyword: "{keyword}")')
            t_edges, t_solo, t_docs, t_titles = build_cooccurrence(meetings, base, title_filter=keyword)
            t_G = build_graph(t_edges, t_docs, top_n=args.top_n, min_weight=1)
            if len(t_G) < 2:
                print(f'    Skipped (not enough nodes)')
                continue
            t_part = detect_communities(t_G)
            draw_network(t_G, t_part, os.path.join(out_folder, f'alliance_network_{prefix}_{label}.png'),
                         title=f'{prefix} Co-sponsorship: {label}')

    print('\nDone!')


if __name__ == '__main__':
    main()
