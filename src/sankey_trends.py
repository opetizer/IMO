#####
# Sankey diagram generation for document trends across meetings
# 单一议题的文档流向桑基图可视化
#
# Usage:
# python sankey_trends.py --meeting_folder <meeting_folder> [options]
#
# example:
# python src/sankey_trends.py --meeting_folder "output/MEPC" --top_k_titles 6 --top_n_countries 8 --min_total_count 1 --normalize --flow_method min
#####

import os
import re
import json
import argparse
from collections import defaultdict, Counter
from operator import itemgetter
from pprint import pprint

import pandas as pd
import plotly.graph_objects as go

from json_read import load_data


def natural_meeting_sort_key(name):
    """Sort meeting names naturally (e.g., MEPC 77, MEPC 78, ...)"""
    m = re.search(r"(\d+)", name)
    if m:
        return int(m.group(1))
    return name


def find_meeting_dirs(base_folder):
    """Find all meeting subdirectories containing data_parsed.json or data.json"""
    items = []
    for entry in os.listdir(base_folder):
        path = os.path.join(base_folder, entry)
        if os.path.isdir(path):
            # check for data_parsed.json first, then data.json
            if os.path.exists(os.path.join(path, 'data_parsed.json')) or \
               os.path.exists(os.path.join(path, 'data.json')):
                items.append(entry)
    items_sorted = sorted(items, key=natural_meeting_sort_key)
    return items_sorted


def load_meeting_df(base_folder, meeting_dir):
    """Load data from a meeting's data_parsed.json file (fallback to data.json)"""
    # Try data_parsed.json first, fallback to data.json
    path = os.path.join(base_folder, meeting_dir, 'data_parsed.json')
    if not os.path.exists(path):
        path = os.path.join(base_folder, meeting_dir, 'data.json')
    if not os.path.exists(path):
        return pd.DataFrame()
    df = load_data(path)
    # add meeting dir as column
    df['meeting_dir'] = meeting_dir
    return df


def normalize_country_field(originator_raw):
    """Normalize and extract country from Originator field"""
    if not originator_raw or not isinstance(originator_raw, str):
        return None
    # split on common separators and take stripped tokens
    parts = re.split(r"[;,/|]", originator_raw)
    parts = [p.strip() for p in parts if p and p.strip()]
    if not parts:
        return None
    # return first as canonical
    return parts[0]


def build_counts(meetings_ordered, base_folder, title_field='Title', country_field='Originator', exclude=None):
    """
    Build counts: counts[title][meeting][country] = count

    Args:
        exclude: list of countries to exclude (e.g., ['Secretariat', 'Unknown', 'Other'])
    """
    if exclude is None:
        exclude = []

    # counts[title][meeting][country] = count
    counts = defaultdict(lambda: defaultdict(Counter))
    for m in meetings_ordered:
        df = load_meeting_df(base_folder, m)
        if df.empty:
            continue
        for _, row in df.iterrows():
            title = row.get(title_field, '')
            if not title or not isinstance(title, str):
                continue
            title = ' '.join(title.split())
            origin = row.get(country_field, '')
            country = normalize_country_field(origin) or 'Unknown'

            # Skip excluded countries
            if country in exclude:
                continue

            counts[title][m][country] += 1
    return counts


def select_top_titles(counts, top_k=6, min_total=1):
    """Select top titles by total document count"""
    totals = []
    for title, meetdict in counts.items():
        total = sum(sum(c.values()) for c in meetdict.values())
        totals.append((title, total))
    totals_sorted = sorted(totals, key=itemgetter(1), reverse=True)
    selected = [t for t, s in totals_sorted if s >= min_total][:top_k]
    return selected


def compute_sankey_components(meetings_ordered, counts_for_title, top_n_countries=8, normalize=False, flow_method='min'):
    """Compute deterministic nodes, positions and links for a Sankey chart."""
    def canonical_meeting_label(name):
        m = re.search(r"^(\D*?)\s*(\d+)", name)
        if m:
            prefix = m.group(1).strip()
            num = m.group(2)
            return f"{prefix} {num}".strip()
        return name

    per_meeting_counts = {}
    meetings_grouped = []
    for m in meetings_ordered:
        g = canonical_meeting_label(m)
        cdict = counts_for_title.get(m, {})
        if g not in per_meeting_counts:
            per_meeting_counts[g] = Counter()
            meetings_grouped.append(g)
        per_meeting_counts[g].update(cdict)

    agg = Counter()
    for g in meetings_grouped:
        agg.update(per_meeting_counts.get(g, {}))
    top_countries = [c for c, _ in agg.most_common(top_n_countries)]

    for g in meetings_grouped:
        cdict = per_meeting_counts.get(g, {})
        keep = {k: v for k, v in cdict.items() if k in top_countries}
        # Removed: aggregation of non-top countries into 'Other'
        per_meeting_counts[g] = keep

    per_meeting_percent = {}
    if normalize:
        for g in meetings_grouped:
            total = sum(per_meeting_counts.get(g, {}).values())
            if total > 0:
                per_meeting_percent[g] = {c: (v / total * 100.0) for c, v in per_meeting_counts[g].items()}
            else:
                per_meeting_percent[g] = {}

    countries_union = [c for c in top_countries if any(c in per_meeting_counts[g] for g in meetings_grouped)]

    nodes = []
    node_idx = {}
    node_meta_counts = []
    nodes_per_meeting = {g: [] for g in meetings_grouped}
    for g in meetings_grouped:
        for c in countries_union:
            if c in per_meeting_counts.get(g, {}):
                name = f"{g} - {c}"
                node_idx[name] = len(nodes)
                nodes.append(name)
                node_meta_counts.append(per_meeting_counts[g][c])
                nodes_per_meeting[g].append((name, per_meeting_counts[g][c]))

    sources = []
    targets = []
    values = []
    link_countries = []
    for i in range(len(meetings_grouped) - 1):
        g1 = meetings_grouped[i]
        g2 = meetings_grouped[i + 1]
        c1 = per_meeting_counts.get(g1, {})
        c2 = per_meeting_counts.get(g2, {})
        for country in countries_union:
            n1 = f"{g1} - {country}"
            n2 = f"{g2} - {country}"
            if n1 not in node_idx or n2 not in node_idx:
                continue
            v1 = c1.get(country, 0)
            v2 = c2.get(country, 0)
            if flow_method == 'min':
                val_raw = min(v1, v2)
            elif flow_method == 'avg':
                val_raw = (v1 + v2) / 2.0
            else:
                val_raw = v2
            if val_raw and val_raw > 0:
                if normalize:
                    total2 = sum(per_meeting_counts.get(g2, {}).values())
                    val = (v2 / total2 * 100.0) if total2 > 0 else 0
                else:
                    val = val_raw
                sources.append(node_idx[n1])
                targets.append(node_idx[n2])
                values.append(val)
                link_countries.append(country)

    n_meet = len(meetings_grouped)
    node_x = [0.0] * len(nodes)
    node_y = [0.5] * len(nodes)
    gap = 0.01
    meeting_x = {g: (idx / (n_meet - 1) if n_meet > 1 else 0.5) for idx, g in enumerate(meetings_grouped)}

    # Calculate global max count for proper scaling
    global_max_count = max(node_meta_counts) if node_meta_counts else 1

    for g in meetings_grouped:
        items = nodes_per_meeting.get(g, [])
        if not items:
            continue
        n_items = len(items)
        x_pos = meeting_x[g]
        total = sum(cnt for _, cnt in items)
        if total <= 0:
            for j, (name, _) in enumerate(items):
                ni = node_idx[name]
                node_x[ni] = x_pos
                node_y[ni] = (j + 1) / (n_items + 1)
            continue

        # Calculate initial heights
        raw_heights = [(cnt / global_max_count) * 0.8 for _, cnt in items]
        total_height = sum(raw_heights) + gap * (len(items) + 1)

        # Scale down if total height exceeds 0.9 to prevent going out of bounds
        scale_factor = min(1.0, 0.9 / total_height) if total_height > 0 else 1.0

        cursor = gap
        for (name, cnt), raw_h in zip(items, raw_heights):
            ni = node_idx[name]
            # Scale each node's height
            height = raw_h * scale_factor
            center = cursor + height / 2.0
            node_x[ni] = x_pos
            node_y[ni] = center
            cursor += height + gap

    return {
        'nodes': nodes,
        'node_meta_counts': node_meta_counts,
        'node_x': node_x,
        'node_y': node_y,
        'sources': sources,
        'targets': targets,
        'values': values,
        'link_countries': link_countries,
        'meetings_grouped': meetings_grouped,
        'per_meeting_percent': per_meeting_percent
    }


def _extract_country_from_node(node_name):
    """Extract country name from node name format 'MEETING - COUNTRY'."""
    return node_name.split(' - ', 1)[1]


def _hex_to_rgba(hexcolor, alpha=1.0):
    """Convert hex color to rgba string with specified alpha."""
    if hexcolor.startswith('rgba') or hexcolor.startswith('rgb'):
        return hexcolor
    h = hexcolor.lstrip('#')
    lv = len(h)
    r, g, b = tuple(int(h[i:i+lv//3], 16) for i in range(0, lv, lv//3))
    return f'rgba({r},{g},{b},{alpha})'


def _create_color_map(nodes):
    """Create a mapping of countries to colors."""
    import plotly.express as px
    palettes = px.colors.qualitative.Plotly
    country_unique = sorted(set(_extract_country_from_node(n) for n in nodes))
    return {c: palettes[i % len(palettes)] for i, c in enumerate(country_unique)}


def _build_node_labels_and_hover(nodes, node_meta_counts, per_meeting_percent, normalize):
    """Build labels and hover text for nodes."""
    labels = []
    hover_text = []

    for node_name, count in zip(nodes, node_meta_counts):
        meeting, country = node_name.split(' - ', 1)

        if normalize:
            pct = per_meeting_percent.get(meeting, {}).get(country, 0)
            labels.append(f"{country}<br>{pct:.1f}% ({count})")
            hover_text.append(f"{meeting}<br>{country}<br>{count} docs ({pct:.1f}%)")
        else:
            labels.append(f"{country}<br>{count}")
            hover_text.append(f"{meeting}<br>{country}<br>{count} documents")

    return labels, hover_text


def _build_link_hover(link_countries, values):
    """Build hover text for links."""
    return [f"{country}<br>{value:.1f}" for country, value in zip(link_countries, values)]


def build_sankey_for_title(title, meetings_ordered, counts_for_title, top_n_countries=8, out_folder='output', out_name=None, normalize=False, flow_method='min', node_pad=12, node_thickness=22):
    """Build a Sankey diagram visualization for a single title.

    Parameters:
    -----------
    title : str
        The document title to visualize
    meetings_ordered : list
        List of meeting directories in order
    counts_for_title : dict
        Count data for this title across meetings and countries
    top_n_countries : int
        Number of top countries to display (default: 8)
    out_folder : str
        Output folder for HTML file (default: 'output')
    out_name : str, optional
        Custom output filename; generated if not provided
    normalize : bool
        If True, normalize values to percentages per meeting (default: False)
    flow_method : str
        Method for computing link values: 'min' (default), 'avg', or 'next'
    node_pad : int
        Padding between nodes (plotly Sankey parameter, default: 12)
    node_thickness : int
        Thickness of nodes (plotly Sankey parameter, default: 22)

    Returns:
    --------
    str or None
        Path to generated HTML file, or None if no valid data
    """
    # Compute Sankey diagram components
    comps = compute_sankey_components(
        meetings_ordered,
        counts_for_title,
        top_n_countries=top_n_countries,
        normalize=normalize,
        flow_method=flow_method
    )
    if not comps or not comps['values']:
        return None

    # Extract components
    nodes = comps['nodes']
    node_meta_counts = comps['node_meta_counts']
    node_x = comps['node_x']
    node_y = comps['node_y']
    sources = comps['sources']
    targets = comps['targets']
    values = comps['values']
    link_countries = comps['link_countries']
    per_meeting_percent = comps['per_meeting_percent']

    # Build node visualization properties
    labels, node_hover = _build_node_labels_and_hover(
        nodes, node_meta_counts, per_meeting_percent, normalize
    )

    # Build color scheme
    color_map = _create_color_map(nodes)
    country_list = [_extract_country_from_node(n) for n in nodes]
    node_colors = [_hex_to_rgba(color_map[c], alpha=0.95) for c in country_list]
    link_colors = [_hex_to_rgba(color_map[c], alpha=0.6) for c in link_countries]

    # Build link hover text
    link_hover = _build_link_hover(link_countries, values)

    # Create Sankey figure
    fig = go.Figure(go.Sankey(
        arrangement='fixed',
        node=dict(
            pad=node_pad,
            thickness=node_thickness,
            line=dict(color="rgba(0,0,0,0.25)", width=0.5),
            label=labels,
            color=node_colors,
            x=[round(x, 4) for x in node_x],
            y=[round(y, 4) for y in node_y],
            customdata=[[text] for text in node_hover],
            hovertemplate=['%{customdata[0]}<extra></extra>' for _ in nodes]
        ),
        link=dict(
            source=sources,
            target=targets,
            value=values,
            color=link_colors,
            customdata=[[text] for text in link_hover],
            hovertemplate=['%{customdata[0]}<extra></extra>' for _ in link_hover]
        )
    ))

    # Add meeting column annotations
    meetings_grouped = comps['meetings_grouped']
    n_meet = len(meetings_grouped)
    meeting_annotations = []
    for idx, g in enumerate(meetings_grouped):
        x_frac = idx / (n_meet - 1) if n_meet > 1 else 0.5
        meeting_annotations.append(dict(
            x=x_frac, y=1.08, xref='paper', yref='paper',
            text=f"<b>{g}</b>", showarrow=False,
            font=dict(size=12, color='#333'),
            xanchor='center', yanchor='bottom'
        ))

    # Configure layout
    fig.update_layout(
        title_text=f"Sankey: {title}",
        font_size=11,
        height=700,
        margin=dict(l=140, r=140, t=120, b=60),
        plot_bgcolor='white',
        paper_bgcolor='white',
        hovermode='closest',
        annotations=meeting_annotations
    )

    # Determine output filename and save
    os.makedirs(out_folder, exist_ok=True)
    if not out_name:
        safe_title = re.sub(r"[^0-9A-Za-z\-_ ]+", '', title)[:60].replace(' ', '_')
        out_name = os.path.join(out_folder, f"sankey_{safe_title}.png")

    fig.write_image(out_name, scale=2, width=1400, height=700)
    return out_name


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--meeting_folder', type=str, required=True, help='Path to folder containing meeting subfolders (e.g., output/MEPC)')
    parser.add_argument('--top_k_titles', type=int, default=6)
    parser.add_argument('--top_n_countries', type=int, default=8)
    parser.add_argument('--min_total_count', type=int, default=3, help='Minimum total docs per title to be considered hot')
    parser.add_argument('--out_folder', type=str, default=None)
    parser.add_argument('--title_field', type=str, default='Title')
    parser.add_argument('--country_field', type=str, default='Originator')
    parser.add_argument('--normalize', action='store_true', help='Normalize per meeting to percentages')
    parser.add_argument('--flow_method', type=str, default='min', choices=['min', 'avg', 'next'],
                        help="How to compute link values: 'min' (default), 'avg' or 'next' (use next meeting count)")
    parser.add_argument('--node_pad', type=int, default=12, help='Node pad (spacing)')
    parser.add_argument('--node_thickness', type=int, default=22, help='Node thickness')
    parser.add_argument('--exclude_countries', type=str, nargs='*', default=['Secretariat', 'Unknown', 'Other'],
                        help='Countries to exclude from visualization (default: Secretariat Unknown Other)')
    args = parser.parse_args()

    base = args.meeting_folder
    meetings = find_meeting_dirs(base)
    if not meetings:
        print('No meeting subfolders with data_parsed.json or data.json found in', base)
        return

    print('Meetings found:', meetings)

    counts = build_counts(meetings, base, title_field=args.title_field, country_field=args.country_field,
                          exclude=args.exclude_countries)
    selected_titles = select_top_titles(counts, top_k=args.top_k_titles, min_total=args.min_total_count)

    print('Selected hot titles:')
    pprint(selected_titles)

    out_folder = args.out_folder or os.path.join(base, 'sankey')
    os.makedirs(out_folder, exist_ok=True)

    meta = []
    for title in selected_titles:
        print('Building sankey for:', title)
        out_file = build_sankey_for_title(
            title,
            meetings,
            counts.get(title, {}),
            top_n_countries=args.top_n_countries,
            out_folder=out_folder,
            normalize=args.normalize,
            flow_method=args.flow_method,
            node_pad=args.node_pad,
            node_thickness=args.node_thickness
        )
        meta.append({'title': title, 'file': out_file})

    meta_file = os.path.join(out_folder, 'sankey_meta.json')
    with open(meta_file, 'w', encoding='utf-8') as mf:
        json.dump(meta, mf, ensure_ascii=False, indent=2)

    print('Done. Outputs in', out_folder)


if __name__ == '__main__':
    main()
