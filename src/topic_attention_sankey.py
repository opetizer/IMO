#####
# Hotspot Topic Attention Sankey Diagram
# 热点议题关注度时序变化桑基图
#
# 为每个议题和每个国家分别生成独立的桑基图
# 关注度使用议题数量（文档数量）定义
#
# Usage:
# python topic_attention_sankey.py --meeting_folder <meeting_folder> [options]
#
# example:
# python src/topic_attention_sankey.py --meeting_folder "output/MEPC" --top_n_titles 5 --top_n_countries 8 --min_count 1
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
import plotly.express as px

from json_read import load_data


def natural_meeting_sort_key(name):
    """Sort meeting names naturally (e.g., MEPC 77, MEPC 78, ...)"""
    m = re.search(r"(\d+)", name)
    if m:
        return int(m.group(1))
    return name


def find_meeting_dirs(base_folder):
    """Find all meeting subdirectories containing data.json"""
    items = []
    for entry in os.listdir(base_folder):
        path = os.path.join(base_folder, entry)
        if os.path.isdir(path):
            if os.path.exists(os.path.join(path, 'data.json')):
                items.append(entry)
    items_sorted = sorted(items, key=natural_meeting_sort_key)
    return items_sorted


def load_meeting_df(base_folder, meeting_dir):
    """Load data from a meeting's data_parsed.json file"""
    # Try data_parsed.json first, fallback to data.json
    path = os.path.join(base_folder, meeting_dir, 'data_parsed.json')
    if not os.path.exists(path):
        path = os.path.join(base_folder, meeting_dir, 'data.json')
    if not os.path.exists(path):
        return pd.DataFrame()
    df = load_data(path)
    df['meeting_dir'] = meeting_dir
    return df


def normalize_country_field(originator_raw):
    """Normalize and extract country from Originator field"""
    if not originator_raw or not isinstance(originator_raw, str):
        return None
    parts = re.split(r"[;,/|]", originator_raw)
    parts = [p.strip() for p in parts if p and p.strip()]
    if not parts:
        return None
    return parts[0]


def build_attention_counts(meetings_ordered, base_folder, title_field='Title', country_field='Originator', exclude=None):
    """
    Build attention counts: counts[meeting][title][country] = document_count

    Args:
        exclude: list of countries to exclude (e.g., ['Secretariat', 'Unknown', 'Other'])
    """
    if exclude is None:
        exclude = []

    counts = defaultdict(lambda: defaultdict(Counter))

    for meeting in meetings_ordered:
        df = load_meeting_df(base_folder, meeting)
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

            counts[meeting][title][country] += 1

    return counts


def get_top_titles_and_countries(counts, top_n_titles=5, top_n_countries=8, min_count=1):
    """
    Select top titles and countries based on total document counts
    """
    # Aggregate title counts across all meetings
    title_totals = defaultdict(int)
    country_totals = defaultdict(int)

    for meeting, title_dict in counts.items():
        for title, country_counter in title_dict.items():
            title_total = sum(country_counter.values())
            title_totals[title] += title_total

            for country, count in country_counter.items():
                country_totals[country] += count

    # Select top titles
    top_titles = [
        title for title, total in sorted(
            title_totals.items(),
            key=itemgetter(1),
            reverse=True
        )
        if total >= min_count
    ][:top_n_titles]

    # Select top countries
    top_countries = [
        country for country, total in sorted(
            country_totals.items(),
            key=itemgetter(1),
            reverse=True
        )
        if total >= min_count
    ][:top_n_countries]

    return top_titles, top_countries


def canonical_meeting_label(name):
    """Extract canonical meeting label (e.g., 'MEPC 77' from 'MEPC 77')"""
    m = re.search(r"^(\D*?)\s*(\d+)", name)
    if m:
        prefix = m.group(1).strip()
        num = m.group(2)
        return f"{prefix} {num}".strip()
    return name


def sanitize_filename(name):
    """Sanitize string for use in filename"""
    # Remove or replace invalid characters
    name = re.sub(r'[<>:"/\\|?*]', '', name)
    name = name.replace(' ', '_')
    # Limit length
    return name[:50]


def build_per_title_sankey(meetings_ordered, counts, title, top_countries, out_folder, meeting_prefix, min_count=1):
    """
    Build Sankey diagram for a single title, showing country attention over meetings

    Structure: Meeting -> Country
    Each node: "{Meeting} - {Country}"
    """
    # Group meetings by canonical label
    meetings_grouped = []
    per_meeting_data = {}

    for m in meetings_ordered:
        g = canonical_meeting_label(m)
        if g not in per_meeting_data:
            per_meeting_data[g] = Counter()
            meetings_grouped.append(g)
        # Get country counts for this title
        per_meeting_data[g].update(counts.get(m, {}).get(title, {}))

    # Build nodes: for each meeting, create country nodes
    nodes = []
    node_idx = {}
    node_counts = []
    nodes_per_meeting = {g: [] for g in meetings_grouped}
    all_counts = []

    for g in meetings_grouped:
        for country in top_countries:
            count = per_meeting_data[g].get(country, 0)
            if count >= min_count:
                node_name = f"{g} - {country}"
                node_idx[node_name] = len(nodes)
                nodes.append(node_name)
                node_counts.append(count)
                all_counts.append(count)
                nodes_per_meeting[g].append({
                    'name': node_name,
                    'count': count,
                    'country': country
                })

    if not nodes:
        return None

    global_max_count = max(all_counts) if all_counts else 1

    # Calculate node positions
    n_meet = len(meetings_grouped)
    node_x = [0.0] * len(nodes)
    node_y = [0.5] * len(nodes)
    gap = 0.01
    meeting_x = {g: (idx / (n_meet - 1) if n_meet > 1 else 0.5) for idx, g in enumerate(meetings_grouped)}

    # Position nodes within each meeting column
    for g in meetings_grouped:
        items = nodes_per_meeting.get(g, [])
        if not items:
            continue

        x_pos = meeting_x[g]

        # Sort items by count for better visual
        items_sorted = sorted(items, key=lambda x: x['count'], reverse=True)

        # Calculate initial heights
        raw_heights = [(item['count'] / global_max_count) * 0.8 for item in items_sorted]
        total_height = sum(raw_heights) + gap * (len(items_sorted) + 1)

        # Scale down if total height exceeds 0.9 to prevent going out of bounds
        scale_factor = min(1.0, 0.9 / total_height) if total_height > 0 else 1.0

        cursor = gap
        for item, raw_h in zip(items_sorted, raw_heights):
            ni = node_idx[item['name']]
            height = raw_h * scale_factor
            center = cursor + height / 2.0
            node_x[ni] = x_pos
            node_y[ni] = center
            cursor += height + gap

    # Build links between adjacent meetings
    sources = []
    targets = []
    values = []
    link_colors = []

    # Create color map for countries
    palettes = px.colors.qualitative.Plotly
    country_colors = {c: palettes[i % len(palettes)] for i, c in enumerate(sorted(top_countries))}

    for i in range(len(meetings_grouped) - 1):
        g1 = meetings_grouped[i]
        g2 = meetings_grouped[i + 1]

        for country in top_countries:
            node1_name = f"{g1} - {country}"
            node2_name = f"{g2} - {country}"

            if node1_name in node_idx and node2_name in node_idx:
                count1 = per_meeting_data[g1].get(country, 0)
                count2 = per_meeting_data[g2].get(country, 0)

                flow_val = min(count1, count2)
                if flow_val > 0:
                    sources.append(node_idx[node1_name])
                    targets.append(node_idx[node2_name])
                    values.append(flow_val)
                    # Color links by country
                    color = country_colors.get(country, 'rgba(100, 100, 100, 0.8)')
                    if color.startswith('#'):
                        h = color.lstrip('#')
                        r, g_c, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
                        link_colors.append(f'rgba({r},{g_c},{b},0.4)')
                    elif 'rgb(' in color:
                        link_colors.append(color.replace('rgb(', 'rgba(').replace(')', ',0.4)'))
                    else:
                        link_colors.append('rgba(100, 100, 100, 0.4)')

    # Build labels and colors
    labels = []
    hover_text = []
    node_colors = []

    for node, count in zip(nodes, node_counts):
        # node format: "{Meeting} - {Country}"
        parts = node.split(' - ')
        if len(parts) == 2:
            meeting, country_name = parts
            labels.append(f"{country_name}<br>{count} docs")
            hover_text.append(f"{meeting}<br>{country_name}<br>Documents: {count}")

            color = country_colors.get(country_name, 'rgba(100, 100, 100, 0.8)')
            if color.startswith('#'):
                h = color.lstrip('#')
                r, g_c, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
                node_colors.append(f'rgba({r},{g_c},{b},0.85)')
            elif 'rgb(' in color:
                node_colors.append(color.replace('rgb(', 'rgba(').replace(')', ',0.85)'))
            else:
                node_colors.append(color)
        else:
            labels.append(node)
            hover_text.append(f"{node}<br>Documents: {count}")
            node_colors.append('rgba(100, 100, 100, 0.8)')

    # Create Sankey figure
    fig = go.Figure(go.Sankey(
        arrangement='fixed',
        node=dict(
            pad=10,
            thickness=20,
            line=dict(color="rgba(0,0,0,0.3)", width=1),
            label=labels,
            color=node_colors,
            x=[round(x, 4) for x in node_x],
            y=[round(y, 4) for y in node_y],
            customdata=[[text] for text in hover_text],
            hovertemplate=['%{customdata[0]}<extra></extra>' for _ in nodes]
        ),
        link=dict(
            source=sources,
            target=targets,
            value=values,
            color=link_colors,
            hovertemplate=['Flow: %{value}<extra></extra>' for _ in values]
        )
    ))

    # Add meeting column annotations
    meeting_annotations_t = []
    for idx, g in enumerate(meetings_grouped):
        x_frac = idx / (n_meet - 1) if n_meet > 1 else 0.5
        meeting_annotations_t.append(dict(
            x=x_frac, y=1.08, xref='paper', yref='paper',
            text=f"<b>{g}</b>", showarrow=False,
            font=dict(size=12, color='#333'),
            xanchor='center', yanchor='bottom'
        ))

    # Update layout
    fig.update_layout(
        title_text=f"Topic: {title}<br><sub>Country attention over meetings (document count)</sub>",
        font_size=11,
        height=700,
        margin=dict(l=20, r=20, t=100, b=20),
        plot_bgcolor='white',
        paper_bgcolor='white',
        hovermode='closest',
        annotations=meeting_annotations_t
    )

    # Save
    os.makedirs(out_folder, exist_ok=True)
    safe_title = sanitize_filename(title)
    out_name = f"Sankey_{meeting_prefix}_{safe_title}.png"
    out_path = os.path.join(out_folder, out_name)
    fig.write_image(out_path, scale=2, width=1400, height=700)
    return out_path


def build_per_country_sankey(meetings_ordered, counts, country, top_titles, out_folder, meeting_prefix, min_count=1):
    """
    Build Sankey diagram for a single country, showing topic attention over meetings

    Structure: Meeting -> Topic
    Each node: "{Meeting} - {Topic}"
    """
    # Group meetings by canonical label
    meetings_grouped = []
    per_meeting_data = {}

    for m in meetings_ordered:
        g = canonical_meeting_label(m)
        if g not in per_meeting_data:
            per_meeting_data[g] = Counter()
            meetings_grouped.append(g)
        # Get title counts for this country
        for title, country_counter in counts.get(m, {}).items():
            per_meeting_data[g][title] += country_counter.get(country, 0)

    # Build nodes: for each meeting, create title nodes
    nodes = []
    node_idx = {}
    node_counts = []
    nodes_per_meeting = {g: [] for g in meetings_grouped}
    all_counts = []

    for g in meetings_grouped:
        for title in top_titles:
            count = per_meeting_data[g].get(title, 0)
            if count >= min_count:
                # Use full title as node key to avoid collisions
                node_name = f"{g} - {title}"
                node_idx[node_name] = len(nodes)
                nodes.append(node_name)
                node_counts.append(count)
                all_counts.append(count)
                nodes_per_meeting[g].append({
                    'name': node_name,
                    'count': count,
                    'title': title
                })

    if not nodes:
        return None

    global_max_count = max(all_counts) if all_counts else 1

    # Calculate node positions
    n_meet = len(meetings_grouped)
    node_x = [0.0] * len(nodes)
    node_y = [0.5] * len(nodes)
    gap = 0.01
    meeting_x = {g: (idx / (n_meet - 1) if n_meet > 1 else 0.5) for idx, g in enumerate(meetings_grouped)}

    # Position nodes within each meeting column
    for g in meetings_grouped:
        items = nodes_per_meeting.get(g, [])
        if not items:
            continue

        x_pos = meeting_x[g]

        # Sort items by count for better visual
        items_sorted = sorted(items, key=lambda x: x['count'], reverse=True)

        # Calculate initial heights
        raw_heights = [(item['count'] / global_max_count) * 0.8 for item in items_sorted]
        total_height = sum(raw_heights) + gap * (len(items_sorted) + 1)

        # Scale down if total height exceeds 0.9 to prevent going out of bounds
        scale_factor = min(1.0, 0.9 / total_height) if total_height > 0 else 1.0

        cursor = gap
        for item, raw_h in zip(items_sorted, raw_heights):
            ni = node_idx[item['name']]
            height = raw_h * scale_factor
            center = cursor + height / 2.0
            node_x[ni] = x_pos
            node_y[ni] = center
            cursor += height + gap

    # Build links between adjacent meetings
    sources = []
    targets = []
    values = []
    link_colors = []

    # Create color map for titles
    palettes = px.colors.qualitative.Plotly
    title_colors = {t: palettes[i % len(palettes)] for i, t in enumerate(sorted(top_titles))}

    for i in range(len(meetings_grouped) - 1):
        g1 = meetings_grouped[i]
        g2 = meetings_grouped[i + 1]

        for title in top_titles:
            node1_name = f"{g1} - {title}"
            node2_name = f"{g2} - {title}"

            if node1_name in node_idx and node2_name in node_idx:
                count1 = per_meeting_data[g1].get(title, 0)
                count2 = per_meeting_data[g2].get(title, 0)

                flow_val = min(count1, count2)
                if flow_val > 0:
                    sources.append(node_idx[node1_name])
                    targets.append(node_idx[node2_name])
                    values.append(flow_val)
                    # Color links by title
                    color = title_colors.get(title, 'rgba(100, 100, 100, 0.8)')
                    if color.startswith('#'):
                        h = color.lstrip('#')
                        r, g_c, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
                        link_colors.append(f'rgba({r},{g_c},{b},0.4)')
                    elif 'rgb(' in color:
                        link_colors.append(color.replace('rgb(', 'rgba(').replace(')', ',0.4)'))
                    elif 'rgba(' in color:
                        # Replace alpha
                        import re as _re
                        link_colors.append(_re.sub(r',\s*[\d.]+\)$', ',0.4)', color))
                    else:
                        link_colors.append('rgba(100, 100, 100, 0.4)')

    # Build labels and colors
    labels = []
    hover_text = []
    node_colors = []

    for node, count in zip(nodes, node_counts):
        # node format: "{Meeting} - {Title}" (full title)
        parts = node.split(' - ', 1)
        if len(parts) == 2:
            meeting, full_title = parts
            # Truncate only for display (40 chars)
            display_title = full_title[:40] + '...' if len(full_title) > 40 else full_title
            labels.append(f"{display_title}<br>{count}")
            hover_text.append(f"{meeting}<br>{full_title}<br>Documents: {count}")

            color = title_colors.get(full_title, 'rgba(100, 100, 100, 0.8)')
            if color.startswith('#'):
                h = color.lstrip('#')
                r, g_c, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
                node_colors.append(f'rgba({r},{g_c},{b},0.85)')
            elif 'rgb(' in color:
                node_colors.append(color.replace('rgb(', 'rgba(').replace(')', ',0.85)'))
            else:
                node_colors.append(color)
        else:
            labels.append(node)
            hover_text.append(f"{node}<br>Documents: {count}")
            node_colors.append('rgba(100, 100, 100, 0.8)')

    # Create Sankey figure
    fig = go.Figure(go.Sankey(
        arrangement='fixed',
        node=dict(
            pad=10,
            thickness=20,
            line=dict(color="rgba(0,0,0,0.3)", width=1),
            label=labels,
            color=node_colors,
            x=[round(x, 4) for x in node_x],
            y=[round(y, 4) for y in node_y],
            customdata=[[text] for text in hover_text],
            hovertemplate=['%{customdata[0]}<extra></extra>' for _ in nodes]
        ),
        link=dict(
            source=sources,
            target=targets,
            value=values,
            color=link_colors,
            hovertemplate=['Flow: %{value}<extra></extra>' for _ in values]
        )
    ))

    # Add meeting column annotations
    meeting_annotations = []
    for idx, g in enumerate(meetings_grouped):
        x_frac = idx / (n_meet - 1) if n_meet > 1 else 0.5
        meeting_annotations.append(dict(
            x=x_frac, y=1.08, xref='paper', yref='paper',
            text=f"<b>{g}</b>", showarrow=False,
            font=dict(size=12, color='#333'),
            xanchor='center', yanchor='bottom'
        ))

    # Update layout
    fig.update_layout(
        title_text=f"Country: {country}<br><sub>Topic attention over meetings (document count)</sub>",
        font_size=11,
        height=700,
        margin=dict(l=20, r=20, t=100, b=20),
        plot_bgcolor='white',
        paper_bgcolor='white',
        hovermode='closest',
        annotations=meeting_annotations
    )

    # Save
    os.makedirs(out_folder, exist_ok=True)
    safe_country = sanitize_filename(country)
    out_name = f"Sankey_{meeting_prefix}_{safe_country}.png"
    out_path = os.path.join(out_folder, out_name)
    fig.write_image(out_path, scale=2, width=1400, height=700)
    return out_path


def main():
    parser = argparse.ArgumentParser(
        description='Generate Sankey diagrams for each topic and country showing attention over meetings'
    )
    parser.add_argument('--meeting_folder', type=str, required=True,
                        help='Path to folder containing meeting subfolders (e.g., output/MEPC)')
    parser.add_argument('--top_n_titles', type=int, default=5,
                        help='Number of top titles to visualize (default: 5)')
    parser.add_argument('--top_n_countries', type=int, default=8,
                        help='Number of top countries to visualize (default: 8)')
    parser.add_argument('--min_count', type=int, default=1,
                        help='Minimum document count for a node to be included (default: 1)')
    parser.add_argument('--out_folder', type=str, default=None,
                        help='Output folder (default: <meeting_folder>/sankey)')
    parser.add_argument('--title_field', type=str, default='Title',
                        help='Field name for title (default: Title)')
    parser.add_argument('--country_field', type=str, default='Originator',
                        help='Field name for country/originator (default: Originator)')
    parser.add_argument('--exclude_countries', type=str, nargs='*', default=['Secretariat', 'Unknown', 'Other'],
                        help='Countries to exclude from visualization (default: Secretariat Unknown Other)')
    args = parser.parse_args()

    # Find meetings
    base = args.meeting_folder
    meetings = find_meeting_dirs(base)
    if not meetings:
        print(f'No meeting subfolders with data.json found in {base}')
        return

    print(f'Meetings found: {meetings}')

    # Build attention counts
    counts = build_attention_counts(meetings, base, args.title_field, args.country_field, exclude=args.exclude_countries)

    # Select top titles and countries
    top_titles, top_countries = get_top_titles_and_countries(
        counts, args.top_n_titles, args.top_n_countries, args.min_count
    )

    print(f'\nTop {len(top_titles)} titles:')
    for i, title in enumerate(top_titles, 1):
        print(f'  {i}. {title}')

    print(f'\nTop {len(top_countries)} countries:')
    for i, country in enumerate(top_countries, 1):
        print(f'  {i}. {country}')

    # Determine output folder and meeting prefix
    out_folder = args.out_folder or os.path.join(base, 'sankey')
    meeting_prefix = os.path.basename(base).upper()

    os.makedirs(out_folder, exist_ok=True)

    # Generate Sankey diagrams for each title
    print(f'\n{"="*60}')
    print(f'Generating Sankey diagrams for each title...')
    print(f'{"="*60}')

    title_files = []
    for title in top_titles:
        print(f'  Building: {title[:50]}...')
        out_file = build_per_title_sankey(
            meetings, counts, title, top_countries,
            out_folder, meeting_prefix, args.min_count
        )
        if out_file:
            title_files.append({'title': title, 'file': out_file})
            print(f'    Saved: {os.path.basename(out_file)}')

    # Generate Sankey diagrams for each country
    print(f'\n{"="*60}')
    print(f'Generating Sankey diagrams for each country...')
    print(f'{"="*60}')

    country_files = []
    for country in top_countries:
        print(f'  Building: {country}...')
        out_file = build_per_country_sankey(
            meetings, counts, country, top_titles,
            out_folder, meeting_prefix, args.min_count
        )
        if out_file:
            country_files.append({'country': country, 'file': out_file})
            print(f'    Saved: {os.path.basename(out_file)}')

    # Save metadata
    metadata = {
        'titles': title_files,
        'countries': country_files
    }

    meta_file = os.path.join(out_folder, 'sankey_metadata.json')
    with open(meta_file, 'w', encoding='utf-8') as mf:
        json.dump(metadata, mf, ensure_ascii=False, indent=2)

    print(f'\n{"="*60}')
    print(f'Done! Generated {len(title_files)} title diagrams and {len(country_files)} country diagrams')
    print(f'Output folder: {out_folder}')
    print(f'Metadata saved to: {meta_file}')
    print(f'{"="*60}')


if __name__ == '__main__':
    main()
