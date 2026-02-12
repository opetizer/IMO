"""
Cross-Committee Dynamic Topic Analysis (跨委员会动态主题演进)
============================================================
Analyze how key maritime policy themes evolve over time across committees.
Focus on cross-cutting themes: GHG, alternative fuels, autonomous ships, etc.

Outputs:
  1. topic_evolution_heatmap.html  — All topics over time (per committee)
  2. cross_theme_trends.html       — Key themes across committees
  3. emerging_topics.html          — Newly emerging vs declining topics
  4. topic_dynamics_report.json    — Summary statistics
"""

import os
import json
import argparse
from collections import defaultdict

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def load_data(committee, base_dir="output"):
    """Load assignments and topic info."""
    csv_path = os.path.join(base_dir, committee, "bertopic", f"bertopic_assignments_{committee}.csv")
    json_path = os.path.join(base_dir, committee, "bertopic", f"bertopic_analysis_{committee}.json")
    
    if not os.path.exists(csv_path):
        return None, {}
    
    df = pd.read_csv(csv_path, encoding='utf-8-sig')
    df['committee'] = committee
    
    topic_map = {}
    if os.path.exists(json_path):
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        for t in data.get('topics', []):
            words = [w['word'] for w in t['top_words'][:3]]
            topic_map[t['id']] = ' / '.join(words)
    
    return df, topic_map


def extract_meeting_number(meeting_str, committee):
    """Extract numeric session from meeting string for ordering."""
    import re
    prefix = committee.split('-')[0]
    m = re.search(rf'{prefix}\s*(\d+)', str(meeting_str))
    if m:
        return int(m.group(1))
    return 0


def build_topic_timeline(df, topic_map, committee):
    """Build topic proportion over meetings."""
    if df is None or df.empty:
        return pd.DataFrame()
    
    df = df.copy()
    df['meeting_num'] = df['meeting'].apply(lambda x: extract_meeting_number(x, committee))
    df = df[df['topic_id'] != -1]  # exclude outliers
    
    # Count docs per meeting per topic
    counts = df.groupby(['meeting', 'meeting_num', 'topic_id']).size().reset_index(name='count')
    
    # Total per meeting
    totals = counts.groupby(['meeting', 'meeting_num'])['count'].sum().reset_index(name='total')
    
    # Merge
    merged = counts.merge(totals, on=['meeting', 'meeting_num'])
    merged['proportion'] = merged['count'] / merged['total']
    
    # Add topic labels
    merged['topic_label'] = merged['topic_id'].map(
        lambda x: f"T{x}: {topic_map.get(x, '?')[:35]}"
    )
    
    return merged.sort_values('meeting_num')


def plot_topic_evolution_heatmap(timeline_data, committee, out_dir):
    """Heatmap: meetings (x) vs topics (y), color = proportion."""
    if timeline_data.empty:
        return
    
    pivot = timeline_data.pivot_table(
        index='topic_label', columns='meeting', values='proportion', fill_value=0
    )
    
    # Sort columns by meeting number
    meeting_order = timeline_data.drop_duplicates('meeting').sort_values('meeting_num')['meeting'].tolist()
    pivot = pivot.reindex(columns=meeting_order)
    
    fig = px.imshow(
        pivot.values,
        x=pivot.columns.tolist(),
        y=pivot.index.tolist(),
        color_continuous_scale='Viridis',
        aspect='auto',
        title=f'{committee} - Topic Evolution Over Sessions',
        labels=dict(color='Proportion')
    )
    fig.update_layout(
        width=1100, height=max(400, len(pivot) * 30),
        xaxis_tickangle=-30,
        margin=dict(l=300)
    )
    
    path = os.path.join(out_dir, f'topic_evolution_heatmap_{committee}.html')
    fig.write_html(path)
    print(f"  Saved: {path}")


def detect_trends(timeline_data, committee):
    """Detect rising and declining topics based on linear regression slope."""
    if timeline_data.empty:
        return []
    
    trends = []
    for topic_id in timeline_data['topic_id'].unique():
        sub = timeline_data[timeline_data['topic_id'] == topic_id].sort_values('meeting_num')
        if len(sub) < 3:
            continue
        
        x = sub['meeting_num'].values.astype(float)
        y = sub['proportion'].values
        
        # Simple linear regression
        if len(x) > 1:
            try:
                slope = np.polyfit(x, y, 1)[0]
            except (np.linalg.LinAlgError, ValueError):
                slope = 0.0
            mean_prop = y.mean()
            label = sub['topic_label'].iloc[0]
            
            # Relative change
            first_half = y[:len(y)//2].mean() if len(y) > 1 else y[0]
            second_half = y[len(y)//2:].mean() if len(y) > 1 else y[-1]
            rel_change = (second_half - first_half) / max(first_half, 0.001)
            
            trends.append({
                'committee': committee,
                'topic_id': int(topic_id),
                'topic_label': label,
                'slope': round(float(slope), 5),
                'mean_proportion': round(float(mean_prop), 4),
                'relative_change': round(float(rel_change), 3),
                'sessions': int(len(sub)),
                'trend': 'rising' if slope > 0.005 else ('declining' if slope < -0.005 else 'stable')
            })
    
    return sorted(trends, key=lambda x: x['slope'], reverse=True)


def plot_emerging_declining(all_trends, out_dir):
    """Bar chart of top rising and declining topics across all committees."""
    if not all_trends:
        return
    
    df = pd.DataFrame(all_trends)
    
    # Top 10 rising and top 10 declining
    rising = df[df['slope'] > 0].nlargest(15, 'slope')
    declining = df[df['slope'] < 0].nsmallest(15, 'slope')
    combined = pd.concat([rising, declining])
    
    combined['display'] = combined['committee'] + ': ' + combined['topic_label']
    combined = combined.sort_values('slope')
    
    fig = go.Figure()
    
    colors = ['#e74c3c' if s < 0 else '#2ecc71' for s in combined['slope']]
    
    fig.add_trace(go.Bar(
        y=combined['display'],
        x=combined['slope'],
        orientation='h',
        marker_color=colors,
        text=[f"{s:+.4f}" for s in combined['slope']],
        textposition='outside'
    ))
    
    fig.update_layout(
        title='Emerging vs Declining Topics Across Committees',
        xaxis_title='Trend Slope (proportion per session)',
        yaxis_title='',
        width=1100, height=max(500, len(combined) * 28),
        margin=dict(l=400)
    )
    
    path = os.path.join(out_dir, 'emerging_declining_topics.html')
    fig.write_html(path)
    print(f"  Saved: {path}")


def plot_cross_theme_trends(all_timelines, out_dir):
    """Track key cross-cutting themes across committees."""
    
    # Define cross-cutting keywords to track
    themes = {
        'GHG/Emissions': ['emission', 'ghg', 'co2', 'carbon', 'greenhouse'],
        'Alternative Fuels': ['fuel', 'methanol', 'ammonia', 'hydrogen', 'lng', 'alternative fuel'],
        'Autonomous Ships': ['autonomous', 'mass', 'maritime autonomous'],
        'Digitalization': ['digital', 'e-navigation', 'cyber', 'electronic'],
        'Safety Equipment': ['lifeboat', 'life-saving', 'survival craft', 'fire'],
        'Ballast Water': ['ballast water', 'bwm'],
        'Polar/Arctic': ['polar', 'arctic', 'ice'],
    }
    
    results = []
    
    for committee, (df, topic_map) in all_timelines.items():
        if df is None or df.empty:
            continue
        
        df_filtered = df[df['topic_id'] != -1].copy()
        df_filtered['meeting_num'] = df_filtered['meeting'].apply(
            lambda x: extract_meeting_number(x, committee)
        )
        
        for theme_name, keywords in themes.items():
            # Find topics that match this theme
            matching_topics = []
            for tid, label in topic_map.items():
                label_lower = label.lower()
                if any(kw in label_lower for kw in keywords):
                    matching_topics.append(tid)
            
            if not matching_topics:
                continue
            
            # Aggregate by meeting
            theme_docs = df_filtered[df_filtered['topic_id'].isin(matching_topics)]
            total_by_meeting = df_filtered.groupby('meeting_num').size()
            theme_by_meeting = theme_docs.groupby('meeting_num').size()
            
            for meeting_num in sorted(total_by_meeting.index):
                t_count = theme_by_meeting.get(meeting_num, 0)
                total = total_by_meeting.get(meeting_num, 1)
                results.append({
                    'committee': committee,
                    'theme': theme_name,
                    'meeting_num': meeting_num,
                    'count': int(t_count),
                    'proportion': round(t_count / total, 4)
                })
    
    if not results:
        return
    
    res_df = pd.DataFrame(results)
    
    # Plot each theme as a subplot
    active_themes = res_df['theme'].unique()
    n_themes = len(active_themes)
    
    if n_themes == 0:
        return
    
    fig = make_subplots(
        rows=n_themes, cols=1,
        subplot_titles=list(active_themes),
        shared_xaxes=False,
        vertical_spacing=0.04
    )
    
    colors = px.colors.qualitative.Set2
    committee_colors = {}
    
    for row_idx, theme in enumerate(active_themes):
        theme_data = res_df[res_df['theme'] == theme]
        
        for ci, committee in enumerate(theme_data['committee'].unique()):
            if committee not in committee_colors:
                committee_colors[committee] = colors[len(committee_colors) % len(colors)]
            
            comm_data = theme_data[theme_data['committee'] == committee].sort_values('meeting_num')
            
            fig.add_trace(
                go.Scatter(
                    x=comm_data['meeting_num'],
                    y=comm_data['proportion'],
                    name=committee,
                    mode='lines+markers',
                    line=dict(color=committee_colors[committee]),
                    legendgroup=committee,
                    showlegend=(row_idx == 0)
                ),
                row=row_idx + 1, col=1
            )
    
    fig.update_layout(
        title='Cross-Committee Theme Trends Over Time',
        width=1100, height=250 * n_themes,
        showlegend=True
    )
    
    path = os.path.join(out_dir, 'cross_theme_trends.html')
    fig.write_html(path)
    print(f"  Saved: {path}")


def main():
    parser = argparse.ArgumentParser(description='Cross-Committee Dynamic Topic Analysis')
    parser.add_argument('--committees', nargs='+', default=['MEPC', 'MSC', 'CCC', 'SSE', 'ISWG-GHG'])
    parser.add_argument('--base-dir', default='output')
    args = parser.parse_args()
    
    out_dir = os.path.join(args.base_dir, 'dynamic_analysis')
    os.makedirs(out_dir, exist_ok=True)
    
    all_timelines = {}
    all_trends = []
    
    for committee in args.committees:
        print(f"\n{'='*60}")
        print(f"Analyzing {committee}...")
        print(f"{'='*60}")
        
        df, topic_map = load_data(committee, args.base_dir)
        if df is None:
            print(f"  No data found, skipping")
            continue
        
        all_timelines[committee] = (df, topic_map)
        
        print(f"  Loaded {len(df)} docs, {len(topic_map)} topics")
        
        # Build timeline
        timeline = build_topic_timeline(df, topic_map, committee)
        
        # Per-committee heatmap
        comm_out = os.path.join(out_dir, committee)
        os.makedirs(comm_out, exist_ok=True)
        plot_topic_evolution_heatmap(timeline, committee, comm_out)
        
        # Detect trends
        trends = detect_trends(timeline, committee)
        all_trends.extend(trends)
        
        # Print top trends
        rising = [t for t in trends if t['trend'] == 'rising']
        declining = [t for t in trends if t['trend'] == 'declining']
        print(f"  Rising topics: {len(rising)}, Declining: {len(declining)}, Stable: {len(trends)-len(rising)-len(declining)}")
        
        if rising:
            print(f"    Top rising: {rising[0]['topic_label']} (slope={rising[0]['slope']:.4f})")
        if declining:
            print(f"    Top declining: {declining[-1]['topic_label']} (slope={declining[-1]['slope']:.4f})")
    
    # Cross-committee analyses
    print(f"\n{'='*60}")
    print(f"Cross-Committee Analysis...")
    print(f"{'='*60}")
    
    plot_emerging_declining(all_trends, out_dir)
    plot_cross_theme_trends(all_timelines, out_dir)
    
    # Save report
    report = {
        'committees': args.committees,
        'trends_by_committee': {}
    }
    for committee in args.committees:
        comm_trends = [t for t in all_trends if t['committee'] == committee]
        report['trends_by_committee'][committee] = {
            'total_topics': len(comm_trends),
            'rising': [t for t in comm_trends if t['trend'] == 'rising'],
            'declining': [t for t in comm_trends if t['trend'] == 'declining'],
            'stable': [t for t in comm_trends if t['trend'] == 'stable']
        }
    
    path = os.path.join(out_dir, 'topic_dynamics_report.json')
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(f"\n  Saved: {path}")
    
    print("\nDynamic topic analysis complete!")


if __name__ == '__main__':
    main()
