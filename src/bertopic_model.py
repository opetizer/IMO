"""
BERTopic 主题建模 (BERTopic Topic Modeling)
============================================
使用 BERTopic (transformer embeddings + HDBSCAN + c-TF-IDF) 对 IMO 会议文档
进行主题建模，替代传统 LDA，生成更高质量的语义主题。

Features:
  1. Sentence-BERT embeddings for semantic document representation
  2. UMAP dimensionality reduction + HDBSCAN clustering
  3. c-TF-IDF topic representation
  4. Dynamic topic modeling: track topic evolution across sessions
  5. Per-country topic distribution analysis
  6. Interactive visualizations (topic hierarchy, barchart, over time)

Usage:
    python src/bertopic_model.py --meeting_folder output/MEPC [--min_topic_size 10] [--embedding_model all-MiniLM-L6-v2]
    python src/bertopic_model.py --meeting_folder output/MEPC --dynamic   # temporal evolution
"""

import os
import re
import json
import argparse
from collections import defaultdict, Counter
from datetime import datetime

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from bertopic import BERTopic
from bertopic.representation import KeyBERTInspired, MaximalMarginalRelevance
from sentence_transformers import SentenceTransformer
from umap import UMAP
from hdbscan import HDBSCAN
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import (
    adjusted_rand_score,
    calinski_harabasz_score,
    davies_bouldin_score,
    silhouette_score,
)

try:
    import plotly.express as px
    import plotly.graph_objects as go
    HAS_PLOTLY = True
except ImportError:
    HAS_PLOTLY = False

try:
    from wordcloud import WordCloud
    HAS_WORDCLOUD = True
except ImportError:
    HAS_WORDCLOUD = False

try:
    from gensim.corpora import Dictionary
    from gensim.models.coherencemodel import CoherenceModel
    HAS_GENSIM = True
except ImportError:
    HAS_GENSIM = False


# ───────────────────── helpers ─────────────────────

def natural_sort_key(name):
    m = re.search(r"(\d+)", str(name))
    return (0, int(m.group(1))) if m else (1, str(name))


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


def load_meeting_records(base_folder, meeting_dir):
    for fname in ('data_processed.json', 'data_parsed.json', 'data.json'):
        path = os.path.join(base_folder, meeting_dir, fname)
        if os.path.exists(path):
            with open(path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            if isinstance(data, list):
                return data
    return []


def normalize_symbol(sym):
    if not sym:
        return ''
    return re.sub(r'\s+', ' ', sym.strip().upper())


# ───────────────────── stopwords ─────────────────────

from stopwords import STOPWORDS
from spacy_pipeline import build_topic_analyzer


# ───────────────────── data loading ─────────────────────

def load_all_documents(base_folder, meetings):
    """Load all documents, return list of dicts."""
    docs = []
    symbol_set = set()
    for m in meetings:
        records = load_meeting_records(base_folder, m)
        for row in records:
            symbol = normalize_symbol(row.get('Symbol', ''))
            if not symbol or symbol in symbol_set:
                continue
            symbol_set.add(symbol)

            full_text = row.get('full_text', '') or ''
            if len(full_text) < 100:
                parts = []
                for field in ['Summary', 'Introduction', 'Annex_Content', 'Subject']:
                    v = row.get(field, '')
                    if v and isinstance(v, str):
                        parts.append(v)
                if parts:
                    full_text = ' '.join(parts)

            if len(full_text.strip()) < 50:
                continue  # Skip docs with no meaningful text

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


def extract_session_number(meeting_name):
    """Extract numeric session from meeting name, e.g. 'MEPC 77' -> 77"""
    m = re.search(r'(\d+)', meeting_name)
    return int(m.group(1)) if m else 0


# ───────────────────── BERTopic pipeline ─────────────────────

def build_bertopic_model(docs, min_topic_size=10, embedding_model_name='all-MiniLM-L6-v2',
                         nr_topics='auto', random_state=42, embedding_batch_size=32):
    """
    Build and fit BERTopic model.
    Returns: topic_model, topics, probs, embeddings
    """
    texts = [d['full_text'] for d in docs]

    print(f'  Loading embedding model: {embedding_model_name}...')
    embedding_model = SentenceTransformer(embedding_model_name)

    print(f'  Computing embeddings for {len(texts)} documents...')
    embeddings = embedding_model.encode(
        texts,
        show_progress_bar=True,
        batch_size=embedding_batch_size,
    )

    # UMAP for dimensionality reduction
    umap_model = UMAP(
        n_neighbors=15,
        n_components=5,
        min_dist=0.0,
        metric='cosine',
        random_state=random_state,
    )

    # HDBSCAN for clustering
    hdbscan_model = HDBSCAN(
        min_cluster_size=min_topic_size,
        min_samples=3,
        metric='euclidean',
        cluster_selection_method='eom',
        prediction_data=True,
    )

    # Vectorizer with shared spaCy tokenizer so domain phrases stay intact.
    vectorizer_model = CountVectorizer(
        analyzer=build_topic_analyzer(stopwords=STOPWORDS),
        min_df=3,
        max_df=0.95,
        lowercase=False,
        token_pattern=None,
    )

    # Representation models for better topic labels
    representation_model = [
        KeyBERTInspired(top_n_words=10),
        MaximalMarginalRelevance(diversity=0.3),
    ]

    print(f'  Fitting BERTopic (min_topic_size={min_topic_size})...')
    topic_model = BERTopic(
        embedding_model=embedding_model,
        umap_model=umap_model,
        hdbscan_model=hdbscan_model,
        vectorizer_model=vectorizer_model,
        representation_model=representation_model,
        nr_topics=nr_topics,
        top_n_words=15,
        verbose=True,
        calculate_probabilities=True,
    )

    try:
        topics, probs = topic_model.fit_transform(texts, embeddings)
    except ValueError as e:
        if 'max_df corresponds to < documents than min_df' not in str(e):
            raise
        print('  Retrying BERTopic with fallback vectorizer (min_df=1, max_df=1.0)...')
        fallback_vectorizer = CountVectorizer(
            analyzer=build_topic_analyzer(stopwords=STOPWORDS),
            min_df=1,
            max_df=1.0,
            lowercase=False,
            token_pattern=None,
        )
        topic_model = BERTopic(
            embedding_model=embedding_model,
            umap_model=umap_model,
            hdbscan_model=hdbscan_model,
            vectorizer_model=fallback_vectorizer,
            representation_model=representation_model,
            nr_topics=nr_topics,
            top_n_words=15,
            verbose=True,
            calculate_probabilities=True,
        )
        topics, probs = topic_model.fit_transform(texts, embeddings)

    # Ensure topics is a list for consistent .count() usage
    if not isinstance(topics, list):
        topics = list(topics)

    return topic_model, topics, probs, embeddings


def _build_topic_model_for_embeddings(min_topic_size=10, nr_topics='auto', random_state=42):
    """Build BERTopic model that consumes precomputed embeddings."""
    umap_model = UMAP(
        n_neighbors=15,
        n_components=5,
        min_dist=0.0,
        metric='cosine',
        random_state=random_state,
    )

    hdbscan_model = HDBSCAN(
        min_cluster_size=min_topic_size,
        min_samples=3,
        metric='euclidean',
        cluster_selection_method='eom',
        prediction_data=True,
    )

    vectorizer_model = CountVectorizer(
        analyzer=build_topic_analyzer(stopwords=STOPWORDS),
        min_df=1,
        max_df=1.0,
        lowercase=False,
        token_pattern=None,
    )

    return BERTopic(
        embedding_model=None,
        umap_model=umap_model,
        hdbscan_model=hdbscan_model,
        vectorizer_model=vectorizer_model,
        representation_model=None,
        nr_topics=nr_topics,
        top_n_words=15,
        verbose=False,
        calculate_probabilities=False,
    )


def run_validation_experiments(topic_model, docs, topics, embeddings, out_folder, prefix,
                               min_topic_size=10, nr_topics='auto', seed_list=None):
    """Run supplementary validation experiments for BERTopic outputs."""
    if seed_list is None:
        seed_list = [13, 42, 77]

    texts = [d['full_text'] for d in docs]
    labels = np.array(topics)
    mask = labels != -1

    quality = {}
    if mask.sum() >= 10 and len(np.unique(labels[mask])) >= 2:
        emb = np.asarray(embeddings)
        quality['silhouette'] = float(silhouette_score(emb[mask], labels[mask], metric='cosine'))
        quality['calinski_harabasz'] = float(calinski_harabasz_score(emb[mask], labels[mask]))
        quality['davies_bouldin'] = float(davies_bouldin_score(emb[mask], labels[mask]))
    else:
        quality['silhouette'] = None
        quality['calinski_harabasz'] = None
        quality['davies_bouldin'] = None

    # Topic diversity: unique words among top-10 words across all valid topics.
    topic_info = topic_model.get_topic_info()
    valid_topics = topic_info[topic_info['Topic'] != -1]['Topic'].tolist()
    all_top_words = []
    for tid in valid_topics:
        words = topic_model.get_topic(tid) or []
        all_top_words.extend([w for w, _ in words[:10]])
    diversity = len(set(all_top_words)) / max(len(all_top_words), 1)
    quality['topic_diversity_top10'] = round(float(diversity), 4)

    # Coherence with gensim (optional)
    if HAS_GENSIM:
        tokenized_docs = [build_topic_analyzer(stopwords=STOPWORDS)(t) for t in texts]
        dictionary = Dictionary(tokenized_docs)
        topic_word_lists = []
        for tid in valid_topics:
            words = topic_model.get_topic(tid) or []
            clean_words = [w for w, _ in words[:10] if w in dictionary.token2id]
            if len(clean_words) >= 2:
                topic_word_lists.append(clean_words)

        if topic_word_lists and len(dictionary) > 0:
            try:
                c_v = CoherenceModel(
                    topics=topic_word_lists,
                    texts=tokenized_docs,
                    dictionary=dictionary,
                    coherence='c_v',
                ).get_coherence()
                quality['coherence_c_v'] = float(c_v)
            except Exception:
                quality['coherence_c_v'] = None

            try:
                corpus = [dictionary.doc2bow(doc) for doc in tokenized_docs]
                u_mass = CoherenceModel(
                    topics=topic_word_lists,
                    corpus=corpus,
                    dictionary=dictionary,
                    coherence='u_mass',
                ).get_coherence()
                quality['coherence_u_mass'] = float(u_mass)
            except Exception:
                quality['coherence_u_mass'] = None
        else:
            quality['coherence_c_v'] = None
            quality['coherence_u_mass'] = None
    else:
        quality['coherence_c_v'] = None
        quality['coherence_u_mass'] = None

    stability_rows = []
    baseline = np.array(topics)
    for seed in seed_list:
        tmp_model = _build_topic_model_for_embeddings(
            min_topic_size=min_topic_size,
            nr_topics=nr_topics,
            random_state=int(seed),
        )
        tmp_topics, _ = tmp_model.fit_transform(texts, embeddings)
        tmp_topics = np.array(tmp_topics)

        ari_all = float(adjusted_rand_score(baseline, tmp_topics))
        both_valid = (baseline != -1) & (tmp_topics != -1)
        if both_valid.sum() >= 10 and len(np.unique(baseline[both_valid])) >= 2 and len(np.unique(tmp_topics[both_valid])) >= 2:
            ari_no_outlier = float(adjusted_rand_score(baseline[both_valid], tmp_topics[both_valid]))
        else:
            ari_no_outlier = None

        n_topics_seed = len(set(tmp_topics.tolist())) - (1 if -1 in tmp_topics else 0)
        outlier_rate_seed = float((tmp_topics == -1).sum() / max(len(tmp_topics), 1))

        stability_rows.append({
            'seed': int(seed),
            'ari_vs_baseline_all': round(ari_all, 4),
            'ari_vs_baseline_no_outlier': (round(ari_no_outlier, 4) if ari_no_outlier is not None else None),
            'n_topics': int(n_topics_seed),
            'outlier_rate': round(outlier_rate_seed, 4),
        })

    stab_df = pd.DataFrame(stability_rows)
    stab_csv = os.path.join(out_folder, f'bertopic_validation_stability_{prefix}.csv')
    stab_df.to_csv(stab_csv, index=False, encoding='utf-8-sig')
    print(f'  Saved: {os.path.basename(stab_csv)}')

    if not stab_df.empty:
        fig, ax = plt.subplots(figsize=(8, 4.8))
        x = np.arange(len(stab_df))
        ax.plot(x, stab_df['ari_vs_baseline_all'], marker='o', label='ARI (all)')
        if stab_df['ari_vs_baseline_no_outlier'].notna().any():
            ax.plot(x, stab_df['ari_vs_baseline_no_outlier'], marker='s', label='ARI (no outlier)')
        ax.set_xticks(x, [str(s) for s in stab_df['seed']])
        ax.set_ylim(0, 1)
        ax.set_xlabel('Random seed')
        ax.set_ylabel('Agreement with baseline')
        ax.set_title(f'{prefix} BERTopic Seed Stability')
        ax.grid(alpha=0.25)
        ax.legend(loc='lower right')
        fig.tight_layout()
        stab_png = os.path.join(out_folder, f'bertopic_validation_stability_{prefix}.png')
        fig.savefig(stab_png, dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f'  Saved: {os.path.basename(stab_png)}')

    report = {
        'quality_metrics': quality,
        'baseline': {
            'n_documents': int(len(docs)),
            'n_topics': int(len(set(topics)) - (1 if -1 in topics else 0)),
            'outlier_rate': round(float(np.mean(np.array(topics) == -1)), 4),
        },
        'seed_stability': stability_rows,
        'seed_list': [int(s) for s in seed_list],
    }

    out_json = os.path.join(out_folder, f'bertopic_validation_{prefix}.json')
    with open(out_json, 'w', encoding='utf-8') as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f'  Saved: {os.path.basename(out_json)}')


# ───────────────────── dynamic topics ─────────────────────

def run_dynamic_topics(topic_model, docs, topics, embeddings):
    """
    Run dynamic topic modeling to track evolution across sessions.
    Returns topics_over_time DataFrame.
    """
    texts = [d['full_text'] for d in docs]

    # Create timestamps from meeting names (use session number as proxy)
    # BERTopic needs actual timestamps; we map sessions to dates
    session_dates = {}
    for d in docs:
        meeting = d['meeting']
        if meeting not in session_dates:
            # Use date from first doc of that meeting, or generate from session number
            date_str = d.get('date', '')
            if date_str:
                try:
                    # Try DD/MM/YYYY format
                    dt = datetime.strptime(date_str, '%d/%m/%Y')
                    session_dates[meeting] = dt
                except ValueError:
                    pass
            if meeting not in session_dates:
                sess_num = extract_session_number(meeting)
                # Approximate: MEPC sessions roughly every 6 months
                session_dates[meeting] = datetime(2021 + (sess_num - 77) // 2,
                                                   6 if sess_num % 2 else 12, 1)

    timestamps = [session_dates.get(d['meeting'], datetime(2023, 1, 1)) for d in docs]

    print(f'  Computing dynamic topics over {len(set(timestamps))} time points...')
    topics_over_time = topic_model.topics_over_time(
        texts, timestamps, nr_bins=len(set(timestamps)),
    )

    return topics_over_time


# ───────────────────── country analysis ─────────────────────

def analyze_country_topics(docs, topics, topic_model):
    """Analyze topic distribution per country/originator."""
    country_topics = defaultdict(lambda: Counter())

    for doc, topic_id in zip(docs, topics):
        if topic_id == -1:  # Outlier
            continue
        originators = doc.get('originator_split', [])
        if not originators:
            raw = doc.get('originator', '')
            if raw:
                originators = [raw]
        for org in originators:
            org = org.strip()
            if org and org not in {'Secretariat', 'Unknown', ''}:
                country_topics[org][topic_id] += 1

    # Build DataFrame
    topic_info = topic_model.get_topic_info()
    topic_names = {row['Topic']: row['Name'] for _, row in topic_info.iterrows()}

    rows = []
    for country, topics_counter in country_topics.items():
        total = sum(topics_counter.values())
        for tid, count in topics_counter.items():
            rows.append({
                'country': country,
                'topic_id': tid,
                'topic_name': topic_names.get(tid, f'Topic {tid}'),
                'count': count,
                'share': round(count / total, 4),
            })

    df = pd.DataFrame(rows)
    return df, country_topics


# ───────────────────── visualization ─────────────────────

def generate_wordclouds(topic_model, out_folder, prefix, n_topics=6):
    """Generate word cloud PNG for the top n_topics representative topics."""
    if not HAS_WORDCLOUD:
        print('  Skipping word clouds: wordcloud package not installed')
        return

    topic_info = topic_model.get_topic_info()
    # Filter out outlier topic (-1), take top n by doc count (already sorted)
    valid_topics = topic_info[topic_info['Topic'] != -1].head(n_topics)
    n = len(valid_topics)
    if n == 0:
        return

    cols = 3
    rows = 2
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 5, rows * 4))
    axes = axes.reshape(rows, cols)

    for idx, (_, row) in enumerate(valid_topics.iterrows()):
        tid = row['Topic']
        topic_words = topic_model.get_topic(tid)
        if not topic_words:
            continue
        # Build frequency dict from c-TF-IDF scores (shift to positive)
        min_score = min(s for _, s in topic_words)
        freq = {w.replace('_', ' '): float(s - min_score + 1e-6) for w, s in topic_words}

        _reds  = ['#B71C1C', '#C62828', '#D32F2F', '#E53935', '#F44336']
        _blues = ['#0D47A1', '#1565C0', '#1976D2', '#1E88E5', '#2196F3']
        import random as _rnd
        def _rb_color(word, font_size, position, orientation, random_state=None, **kw):
            pool = _reds if font_size > 60 else _blues
            return _rnd.choice(pool)

        wc = WordCloud(
            width=800, height=600,
            background_color='white',
            max_words=15,
            color_func=_rb_color,
            prefer_horizontal=1.0,
            min_font_size=14,
            max_font_size=200,
            relative_scaling=0.3,
        ).generate_from_frequencies(freq)

        r, c = divmod(idx, cols)
        ax = axes[r][c]
        ax.imshow(wc, interpolation='bilinear')
        ax.axis('off')
        short_name = row['Name'][:45] + '…' if len(row['Name']) > 45 else row['Name']
        ax.set_title(f'Topic {tid}: {short_name}\n({int(row["Count"])} docs)',
                     fontsize=8, pad=4)

    # Hide unused axes
    for idx in range(n, rows * cols):
        r, c = divmod(idx, cols)
        axes[r][c].axis('off')

    fig.suptitle(f'{prefix} – BERTopic Word Clouds', fontsize=13, y=1.01)
    fig.tight_layout()
    out_path = os.path.join(out_folder, f'bertopic_wordclouds_{prefix}.png')
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'  Saved: bertopic_wordclouds_{prefix}.png')


def generate_topic_doc_stats(topic_model, topics, out_folder, prefix):
    """Generate a bar chart of document count per topic (sorted descending)."""
    topic_info = topic_model.get_topic_info()
    # Exclude outliers
    df = topic_info[topic_info['Topic'] != -1].copy()
    if df.empty:
        return
    df = df.sort_values('Count', ascending=True)  # horizontal bar: largest at top

    fig, ax = plt.subplots(figsize=(10, max(5, len(df) * 0.35)))
    colors = plt.cm.RdYlGn(np.linspace(0.2, 0.85, len(df)))
    bars = ax.barh(range(len(df)), df['Count'], color=colors, edgecolor='white', linewidth=0.5)

    # Labels on bars
    for bar, val in zip(bars, df['Count']):
        ax.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height() / 2,
                str(int(val)), va='center', ha='left', fontsize=7)

    short_names = [
        (n[:50] + '…' if len(n) > 50 else n) for n in df['Name']
    ]
    ax.set_yticks(range(len(df)))
    ax.set_yticklabels(short_names, fontsize=7)
    ax.set_xlabel('Number of Documents', fontsize=10)
    ax.set_title(f'{prefix} – Documents per Topic', fontsize=12, pad=12)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_xlim(0, df['Count'].max() * 1.15)

    # Outlier annotation
    n_outliers = topics.count(-1)
    outlier_pct = n_outliers / max(len(topics), 1) * 100
    fig.text(0.98, 0.02,
             f'Outliers (topic -1): {n_outliers} docs ({outlier_pct:.1f}%)',
             ha='right', va='bottom', fontsize=8, color='grey')

    fig.tight_layout()
    out_path = os.path.join(out_folder, f'bertopic_topic_stats_{prefix}.png')
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'  Saved: bertopic_topic_stats_{prefix}.png')


def save_visualizations(topic_model, docs, topics, probs, embeddings,
                        topics_over_time, country_df, out_folder, prefix):
    """Generate and save all visualizations."""

    # 1. Topic info summary
    topic_info = topic_model.get_topic_info()
    topic_info.to_csv(os.path.join(out_folder, f'bertopic_topics_{prefix}.csv'),
                      index=False, encoding='utf-8-sig')
    print(f'  Saved: bertopic_topics_{prefix}.csv')

    # 2. Topic barchart (top words per topic)
    if HAS_PLOTLY:
        try:
            fig = topic_model.visualize_barchart(top_n_topics=20, n_words=10)
            fig.write_html(os.path.join(out_folder, f'bertopic_barchart_{prefix}.html'))
            print(f'  Saved: bertopic_barchart_{prefix}.html')
        except Exception as e:
            print(f'  Warning: barchart failed: {e}')

    # 3. Topic hierarchy
    if HAS_PLOTLY:
        try:
            fig = topic_model.visualize_hierarchy()
            fig.write_html(os.path.join(out_folder, f'bertopic_hierarchy_{prefix}.html'))
            print(f'  Saved: bertopic_hierarchy_{prefix}.html')
        except Exception as e:
            print(f'  Warning: hierarchy failed: {e}')

    # 4. Intertopic distance map
    if HAS_PLOTLY:
        try:
            fig = topic_model.visualize_topics()
            fig.write_html(os.path.join(out_folder, f'bertopic_intertopic_{prefix}.html'))
            print(f'  Saved: bertopic_intertopic_{prefix}.html')
        except Exception as e:
            print(f'  Warning: intertopic map failed: {e}')

    # 5. Document map (UMAP 2D projection)
    if HAS_PLOTLY:
        try:
            fig = topic_model.visualize_documents(
                [d['full_text'][:200] for d in docs],
                embeddings=embeddings,
            )
            fig.write_html(os.path.join(out_folder, f'bertopic_docmap_{prefix}.html'))
            print(f'  Saved: bertopic_docmap_{prefix}.html')
        except Exception as e:
            print(f'  Warning: docmap failed: {e}')

    # 6. Topics over time
    if topics_over_time is not None and HAS_PLOTLY:
        try:
            fig = topic_model.visualize_topics_over_time(topics_over_time, top_n_topics=15)
            fig.write_html(os.path.join(out_folder, f'bertopic_overtime_{prefix}.html'))
            print(f'  Saved: bertopic_overtime_{prefix}.html')
        except Exception as e:
            print(f'  Warning: topics over time failed: {e}')

    # 7. Heatmap: topic similarity
    if HAS_PLOTLY:
        try:
            fig = topic_model.visualize_heatmap(top_n_topics=30)
            fig.write_html(os.path.join(out_folder, f'bertopic_heatmap_{prefix}.html'))
            print(f'  Saved: bertopic_heatmap_{prefix}.html')
        except Exception as e:
            print(f'  Warning: heatmap failed: {e}')

    # 8. Country-topic heatmap
    if country_df is not None and not country_df.empty and HAS_PLOTLY:
        try:
            # Top 20 countries by document count
            top_countries = country_df.groupby('country')['count'].sum().nlargest(20).index.tolist()
            # Top 15 topics
            top_topics = country_df.groupby('topic_id')['count'].sum().nlargest(15).index.tolist()

            filtered = country_df[
                (country_df['country'].isin(top_countries)) &
                (country_df['topic_id'].isin(top_topics))
            ]
            pivot = filtered.pivot_table(index='country', columns='topic_name',
                                         values='count', fill_value=0, aggfunc='sum')

            fig = px.imshow(pivot, color_continuous_scale='YlOrRd',
                            title=f'{prefix} Country-Topic Distribution',
                            labels=dict(x='Topic', y='Country', color='Documents'),
                            aspect='auto')
            fig.update_layout(width=1200, height=700)
            fig.write_html(os.path.join(out_folder, f'bertopic_country_topic_{prefix}.html'))
            print(f'  Saved: bertopic_country_topic_{prefix}.html')
        except Exception as e:
            print(f'  Warning: country-topic heatmap failed: {e}')

    # 9. Save document-topic assignments
    topic_name_map = {row['Topic']: row['Name'] for _, row in topic_info.iterrows()}
    assignments = []
    for i, (doc, topic_id) in enumerate(zip(docs, topics)):
        assignments.append({
            'symbol': doc['symbol'],
            'meeting': doc['meeting'],
            'title': doc['title'],
            'originator': doc['originator'],
            'topic_id': topic_id,
            'topic_name': topic_name_map.get(topic_id, f'Topic {topic_id}'),
        })
    pd.DataFrame(assignments).to_csv(
        os.path.join(out_folder, f'bertopic_assignments_{prefix}.csv'),
        index=False, encoding='utf-8-sig'
    )
    print(f'  Saved: bertopic_assignments_{prefix}.csv')

    # 10. Word clouds (new)
    generate_wordclouds(topic_model, out_folder, prefix)

    # 11. Topic-document statistics bar chart (new)
    generate_topic_doc_stats(topic_model, topics, out_folder, prefix)


# ───────────────────── analysis stats ─────────────────────

def save_analysis(topic_model, docs, topics, country_df, out_path, prefix):
    """Save comprehensive analysis JSON."""
    topic_info = topic_model.get_topic_info()

    analysis = {
        'summary': {
            'total_documents': len(docs),
            'total_topics': len(set(topics)) - (1 if -1 in topics else 0),
            'outliers': topics.count(-1),
            'outlier_rate': round(topics.count(-1) / max(len(docs), 1), 4),
        },
        'topics': [],
        'meeting_distribution': {},
    }

    # Topic details
    for _, row in topic_info.iterrows():
        if row['Topic'] == -1:
            continue
        topic_words = topic_model.get_topic(row['Topic'])
        analysis['topics'].append({
            'id': int(row['Topic']),
            'name': row['Name'],
            'count': int(row['Count']),
            'top_words': [{'word': w, 'score': round(float(s), 4)} for w, s in topic_words[:10]],
        })

    # Meeting distribution
    meeting_topic_counts = defaultdict(lambda: Counter())
    for doc, tid in zip(docs, topics):
        if tid != -1:
            meeting_topic_counts[doc['meeting']][tid] += 1
    for m in sorted(meeting_topic_counts.keys(), key=natural_sort_key):
        analysis['meeting_distribution'][m] = dict(meeting_topic_counts[m])

    # Country stats (top 20)
    if country_df is not None and not country_df.empty:
        country_summary = country_df.groupby('country')['count'].sum().nlargest(20)
        analysis['top_countries'] = [
            {'country': c, 'total_docs': int(n)} for c, n in country_summary.items()
        ]

    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(analysis, f, ensure_ascii=False, indent=2)
    print(f'  Saved: {os.path.basename(out_path)}')


# ───────────────────── main ─────────────────────

def main():
    parser = argparse.ArgumentParser(description='BERTopic analysis for IMO meeting documents')
    parser.add_argument('--meeting_folder', type=str, required=True,
                        help='Path to meeting output folder (e.g. output/MEPC)')
    parser.add_argument('--min_topic_size', type=int, default=10,
                        help='Minimum cluster size for HDBSCAN')
    parser.add_argument('--embedding_model', type=str, default='all-MiniLM-L6-v2',
                        help='Sentence-BERT model name')
    parser.add_argument('--embedding_batch_size', type=int, default=32,
                        help='Embedding batch size for SentenceTransformer.encode')
    parser.add_argument('--nr_topics', type=str, default='auto',
                        help='Number of topics ("auto" or integer)')
    parser.add_argument('--dynamic', action='store_true',
                        help='Run dynamic topic modeling (evolution over time)')
    parser.add_argument('--out_folder', type=str, default=None)
    parser.add_argument('--validation', action='store_true',
                        help='Run supplementary validation experiments')
    parser.add_argument('--validation_seeds', type=str, default='13,42,77',
                        help='Comma-separated random seeds for stability checks')
    args = parser.parse_args()

    base = args.meeting_folder
    meetings = find_meeting_dirs(base)
    if not meetings:
        print(f'No meeting dirs found in {base}')
        return

    out_folder = args.out_folder or os.path.join(base, 'bertopic')
    os.makedirs(out_folder, exist_ok=True)
    prefix = os.path.basename(base).upper()

    nr_topics = args.nr_topics
    if nr_topics != 'auto':
        try:
            nr_topics = int(nr_topics)
        except ValueError:
            nr_topics = 'auto'

    print(f'Meetings: {meetings}')
    print(f'Output: {out_folder}')
    print(f'Embedding model: {args.embedding_model}')

    # ── Step 1: Load documents ──
    print('\n[1/5] Loading documents...')
    docs = load_all_documents(base, meetings)
    print(f'  Loaded {len(docs)} documents with text')

    # ── Step 2: Fit BERTopic ──
    print('\n[2/5] Building BERTopic model...')
    topic_model, topics, probs, embeddings = build_bertopic_model(
        docs,
        min_topic_size=args.min_topic_size,
        embedding_model_name=args.embedding_model,
        nr_topics=nr_topics,
        random_state=42,
        embedding_batch_size=args.embedding_batch_size,
    )
    n_topics = len(set(topics)) - (1 if -1 in topics else 0)
    n_outliers = topics.count(-1)
    print(f'  Found {n_topics} topics, {n_outliers} outliers ({n_outliers/max(len(docs),1)*100:.1f}%)')

    # ── Step 3: Dynamic topics ──
    topics_over_time = None
    if args.dynamic:
        print('\n[3/5] Dynamic topic modeling...')
        topics_over_time = run_dynamic_topics(topic_model, docs, topics, embeddings)
        print(f'  Generated {len(topics_over_time)} time-topic records')
    else:
        print('\n[3/5] Dynamic modeling skipped (use --dynamic to enable)')

    # ── Step 4: Country analysis ──
    print('\n[4/5] Analyzing country-topic distribution...')
    country_df, country_topics = analyze_country_topics(docs, topics, topic_model)
    print(f'  {len(country_topics)} countries/organizations analyzed')

    # ── Step 5: Save & visualize ──
    print('\n[5/5] Generating outputs...')
    save_visualizations(topic_model, docs, topics, probs, embeddings,
                        topics_over_time, country_df, out_folder, prefix)
    save_analysis(topic_model, docs, topics, country_df,
                  os.path.join(out_folder, f'bertopic_analysis_{prefix}.json'), prefix)

    if args.validation:
        print('\n[extra] Running validation experiments...')
        try:
            seed_list = [int(s.strip()) for s in args.validation_seeds.split(',') if s.strip()]
        except ValueError:
            seed_list = [13, 42, 77]
        run_validation_experiments(
            topic_model,
            docs,
            topics,
            embeddings,
            out_folder,
            prefix,
            min_topic_size=args.min_topic_size,
            nr_topics=nr_topics,
            seed_list=seed_list,
        )

    # Save the model
    model_path = os.path.join(out_folder, f'bertopic_model_{prefix}')
    topic_model.save(model_path, serialization='safetensors', save_ctfidf=True,
                     save_embedding_model=args.embedding_model)
    print(f'  Saved model: {model_path}')

    print('\nDone!')


if __name__ == '__main__':
    main()
