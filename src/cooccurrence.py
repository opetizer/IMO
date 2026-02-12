import os
import nltk
import json
import networkx as nx
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from collections import defaultdict
from nltk.corpus import wordnet, stopwords
from nltk.stem import WordNetLemmatizer
import argparse
import logging
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
import community as community_louvain
from adjustText import adjust_text
from gensim.models import Phrases
from datetime import datetime

# --- 引入新的数据读取模块 ---
from json_read import load_data

# 设置通用绘图风格 (SCI 论文风格：白色背景，无网格或少网格)
sns.set_theme(style="ticks", context="paper") # context="paper" 适合论文发表，字体大小适中
# 尝试设置中文字体
try:
    plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'Microsoft YaHei', 'sans-serif']
except Exception:
    pass
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['pdf.fonttype'] = 42 # 确保导出PDF时文字可编辑
plt.rcParams['ps.fonttype'] = 42

# --- 边缘权重阈值 ---
# 建议适当提高此值以减少连线密度，使图更清晰
EDGE_WEIGHT_THRESHOLD = 3.0 

parser = argparse.ArgumentParser()
parser.add_argument('--title', type=str, required=True)
parser.add_argument('--subtitle', type=str, required=True)
parser.add_argument('--logging', type=str, required=True)
parser.add_argument('--text_extracted_folder', type=str, required=True)
parser.add_argument('--symbol', type=str, default=None,
                    help="筛选会议 Symbol（例如 'MEPC 81'）。")
parser.add_argument('--per_agenda', action='store_true',
                    help="若设置，则在总体图之外为每个议题生成单独的共现图")
parser.add_argument('--per_agenda_topk', type=int, default=40,
                    help="每个议题图的 top-k 关键词数量（默认 40）")
parser.add_argument('--min_agenda_nodes', type=int, default=20,
                    help="每个议题共现图的最小节点数量阈值，小于该值的议题将被跳过（默认 5）")

parser.add_argument('--countries', type=str, nargs='*', default=None,
                    help="用于筛选的国家列表 (Originator)。例如: --countries China Japan")
parser.add_argument('--start_date', type=str, default=None,
                    help="筛选的开始日期 (格式: YYYY-MM-DD)。")
parser.add_argument('--end_date', type=str, default=None,
                    help="筛选的结束日期 (格式: YYYY-MM-DD)。")
args = parser.parse_args()

text_folder = args.text_extracted_folder
top_k = 80 # 略微减少节点数，使图更易读 (原100)
window_size = 5
output_path = f'output/{args.title}/{args.subtitle}'
co_output_path = os.path.join(output_path, 'cooccurrence_graph.png')
freq_output_path = os.path.join(output_path, 'word_freq.json')

stop_words = set(stopwords.words('english'))
from stopword import additional_stopwords
stop_words.update(additional_stopwords)
json_file_path = os.path.join(text_folder, "data.json")

def setup_logger(log_file):
    logger = logging.getLogger(args.title)
    if logger.hasHandlers():
        logger.handlers.clear()
    logger.setLevel(logging.INFO)
    file_handler = logging.FileHandler(log_file, mode='w', encoding='utf-8')
    file_handler.setLevel(logging.INFO)
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    return logger

logger = setup_logger(args.logging)

def download_nltk_data():
    nltk.download('averaged_perceptron_tagger', quiet=True)
    nltk.download('punkt', quiet=True)
    nltk.download('stopwords', quiet=True) 
    nltk.download('wordnet', quiet=True)

lemmatizer = WordNetLemmatizer()
def get_wordnet_pos(treebank_tag):
    if treebank_tag.startswith('J'): return wordnet.ADJ
    elif treebank_tag.startswith('V'): return wordnet.VERB
    elif treebank_tag.startswith('N'): return wordnet.NOUN
    elif treebank_tag.startswith('R'): return wordnet.ADV
    else: return wordnet.NOUN

def lemmatizing_tokenizer(text):
    if not isinstance(text, str): return []
    tagged_tokens = nltk.pos_tag(nltk.word_tokenize(text))
    lemmas = []
    for word, tag in tagged_tokens:
        if tag.startswith('NN'): 
            pos = get_wordnet_pos(tag)
            lemma = lemmatizer.lemmatize(word.lower(), pos=pos)
            if len(lemma) > 1 and lemma.isalpha() and lemma not in stop_words:
                lemmas.append(lemma)
    return lemmas

def build_cooccurrence_graph(token_docs, top_k_local=80, edge_threshold=EDGE_WEIGHT_THRESHOLD, window_local=window_size):
    """基于已处理的 token 列表（每个元素是一篇文档的 tokens）构建共现网络并返回 NetworkX 图对象。
    返回 None 如果图为空或无有效边。
    """
    if not token_docs:
        return None

    # TF-IDF 以决定 top_k
    try:
        vectorizer_local = TfidfVectorizer(
            tokenizer=lambda x: x,
            preprocessor=lambda x: x,
            lowercase=False,
            max_features=top_k_local
        )
        tfidf_matrix_local = vectorizer_local.fit_transform(token_docs)
    except Exception as e:
        logger.warning(f"TF-IDF 构建失败: {e}")
        return None

    feature_names_local = vectorizer_local.get_feature_names_out()
    tfidf_scores_local = tfidf_matrix_local.sum(axis=0).A1
    word_tfidf_local = dict(zip(feature_names_local, tfidf_scores_local))

    sorted_words_local = sorted(word_tfidf_local.items(), key=lambda x: x[1], reverse=True)
    top_words_local = [w for w, s in sorted_words_local]

    # 构建共现
    word_counts_local = defaultdict(int)
    cooccurrence_local = defaultdict(int)

    for tokens in token_docs:
        filtered_tokens = [w for w in tokens if w in top_words_local]
        for word in filtered_tokens:
            word_counts_local[word] += 1
        for i in range(len(filtered_tokens)):
            for j in range(i + 1, min(i + window_local, len(filtered_tokens))):
                w1, w2 = filtered_tokens[i], filtered_tokens[j]
                if w1 == w2:
                    continue
                pair = tuple(sorted((w1, w2)))
                cooccurrence_local[pair] += 1

    G_local = nx.Graph()
    for word in top_words_local:
        if word_counts_local[word] > 0:
            G_local.add_node(word, size=word_counts_local[word], weight=word_tfidf_local.get(word, 1.0))

    for (w1, w2), freq in cooccurrence_local.items():
        if w1 in G_local and w2 in G_local and freq >= edge_threshold:
            G_local.add_edge(w1, w2, weight=freq)

    # 移除孤立节点
    isolates_local = list(nx.isolates(G_local))
    if isolates_local:
        G_local.remove_nodes_from(isolates_local)

    if not G_local.nodes():
        return None

    return G_local


def draw_graph(G, out_file, title_text, subtitle_text=None):
    """绘制并保存共现图，样式与之前一致。"""
    if G is None or not G.nodes():
        logger.warning(f"图为空，跳过保存: {out_file}")
        return

    plt.figure(figsize=(12, 10), facecolor='white', dpi=300)

    layout_k = 2.0 / np.sqrt(len(G.nodes())) if len(G.nodes()) > 0 else 1.0
    pos_local = nx.spring_layout(G, k=layout_k, seed=42, iterations=100)

    node_sizes_raw = np.array([G.nodes[n]['size'] for n in G.nodes()])
    if len(node_sizes_raw) > 0 and node_sizes_raw.max() > node_sizes_raw.min():
        norm_sizes = (node_sizes_raw - node_sizes_raw.min()) / (node_sizes_raw.max() - node_sizes_raw.min())
        node_sizes_local = 100 + norm_sizes * 800
    else:
        node_sizes_local = [300] * len(G.nodes())

    partition_local = community_louvain.best_partition(G)
    unique_comms_local = sorted(list(set(partition_local.values())))
    colors_local = plt.cm.tab20(np.linspace(0, 1, max(1, len(unique_comms_local))))
    color_map_local = {com: colors_local[i] for i, com in enumerate(unique_comms_local)}
    node_colors_local = [color_map_local[partition_local[n]] for n in G.nodes()]

    edges_local = G.edges()
    weights_local = [G[u][v]['weight'] for u, v in edges_local]
    if weights_local:
        max_w_local = max(weights_local)
        width_local = [(w / max_w_local) * 2.5 + 0.5 for w in weights_local]
    else:
        width_local = 1.0

    nx.draw_networkx_edges(G, pos_local, width=width_local, edge_color='#B0B0B0', alpha=0.4)
    nx.draw_networkx_nodes(G, pos_local, node_size=node_sizes_local, node_color=node_colors_local, alpha=0.9,
                           linewidths=1.5, edgecolors='white')

    texts_local = []
    for node, (x, y) in pos_local.items():
        n_size = G.nodes[node]['size']
        size_range = node_sizes_raw.max() - node_sizes_raw.min()
        if len(node_sizes_raw) > 0 and size_range > 0:
            font_size = 9 + (n_size - node_sizes_raw.min()) / size_range * 6
        else:
            font_size = 10
        texts_local.append(plt.text(x, y, node.replace('_', ' '), fontsize=font_size,
                                    fontweight='medium', ha='center', va='center',
                                    bbox=dict(boxstyle="round,pad=0.1", fc="white", ec="none", alpha=0.6)))

    adjust_text(texts_local, expand_points=(1.5, 1.5), arrowprops=dict(arrowstyle='-', color='gray', lw=0.5, alpha=0.5),
                force_text=0.5, force_points=0.5)

    plt.axis('off')
    title_full = title_text if not subtitle_text else f"{title_text} — {subtitle_text}"
    plt.title(title_full, fontsize=14, fontweight='bold', pad=12, loc='left')
    plt.figtext(0.1, 0.02, f"Nodes: {len(G.nodes())}    Edges: {len(G.edges())}    Threshold: {EDGE_WEIGHT_THRESHOLD}", fontsize=8, color='gray')

    os.makedirs(os.path.dirname(out_file), exist_ok=True)
    plt.savefig(out_file, dpi=300, bbox_inches='tight')
    plt.close()
    logger.info(f"Saved graph: {out_file}")


def main():
    download_nltk_data()

    try:
        start_date_obj = datetime.strptime(args.start_date, '%Y-%m-%d') if args.start_date else None
        end_date_obj = datetime.strptime(args.end_date, '%Y-%m-%d') if args.end_date else None
    except ValueError:
        logger.error("无效的日期格式。请使用 YYYY-MM-DD。")
        return

    countries_to_check = [c.lower() for c in args.countries] if args.countries else None
    logger.info(f"应用筛选器 - 国家: {args.countries}, 开始日期: {args.start_date}, 结束日期: {args.end_date}")

    if not os.path.exists(json_file_path):
        logger.error(f"未找到 data.json: {json_file_path}")
        return

    df = load_data(json_file_path)
    records = df.to_dict('records')

    # 筛选指定会议（Symbol）
    meeting_symbol = args.symbol.lower() if args.symbol else None
    meeting_records = []
    for item in records:
        if meeting_symbol:
            sym = str(item.get('Symbol', '')).lower()
            if meeting_symbol not in sym:
                continue
        # apply country/date filters
        if countries_to_check:
            originator_str = str(item.get('Originator', '')).lower()
            if not any(country in originator_str for country in countries_to_check):
                continue
        if start_date_obj or end_date_obj:
            item_date_str = item.get('Date')
            if item_date_str:
                try:
                    item_date_obj = datetime.strptime(item_date_str, '%d/%m/%Y')
                    if start_date_obj and item_date_obj < start_date_obj: continue
                    if end_date_obj and item_date_obj > end_date_obj: continue
                except (ValueError, TypeError):
                    pass
        meeting_records.append(item)

    logger.info(f"选中 {len(meeting_records)} 条记录用于会议级共现分析。")
    if not meeting_records:
        logger.warning("没有符合筛选条件的记录，结束。")
        return

    # 构建文档与议题映射
    documents = []
    agendas = []
    subjects = []
    for it in meeting_records:
        text = it.get('full_text', '')
        if text and isinstance(text, str) and text.strip():
            documents.append(text)
            agendas.append(it.get('Agenda_Item', 'INFO'))
            subjects.append(it.get('Title', '') if it.get('Title') else '')

    logger.info(f"有效文本数: {len(documents)}")
    if not documents:
        logger.warning("没有有效文本，结束。")
        return

    # NLP 处理
    logger.info("分词与短语提取...")
    all_tokens = [lemmatizing_tokenizer(doc) for doc in documents]
    all_tokens = [t for t in all_tokens if t]
    if not all_tokens:
        logger.warning("分词后没有有效 tokens，结束。")
        return

    bigram_phraser = Phrases(all_tokens, min_count=2, threshold=4)
    tokens_with_bigrams = [bigram_phraser[doc] for doc in all_tokens]
    trigram_phraser = Phrases(tokens_with_bigrams, min_count=2, threshold=4)
    processed_tokens_all = [trigram_phraser[doc] for doc in tokens_with_bigrams]

    os.makedirs(output_path, exist_ok=True)
    token_output_path = os.path.join(output_path, 'processed_tokens.json')
    with open(token_output_path, 'w', encoding='utf-8') as f:
        json.dump(processed_tokens_all, f, ensure_ascii=False, indent=4)

    # 1) Overall graph for the meeting
    logger.info("构建会议总体共现图...")
    G_overall = build_cooccurrence_graph(processed_tokens_all, top_k_local=top_k)
    overall_out = os.path.join(output_path, 'cooccurrence_overall.png')
    draw_graph(G_overall, overall_out, f"{args.subtitle} (Overall)", subtitle_text=args.title)

    # 2) Per-agenda graphs
    if args.per_agenda:
        per_out_dir = os.path.join(output_path, 'per_agenda')
        os.makedirs(per_out_dir, exist_ok=True)
        tokens_by_agenda = defaultdict(list)
        subjects_by_agenda = defaultdict(list)
        for ag, sub, toks in zip(agendas, subjects, processed_tokens_all):
            tokens_by_agenda[ag].append(toks)
            if sub and isinstance(sub, str) and sub.strip():
                subjects_by_agenda[ag].append(sub.strip())

        logger.info(f"为 {len(tokens_by_agenda)} 个议题生成单独的共现图")
        per_agenda_meta = []
        for ag, toks_list in tokens_by_agenda.items():
            if not toks_list:
                continue
            subjects_list = list(dict.fromkeys(subjects_by_agenda.get(ag, [])))
            subject_text_full = '; '.join(subjects_list)
            subject_text_short = (subject_text_full[:120] + '...') if len(subject_text_full) > 120 else subject_text_full

            G_ag = build_cooccurrence_graph(toks_list, top_k_local=args.per_agenda_topk)
            n_nodes = 0 if (G_ag is None) else G_ag.number_of_nodes()
            n_edges = 0 if (G_ag is None) else G_ag.number_of_edges()

            # 跳过节点数量太少的议题
            if G_ag is None or n_nodes < args.min_agenda_nodes:
                logger.info(f"跳过议题 '{ag}'（节点数: {n_nodes}），Subjects: {subject_text_short}")
                per_agenda_meta.append({
                    'agenda': ag,
                    'subjects': subjects_list,
                    'nodes': n_nodes,
                    'edges': n_edges,
                    'status': 'skipped',
                    'reason': 'too_few_nodes'
                })
                continue

            safe_name = str(ag).replace('/', '_').replace(' ', '_')[:80]
            out_file_ag = os.path.join(per_out_dir, f'cooccurrence_agenda_{safe_name}.png')
            draw_graph(G_ag, out_file_ag, f"{args.subtitle} (Agenda: {ag})", subtitle_text=f"{args.title} — {subject_text_short}")

            per_agenda_meta.append({
                'agenda': ag,
                'subjects': subjects_list,
                'nodes': n_nodes,
                'edges': n_edges,
                'status': 'ok',
                'file': out_file_ag
            })

        # 保存 per-agenda 的元数据
        meta_out = os.path.join(per_out_dir, 'per_agenda_meta.json')
        with open(meta_out, 'w', encoding='utf-8') as mf:
            json.dump(per_agenda_meta, mf, ensure_ascii=False, indent=2)
        logger.info(f"Per-agenda metadata saved: {meta_out}")

    logger.info("共现图生成完毕。")

if __name__ == "__main__":
    main()