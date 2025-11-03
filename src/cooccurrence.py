import os
import nltk
import json
import networkx as nx
import seaborn as sns
import matplotlib.pyplot as plt
from collections import defaultdict
from nltk.corpus import wordnet, stopwords
from nltk.stem import WordNetLemmatizer
import argparse
import logging
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
import community as community_louvain
from adjustText import adjust_text
# --- 新增：导入gensim 和 datetime ---
from gensim.models import Phrases
from datetime import datetime

# 设置seaborn样式
sns.set_theme(style="whitegrid", palette="pastel")
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
EDGE_WEIGHT_THRESHOLD = 5.0

parser = argparse.ArgumentParser()
parser.add_argument('--title', type=str, required=True)
parser.add_argument('--subtitle', type=str, required=True)
parser.add_argument('--logging', type=str, required=True)
parser.add_argument('--text_extracted_folder', type=str, required=True)

parser.add_argument('--countries', type=str, nargs='*', default=None,
                    help="用于筛选的国家列表 (Originator)。例如: --countries China Japan")
parser.add_argument('--start_date', type=str, default=None,
                    help="筛选的开始日期 (格式: YYYY-MM-DD)。")
parser.add_argument('--end_date', type=str, default=None,
                    help="筛选的结束日期 (格式: YYYY-MM-DD)。")
args = parser.parse_args()

text_folder = args.text_extracted_folder
top_k = 100
window_size = 5
output_path = f'output/{args.title}/{args.subtitle}'
co_output_path = os.path.join(output_path, 'cooccurrence_graph.png')
freq_output_path = os.path.join(output_path, 'word_freq.json')
stop_words = set(stopwords.words('english'))
from stopword import additional_stopwords

# Add these new words to the main stopword set
stop_words.update(additional_stopwords)
# 设定 data.json 的文件路径
json_file_path = os.path.join(text_folder, "data.json")

def setup_logger(log_file):
    # (这部分代码保持不变)
    logger = logging.getLogger(args.title)
    if logger.hasHandlers(): # 防止重复添加handler
        logger.handlers.clear()
    logger.setLevel(logging.INFO)
    file_handler = logging.FileHandler(log_file, mode='w') # 每次运行覆盖日志
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
    # (这部分代码保持不变)
    nltk.download('averaged_perceptron_tagger', quiet=True)
    nltk.download('punkt', quiet=True)
    nltk.download('stopwords', quiet=True) 
    nltk.download('wordnet', quiet=True)

# 定义词形还原器和词性转换函数（全局，避免重复创建）
lemmatizer = WordNetLemmatizer()
def get_wordnet_pos(treebank_tag):
    if treebank_tag.startswith('J'):
        return wordnet.ADJ
    elif treebank_tag.startswith('V'):
        return wordnet.VERB
    elif treebank_tag.startswith('N'):
        return wordnet.NOUN
    elif treebank_tag.startswith('R'):
        return wordnet.ADV
    else:
        return wordnet.NOUN

def lemmatizing_tokenizer(text):
    tagged_tokens = nltk.pos_tag(nltk.word_tokenize(text))
    lemmas = []
    for word, tag in tagged_tokens:
        # Only process words that are nouns (NN, NNS, NNP, NNPS)
        if tag.startswith('NN'): 
            pos = get_wordnet_pos(tag)
            lemma = lemmatizer.lemmatize(word.lower(), pos=pos)
            # Check for stopwords *after* lemmatizing
            if len(lemma) > 1 and lemma.isalpha() and lemma not in stop_words:
                lemmas.append(lemma)
    return lemmas

def main():
    download_nltk_data()
    documents = []
    
    try:
        # 解析 YYYY-MM-DD 格式的日期参数
        start_date_obj = datetime.strptime(args.start_date, '%Y-%m-%d') if args.start_date else None
        end_date_obj = datetime.strptime(args.end_date, '%Y-%m-%d') if args.end_date else None
    except ValueError:
        logger.error("无效的日期格式。请使用 YYYY-MM-DD。")
        return

    # 准备国家列表以进行不区分大小写的检查
    countries_to_check = [c.lower() for c in args.countries] if args.countries else None
    
    logger.info(f"应用筛选器 - 国家: {args.countries}, 开始日期: {args.start_date}, 结束日期: {args.end_date}")

    # --- 步骤 2: 加载和筛选文档 ---
    if os.path.exists(json_file_path):
        with open(json_file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            
            filtered_doc_count = 0  # 用于记录通过筛选的文档数
            
            for file_content in data:
                
                # --- 应用筛选器 ---
                
                # 1. 国家 (Originator) 筛选
                if countries_to_check:
                    originator_str = file_content.get('Originator', '').lower()
                    # 检查 'Originator' 字符串中是否包含 *任何* 目标国家
                    if not any(country in originator_str for country in countries_to_check):
                        continue  # 如果不匹配，则跳过此文档

                # 2. 日期 (Date) 筛选
                item_date_str = file_content.get('Date')
                if (start_date_obj or end_date_obj) and item_date_str:
                    try:
                        # 解析 data.json 中的 DD/MM/YYYY 格式
                        item_date_obj = datetime.strptime(item_date_str, '%d/%m/%Y')
                        
                        if start_date_obj and item_date_obj < start_date_obj:
                            continue  # 跳过：早于开始日期
                        if end_date_obj and item_date_obj > end_date_obj:
                            continue  # 跳过：晚于结束日期
                            
                    except ValueError:
                        logger.warning(f"无法解析文档 {file_content.get('Symbol')} 的日期 '{item_date_str}'")
                        continue  # 如果日期格式错误，跳过

                # --- 筛选器应用完毕 ---

                # --- 步骤 3: 提取文本（去除最后一段） ---
                content_list = file_content.get('content')
                if not content_list:  # 如果没有内容，跳过
                    continue
                
                # 使用切片 [:-1] 来获取除最后一个元素外的所有段落
                text = " ".join(item['text'] for item in content_list[:-1])
                
                if not text:  # 如果（在去除最后一段后）文本为空，则跳过
                    continue
                
                # --- 提取完毕 ---

                documents.append(text)
                filtered_doc_count += 1
                
    else:
        logger.error(f"在文件夹中未找到 data.json: {text_folder}")
        return
    
    logger.info(f"应用筛选器后加载了 {len(documents)} 篇文档 (共找到 {filtered_doc_count} 篇匹配的文档)。")
    
    # --- ^ ^ ^ --- 修改结束 --- ^ ^ ^ ---

    # --- 步骤 4: 对所有文档进行分词和词形还原 ---
    logger.info("正在对所有文档进行分词和词形还原...")
    all_tokens = [lemmatizing_tokenizer(doc) for doc in documents]
    
    # --- 步骤 5: 使用 Gensim 训练和应用 Bigram 模型 ---
    logger.info("正在使用 gensim 训练 bigram 模型...")
    # min_count: 词组至少出现的次数, threshold: 成组的评分阈值，越高越难成组
    phrases = Phrases(all_tokens, min_count=2, threshold=4) 
    bigram_phraser = phrases
    # 将分词列表转换为包含bigram的列表
    tokens_with_bigrams = [bigram_phraser[doc] for doc in all_tokens]
    logger.info("Bigram 模型训练和应用完成。")
    # 打印一个样本看看效果
    if tokens_with_bigrams and len(tokens_with_bigrams[0]) > 0:
        logger.info(f"包含 bigram 的词元示例: {';'.join(tokens_with_bigrams[0][:20])}...")


    # --- 步骤 6: 改造 TF-IDF Vectorizer ---
    logger.info("正在对包含 bigram 的词元运行 TF-IDF...")
    
    vectorizer = TfidfVectorizer(
        tokenizer=lambda x: x,
        preprocessor=lambda x: x,
        lowercase=False,
        max_features=top_k
    )
    # 用包含bigram的列表来训练
    tfidf_matrix = vectorizer.fit_transform(tokens_with_bigrams)
    
    feature_names = vectorizer.get_feature_names_out()
    tfidf_scores = tfidf_matrix.sum(axis=0).A1
    word_tfidf = dict(zip(feature_names, tfidf_scores))
    sorted_words = sorted(word_tfidf.items(), key=lambda x: x[1], reverse=True)
    top_words = [word for word, score in sorted_words] # top_k已经在vectorizer里限制了

    with open(freq_output_path, 'w', encoding='utf-8') as f:
        json.dump(sorted_words, f, ensure_ascii=False, indent=4)
    logger.info(f"词频数据已保存至: {freq_output_path}")
    logger.info(f"基于 TF-IDF 的前 {len(top_words)} 个词/短语: {top_words[:top_k]}")

    # --- 步骤 7: 更新共现网络的数据源 ---
    logger.info("正在构建共现图...")
    word_counts = defaultdict(int)
    cooccurrence = defaultdict(int)
    
    # 直接使用已经处理好的 tokens_with_bigrams，不再重新分词
    for tokens in tokens_with_bigrams:
        filtered_tokens = [w for w in tokens if w in top_words]
        for word in filtered_tokens:
            word_counts[word] += 1
        for i in range(len(filtered_tokens)):
            for j in range(i + 1, min(i + window_size, len(filtered_tokens))):
                pair = tuple(sorted((filtered_tokens[i], filtered_tokens[j])))
                cooccurrence[pair] += 1

    # --- 网络图构建与可视化 (这部分代码保持不变) ---
    G = nx.Graph()
    for word in top_words:
        if word_counts[word] > 0:
            G.add_node(word, size=np.sqrt(word_counts[word]))

    for (w1, w2), freq in cooccurrence.items():
        if w1 in G and w2 in G and w1 != w2 and freq > EDGE_WEIGHT_THRESHOLD:
            G.add_edge(w1, w2, weight=freq)

    if not G.nodes():
        logger.warning("图中没有节点。边缘权重阈值可能太高或未找到共现关系。")
        return

    # 社群发现
    partition = community_louvain.best_partition(G)
    num_communities = len(set(partition.values()))
    cmap = plt.get_cmap('Set3', max(num_communities, 8)) # 确保颜色数量足够
    node_colors = [cmap(partition[n]) for n in G.nodes()]

    # 可视化
    plt.figure(figsize=(18, 14), facecolor='white')
    pos = nx.spring_layout(G, k=0.9, seed=42, iterations=150)
    node_sizes = [G.nodes[n]['size'] * 100 for n in G.nodes()]
    edge_weights = list(nx.get_edge_attributes(G, 'weight').values())
    
    if edge_weights: # 避免权重为空列表
        min_w, max_w = min(edge_weights), max(edge_weights)
        edge_alphas = [(w - min_w) / (max_w - min_w) * 0.7 + 0.1 if max_w > min_w else 0.5 for w in edge_weights]
    else:
        edge_alphas = 0.5

    nx.draw_networkx_nodes(G, pos, node_size=node_sizes, node_color=node_colors, alpha=0.9)
    nx.draw_networkx_edges(G, pos, width=2.0, edge_color='grey', alpha=edge_alphas)

    texts = []
    for node, (x, y) in pos.items():
        font_size = 10 + np.sqrt(G.nodes[node]['size'])
        texts.append(plt.text(x, y, node.replace('_', ' '), fontsize=font_size, ha='center', va='center'))

    adjust_text(texts, arrowprops=dict(arrowstyle='-', color='gray', lw=0.5))

    plt.title(f"{args.subtitle} 关键词共现图", fontsize=20, pad=20)
    plt.axis('off')
    plt.savefig(co_output_path, dpi=300, bbox_inches='tight')
    
    logger.info(f"优化后的共现图已保存至: {co_output_path}")

if __name__ == "__main__":
    main()