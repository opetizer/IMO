# src/topic.py

import os
import nltk
import json
import argparse
import logging
from datetime import datetime
import pandas as pd

# --- Gensim for Topic Modeling ---
import gensim
import gensim.corpora as corpora
from gensim.models import Phrases, LdaModel

from nltk.corpus import wordnet, stopwords
from nltk.stem import WordNetLemmatizer

# --- 引入新的数据读取模块 ---
from json_read import load_data

# --- Setup: Argument Parser, Logger, Stopwords ---

parser = argparse.ArgumentParser(description="Perform Topic Modeling on MEPC documents.")
parser.add_argument('--title', type=str, required=True)
parser.add_argument('--subtitle', type=str, required=True)
parser.add_argument('--logging', type=str, required=True)
parser.add_argument('--text_extracted_folder', type=str, required=True)
parser.add_argument('--num_topics', type=int, default=10, help="The number of topics to discover.")
parser.add_argument('--countries', type=str, nargs='*', default=None,
                    help="List of countries (Originator) to filter by. e.g., --countries China Japan")
parser.add_argument('--start_date', type=str, default=None,
                    help="Start date for filtering (YYYY-MM-DD).")
parser.add_argument('--end_date', type=str, default=None,
                    help="End date for filtering (YYYY-MM-DD).")
args = parser.parse_args()

text_folder = args.text_extracted_folder
output_path = f'output/{args.title}/{args.subtitle}'
json_file_path = os.path.join(text_folder, "data.json")

stop_words = set(stopwords.words('english'))
from stopword import additional_stopwords
stop_words.update(additional_stopwords)

def setup_logger(log_file):
    logger = logging.getLogger(args.title)
    if logger.hasHandlers():
        logger.handlers.clear()
    logger.setLevel(logging.INFO)
    file_handler = logging.FileHandler(log_file, mode='w')
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
    if not isinstance(text, str):
        return []
    tagged_tokens = nltk.pos_tag(nltk.word_tokenize(text))
    lemmas = []
    for word, tag in tagged_tokens:
        if tag.startswith('NN'): 
            pos = get_wordnet_pos(tag)
            lemma = lemmatizer.lemmatize(word.lower(), pos=pos)
            if len(lemma) > 1 and lemma.isalpha() and lemma not in stop_words:
                lemmas.append(lemma)
    return lemmas


# --- Main Execution ---

def main():
    download_nltk_data()
    
    # --- Step 1: Load and Filter Data using json_read ---
    logger.info("--- Step 1: Loading and Filtering Documents ---")
    
    if not os.path.exists(json_file_path):
        logger.error(f"data.json not found in folder: {text_folder}")
        return

    # 使用 json_read 加载并处理数据
    df = load_data(json_file_path)
    
    if df.empty:
        logger.warning("Loaded DataFrame is empty. Exiting.")
        return

    # --- Apply Filters ---
    
    # 1. Date Filter
    try:
        start_date_obj = datetime.strptime(args.start_date, '%Y-%m-%d') if args.start_date else None
        end_date_obj = datetime.strptime(args.end_date, '%Y-%m-%d') if args.end_date else None
    except ValueError:
        logger.error("Invalid date format. Please use YYYY-MM-DD.")
        return

    if start_date_obj or end_date_obj:
        # Convert DataFrame 'Date' to datetime objects for filtering
        # Assuming format in JSON is 'DD/MM/YYYY' or similar. 
        # But data.json might contain raw strings like '22 November 2021'.
        # For simplicity, we try parsing; if fail, we skip date filtering or handle carefully.
        # json_read keeps 'Date' as string.
        
        # Helper to parse date safely
        def parse_date_safe(d_str):
            try:
                return datetime.strptime(d_str, '%d/%m/%Y')
            except (ValueError, TypeError):
                return None
                
        df['dt_obj'] = df['Date'].apply(parse_date_safe)
        
        if start_date_obj:
            df = df[df['dt_obj'] >= start_date_obj]
        if end_date_obj:
            df = df[df['dt_obj'] <= end_date_obj]

    # 2. Country Filter
    if args.countries:
        countries_to_check = [c.lower() for c in args.countries]
        # Check if Originator contains any of the countries
        # Originator might be NaN
        df['Originator'] = df['Originator'].fillna('')
        mask = df['Originator'].str.lower().apply(lambda x: any(c in x for c in countries_to_check))
        df = df[mask]
        
    logger.info(f"Loaded and filtered {len(df)} documents.")
    
    # Ensure full_text is not empty
    df = df[df['full_text'].notna() & (df['full_text'].str.strip() != '')]
    
    if df.empty:
        logger.warning("No documents left after filtering. Exiting.")
        return

    documents_metadata = df.to_dict('records') # Convert back to list of dicts for iteration

    # --- Step 2: Tokenizing and Phrase Detection ---
    logger.info("--- Step 2: Tokenizing and Phrase Detection ---")
    
    all_docs_text = [doc['full_text'] for doc in documents_metadata]
    all_tokens = [lemmatizing_tokenizer(doc) for doc in all_docs_text]
    
    # Remove empty token lists
    valid_indices = [i for i, tokens in enumerate(all_tokens) if tokens]
    all_tokens = [all_tokens[i] for i in valid_indices]
    documents_metadata = [documents_metadata[i] for i in valid_indices]
    
    if not all_tokens:
        logger.error("No valid tokens found after preprocessing.")
        return

    phrases = Phrases(all_tokens, min_count=2, threshold=4)
    tokens_with_bigrams = [phrases[doc] for doc in all_tokens]
    
    logger.info("Tokenization and phrase (bigram) detection complete.")

    # --- Step 3: Topic Modeling with LDA ---
    logger.info("--- Step 3: Building Dictionary and Corpus for LDA ---")
    
    id2word = corpora.Dictionary(tokens_with_bigrams)
    # Filter extremes
    id2word.filter_extremes(no_below=3, no_above=0.6) # Adjusted slightly for smaller datasets
    
    corpus = [id2word.doc2bow(text) for text in tokens_with_bigrams]

    logger.info(f"Dictionary contains {len(id2word)} unique tokens. Corpus contains {len(corpus)} documents.")

    logger.info(f"--- Step 4: Training LDA Topic Model with {args.num_topics} topics ---")
    
    lda_model = LdaModel(corpus=corpus,
                         id2word=id2word,
                         num_topics=args.num_topics,
                         random_state=100,
                         update_every=1,
                         chunksize=100,
                         passes=10,
                         alpha='auto',
                         per_word_topics=True)

    logger.info("LDA model training complete.")

    # --- Step 5: Saving Outputs ---
    
    # 确保输出目录存在
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    # 1. Save topics summary
    topics_summary_path = os.path.join(output_path, 'topics_summary.txt')
    with open(topics_summary_path, 'w', encoding='utf-8') as f:
        f.write(f"LDA Model Results: {args.num_topics} Topics\n")
        f.write("-" * 30 + "\n")
        topics = lda_model.print_topics(num_words=10)
        for topic in topics:
            f.write(f"Topic #{topic[0]}: \n{topic[1]}\n\n")
    logger.info(f"Topics summary saved to: {topics_summary_path}")

    # 2. Save topics as CSV
    csv_path = os.path.join(output_path, 'topics_table.csv')
    try:
        with open(csv_path, 'w', encoding='utf-8') as f:
            f.write("Topic_ID")
            for i in range(10): # Top 10 words
                f.write(f",Word_{i+1},Probability_{i+1}")
            f.write("\n")
            for topic in topics:
                f.write(f"{topic[0]}")
                # Parse "0.050*word + 0.030*word2..."
                parts = topic[1].split("+")
                for part in parts:
                    if "*" in part:
                        prob, word = part.split("*")
                        word = word.strip().strip('"')
                        f.write(f",{word},{prob.strip()}")
                f.write("\n")
    except Exception as e:
        logger.error(f"Error saving topics table: {e}")

    # 3. Save document-topic distribution
    topic_dist_list = []
    for i, doc_corpus in enumerate(corpus):
        doc_topics = lda_model.get_document_topics(doc_corpus, minimum_probability=0)
        
        # Sort to find dominant topic
        doc_topics_sorted = sorted(doc_topics, key=lambda x: x[1], reverse=True)
        dominant_topic = doc_topics_sorted[0][0] if doc_topics_sorted else -1

        row = {
            "file_name": documents_metadata[i].get("file_name", ""),
            "Date": documents_metadata[i].get("Date", ""),
            "Originator": documents_metadata[i].get("Originator", ""),
            "Title": documents_metadata[i].get("Title", ""),
            "Dominant_Topic": dominant_topic
        }
        for topic_num, prop in doc_topics:
            row[f"Topic_{topic_num}"] = round(prop, 4)
        
        topic_dist_list.append(row)

    df_topic_dist = pd.DataFrame(topic_dist_list)
    
    # Fill missing columns
    for i in range(args.num_topics):
        if f"Topic_{i}" not in df_topic_dist.columns:
            df_topic_dist[f"Topic_{i}"] = 0.0
            
    # Reorder columns
    cols = ["file_name", "Date", "Originator", "Title", "Dominant_Topic"] + [f"Topic_{i}" for i in range(args.num_topics)]
    # Filter cols that actually exist in df
    cols = [c for c in cols if c in df_topic_dist.columns]
    df_topic_dist = df_topic_dist[cols]

    doc_topic_dist_path = os.path.join(output_path, 'document_topic_distribution.csv')
    df_topic_dist.to_csv(doc_topic_dist_path, index=False, encoding='utf-8-sig')
    logger.info(f"Document-topic distributions saved to: {doc_topic_dist_path}")
    logger.info("Topic modeling process finished successfully!")

if __name__ == "__main__":
    main()