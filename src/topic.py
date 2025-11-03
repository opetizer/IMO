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

# --- Setup: Argument Parser, Logger, Stopwords (Copied from cooccurrence.py) ---

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
    
    # --- Step 1: Load, Filter, and Preprocess Data (Re-used from cooccurrence.py) ---
    logger.info("--- Step 1: Loading and Filtering Documents ---")
    documents_metadata = [] # Store text along with metadata

    try:
        start_date_obj = datetime.strptime(args.start_date, '%Y-%m-%d') if args.start_date else None
        end_date_obj = datetime.strptime(args.end_date, '%Y-%m-%d') if args.end_date else None
    except ValueError:
        logger.error("Invalid date format. Please use YYYY-MM-DD.")
        return

    countries_to_check = [c.lower() for c in args.countries] if args.countries else None
    logger.info(f"Applying filters - Countries: {args.countries}, Start Date: {args.start_date}, End Date: {args.end_date}")

    if not os.path.exists(json_file_path):
        logger.error(f"data.json not found in folder: {text_folder}")
        return

    with open(json_file_path, "r", encoding="utf-8") as f:
        data = json.load(f)
        for file_content in data:
            # Apply filters
            if countries_to_check:
                originator_str = file_content.get('Originator', '').lower()
                if not any(country in originator_str for country in countries_to_check):
                    continue
            
            item_date_str = file_content.get('Date')
            if (start_date_obj or end_date_obj) and item_date_str:
                try:
                    item_date_obj = datetime.strptime(item_date_str, '%d/%m/%Y')
                    if (start_date_obj and item_date_obj < start_date_obj) or \
                       (end_date_obj and item_date_obj > end_date_obj):
                        continue
                except ValueError:
                    logger.warning(f"Could not parse date '{item_date_str}' for document {file_content.get('Symbol')}")
                    continue

            # Extract text (excluding last paragraph)
            content_list = file_content.get('content')
            if not content_list: continue
            
            text = " ".join(item['text'] for item in content_list[:-1])
            if not text: continue
            
            documents_metadata.append({
                "file_name": file_content.get("file_name", "N/A"),
                "Date": file_content.get("Date", "N/A"),
                "Originator": file_content.get("Originator", "N/A"),
                "text": text
            })

    logger.info(f"Loaded and filtered {len(documents_metadata)} documents.")
    if not documents_metadata:
        logger.warning("No documents left after filtering. Exiting.")
        return

    logger.info("--- Step 2: Tokenizing and Phrase Detection ---")
    all_docs_text = [doc['text'] for doc in documents_metadata]
    all_tokens = [lemmatizing_tokenizer(doc) for doc in all_docs_text]
    
    phrases = Phrases(all_tokens, min_count=2, threshold=4)
    tokens_with_bigrams = [phrases[doc] for doc in all_tokens]
    
    logger.info("Tokenization and phrase (bigram) detection complete.")

    # --- Step 3: Topic Modeling with LDA ---
    logger.info("--- Step 3: Building Dictionary and Corpus for LDA ---")
    
    # Create Dictionary
    id2word = corpora.Dictionary(tokens_with_bigrams)
    # Filter out extremes
    id2word.filter_extremes(no_below=7, no_above=0.5)
    # Create Corpus: Term Document Frequency
    corpus = [id2word.doc2bow(text) for text in tokens_with_bigrams]

    logger.info(f"Dictionary contains {len(id2word)} unique tokens. Corpus contains {len(corpus)} documents.")

    logger.info(f"--- Step 4: Training LDA Topic Model with {args.num_topics} topics ---")
    
    # Build LDA model
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
    logger.info("--- Step 5: Saving topic summary and document distributions ---")

    # 1. Save the topics summary
    topics_summary_path = os.path.join(output_path, 'topics_summary.txt')
    with open(topics_summary_path, 'w', encoding='utf-8') as f:
        f.write(f"LDA Model Results: {args.num_topics} Topics\n")
        f.write("-" * 30 + "\n")
        topics = lda_model.print_topics(num_words=10)
        for topic in topics:
            f.write(f"Topic #{topic[0]}: \n{topic[1]}\n\n")
    logger.info(f"Topics summary saved to: {topics_summary_path}")

    # 2. 生成并保存为表格 (CSV 文件)
    csv_path = os.path.join(output_path, 'topics_table.csv')
    logger.info(f"正在将主题结果保存为表格: {csv_path}")
    
    try:
        with open(csv_path, 'w', encoding='utf-8') as f:
            f.write("Topics")
            for i in range(args.num_topics):
                f.write(f",Word_{i+1},Probability_{i+1}")
            f.write("\n")
            for topic in topics:
                f.write(f"{topic[0]}")
                for word_prob in topic[1].split(" + "):
                    prob, word = word_prob.split("*")
                    word = word.strip().strip('"')
                    f.write(f",{word},{prob}")
                f.write("\n")
        logger.info(f"主题表格已保存至: {csv_path}")

    except Exception as e:
        logger.error(f"保存 CSV 表格时出错: {e}")

    # 2. Save the document-topic distribution to CSV
    topic_dist_list = []
    for i, doc_corpus in enumerate(corpus):
        doc_topics = lda_model.get_document_topics(doc_corpus, minimum_probability=0)
        
        # Find the dominant topic
        dominant_topic = sorted(doc_topics, key=lambda x: x[1], reverse=True)[0][0]

        # Create a dictionary for the row
        row = {
            "file_name": documents_metadata[i]["file_name"],
            "Date": documents_metadata[i]["Date"],
            "Originator": documents_metadata[i]["Originator"],
            "Dominant_Topic": dominant_topic
        }
        # Add scores for each topic
        for topic_num, prop in doc_topics:
            row[f"Topic_{topic_num}"] = round(prop, 4)
        
        topic_dist_list.append(row)

    df_topic_dist = pd.DataFrame(topic_dist_list)
    # Fill NaN for topics not present in a document
    for i in range(args.num_topics):
        if f"Topic_{i}" not in df_topic_dist.columns:
            df_topic_dist[f"Topic_{i}"] = 0
    df_topic_dist.fillna(0, inplace=True)
    
    # Reorder columns
    cols = ["file_name", "Date", "Originator", "Dominant_Topic"] + [f"Topic_{i}" for i in range(args.num_topics)]
    df_topic_dist = df_topic_dist[cols]


    doc_topic_dist_path = os.path.join(output_path, 'document_topic_distribution.csv')
    df_topic_dist.to_csv(doc_topic_dist_path, index=False, encoding='utf-8-sig')
    logger.info(f"Document-topic distributions saved to: {doc_topic_dist_path}")
    logger.info("-" * 30)
    logger.info("Topic modeling process finished successfully!")


if __name__ == "__main__":
    main()