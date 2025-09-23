# main.py

import os
import fitz  # PyMuPDF
import json
import argparse
import pandas as pd
from bs4 import BeautifulSoup
import nltk
from nltk.corpus import stopwords
from collections import Counter
from tqdm import tqdm
import logging

parser = argparse.ArgumentParser()
parser.add_argument('--title', type=str, required=True)
parser.add_argument('--subtitle', type=str, required=True)
parser.add_argument('--logging', type=str, required=True)
args = parser.parse_args()

def setup_logger(log_file):
    logger = logging.getLogger(args.title)
    logger.setLevel(logging.INFO)
    # 创建文件处理器
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.INFO)
    # 创建控制台处理器
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    # 创建格式化器
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)
    # 添加处理器到logger
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    return logger

logger = setup_logger(args.logging)

# 初始化nltk停用词
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

# 读取 index.html，提取PDF文件名
def extract_pdf_list(index_html_path):
    with open(index_html_path, 'r', encoding='utf-8') as f:
        soup = BeautifulSoup(f, 'lxml')
    pdf_files = []
    for link in soup.find_all('a'):
        href = link.get('href')
        if href and href.endswith('.pdf'):
            pdf_files.append(href)
    return pdf_files

# 提取PDF文本并保存为txt
def extract_text_from_pdfs(pdf_list, pdf_folder_path, output_folder):
    os.makedirs(output_folder, exist_ok=True)
    for pdf_file in tqdm(pdf_list, desc="Extracting PDFs"):
        pdf_path = os.path.join(pdf_folder_path, pdf_file)
        output_txt_path = os.path.join(output_folder, pdf_file.replace('.pdf', '.txt'))
        if not os.path.exists(pdf_path):
            logger.info(f"Warning: {pdf_file} not found.")
            continue
        try:
            doc = fitz.open(pdf_path)
            text = ""
            for page in doc:
                text += page.get_text()
            doc.close()
            with open(output_txt_path, 'w', encoding='utf-8') as f:
                f.write(text)
        except Exception as e:
            logger.info(f"Error processing {pdf_file}: {e}")

# 统计词频
def simple_word_count(text_folder):
    all_text = ""
    for txt_file in os.listdir(text_folder):
        if txt_file.endswith('.txt'):
            with open(os.path.join(text_folder, txt_file), 'r', encoding='utf-8') as f:
                all_text += f.read().lower()
    words = [word for word in all_text.split() if word.isalpha()]
    words = [word for word in words if word not in stop_words]
    counter = Counter(words)
    return dict(counter.most_common(50))

def main():
    
    title = args.title
    subtitle = args.subtitle

    # 路径设定
    index_html_path = f'data/{title}/{subtitle}/Index.htm'
    pdf_folder_path = f'data/{title}/{subtitle}/'
    text_output_folder = f'extracted_texts/{title}/{subtitle}/'
    json_output_path = f'output/{title}/{subtitle}/word_freq.json'

    os.makedirs(os.path.dirname(json_output_path), exist_ok=True)

    logger.info(f"Step 1: Extracting PDF list from {index_html_path}")
    pdf_list = extract_pdf_list(index_html_path)

    logger.info(f"Step 2: Extracting text from PDFs into {text_output_folder}")
    extract_text_from_pdfs(pdf_list, pdf_folder_path, text_output_folder)

    logger.info(f"Step 3: Performing word frequency analysis")
    word_freq = simple_word_count(text_output_folder)
    with open(json_output_path, 'w', encoding='utf-8') as f:
        json.dump(word_freq, f, indent=2)

if __name__ == "__main__":
    main()
