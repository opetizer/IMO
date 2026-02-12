import json
import argparse
import os
import re
from collections import defaultdict
import pandas as pd
import logging

# 尝试导入通用读取模块
try:
    from json_read import load_data
except ImportError:
    # 如果找不到模块，提供一个简单的 fallback 或者报错
    print("错误: 找不到 json_read.py 模块。请确保该文件在 src 目录下。")
    exit(1)

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def parse_args():
    parser = argparse.ArgumentParser(description="Process JSON data for word frequency and metadata splitting.")
    parser.add_argument('--title', type=str, required=True, help="Main topic folder, e.g., MEPC")
    parser.add_argument('--subtitle', type=str, required=True, help="Sub topic folder, e.g., MEPC 77")
    return parser.parse_args()

def split_originator(originator_str):
    """拆分 Originator 字段中的多个国家/组织"""
    if not isinstance(originator_str, str):
        return []
    # 替换常见的连接词
    cleaned = originator_str.replace(' and ', ', ').replace(';', ',')
    # 拆分并去除空格
    return [item.strip() for item in cleaned.split(',') if item.strip()]

def get_stopwords():
    """定义停用词集合"""
    # 基础停用词
    from nltk.corpus import stopwords
    from stopword import additional_stopwords
    # 添加自定义停用词
    custom_stopwords = additional_stopwords.copy()
    custom_stopwords.update(set(stopwords.words('english')))
    return custom_stopwords

def main():
    args = parse_args()
    
    # 构建路径
    input_path = os.path.join('output', args.title, args.subtitle, 'data.json')
    output_processed_path = os.path.join('output', args.title, args.subtitle, 'data_processed.json')
    
    logger.info(f"正在处理数据: {input_path}")
    
    # 1. 使用 load_data 加载数据 (自动处理 content 里的 JSON 字符串)
    df = load_data(input_path)
    
    if df.empty:
        logger.error("数据为空或无法加载，退出。")
        return

    # 2. 数据处理
    logger.info("正在执行字段拆分和清洗...")
    
    # 拆分 Originator
    df['Originator_split'] = df['Originator'].apply(split_originator)
    
    # 3. 词频统计 (基于 full_text)
    logger.info("正在计算词频...")
    all_text = " ".join(df['full_text'].dropna().astype(str).tolist())
    
    # 简单的正则分词
    words = re.findall(r'\b[a-z]{2,}\b', all_text.lower())
    
    stop_words = get_stopwords()
    word_counts = defaultdict(int)
    
    for word in words:
        if word not in stop_words:
            word_counts[word] += 1
            
    sorted_word_counts = sorted(word_counts.items(), key=lambda item: item[1], reverse=True)

    # 打印 Top 30 词频
    print("\n--- 词频统计 (Top 30) ---")
    for word, count in sorted_word_counts[:30]:
        print(f"{word}: {count}")

    # 4. 保存处理后的数据
    # 将 DataFrame 转换回字典列表以便保存为 JSON
    # 注意：pandas 的 to_json 可能不会完美保留原有嵌套结构，
    # 这里我们构建一个新的 dict 列表，保留关键字段用于前端或后续使用
    
    output_records = df.to_dict(orient='records')
    
    # 清理一下不需要的大文本字段（如果需要减小文件体积），或者保留
    # 这里选择保留并添加处理后的字段
    
    with open(output_processed_path, 'w', encoding='utf-8') as f:
        json.dump(output_records, f, ensure_ascii=False, indent=4)
        
    logger.info(f"\n处理完成！已保存至: {output_processed_path}")
    
    # 打印样本数据预览
    print("\n--- 处理后数据样本 (前1条) ---")
    if output_records:
        sample = output_records[0].copy()
        # 截断长文本以便打印预览
        if sample.get('full_text'):
            sample['full_text'] = sample['full_text'][:100] + "..."
        print(json.dumps(sample, indent=2, ensure_ascii=False))

if __name__ == "__main__":
    main()