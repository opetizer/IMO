import json
import re
from collections import defaultdict

DATA_DIR = 'data'
TOPIC = 'MEPC'
SESSION = 'MEPC 77'
INDEX_FILE = 'Index.htm'
OUTPUT_PATH = f'output/{SESSION}'
OUTPUT_JSON_PATH = f'{OUTPUT_PATH}/data.json'
PROCESSED_JSON_PATH = f'{OUTPUT_PATH}/data_processed.json'

def split_related_documents_final(related_docs):
    """
    Splits the related_document field based on specific document prefixes
    and delimiters like 'and', ';'.
    """
    if not related_docs:
        return []

    # If it's a list with multiple items, assume it's pre-processed
    if isinstance(related_docs, list) and len(related_docs) > 1:
        return related_docs
    
    # If it's a list with one item, treat it as a string
    if isinstance(related_docs, list) and len(related_docs) == 1:
        text = related_docs[0]
    elif isinstance(related_docs, str):
        text = related_docs
    else:
        return []

    # Normalize delimiters to a single character for splitting
    text = text.replace(' and ', ';').replace('; ', ';')
    
    # Initial split by the normalized delimiter
    initial_split = text.split(';')

    final_docs = []
    # Define the prefixes to identify the start of a new document
    prefixes = ["MEPC", "MSC", "SSE", "ISWG-GHG", "CCC", "Resolution", "PPR"]

    for item in initial_split:
        item = item.strip()
        
        # Create a regex pattern to find prefixes within a single item
        pattern = r'(' + '|'.join(prefixes) + r')'
        
        # Find all start indices of the prefixes
        indices = [m.start() for m in re.finditer(pattern, item)]

        if len(indices) > 1:
            # If multiple prefixes are found in one item (e.g., "MEPC 76/9/1 MEPC 76/9/2")
            for i in range(len(indices)):
                start = indices[i]
                end = indices[i+1] if i + 1 < len(indices) else len(item)
                doc_str = item[start:end].strip()
                final_docs.append(doc_str)
        elif item:
            # If only one or no prefix is found, add the item as is
            final_docs.append(item)
            
    # Clean up any empty strings that might have been added
    final_docs = [doc for doc in final_docs if doc]
    
    return final_docs


try:
    # 1. 读取 data.json 文件
    with open(OUTPUT_JSON_PATH, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # 2. 将 Originator 里面的国家根据 "," 和 "and" 拆分
    for item in data:
        originator = item.get('Originator', '')
        if originator:
            originator = originator.replace(' and ', ', ')
            item['Originator_split'] = [country.strip() for country in originator.split(',') if country.strip()]

    # 3. 将 related document 拆分
    for item in data:
        related_docs_raw = item.get('summary', {}).get('related_document')
        item['related_document_split'] = split_related_documents_final(related_docs_raw)

    # 准备文本内容
    all_text = ""
    for item in data:
        text_content = []
        text_content.append(item.get('Title', ''))
        
        summary = item.get('summary', {})
        if summary and isinstance(summary, dict):
            text_content.append(summary.get('text', ''))

        content = item.get('content', [])
        if content and isinstance(content, list):
            for paragraph in content:
                if isinstance(paragraph, dict):
                    text_content.append(paragraph.get('text', ''))
        all_text += "\n".join(text_content)
        
    # 4. 生成 word frequency
    stop_words = set(['i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', 'your', 'yours', 'yourself', 'yourselves', 'he', 'him', 'his', 'himself', 'she', 'her', 'hers', 'herself', 'it', 'its', 'itself', 'they', 'them', 'their', 'theirs', 'themselves', 'what', 'which', 'who', 'whom', 'this', 'that', 'these', 'those', 'am', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having', 'do', 'does', 'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until', 'while', 'of', 'at', 'by', 'for', 'with', 'about', 'against', 'between', 'into', 'through', 'during', 'before', 'after', 'above', 'below', 'to', 'from', 'up', 'down', 'in', 'out', 'on', 'off', 'over', 'under', 'again', 'further', 'then', 'once', 'here', 'there', 'when', 'where', 'why', 'how', 'all', 'any', 'both', 'each', 'few', 'more', 'most', 'other', 'some', 'such', 'no', 'nor', 'not', 'only', 'own', 'same', 'so', 'than', 'too', 'very', 's', 't', 'can', 'will', 'just', 'don', 'should', 'now', 'd', 'll', 'm', 'o', 're', 've', 'y', 'ain', 'aren', 'couldn', 'didn', 'doesn', 'hadn', 'hasn', 'haven', 'isn', 'ma', 'mightn', 'mustn', 'needn', 'shan', 'shouldn', 'wasn', 'weren', 'won', 'wouldn'])
    word_counts = defaultdict(int)

    words = re.findall(r'\b[a-z]{2,}\b', all_text.lower())
    for word in words:
        if word not in stop_words:
            word_counts[word] += 1
            
    sorted_word_counts = sorted(word_counts.items(), key=lambda item: item[1], reverse=True)

    # 打印结果
    print("--- 词频 (Top 30) ---")
    for word, count in sorted_word_counts[:30]:
        print(f"{word}: {count}")

    print("\n--- 已处理的数据 (前3条) ---")
    output_data_sample = []
    for item in data[:3]:
        output_item = {
            "Date": item.get("Date"),
            "Symbol": item.get("Symbol"),
            "Title": item.get("Title"),
            "Originator": item.get("Originator"),
            "Originator_split": item.get("Originator_split"),
            "related_document_split": item.get("related_document_split"),
            "summary_text": item.get("summary", {}).get("text"),
        }
        output_data_sample.append(output_item)
    
    print(json.dumps(output_data_sample, indent=2, ensure_ascii=False))
    
    # 将完整处理过的数据保存到新文件
    with open(PROCESSED_JSON_PATH, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)
    print(f"\n完整处理过的数据已保存到 {PROCESSED_JSON_PATH} 文件中。")


except FileNotFoundError:
    print(f"错误：找不到 {OUTPUT_JSON_PATH} 文件。请确保该文件与脚本位于同一目录中。")
except Exception as e:
    print(f"发生错误：{e}")