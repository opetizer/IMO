import json
import pandas as pd
import logging
import os

# 配置 logger
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def parse_nested_content(content_raw):
    """
    解析嵌套在 content 字段中的 JSON 字符串 (通常由 LLM API 生成)。
    处理可能存在的 Markdown 代码块标记。
    """
    if isinstance(content_raw, dict):
        return content_raw
    
    if not isinstance(content_raw, str):
        return {}

    cleaned_str = content_raw.strip()
    # 移除 Markdown 代码块标记
    if cleaned_str.startswith("```json"):
        cleaned_str = cleaned_str[7:].strip()
    elif cleaned_str.startswith("```"):
        cleaned_str = cleaned_str[3:].strip()
    
    if cleaned_str.endswith("```"):
        cleaned_str = cleaned_str[:-3].strip()

    try:
        return json.loads(cleaned_str)
    except json.JSONDecodeError:
        # 如果解析失败，返回空字典，避免程序崩溃
        return {}

def load_data(file_path):
    """
    读取 data.json 文件，并将其转换为扁平化的 Pandas DataFrame。
    自动处理嵌套的 JSON 字符串和元数据提取。
    
    Args:
        file_path (str): data.json 文件的路径
        
    Returns:
        pd.DataFrame: 包含处理后数据的 DataFrame
    """
    if not os.path.exists(file_path):
        logger.error(f"文件未找到: {file_path}")
        return pd.DataFrame()

    logger.info(f"正在加载数据文件: {file_path} ...")
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except json.JSONDecodeError as e:
        logger.error(f"JSON 解析错误: {e}")
        return pd.DataFrame()

    processed_rows = []

    for item in data:
        # 1. 获取基础文件信息
        row = {
            'file_name': item.get('file_name'),
            'Date': item.get('Date'),
            'Symbol': item.get('Symbol'),
            'Originator': item.get('Originator'),
            'Title_Raw': item.get('Title'), # 原始标题（来自索引）
        }

        # 2. 解析 LLM 生成的内容 (content 字段)
        content_data = parse_nested_content(item.get('content', '{}'))
        
        # 3. 提取元数据 (Metadata)
        meta = content_data.get('metadata', {})
        row['Agenda_Item'] = meta.get('agenda_item')
        row['Subject'] = meta.get('subject')
        # 优先使用 LLM 提取的标题，如果没有则使用原始标题
        row['Title'] = meta.get('title') if meta.get('title') else row['Title_Raw']
        row['Session'] = meta.get('session')
        
        # 4. 提取章节内容 (Sections)
        sections = content_data.get('sections', {})
        row['Summary'] = sections.get('summary')
        row['Introduction'] = sections.get('introduction')
        row['Action_Requested'] = sections.get('action_requested')
        row['Annex_Content'] = sections.get('annex_content')

        # 5. 构建用于 NLP 分析的完整文本 (合并主要章节)
        # 将各部分拼接起来，用换行符分隔
        text_parts = []
        if row['Title']: text_parts.append(row['Title'])
        if row['Summary']: text_parts.append(row['Summary'])
        if row['Introduction']: text_parts.append(row['Introduction'])
        if row['Annex_Content']: text_parts.append(row['Annex_Content'])
        # 也可以选择加入 action_requested
        
        row['full_text'] = "\n\n".join([t for t in text_parts if t and isinstance(t, str)])
        
        processed_rows.append(row)

    df = pd.DataFrame(processed_rows)
    logger.info(f"成功加载 {len(df)} 条记录。")
    return df