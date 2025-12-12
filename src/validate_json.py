import json
import os
import argparse

def validate_data_structure(data, file_path):
    """
    检验加载后的 JSON 数据结构。
    现在适应 LLM API 输出的新格式，其中 'content' 字段是一个包含 JSON 的字符串。
    """
    if not isinstance(data, list):
        print(f"❌ 验证失败: JSON文件的根结构不是一个列表。文件: '{file_path}'")
        return False

    # 必需的顶级字段
    required_keys = ["Date", "Symbol", "Title", "Originator", "file_name", "content"]
    
    # 必需的嵌套 content 字段 (LLM 生成部分)
    required_content_keys = ["metadata", "sections"]

    for i, item in enumerate(data):
        if not isinstance(item, dict):
            print(f"❌ 验证失败: 索引 {i} 处的项目不是一个字典。")
            return False

        # 1. 检查顶级字段
        for key in required_keys:
            if key not in item:
                print(f"❌ 验证失败: 索引 {i} 处的项目缺少顶级键: '{key}'")
                return False

        # 2. 检查 content 字段
        content_raw = item['content']
        content_json = {}

        if isinstance(content_raw, str):
            # 尝试解析字符串中的 JSON
            try:
                # 清洗 markdown 代码块
                clean_str = content_raw.strip()
                if clean_str.startswith("```json"):
                    clean_str = clean_str[7:].strip()
                content_json = json.loads(clean_str)
            except json.JSONDecodeError as e:
                print(f"❌ 验证失败: 索引 {i} 处的 'content' 字段无法解析为 JSON。")
                print(f"   错误: {e}")
                # 打印前50个字符以便调试
                print(f"   内容片段: {content_raw[:50]}...")
                return False
        elif isinstance(content_raw, dict):
            # 已经是字典了
            content_json = content_raw
        else:
            print(f"❌ 验证失败: 索引 {i} 处的 'content' 字段类型错误 ({type(content_raw)})。应为 string (JSON) 或 dict。")
            return False

        # 3. 检查解析后的 content 内部结构
        for key in required_content_keys:
            if key not in content_json:
                print(f"❌ 验证失败: 索引 {i} 处的 content 数据缺少内部键: '{key}'")
                return False
        
        # 验证 sections 字段
        if not isinstance(content_json.get('sections'), dict):
             print(f"❌ 验证失败: 索引 {i} 处的 content['sections'] 不是一个字典。")
             return False

    return True

def main():
    parser = argparse.ArgumentParser(description="检验 data.json 文件的格式和结构。")
    parser.add_argument('file_path', type=str, help="需要检验的 data.json 文件的路径。")
    args = parser.parse_args()
    
    file_path = args.file_path
    print(f"--- 开始验证文件: '{file_path}' ---")

    if not os.path.exists(file_path):
        print(f"❌ 验证失败: 文件未找到 '{file_path}'")
        return

    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except json.JSONDecodeError as e:
        print(f"❌ 验证失败: 文件本身不是有效的JSON格式。{e}")
        return

    if validate_data_structure(data, file_path):
        print(f"✅ 验证成功! 文件 '{file_path}' 的格式和结构均正确 (符合 LLM 输出规范)。")
    else:
        print(f"--- 文件 '{file_path}' 验证未通过 ---")

if __name__ == "__main__":
    main()