import json
import os
import argparse

def validate_data_structure(data, file_path):
    """
    检验加载后的 JSON 数据结构。
    Checks the structure of the loaded JSON data.
    """
    # 根结构必须是列表
    # The root structure must be a list.
    if not isinstance(data, list):
        print(f"❌ 验证失败: JSON文件的根结构不是一个列表。文件: '{file_path}'")
        return False

    # 定义每个字典对象必须包含的顶级键
    # Define the required top-level keys for each dictionary object.
    required_keys = {
        "Date": str, "Symbol": str, "Title": str, "Originator": str, 
        "file_name": str, "summary": dict, "content": list
    }
    
    # 定义 'summary' 字典内部必须包含的键
    # Define the required keys within the 'summary' dictionary.
    required_summary_keys = {
        "text": str, "strategic_direction": list, "output": list,
        "action_to_be_taken": list, "related_document": list
    }

    # 遍历列表中的每一个项目
    # Iterate over each item in the list.
    for i, item in enumerate(data):
        if not isinstance(item, dict):
            print(f"❌ 验证失败: 索引 {i} 处的项目不是一个字典。")
            return False

        # 检查顶级键是否存在以及类型是否正确
        # Check for the existence and correct type of top-level keys.
        for key, expected_type in required_keys.items():
            if key not in item:
                print(f"❌ 验证失败: 索引 {i} 处的项目缺少必需的键: '{key}'")
                return False
            if not isinstance(item[key], expected_type):
                print(f"❌ 验证失败: 索引 {i} 处的键 '{key}' 类型错误。期望类型: {expected_type.__name__}, 实际类型: {type(item[key]).__name__}")
                return False

        # 检查 'summary' 字典的内部结构
        # Check the internal structure of the 'summary' dictionary.
        summary_obj = item['summary']
        for key, expected_type in required_summary_keys.items():
            if key not in summary_obj:
                print(f"❌ 验证失败: 索引 {i} 处项目的 'summary' 字典缺少键: '{key}'")
                return False
            if not isinstance(summary_obj[key], expected_type):
                print(f"❌ 验证失败: 索引 {i} 处项目的 'summary' 中的键 '{key}' 类型错误。期望类型: {expected_type.__name__}, 实际类型: {type(summary_obj[key]).__name__}")
                return False

        # 检查 'content' 列表的内部结构
        # Check the internal structure of the 'content' list.
        content_list = item['content']
        for content_index, content_item in enumerate(content_list):
            if not isinstance(content_item, dict):
                print(f"❌ 验证失败: 索引 {i} 处项目的 'content' 列表中的项目 {content_index} 不是字典。")
                return False
            if "paragraph" not in content_item or "text" not in content_item:
                print(f"❌ 验证失败: 索引 {i} 处项目的 'content' 列表中的项目 {content_index} 缺少 'paragraph' 或 'text' 键。")
                return False

    return True

def main():
    """主函数，用于运行验证。"""
    parser = argparse.ArgumentParser(description="检验 data.json 文件的格式和结构。")
    parser.add_argument('file_path', type=str, help="需要检验的 data.json 文件的路径。")
    args = parser.parse_args()
    
    file_path = args.file_path
    
    print(f"--- 开始验证文件: '{file_path}' ---")

    # 检查文件是否存在
    # Check if the file exists.
    if not os.path.exists(file_path):
        print(f"❌ 验证失败: 文件未找到 '{file_path}'")
        return

    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            # 尝试加载JSON
            # Try to load the JSON.
            data = json.load(f)
    except json.JSONDecodeError as e:
        print(f"❌ 验证失败: 文件不是一个有效的JSON格式。")
        print(f"   错误详情: {e}")
        return
    except Exception as e:
        print(f"❌ 读取文件时发生未知错误: {e}")
        return

    # 运行结构验证
    # Run the structure validation.
    if validate_data_structure(data, file_path):
        print(f"✅ 验证成功! 文件 '{file_path}' 的格式和结构均正确。")
    else:
        print(f"--- 文件 '{file_path}' 验证未通过 ---")

if __name__ == "__main__":
    main()
