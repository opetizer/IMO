import os
import json

TOPIC = "MEPC"

def merge_session_data():
    
    # 基于 consolidate.py 和 json_process.py 的上下文设置基础路径
    base_dir = f"output/{TOPIC}"
    output_file = os.path.join(base_dir, "data.json")
    
    # 确定要合并的会话范围
    start_session = 77
    end_session = 83
    session_folders = [f"MEPC{i}" for i in range(start_session, end_session + 1)]
    
    merged_data = []
    total_files = 0
    total_items = 0

    print(f"开始合并 {base_dir} 目录下的 data.json 文件...")
    print(f"目标会话: {start_session} 至 {end_session}")

    for session in session_folders:
        file_path = os.path.join(base_dir, session, "data.json")
        
        if os.path.exists(file_path):
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    # 根据 validate_json.py，我们知道 data.json 是一个列表
                    session_data = json.load(f)
                    
                    if isinstance(session_data, list):
                        items_in_file = 0
                        # 为每一条记录添加会话来源信息
                        for item in session_data:
                            item['session'] = session  # 关键步骤：添加会话标识
                            items_in_file += 1
                        
                        merged_data.extend(session_data)
                        print(f"  [√] 成功加载: {file_path} (包含 {items_in_file} 条记录)")
                        total_files += 1
                        total_items += items_in_file
                    else:
                        print(f"  [!] 警告: {file_path} 的内容不是一个列表，已跳过。")
                        
            except json.JSONDecodeError:
                print(f"  [X] 错误: {file_path} 包含无效的JSON，已跳过。")
            except Exception as e:
                print(f"  [X] 错误: 读取 {file_path} 时发生未知错误: {e}")
        else:
            print(f"  [i] 信息: 未找到文件 {file_path}，已跳过。")

    # 检查是否合并了任何数据
    if not merged_data:
        print("\n合并完成，但未找到任何数据。请检查路径是否正确。")
        return

    # 将合并后的数据写入新文件
    try:
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(merged_data, f, ensure_ascii=False, indent=4)
        
        print("\n" + "="*30)
        print(f"总共合并了 {total_files} 个文件。")
        print(f"总共包含 {total_items} 条记录。")
        print(f"合并后的文件已保存至: {output_file}")
        print("="*30)
        
    except Exception as e:
        print(f"\n[X] 严重错误: 无法写入最终合并文件 {output_file}: {e}")

if __name__ == "__main__":
    # 确保输出目录存在
    base_dir = "output/MEPC"
    if not os.path.exists(base_dir):
        print(f"错误：基础目录 {base_dir} 不存在。请先运行数据处理流程。")
    else:
        merge_session_data()