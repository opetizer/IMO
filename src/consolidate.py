import os
import json
import pandas as pd
import re
import argparse

parser = argparse.ArgumentParser(description="Consolidate word frequency data for a given topic.")
parser.add_argument('--title', type=str, required=True, help="The main topic folder name (e.g., MEPC)")
args = parser.parse_args()
base_path = os.path.join('output', args.title)
output_csv_path = os.path.join(base_path, 'trend_analysis.csv')

def consolidate(base_path, output_csv_path):
    try:
        all_data = {}
        print(f"Starting consolidation from base path: {base_path}")

        # 获取所有子文件夹并进行自然排序
        # Get all subdirectories and sort them naturally
        dir_list = os.listdir(base_path)
        # 筛选掉非目录文件（例如之前生成的CSV文件）
        # Filter out non-directory files (e.g., previously generated CSV files)
        dir_list = [d for d in dir_list if os.path.isdir(os.path.join(base_path, d))]
        dir_list.sort(key=lambda x: int(re.search(r'\d+', x).group()) if re.search(r'\d+', x) else -1)

        for session_folder in dir_list:
            freq_file_path = os.path.join(base_path, session_folder, 'word_freq.json')
            
            if os.path.exists(freq_file_path):
                print(f"Processing: {freq_file_path}")
                with open(freq_file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    scores_dict = {item[0]: item[1] for item in data}
                    column_name = session_folder.replace(" ", "_")
                    all_data[column_name] = scores_dict
            else:
                print(f"Warning: word_freq.json not found in {session_folder}")

        if not all_data:
            print("No data found to process. Exiting.")
            return

        df = pd.DataFrame(all_data)
        df.fillna(0, inplace=True)
        
        # --- 新增的趋势分析逻辑 ---
        # --- New logic for Trend Analysis ---

        # 1. 计算总分 (高频词)
        # 1. Calculate Total Score (High Frequency)
        df['total_score'] = df.sum(axis=1)
        
        # 2. 计算变化指标
        # 2. Calculate Change Metrics
        # 仅使用会议数据列进行计算
        # Use only session columns for calculation
        session_columns = [col for col in df.columns if col != 'total_score']
        
        # 使用标准差衡量波动性
        # Standard Deviation as a measure of volatility
        df['change_std_dev'] = df[session_columns].std(axis=1)

        # 计算第一次和最后一次会议的绝对变化
        # Absolute change between the first and last session
        if len(session_columns) > 1:
            df['abs_change_first_last'] = (df[session_columns[-1]] - df[session_columns[0]]).abs()
        else:
            df['abs_change_first_last'] = 0.0

        # 按总分排序，使最重要的关键词排在前面
        # Sort by total score to keep the most important keywords at the top
        df.sort_values(by='total_score', ascending=False, inplace=True)
        
        # 调整列顺序以便查看
        # Reorder columns for clarity
        cols_order = ['total_score', 'change_std_dev', 'abs_change_first_last'] + session_columns
        df = df[cols_order]

        df.to_csv(output_csv_path, encoding='utf-8-sig')

        print("-" * 30)
        print(f"Consolidation complete!")
        print(f"Trend analysis data saved to: {output_csv_path}")
        
        print("\n--- Top 10 Keywords by Total Frequency ---")
        print(df.head(10))

        print("\n--- Top 10 Keywords by Largest Change (Standard Deviation) ---")
        print(df.sort_values(by='change_std_dev', ascending=False).head(10))
        
        print("\n--- Top 10 Keywords by Absolute Change (First to Last Session) ---")
        print(df.sort_values(by='abs_change_first_last', ascending=False).head(10))


    except FileNotFoundError:
        print(f"Error: Directory not found at {base_path}")
        print("Please make sure you are running this script from the project's root directory.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

if __name__ == "__main__":
    consolidate(base_path, output_csv_path)
