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
        dir_list = os.listdir(base_path)
        dir_list.sort(key=lambda x: int(re.search(r'\d+', x).group()) if re.search(r'\d+', x) else -1)

        for session_folder in dir_list:
            if not os.path.isdir(os.path.join(base_path, session_folder)):
                continue

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
        df['total_score'] = df.sum(axis=1)
        df.sort_values(by='total_score', ascending=False, inplace=True)

        df.to_csv(output_csv_path, encoding='utf-8-sig')

        print("-" * 30)
        print(f"Consolidation complete!")
        print(f"Trend analysis data saved to: {output_csv_path}")
        print("\n--- Top 5 Keywords by Total Score ---")
        print(df.head())

    except FileNotFoundError:
        print(f"Error: Directory not found at {base_path}")
        print("Please make sure you are running this script from the project's root directory.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

# 4. 标准的Python脚本入口
if __name__ == "__main__":
    consolidate(base_path, output_csv_path)