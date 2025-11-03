import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import argparse
# 新增：用于数据归一化的库
from sklearn.preprocessing import MinMaxScaler

# --- Argument Parser Setup ---
parser = argparse.ArgumentParser(description="Visualize specific keyword trends from a consolidated CSV file.")
parser.add_argument('--csv_path', type=str, required=True, 
                    help="Path to the input trend_analysis.csv file.")
parser.add_argument('--output_path', type=str, required=True, 
                    help="Path to save the output trend chart PNG file.")
# --- 修改：改为接受一个关键词列表 ---
parser.add_argument('--keywords', type=str, required=True, nargs='+',
                    help="A list of specific keywords to plot (e.g., ghg_emission ballast_water).")
# --- 新增：归一化选项 ---
parser.add_argument('--normalize', action='store_true',
                    help="Normalize each keyword's trend to a 0-1 scale to compare trajectories.")

args = parser.parse_args()

def main():
    try:
        df = pd.read_csv(args.csv_path, index_col=0)
        print("CSV file loaded successfully.")
    except FileNotFoundError:
        print(f"Error: The file was not found at {args.csv_path}")
        return

    # --- 1. 筛选数据 ---
    
    # 识别出哪些列是会话 (即，不是 consolidate.py 添加的指标列)
    #
    metric_cols = ['total_score', 'change_std_dev', 'abs_change_first_last']
    session_cols = [col for col in df.columns if col not in metric_cols]
    
    # 检查用户请求的关键词哪些在数据中
    available_keywords = [kw for kw in args.keywords if kw in df.index]
    missing_keywords = [kw for kw in args.keywords if kw not in df.index]
    
    if not available_keywords:
        print(f"Error: None of the requested keywords ({args.keywords}) were found in the file's index.")
        return
        
    if missing_keywords:
        print(f"Warning: The following keywords were not found and will be skipped: {missing_keywords}")
    
    print(f"Plotting trends for: {available_keywords}")

    # 提取我们感兴趣的关键词和会话列
    df_plot = df.loc[available_keywords, session_cols]

    # --- 2. 归一化 (如果用户请求) ---
    if args.normalize:
        print("Normalizing keyword trends to 0-1 scale...")
        # 我们需要对每一行（每个关键词）进行归一化
        # MinMaxScaler 是按列工作的，所以我们先转置(T)，再归一化，然后再转置(T)回来
        scaler = MinMaxScaler()
        
        # 转置: 关键词变为列
        df_T = df_plot.T 
        
        # 归一化 (fit_transform)
        scaled_data = scaler.fit_transform(df_T)
        
        # 将归一化后的数据重新放回DataFrame
        df_scaled_T = pd.DataFrame(scaled_data, index=df_T.index, columns=df_T.columns)
        
        # 再次转置，使关键词恢复为行
        df_plot = df_scaled_T.T
        
        # 更新图表标签
        plot_title = f'Normalized Trend Analysis for Selected Keywords (0-1 Scale)'
        y_label = 'Normalized Score (0-1 Scale)'
    else:
        # 保持原始值
        plot_title = 'Trend Analysis for Selected Keywords (Raw TF-IDF Score)'
        y_label = 'TF-IDF Score'

    # --- 3. 绘图 (单一图表) ---
    sns.set_theme(style="whitegrid", palette="pastel")
    # 创建一个更宽的单一图表
    plt.figure(figsize=(16, 8)) 
    
    # 转置DataFrame以便绘图 (会话 session 成为 x 轴)
    df_transposed = df_plot.T
    
    # 为每个关键词（现在是列）绘制一条线
    for keyword in df_transposed.columns:
        plt.plot(df_transposed.index, df_transposed[keyword], marker='o', linestyle='-', label=keyword)

    plt.title(plot_title, fontsize=20, pad=20)
    plt.xlabel('Session', fontsize=12)
    plt.ylabel(y_label, fontsize=12)
    
    # 设置 x 轴刻度的旋转
    ax = plt.gca() # 获取当前坐标轴
    ax.tick_params(axis='x', labelsize=10)
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")

    # 将图例放到图表外部，防止遮挡
    plt.legend(title='Keywords', fontsize='medium', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)

    # 调整布局，为图例腾出空间
    plt.tight_layout(rect=[0, 0, 0.85, 1]) 

    # --- 4. 保存图表 ---
    output_dir = os.path.dirname(args.output_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    plt.savefig(args.output_path, dpi=300, bbox_inches='tight')
    
    print("-" * 30)
    print(f"Trend visualization complete!")
    print(f"Chart saved to: {args.output_path}")

if __name__ == "__main__":
    main()