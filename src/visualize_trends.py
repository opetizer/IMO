import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import argparse

parser = argparse.ArgumentParser(description="Visualize keyword trends from a consolidated CSV file.")
parser.add_argument('--csv_path', type=str, required=True, help="Path to the input trend_analysis.csv file.")
parser.add_argument('--output_path', type=str, required=True, help="Path to save the output trend chart PNG file.")
parser.add_argument('--title', type=str, default="Keyword Trends", help="The title for the chart.")

parser.add_argument('--keywords', type=str, nargs='+', required=True, help="A list of keywords to plot, separated by spaces.")

args = parser.parse_args()

def main():
    try:
        df = pd.read_csv(args.csv_path, index_col=0)
        print("CSV file loaded successfully.")
    except FileNotFoundError:
        print(f"Error: The file was not found at {args.csv_path}")
        return

    available_keywords = [kw for kw in args.keywords if kw in df.index]
    if not available_keywords:
        print("Error: None of the selected keywords were found in the CSV file.")
        print(f"Please choose from available keywords like: {df.index[:10].tolist()}...")
        return
        
    print(f"Plotting trends for the following keywords: {available_keywords}")

    df_plot = df.loc[available_keywords].copy()
    if 'total_score' in df_plot.columns:
        df_plot.drop(columns=['total_score'], inplace=True)
    df_transposed = df_plot.T
    
    sns.set_theme(style="whitegrid")
    plt.figure(figsize=(15, 8))

    for keyword in df_transposed.columns:
        plt.plot(df_transposed.index, df_transposed[keyword], marker='o', linestyle='-', label=keyword)

    plt.title(args.title, fontsize=20) # 使用参数设置标题
    plt.xlabel('MEPC Session', fontsize=12)
    plt.ylabel('TF-IDF Score (Importance)', fontsize=12)
    plt.xticks(rotation=45, ha="right")
    plt.legend(title='Keywords')
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.tight_layout()

    plt.savefig(args.output_path, dpi=300) # 使用参数设置输出路径
    
    print("-" * 30)
    print(f"Trend visualization complete!")
    print(f"Chart saved to: {args.output_path}")

if __name__ == "__main__":
    main()