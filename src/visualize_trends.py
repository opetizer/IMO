import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import argparse

# --- Argument Parser Setup ---
parser = argparse.ArgumentParser(description="Visualize keyword trends from a consolidated CSV file.")
parser.add_argument('--csv_path', type=str, required=True, help="Path to the input trend_analysis.csv file.")
parser.add_argument('--output_path', type=str, required=True, help="Path to save the output trend chart PNG file.")
parser.add_argument('--top_n', type=int, default=10, help="Number of keywords to plot for each category.")
parser.add_argument('--title_prefix', type=str, default="Keyword", help="Prefix for the chart title.")
args = parser.parse_args()

def plot_trends(df, keywords, title, ax):
    """
    Helper function to plot trends for a given list of keywords on a specific subplot axis.
    """
    if not keywords:
        ax.text(0.5, 0.5, 'No keywords to display.', ha='center', va='center', fontsize=12)
        ax.set_title(title, fontsize=16, pad=15)
        ax.axis('off')
        return
        
    # Exclude metric columns from plotting data
    metric_cols = ['total_score', 'change_std_dev', 'abs_change_first_last']
    plot_cols = [col for col in df.columns if col not in metric_cols]
    
    # Ensure all selected keywords exist in the dataframe index
    available_keywords = [kw for kw in keywords if kw in df.index]
    if not available_keywords:
        print(f"Warning: None of the keywords for '{title}' found in the data.")
        return

    df_plot = df.loc[available_keywords, plot_cols]
    df_transposed = df_plot.T
    
    for keyword in df_transposed.columns:
        ax.plot(df_transposed.index, df_transposed[keyword], marker='o', linestyle='-', label=keyword)

    ax.set_title(title, fontsize=16, pad=15)
    ax.set_xlabel('Session', fontsize=12)
    ax.set_ylabel('TF-IDF Score', fontsize=12)
    
    # --- FIX APPLIED HERE ---
    # The 'ha' parameter is not valid in tick_params. 
    # Setting rotation on the labels themselves is the correct way.
    ax.tick_params(axis='x', labelsize=10)
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")

    ax.legend(title='Keywords', fontsize='small')
    ax.grid(True, which='both', linestyle='--', linewidth=0.5)

def main():
    try:
        df = pd.read_csv(args.csv_path, index_col=0)
        print("CSV file loaded successfully.")
    except FileNotFoundError:
        print(f"Error: The file was not found at {args.csv_path}")
        return

    top_n = args.top_n
    
    # --- Identify Top N Keywords for Each Category ---
    top_freq_keywords = df.sort_values(by='total_score', ascending=False).head(top_n).index.tolist()
    top_volatile_keywords = df.sort_values(by='change_std_dev', ascending=False).head(top_n).index.tolist()
    top_change_keywords = df.sort_values(by='abs_change_first_last', ascending=False).head(top_n).index.tolist()

    print(f"\n--- Top {top_n} Keywords ---")
    print(f"By total frequency: {top_freq_keywords}")
    print(f"By volatility (std dev): {top_volatile_keywords}")
    print(f"By trend change (first-last): {top_change_keywords}")

    # --- Create Subplots ---
    sns.set_theme(style="whitegrid", palette="pastel")
    fig, axes = plt.subplots(3, 1, figsize=(16, 24))
    fig.suptitle(f'{args.title_prefix} Trends Analysis (Top {top_n})', fontsize=24, y=0.96)

    # --- Plot on each subplot ---
    plot_trends(df, top_freq_keywords, f'Top {top_n} Keywords by Total Frequency', axes[0])
    plot_trends(df, top_volatile_keywords, f'Top {top_n} Most Volatile Keywords (by Std Dev)', axes[1])
    plot_trends(df, top_change_keywords, f'Top {top_n} Keywords by Trend Change (First to Last Session)', axes[2])

    plt.tight_layout(rect=[0, 0, 1, 0.95]) # Adjust layout for the main title

    # --- Save the Figure ---
    output_dir = os.path.dirname(args.output_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    plt.savefig(args.output_path, dpi=300, bbox_inches='tight')
    
    print("-" * 30)
    print(f"Trend visualization complete!")
    print(f"Dashboard chart saved to: {args.output_path}")

if __name__ == "__main__":
    main()

