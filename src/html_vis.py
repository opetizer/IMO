import plotly.express as px
import os
import argparse
import logging
import pandas as pd
from json_read import load_data  # 导入新的读取模块

# ==========================================
# 默认文件路径
DEFAULT_FILE_PATH = 'output/MEPC/MEPC 77/data.json'
TOP_N_COUNTRIES = 20  # 在图表中只保留前 N 个最活跃的国家/组织

def parse_arguments():
    """Parses command-line arguments."""
    parser = argparse.ArgumentParser(description="Visualize data in HTML format.")
    parser.add_argument('--logging', default="log/logging.log", type=str, required=False, help="Path to the log file.")
    parser.add_argument('--title', default="html_vis", type=str, required=False, help="Title for the logger.")
    parser.add_argument('--file_path', default=DEFAULT_FILE_PATH, type=str, required=False, help="Path to the data.json file.")
    return parser.parse_args()

def setup_logger(log_file, logger_name):
    """Configures and returns a logger."""
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        
        log_dir = os.path.dirname(log_file)
        if log_dir and not os.path.exists(log_dir):
            os.makedirs(log_dir)
            
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
        
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
    return logger

def process_countries(df):
    """处理 Originator 列，拆分多个国家/组织"""
    # 确保 Originator 是字符串
    df['Originator'] = df['Originator'].fillna('')
    df['Originator_Clean'] = df['Originator'].astype(str).str.replace(' and ', ', ', regex=False)
    df['Country_List'] = df['Originator_Clean'].str.split(',')
    df_exploded = df.explode('Country_List')
    df_exploded['Country'] = df_exploded['Country_List'].str.strip()
    # 过滤空值
    df_exploded = df_exploded[df_exploded['Country'] != '']
    return df_exploded

def filter_top_countries(df, top_n=15):
    """保留最活跃的 Top N 个国家"""
    if df.empty:
        return df
    # 去除Secretariat等非国家实体
    df = df[~df['Country'].str.lower().isin(['secretariat', 'secretary-general'])]

    # 统计国家出现的频次
    country_counts = df['Country'].value_counts()
    
    # 获取前 N 名列表
    top_countries = country_counts.head(top_n).index.tolist()
    print(f"🌍 筛选出前 {top_n} 个活跃国家/组织: {top_countries[:5]}...")
    
    # 只保留这些国家的数据
    df_filtered = df[df['Country'].isin(top_countries)].copy()
    return df_filtered

def generate_visualizations(df, output_path="MEPC_Analysis_Report.html"):
    print("📊 正在生成图表...")

    # 数据填充，防止空值导致的绘图错误
    df['Agenda_Item'] = df['Agenda_Item'].fillna('Unknown')
    df['Subject'] = df['Subject'].fillna('No Subject')

    # --- 1. 热力图 (Agenda vs Country) ---
    heatmap_df = df.groupby(['Country', 'Agenda_Item']).size().reset_index(name='Count')
    
    heatmap_matrix = heatmap_df.pivot(index='Country', columns='Agenda_Item', values='Count').fillna(0)
    
    # 按总提案数对国家排序
    heatmap_matrix['total'] = heatmap_matrix.sum(axis=1)
    heatmap_matrix = heatmap_matrix.sort_values('total', ascending=True).drop('total', axis=1)
    
    fig_heatmap = px.imshow(
        heatmap_matrix,
        labels=dict(x="议题 (Agenda Item)", y="国家/组织", color="提案数"),
        x=heatmap_matrix.columns,
        y=heatmap_matrix.index,
        title="<b>关注度热力图 (Top Active Countries)</b>",
        aspect="auto",
        color_continuous_scale="Blues"
    )
    fig_heatmap.update_layout(height=800)

    # --- 2. 旭日图 (Sunburst) ---
    fig_sunburst = px.sunburst(
        df,
        path=['Agenda_Item', 'Country', 'Subject'], 
        hover_data={'Title': True},
        title="<b>议题全景透视</b>",
        color='Agenda_Item',
        height=900,
        maxdepth=2
    )
    
    fig_sunburst.update_traces(
        textinfo="label+percent entry", 
        insidetextorientation='radial',
        textfont_size=12,
    )
    fig_sunburst.update_layout(
        uniformtext=dict(minsize=12),
        margin=dict(t=40, l=0, r=0, b=0)
    )

    # --- 3. 气泡图 (活跃度 vs 广度) ---
    summary_df = df.groupby('Country').agg(
        Total_Docs=('Symbol', 'count'),
        Unique_Agendas=('Agenda_Item', 'nunique')
    ).reset_index()
    
    fig_bubble = px.scatter(
        summary_df,
        x="Total_Docs",
        y="Unique_Agendas",
        size="Total_Docs",
        color="Country",
        hover_name="Country",
        title="<b>参与度分析 (数量 vs 广度)</b>",
        labels={"Total_Docs": "文件总数", "Unique_Agendas": "参与议题数"}
    )
    fig_bubble.update_layout(showlegend=False)

    # --- 输出 HTML ---
    with open(output_path, 'w', encoding="utf-8") as f:
        f.write("<html><head><title>MEPC Analysis</title>")
        f.write("<style>body{font-family: Arial, sans-serif; margin: 40px; background: #f0f2f5;}")
        f.write(".card{background: white; padding: 20px; margin-bottom: 30px; border-radius: 10px; box-shadow: 0 4px 6px rgba(0,0,0,0.1);}</style>")
        f.write("</head><body>")
        f.write("<h1 style='text-align:center'>MEPC 会议数据分析报告</h1>")
        f.write(f"<p style='text-align:center; color: #666'>基于 pandas 清洗 | 仅展示 Top {TOP_N_COUNTRIES} 活跃主体</p>")
        
        for fig in [fig_heatmap, fig_sunburst, fig_bubble]:
            f.write("<div class='card'>")
            f.write(fig.to_html(full_html=False, include_plotlyjs='cdn'))
            f.write("</div>")
            
        f.write("</body></html>")
    
    print(f"🎉 报告已生成: {os.path.abspath(output_path)}")

if __name__ == "__main__":
    args = parse_arguments()
    logger = setup_logger(args.logging, args.title)
    
    # 使用新的读取函数
    df = load_data(args.file_path)
    
    if not df.empty:
        df_exploded = process_countries(df)
        df_filtered = filter_top_countries(df_exploded, top_n=TOP_N_COUNTRIES)
        
        print(f"原始数据行数: {len(df)} -> 拆分后: {len(df_exploded)} -> 筛选后: {len(df_filtered)}")
        
        generate_visualizations(df_filtered)
    else:
        print("❌ 没有数据可处理。")