import json
import plotly.express as px
import os
import argparse
import logging
import pandas as pd # <-- 新增：导入 pandas

# ==========================================

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

# ==========================================

# ==========================================
def load_and_process_data(file_path):
    
    # 读取 JSON 文件
    print(f"📥 正在加载数据文件: {file_path} ...")
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # 【已修改】将加载的列表转换为 DataFrame
    df = pd.DataFrame(data)
    
    # 【已修改】定义元数据提取函数，包含对 content 字段（API返回的JSON字符串）的解析
    def extract_metadata(row):
        # content 字段是 API 返回的 JSON 字符串。
        content_str = row.get('content', '{}')
        
        content = {}
        try:
            # 清理潜在的 Markdown 代码围栏（例如 ```json...```）
            cleaned_str = content_str.strip()
            if cleaned_str.startswith("```json"):
                cleaned_str = cleaned_str[7:].strip()
            if cleaned_str.endswith("```"):
                cleaned_str = cleaned_str[:-3].strip()

            if cleaned_str:
                content = json.loads(cleaned_str)
        except json.JSONDecodeError:
            # 解析失败时，使用空字典
            pass

        title = row.get('title', '')
        
        agenda = 'INF'
        subject = 'No Subject'
        summary = 'No Summary'

        # 从解析后的内容中提取信息
        meta = content.get('metadata', {})
        sections = content.get('sections', {})
        
        # 优先使用 API 解析出的元数据
        agenda = meta.get('agenda_item', 'INF')
        subject = meta.get('subject', 'No Subject')
        # summary 提取自 sections 字典
        summary = sections.get('summary', 'No Summary')
        
        return pd.Series([agenda, subject, title, summary], 
                         index=['Agenda_Item', 'Subject', 'Clean_Title', 'Summary'])

    # 【已修改】将提取出的元数据合并回主 DataFrame
    meta_df = df.apply(extract_metadata, axis=1)
    df = pd.concat([df, meta_df], axis=1)

    # 处理 Originator 列，拆分多个国家/组织

    df['Originator_Clean'] = df['Originator'].astype(str).str.replace(' and ', ', ', regex=False)
    df['Country_List'] = df['Originator_Clean'].str.split(',')
    df_exploded = df.explode('Country_List')
    df_exploded['Country'] = df_exploded['Country_List'].str.strip()
    df_exploded = df_exploded[df_exploded['Country'] != '']
    
    return df_exploded

def filter_top_countries(df, top_n=15):
    """保留最活跃的 Top N 个国家，其余归为 'Others' (或直接过滤掉)"""
    if df.empty:
        return df
    # 去除Secretariat等非国家实体
    df = df[~df['Country'].str.lower().isin(['secretariat'])]

    # 统计国家出现的频次
    country_counts = df['Country'].value_counts()
    
    # 获取前 N 名列表
    top_countries = country_counts.head(top_n).index.tolist()
    print(f"🌍 筛选出前 {top_n} 个活跃国家/组织: {top_countries[:5]}...")
    
    # 方式A：只保留这些国家的数据 (推荐，图表更清晰)
    df_filtered = df[df['Country'].isin(top_countries)].copy()
    
    # 方式B：其他的标记为 'Others' (如果不介意图表里有个巨大的 Others 块)
    # df_filtered = df.copy()
    # df_filtered.loc[~df_filtered['Country'].isin(top_countries), 'Country'] = 'Others'
    
    return df_filtered

# ==========================================

# ==========================================
def generate_visualizations(df, output_path="MEPC_Analysis_Report.html"):
    print("📊 正在生成图表...")

    # --- 1. 热力图 (Agenda vs Country) ---
    # 统计 (Country, Agenda_Item) 组合的数量
    heatmap_df = df.groupby(['Country', 'Agenda_Item']).size().reset_index(name='Count')
    
    # 透视表: 行=Country, 列=Agenda
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
    # 截断过长的 Subject 以防显示不下
    
    fig_sunburst = px.sunburst(
        df,
        path=['Agenda_Item', 'Country', 'Subject'], # 层级：议题 -> 国家 -> 具体主题
        hover_data={'Title': True},
        title="<b>议题全景透视</b>",
        color='Agenda_Item',
        height=900,
        maxdepth=2  # 默认显示层级深度，防止一开始太乱
    )
    
    # # 悬挂提示中清晰地显示 Summary 信息
    # fig_sunburst.update_traces(
    #     hovertemplate='<b>%{label}</b><br>提案数: %{value}<br>概要: %{customdata[0]}',
    #     customdata=df[['Summary']].values,
    # )

    # 优化文字显示,可以超出范围
    fig_sunburst.update_traces(
        textinfo="label+percent entry", 
        insidetextorientation='radial', # 环形排列文字
        textfont_size=12,
    )
    fig_sunburst.update_layout(
        uniformtext=dict(minsize=12), # 确保文字大小一致
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
        # 【已移除】text="Country", # 移除静态文本标签
        hover_name="Country",
        title="<b>参与度分析 (数量 vs 广度)</b>",
        labels={"Total_Docs": "文件总数", "Unique_Agendas": "参与议题数"}
    )
    # 【已移除】fig_bubble.update_traces(textposition='top center') # 移除对应的文本位置更新
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

# ==========================================
# 主程序
# ==========================================
if __name__ == "__main__":
    args = parse_arguments()
    logger = setup_logger(args.logging, args.title)
    
    
    df = load_and_process_data(args.file_path)
    
    if not df.empty:
        
        df_filtered = filter_top_countries(df, top_n=TOP_N_COUNTRIES)
        
        print(f"原始数据行数(Exploded): {len(df)} -> 筛选后行数: {len(df_filtered)}")
        
        
        generate_visualizations(df_filtered)
    else:
        print("❌ 没有数据可处理。")
