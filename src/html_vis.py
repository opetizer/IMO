import plotly.express as px
import plotly.graph_objects as go
import os
import argparse
import logging
import pandas as pd
from json_read import load_data  # 导入通用数据读取模块

# ==========================================
# 配置与常量
TOP_N_COUNTRIES = 20  # 在图表中只保留前 N 个最活跃的国家/组织

def parse_arguments():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="Visualize data in HTML format.")
    
    # 路径定位参数
    parser.add_argument('--title', default="MEPC", type=str, help="主标题 (例如 MEPC)，用于构建文件夹路径。")
    parser.add_argument('--subtitle', type=str, required=False, help="子标题 (例如 'MEPC 77')，用于定位数据和命名输出。")
    parser.add_argument('--file_path', type=str, required=False, help="显式指定 data.json 路径 (如果指定，将忽略 title/subtitle 构建的路径)。")
    
    # 筛选参数
    parser.add_argument('--agenda_items', nargs='*', help="筛选特定的议题 ID (例如: 3 4 7)。如果不填则分析所有议题。")
    
    # 系统参数
    parser.add_argument('--logging', default="log/logging.log", type=str, help="Path to the log file.")
    
    return parser.parse_args()

def setup_logger(log_file, logger_name):
    """配置 Logger"""
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
    # 清洗连接词
    df['Originator_Clean'] = df['Originator'].astype(str).str.replace(' and ', ', ', regex=False).replace(';', ', ', regex=False)
    # 拆分
    df['Country_List'] = df['Originator_Clean'].str.split(',')
    # 炸裂 (Explode) 列表为多行
    df_exploded = df.explode('Country_List')
    df_exploded['Country'] = df_exploded['Country_List'].str.strip()
    # 过滤空值
    df_exploded = df_exploded[df_exploded['Country'] != '']
    return df_exploded

def filter_top_countries(df, top_n=15):
    """保留最活跃的 Top N 个国家"""
    if df.empty:
        return df
    # 去除 Secretariat 等非国家实体 (可根据需要调整)
    exclude_list = ['secretariat', 'secretary-general', 'imo', 'chair']
    df = df[~df['Country'].str.lower().isin(exclude_list)]

    # 统计国家出现的频次
    country_counts = df['Country'].value_counts()
    
    # 获取前 N 名列表
    top_countries = country_counts.head(top_n).index.tolist()
    print(f"🌍 筛选出前 {top_n} 个活跃国家/组织: {top_countries[:5]}...")
    
    # 只保留这些国家的数据
    df_filtered = df[df['Country'].isin(top_countries)].copy()
    return df_filtered

def generate_visualizations(df, output_path, title_suffix=""):
    """生成 Plotly 图表并导出为 HTML"""
    print(f"📊 正在生成图表报告: {output_path}...")

    # --- 数据预处理 ---
    # 创建Angenda_Item和Title的映射字典,只取第一个Title作为代表
    agenda_dict = df.groupby('Agenda_Item')['Title'].apply(lambda x: x.unique().tolist()).to_dict()
    # 确保Title唯一
    for key in agenda_dict:
        if len(agenda_dict[key]) > 1:
            logger.error(f"警告: Agenda Item {key} 对应多个标题: {agenda_dict[key]}")
            # exit(1)


    # 数据填充，防止空值导致的绘图错误
    df['Agenda_Item'] = df['Agenda_Item'].fillna('Unknown')
    # 将 Agenda_Item 转为字符串，保证离散颜色映射
    df['Agenda_Item'] = df['Agenda_Item'].astype(str)
    df['Subject'] = df['Subject'].fillna('No Subject')

    # --- 1. 热力图 (Agenda vs Country) ---
    heatmap_df = df.groupby(['Country', 'Title']).size().reset_index(name='Count')
    
    heatmap_matrix = heatmap_df.pivot(index='Country', columns='Title', values='Count').fillna(0)
    
    # 按总提案数对国家排序
    heatmap_matrix['total'] = heatmap_matrix.sum(axis=1)
    heatmap_matrix = heatmap_matrix.sort_values('total', ascending=True).drop('total', axis=1)
    
    fig_heatmap = px.imshow(
        heatmap_matrix,
        labels=dict(x="议题 (Agenda Item)", y="国家/组织", color="提案数"),
        x=heatmap_matrix.columns,
        y=heatmap_matrix.index,
        title=f"<b>关注度热力图 (Top Active Countries) {title_suffix}</b>",
        aspect="auto",
        color_continuous_scale="Blues"
    )
    fig_heatmap.update_layout(height=800)

    # --- 2. 旭日图 (Sunburst) ---
    # 如果数据量太少，旭日图可能会报错，加个判断
    if len(df) > 0:
        fig_sunburst = px.sunburst(
            df,
            path=['Title', 'Country', 'Subject'], 
            hover_data={'Title': True},
            title=f"<b>议题全景透视 {title_suffix}</b>",
            color='Agenda_Item',
            height=900,
            maxdepth=2
        )
        
        fig_sunburst.update_traces(
            textinfo="label", 
            insidetextorientation='radial',
            textfont_size=12,
            branchvalues='total'
        )
        fig_sunburst.update_layout(
            uniformtext=dict(minsize=12),
            margin=dict(t=40, l=0, r=0, b=0)
        )
    else:
        fig_sunburst = px.scatter(title="数据不足以生成旭日图")

        # --- 3. 玫瑰图 (Rose Diagram for Participation) ---
    summary_df = df.groupby('Country').agg(
        Total_Docs=('Symbol', 'count'),
        Unique_Agendas=('Agenda_Item', 'nunique')
    ).reset_index()

    summary_df = summary_df.sort_values('Total_Docs', ascending=False)
    
    fig_rose = go.Figure(go.Barpolar(
        r=summary_df['Total_Docs'],
        theta=summary_df['Country'],
        marker=dict(
            color=summary_df['Unique_Agendas'],
            colorscale='RdBu',
            colorbar=dict(title="参与议题数")
        ),
        name=f"参与度分析 - 提案总量与广度 (玫瑰图) {title_suffix}",
        ))
    
    fig_rose.update_traces(marker_line_color='white', marker_line_width=1)
    fig_rose.update_layout(
        polar=dict(
            radialaxis=dict(showgrid=True, gridcolor="#DDD"),
            angularaxis=dict(showgrid=True, gridcolor="#DDD", rotation=90, direction="clockwise")
        )
    )

    # --- 输出 HTML ---
    with open(output_path, 'w', encoding="utf-8") as f:
        f.write(f"<html><head><title>IMO Analysis - {title_suffix}</title>")
        f.write("<style>body{font-family: Arial, sans-serif; margin: 40px; background: #f0f2f5;}")
        f.write(".card{background: white; padding: 20px; margin-bottom: 30px; border-radius: 10px; box-shadow: 0 4px 6px rgba(0,0,0,0.1);}")
        f.write("h1{color: #333;} p{color: #666;}</style>")
        f.write("</head><body>")
        f.write(f"<h1 style='text-align:center'>MEPC 会议数据分析报告 {title_suffix}</h1>")
        f.write(f"<p style='text-align:center;'>数据来源: 自动化解析 | 筛选条件: Top {TOP_N_COUNTRIES} 活跃主体</p>")
        
        for fig in [fig_heatmap, fig_sunburst, fig_rose]:
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
    logger = setup_logger(args.logging, "html_vis")
    data_path = os.path.join('output', args.title, args.subtitle, 'data.json')
    output_dir = os.path.join('output', args.title, args.subtitle)
    
    if not os.path.exists(data_path):
        print(f"❌ 错误: 找不到数据文件 {data_path}")
        exit(1)
        
    df = load_data(data_path)
    df_exploded = process_countries(df)
    
    title_suffix = ""
    
    # 筛选 Agenda Items
    if args.agenda_items:
        print(f"🔍 正在筛选议题: {args.agenda_items}")
        # 转换为字符串进行比对，防止类型不匹配 (3 vs "3")
        target_items = [str(item) for item in args.agenda_items]
        
        # 填充 NaN 以免报错
        df_exploded['Agenda_Item'] = df_exploded['Agenda_Item'].fillna('Unknown').astype(str)
        
        # 筛选
        df_exploded = df_exploded[df_exploded['Agenda_Item'].isin(target_items)]
        
        if df_exploded.empty:
            print(f"⚠️ 警告: 筛选议题 {target_items} 后没有剩余数据。")
            exit(0)
            
        title_suffix = f"- 议题 {','.join(target_items)}"
        # 输出文件名加上筛选标识
        output_filename = f"Analysis_Report_Agenda_{'_'.join(target_items)}.html"
    else:
        output_filename = "Analysis_Report_Full.html"

    # 5. 筛选活跃国家 (Top N)
    print(f"筛选前记录数: {len(df_exploded)}")
    df_filtered = filter_top_countries(df_exploded, top_n=TOP_N_COUNTRIES)
    print(f"筛选后记录数: {len(df_filtered)}")

    if df_filtered.empty:
         print("⚠️ 警告: 过滤活跃国家后数据为空。")
         exit(0)

    # 6. 生成图表
    output_full_path = os.path.join(output_dir, output_filename)
    generate_visualizations(df_filtered, output_full_path, title_suffix)