## 使用说明

本项目为 IMO 相关的研究资料整理与分享。

## 目录结构

- `src/`：源代码文件夹
 - json_read.py: 通用数据读取模块，处理嵌套 JSON 和 Markdown 格式。
 - cooccurrence.py: 关键词共现网络分析（SCI 绘图风格）。
 - json_process.py: 数据清洗与词频统计工具。
 - parse_data.py: 文档解析与清洗模块。
 - api_clean.py: 大模型 API 调用模块。
 - validate_json.py: 数据验证模块。
 - html_vis.py: HTML 文档可视化模块。
 - merge_data.py: 数据合并模块。
 - stopwords.py: 停用词模块。
 - topic.py: 主题建模模块。
 - consolidate.py: 数据量化分析模块。
 - sankey_trends.py: 单一议题的文档流向桑基图可视化。
 - topic_attention_sankey.py: 多议题/多国家关注度时序变化桑基图。
 - alliance_network.py: 国家/组织联合提案共现网络分析（Louvain社群发现）。
 - citation_network.py: 文档引用网络 + TF-IDF相似度聚类 + 提案演化链追踪。

- `data/`：数据文件夹
- `readme.md`：项目说明文件

`data`文件夹结构如图所示：
```
data/
└── conference_name/
	├── session1/
	│   ├── paper1.pdf
	│   ├── paper2.pdf
	│   └── ...
```
`output`文件夹结构如图所示：
```
output/
└── conference_name/
	├── session1/
	│   ├── data.json           # 原始结构化数据
	│   ├── data_processed.json # 清洗后的扁平化数据
	│   ├── word_freq.json      # 词频统计
	|   └── cooccurrence_graph.png # 共现网络图
	└── sankey/                 # 桑基图可视化输出
		│   ├── Sankey_MEPC_{议题名}.html        # 按议题分类的桑基图
		│   ├── Sankey_MEPC_{国家名}.html        # 按国家分类的桑基图
		│   ├── sankey_{议题名}.html             # 单议题文档流向图
		│   └── sankey_metadata.json             # 桑基图元数据
```	

## 使用方法

### 下载并安装项目依赖

1. 克隆项目到本地：
```bash
git clone https://github.com/imo-ai/imo-data.git
```

2. 安装项目依赖：
```
pip install -r requirements.txt
```

3. 将会议数据放置在 `data` 文件夹下。

### 环境变量设置

系统使用云端大模型（默认为 Qwen/通义千问）进行文档清洗，必须在运行前配置环境变量：
[Linux/Mac]
```bash
export QWEN_API_KEY="sk-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"
export QWEN_BASE_URL="[https://api.qwen.ai/v1/chat/completions](https://api.qwen.ai/v1/chat/completions)"
```

[Windows PowerShell]
```shell
$env:QWEN_API_KEY="sk-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"
$env:QWEN_BASE_URL="[https://api.qwen.ai/v1/chat/completions](https://api.qwen.ai/v1/chat/completions)"
```

### 智能解析与清洗模块 (Parsing & Cleaning)

对应脚本: `src/parse_data.py, src/api_clean.py, src/validate_json.py， src/json_read.py`

此模块负责将原始 PDF 文档转化为高质量的结构化数据。

**算法流程**

1. 索引读取: 解析 `Index.htm` 提取会议元数据（日期、Symbol、Title、Originator）。

2. 文本提取: 使用 PyMuPDF (fitz) 提取 PDF 原始文本。

3. 正则预清洗 (clean_text_regex):

- 去除时间戳格式（如 12:34）。

- 去除页码和无意义的特殊符号。

- 压缩多余的空格和换行符。

4. LLM 结构化提取 (llm_api_clean):

- 调用大模型 API（如 qwen-plus）。

- 使用 System Prompt 指导模型动态提取 Metadata（文档ID、会话、提交方）、Sections（摘要、正文、附件）和 Tables（Markdown格式表格）。

- 输出: 标准化的 JSON 对象。

5. 数据验证 (`validate_json.py`): 校验生成的 JSON 结构是否完整，字段类型是否正确。

**使用示例**:

```bash
# 初始化并解析 MEPC 77 会议数据
sh src/parse_initialize.sh 77
```

### 数据后处理与统计模块 (Data Processing)

对应脚本: `src/json_process.py`

在解析完成后，对数据进行进一步清洗和统计，生成更适合分析的扁平化数据。

**功能**:

字段拆分: 自动拆分 Originator 字段中的多个国家/组织（如 "China, Japan and Singapore" -> ["China", "Japan", "Singapore"]）。

全文词频: 基于合并后的文档全文本（Full Text）计算词频，去除停用词。

格式转换: 生成 `data_processed.json`。

**使用示例**：

```bash
# 处理 MEPC 77 的数据
sh src/json_process.sh 77
```

### 主题建模模块 (Topic Modeling)

对应脚本: `src/topic.py`

基于 LDA (Latent Dirichlet Allocation) 算法，从文档集中发现隐藏的主题结构。

**算法流程**

1. 预处理:

- 分词 (Tokenization) 与词性标注 (POS Tagging)。

- 词形还原 (Lemmatization): 仅保留名词 (NN)，还原为词根形式。

- 停用词过滤: 加载 nltk 停用词库及 `src/stopword.py` 中的自定义领域停用词。

2. 短语检测 (Bigrams): 使用 gensim.models.Phrases 自动识别常用短语（如 "ghg_emission", "ballast_water"）。

3. 语料构建:

- 建立词典 (Dictionary) 并过滤极端词频（no_below=7, no_above=0.5）。

- 构建词袋模型 (Bag-of-Words Corpus)。

4. LDA 训练: 训练 LdaModel，生成 num_topics 个主题。

5. 文档分类: 计算每篇文档属于各个主题的概率，确定其“主导主题 (Dominant Topic)”。

**输出文件**

- `topics_summary.txt`: 每个主题的关键词列表。

- `document_topic_distribution.csv`: 文档-主题分布矩阵。

**运行示例**:

```bash
# 对MEPC 77，运行 LDA 主题建模，保留 10 个主题
sh src/topic.sh 77 10
```

### 关键词共现与网络分析 (Co-occurrence Network)

对应脚本: `src/cooccurrence.py`

构建关键词共现网络，可视化概念间的关联。

**算法流程**

1. TF-IDF 筛选: 计算文档集中词汇的 TF-IDF 值，选取 Top-K (默认 100) 关键词作为网络节点。

2. 共现矩阵构建:

- 使用滑动窗口 (Window Size = 5)。

- 统计关键词对在同一窗口内出现的频率。

3. 图构建 (Graph Construction):

- 节点大小 $\propto$ 词频。

- 边权重 $\propto$ 共现频率 (过滤掉权重低于阈值的边)。

4. 社群发现 (Community Detection): 使用 Louvain 算法对网络节点进行聚类着色，识别语义群组。

**输出文件**: `cooccurrence_graph.png` (网络图), `word_freq.json` (词频表)。

**运行示例**:

```bash
# 对 MEPC 77 运行关键词共现分析
sh src/cooccurrence.sh 77
```

### 趋势分析与可视化 (Trend Analysis & Visualization)

对应脚本: `src/consolidate.py, src/visualize_trends.py, src/html_vis.py`

分析会议主题随时间的变化趋势，并生成可视化报告。

**算法流程**

1. 数据整合 (consolidate.py):

- 遍历不同会话文件夹（如 MEPC 77, MEPC 78...）。

- 合并 word_freq.json，生成宽表格式的 CSV (trend_analysis.csv)。

2. 趋势绘图 (visualize_trends.py):

- 读取整合后的 CSV。

- 归一化 (可选): 使用 MinMaxScaler 将不同量级的词频缩放到 [0, 1] 区间，便于对比趋势形态。

- 绘制多关键词的时间序列折线图。

3. 交互式仪表盘 (html_vis.py):

- 基于 pandas 和 plotly生成 热力图 (国家 vs 议题)、旭日图 (议题层级分布) 和 气泡图 (参与度分析)。

**输出文件**: `trend_analysis.csv` (趋势数据), `trend_analysis.html` (交互式仪表盘)。

**运行示例**:

```bash
# 对 MEPC 77 运行趋势分析
sh src/consolidate.sh 77
```

### 桑基图可视化 (Sankey Diagram Visualization)

对应脚本: `src/sankey_trends.py, src/topic_attention_sankey.py`

生成桑基图以展示议题关注度随时间的演变和国家参与度的变化。

#### 1. 单一议题文档流向桑基图 (sankey_trends.py)

为每个热点议题生成独立的桑基图，展示不同国家在该议题上的文档流向。

**算法流程**

1. 数据聚合:
   - 遍历所有会议文件夹（如 MEPC 77-83）
   - 统计每个议题在每个会议、每个国家的文档数量
   - 筛选出热点议题（按文档总量排序）

2. 节点构建:
   - 横轴：会议按时间顺序排列
   - 节点：每个会议中的国家节点（格式："{会议} - {国家}"）
   - 节点大小：基于文档数量进行全局缩放

3. 连线构建:
   - 相邻会议间，相同国家的议题文档流向
   - 流向值 = min(上次会议文档数, 本次会议文档数)

4. 可视化:
   - 使用 Plotly Sankey 生成交互式 HTML
   - 支持悬停查看详细信息
   - 颜色区分不同国家

**输出文件**: `sankey_{议题名}.html` (每个议题一个文件)

**运行示例**:
```bash
# 生成前6个热点议题的桑基图
python src/sankey_trends.py --meeting_folder "output/MEPC" --top_k_titles 6 --top_n_countries 8

# 使用归一化和最小流向方法
python src/sankey_trends.py --meeting_folder "output/MEPC" --normalize --flow_method min
```

#### 2. 多维度关注度桑基图 (topic_attention_sankey.py)

为每个议题和每个国家分别生成独立的桑基图，从多角度分析关注度变化。

**算法流程**

1. 热点识别:
   - 统计所有议题的文档总量，选出 Top-N 热点议题
   - 统计所有国家的文档总量，选出 Top-N 活跃国家

2. 按议题分类 (Per-Title Sankey):
   - 为每个热点议题生成一个桑基图
   - 横轴：会议时间序列
   - 节点：参与该议题的国家
   - 展示：不同国家对该议题的关注度演变

3. 按国家分类 (Per-Country Sankey):
   - 为每个活跃国家生成一个桑基图
   - 横轴：会议时间序列
   - 节点：该国关注的热点议题
   - 展示：该国对不同议题的关注度分布

4. 节点定位:
   - 全局最大文档数用于节点高度缩放
   - 节点位置反映实际文档数量差异
   - 同一会议内的节点按文档量排序

**输出文件**:
- 议题图：`Sankey_{会议类型}_{议题名}.html`
- 国家图：`Sankey_{会议类型}_{国家名}.html`
- 元数据：`sankey_metadata.json`

**运行示例**:
```bash
# 生成前5个议题和前8个国家的桑基图
python src/topic_attention_sankey.py --meeting_folder "output/MEPC" --top_n_titles 5 --top_n_countries 8

# 设置最小文档数阈值
python src/topic_attention_sankey.py --meeting_folder "output/MEPC" --min_count 2
```

**可视化特点**

- 节点高度：反映文档数量（关注度）
- 连线透明度：表示流向强度
- 颜色编码：区分国家/议题
- 交互式：悬停显示详细信息

**参数说明**

| 参数 | sankey_trends.py | topic_attention_sankey.py | 默认值 | 说明 |
|------|-----------------|--------------------------|--------|------|
| `--meeting_folder` | ✓ | ✓ | 必需 | 会议文件夹路径 |
| `--top_n_titles` / `--top_k_titles` | ✓ | ✓ | 5/6 | 显示的议题数量 |
| `--top_n_countries` | ✓ | ✓ | 8 | 显示的国家数量 |
| `--min_count` / `--min_total_count` | ✓ | ✓ | 1 | 最小文档数阈值 |
| `--normalize` | ✓ | - | False | 是否归一化为百分比 |
| `--flow_method` | ✓ | - | min | 流向计算方法 (min/avg/next) |
| `--out_folder` | ✓ | ✓ | auto | 输出文件夹 |

## 常见问题

### 停用词调整

若分析结果中包含过多无意义词汇（如 "document", "page"），请编辑 src/stopword.py 中的 additional_stopwords 集合。

### API 错误处理

若 api_clean.py 报错 RateLimitError 或连接超时，请检查 QWEN_API_KEY 是否有效，或在代码中增加重试逻辑。

### PDF 解析乱码

部分旧版 PDF 可能无法通过 fitz 完美提取文本。系统依赖 OCR 或更强的 PDF 解析库（如 pdfplumber）作为替补，但目前主要基于文本层提取。