## 使用说明

本项目为 IMO 相关的研究资料整理与分享。

## 目录结构

- `src/`：源代码文件夹
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
	├── session2/
	│   ├── paper1.pdf
	│   ├── paper2.pdf
	│   └── ...
	└── ...
```
`output`文件夹结构如图所示：
```
output/
└── conference_name/
	├── session1/
	│   ├── data.json
	│   ├── word_freq.json
	|   └── cooccurrence_graph.png
	├── session2/
	│   ├── data.json
	│   ├── word_freq.json
	|   └── cooccurrence_graph.png
	└── ...
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

2. 将会议数据放置在 `data` 文件夹下，结构如图所示：
```
data/
└── conference_name/
	├── session1/
	│   ├── paper1.pdf
	│   ├── paper2.pdf
	│   └── ...
	├── session2/
	│   ├── paper1.pdf
	│   ├── paper2.pdf
	│   └── ...
	└── ...
```

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

对应脚本: `src/parse_data.py, src/api_clean.py, src/validate_json.py`

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

使用示例:

```bash
python src/parse_data.py --title "MEPC" --subtitle "MEPC 77" --logging "log/logging.log"
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

## 自动化工作流脚本 (Workflow Automation)

为了简化操作，系统提供了三个 Shell 脚本来串联上述模块。

1. 初始化与解析 (`parse_initialize.sh`)

功能: 初始化目录结构，运行数据解析和清洗。
用法:
```bash
# ./src/parse_initialize.sh [Session_Number]
sh src/parse_initialize.sh 77
```

此命令会自动创建 output/MEPC/MEPC 77 目录，并调用 `parse_data.py` 处理 data/MEPC/MEPC 77 下的数据。

2. 运行主题分析 (topic.sh)

功能: 对指定会话运行 LDA 主题建模。
用法:
```bash
# ./src/topic.sh [Session_Number] [Num_Topics]
sh src/topic.sh 77 10
```

此命令会对 MEPC 77 的数据提取 10 个主题，并生成相关 CSV 报告。

3. 跨会话趋势整合 (consolidate.sh)

功能: 整合所有已处理会话的数据，生成趋势图。
用法:
```bash
sh src/consolidate.sh
```

此命令会扫描 output/MEPC/ 下的所有会话子目录，合并词频数据，并为预设关键词（如 ghg, lng, oil）生成趋势图。

## 常见问题

### 停用词调整

若分析结果中包含过多无意义词汇（如 "document", "page"），请编辑 src/stopword.py 中的 additional_stopwords 集合。

### API 错误处理

若 api_clean.py 报错 RateLimitError 或连接超时，请检查 QWEN_API_KEY 是否有效，或在代码中增加重试逻辑。

### PDF 解析乱码

部分旧版 PDF 可能无法通过 fitz 完美提取文本。系统依赖 OCR 或更强的 PDF 解析库（如 pdfplumber）作为替补，但目前主要基于文本层提取。