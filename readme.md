# IMO 会议提案文本挖掘与政策分析系统

**IMO Meeting Proposal Text Mining and Policy Analysis System**

基于国际海事组织（IMO）五大委员会/工作组的 2,523 篇会议提案文档，综合运用 BERTopic 主题建模、引用网络分析、共同提案联盟挖掘、政策立场检测等方法，系统性分析国际海事治理中的议题演化、国家参与模式与跨委员会政策关联。

> A comprehensive text mining and policy analysis framework for 2,523 IMO meeting proposals across five committees/working groups (MEPC, MSC, CCC, SSE, ISWG-GHG), employing BERTopic modeling, citation network analysis, co-sponsorship alliance detection, and policy stance classification.

---

## 📊 数据规模

| 委员会/工作组 | 会议届次 | 文档数 | 主题数 | 离群率 |
|:---:|:---:|:---:|:---:|:---:|
| MEPC（海洋环境保护委员会） | 77–83 | 856 | 11 | 14.7% |
| MSC（海上安全委员会） | 102–109 | 837 | 37 | 17.8% |
| CCC（货物运输分委会） | 7–10 | 396 | 19 | 16.4% |
| SSE（船舶系统与设备分委会） | 7–11 | 349 | 12 | 12.9% |
| ISWG-GHG（温室气体减排工作组） | 16–18 | 85 | 9 | 22.4% |
| **合计** | — | **2,523** | **88** | — |

---

## 📁 目录结构

```
IMO/
├── src/                          # 源代码
│   ├── parse_data.py             # PDF文档解析与文本提取
│   ├── parse_all_meetings.py     # 批量解析所有会议
│   ├── api_clean.py              # LLM结构化元数据提取 (Qwen)
│   ├── llm_clean.py              # LLM结构化元数据提取 (Claude)
│   ├── validate_json.py          # 数据验证
│   ├── json_read.py              # 通用数据读取模块
│   ├── json_process.py           # 数据清洗与后处理
│   ├── extract_metadata.py       # 元数据提取
│   ├── merge_data.py             # 数据合并
│   ├── fix_originator.py         # Originator字段修复
│   ├── fix_dates.py              # 日期格式修复
│   ├── stopword.py               # 海事领域停用词
│   ├── download_models.py        # 预训练模型下载
│   ├── bertopic_model.py         # BERTopic主题建模 ★核心
│   ├── country_stance.py         # 国家/组织参与模式与议题偏好
│   ├── alliance_network.py       # 共同提案网络 (Louvain社群发现)
│   ├── citation_network.py       # 引用网络 + TF-IDF相似度 + 演化链
│   ├── dynamic_topics.py         # 动态主题演化趋势
│   ├── cross_committee_deep.py   # 跨委员会深度对比分析
│   ├── cooccurrence.py           # 关键词共现网络
│   ├── sankey_trends.py          # 桑基图可视化（单议题流向）
│   ├── topic_attention_sankey.py # 桑基图可视化（多维关注度）
│   ├── html_vis.py               # HTML交互式可视化
│   ├── consolidate.py            # 数据量化整合
│   ├── visualize_trends.py       # 趋势可视化
│   ├── topic.py                  # [legacy] 早期LDA主题建模
│   ├── cmp.py                    # [legacy] 空文件
│   └── extracted.py              # [legacy] 早期提取脚本
│
├── data/                         # 原始PDF文档
│   └── {委员会}/
│       └── {会议届次}/
│           ├── Index.htm         # 会议文档索引
│           └── *.pdf             # 提案PDF文件
│
├── output/                       # 分析输出
│   ├── {委员会}/
│   │   ├── bertopic/             # BERTopic模型、主题CSV、交互式HTML
│   │   ├── alliance/             # 联盟网络统计JSON
│   │   ├── citation/             # 引用网络分析结果
│   │   └── {会议届次}/           # 各会议的解析数据
│   ├── stance_analysis/          # 5个委员会的国家参与分析
│   ├── dynamic_analysis/         # 动态趋势分析报告
│   ├── deep_analysis/            # 跨委员会分析（相似度矩阵、协作网络）
│   ├── llm_topic_labels.json     # 88个主题的LLM中英文标签
│   ├── all_proposals_metadata.csv/xlsx  # 全量元数据汇总
│   └── ISWG-GHG/
│       └── stance_detection_results.json  # 政策立场检测结果
│
├── imo/                          # Python虚拟环境
├── requirements.txt              # 依赖清单
├── readme.md                     # 本文件
└── 研究日志.md                    # 研究过程日志
```

---

## 🛠 技术栈与依赖

### 核心框架

| 类别 | 工具/库 | 用途 |
|:---|:---|:---|
| 主题建模 | BERTopic, sentence-transformers, UMAP, HDBSCAN | 基于Transformer的主题发现 |
| 网络分析 | NetworkX, python-louvain | 引用/联盟网络构建与社群发现 |
| NLP/ML | scikit-learn, TF-IDF | 文本向量化、相似度计算、聚类 |
| 可视化 | Plotly, Matplotlib | 交互式图表与学术图表 |
| 数据处理 | Pandas, NumPy | 结构化数据处理 |
| PDF解析 | PyMuPDF (fitz) | PDF文本提取 |
| LLM集成 | OpenAI SDK (Qwen/Claude) | 结构化信息提取、主题标签、立场检测 |
| 文档生成 | python-docx | Word文档生成 |

### 环境配置

```bash
# 1. 创建虚拟环境
python -m venv imo

# 2. 激活虚拟环境
# Windows:
imo\Scripts\activate
# Linux/Mac:
source imo/bin/activate

# 3. 安装依赖
pip install -r requirements.txt

# 4. 下载预训练模型（sentence-transformers）
python src/download_models.py
```

### 环境变量

系统使用云端大模型进行文档结构化提取，需配置 API 密钥：

```bash
# Qwen (通义千问) — 用于文档解析
export QWEN_API_KEY="sk-xxxxxxxx"
export QWEN_BASE_URL="https://api.qwen.ai/v1/chat/completions"

# Claude — 用于立场检测（可选）
export ANTHROPIC_API_KEY="sk-ant-xxxxxxxx"
```

---

## 📋 模块功能说明

### 一、数据采集与解析

| 脚本 | 功能 |
|:---|:---|
| `parse_data.py` | 解析 `Index.htm` 提取元数据，使用 PyMuPDF 提取 PDF 文本，正则预清洗 |
| `parse_all_meetings.py` | 批量遍历所有委员会/会议，调用 parse_data 进行全量解析 |
| `api_clean.py` / `llm_clean.py` | 调用 Qwen/Claude API 进行 LLM 结构化提取（Metadata、Sections、Tables） |
| `validate_json.py` | 校验生成的 JSON 结构完整性与字段类型 |
| `json_read.py` / `json_process.py` | 数据读取、字段拆分（Originator多国拆分）、词频统计 |
| `extract_metadata.py` | 元数据字段提取 |
| `merge_data.py` | 多源数据合并 |
| `fix_originator.py` | 从 PDF 文件名修复缺失的 Originator 字段 |
| `fix_dates.py` | 日期格式标准化 |
| `stopword.py` | 海事领域自定义停用词表 |

**数据处理流程：** PDF → PyMuPDF文本提取 → 正则预清洗 → LLM结构化 → JSON验证 → 后处理

### 二、主题建模

| 脚本 | 功能 |
|:---|:---|
| `bertopic_model.py` | **核心脚本。** 基于 BERTopic 的主题建模，使用 sentence-transformers 编码、UMAP 降维、HDBSCAN 聚类，自动发现文档主题结构。替代了早期 LDA 方案。 |
| `dynamic_topics.py` | 分析主题随会议届次的动态演化，识别上升趋势与下降趋势主题 |
| `topic.py` | [legacy] 早期 LDA 主题建模，已被 bertopic_model.py 替代 |

**BERTopic 流程：** 文档 → Sentence-BERT 编码 → UMAP 降维 → HDBSCAN 聚类 → c-TF-IDF 主题表示 → LLM 标签增强

### 三、网络分析

| 脚本 | 功能 |
|:---|:---|
| `country_stance.py` | 分析各国家/组织在不同委员会的提案数量、议题多样性（Shannon Entropy）、议题偏好 |
| `alliance_network.py` | 基于共同提案关系构建合作网络，Louvain 社群发现识别联盟结构 |
| `citation_network.py` | 正则提取文档间引用关系，构建有向引用网络，TF-IDF 相似度矩阵，层次聚类，演化链追踪 |
| `cross_committee_deep.py` | 跨委员会主题相似度矩阵、桥接国家识别、政策关联分析 |
| `cooccurrence.py` | TF-IDF 筛选关键词 → 滑动窗口共现矩阵 → Louvain 聚类着色 |

### 四、可视化

| 脚本 | 功能 |
|:---|:---|
| `sankey_trends.py` | 单议题文档流向桑基图（国家×会议） |
| `topic_attention_sankey.py` | 多维关注度桑基图（按议题/按国家分组） |
| `html_vis.py` | 交互式仪表盘（热力图、旭日图、气泡图） |
| `consolidate.py` / `visualize_trends.py` | 跨会议词频整合与趋势折线图 |

---

## ▶️ 运行说明

### 完整分析流程

```bash
# 1. 数据解析（以MEPC为例）
python src/parse_data.py --committee MEPC --session 77

# 2. 批量解析所有会议
python src/parse_all_meetings.py

# 3. LLM结构化提取
python src/api_clean.py --input output/MEPC/77/data.json

# 4. 数据验证与后处理
python src/validate_json.py --input output/MEPC/77/data.json
python src/json_process.py --input output/MEPC/77/data.json

# 5. BERTopic主题建模
python src/bertopic_model.py --committee MEPC

# 6. 网络分析
python src/alliance_network.py --committee MEPC
python src/citation_network.py --committee MEPC
python src/country_stance.py --committee MEPC

# 7. 跨委员会分析
python src/dynamic_topics.py
python src/cross_committee_deep.py

# 8. 可视化
python src/sankey_trends.py --meeting_folder output/MEPC
python src/topic_attention_sankey.py --meeting_folder output/MEPC
```

---

## 📈 关键结果摘要

### BERTopic 主题建模
- 5 个委员会共发现 **88 个主题**，LLM 生成中英文标签
- MEPC 以 GHG 减排、压载水管理、防污染为核心
- MSC 主题最丰富（37 个），涵盖海上安全各细分领域

### 引用网络
- MEPC 最长引用演化链：**30 篇文档**（压载水管理方向）
- 最高被引文档：**MEPC 80/17**（38 次被引）

### 国家参与
- 中国提案分布：MEPC 45 篇, MSC 56 篇, CCC 30 篇, SSE 59 篇（排名第一）, ISWG-GHG 9 篇
- 跨委员会实体重叠最高：SSE-ISWG-GHG（Jaccard 相似度 0.556）

### Originator 数据覆盖率
- MEPC 100%, MSC 95.8%, CCC 96.7%, SSE 97.7%, ISWG-GHG 100%

### ISWG-GHG 政策立场检测
- GFS（全球燃油标准）29.4%, IMSF&F（IMO基金与融资）17.6%, Emissions Pricing（排放定价）17.6%

---

## 📂 输出文件说明

| 输出路径 | 内容 |
|:---|:---|
| `output/{委员会}/bertopic/` | BERTopic 模型文件、主题词 CSV、交互式 HTML 可视化 |
| `output/{委员会}/alliance/` | 联盟网络统计 JSON、网络图 |
| `output/{委员会}/citation/` | 引用网络分析 JSON、交互式引用图、相似度热力图 |
| `output/{委员会}/{届次}/` | 各会议的解析 JSON、词频统计 |
| `output/stance_analysis/` | 5 个委员会的国家参与 CSV 报告 + 交互式 HTML |
| `output/dynamic_analysis/` | 主题动态趋势报告 |
| `output/deep_analysis/` | 跨委员会相似度矩阵、协作网络、政策关联 |
| `output/llm_topic_labels.json` | 88 个主题的 LLM 生成中英文标签 |
| `output/all_proposals_metadata.csv/.xlsx` | 2,523 篇提案全量元数据 |
| `output/ISWG-GHG/stance_detection_results.json` | ISWG-GHG 85 篇提案的政策立场分类 |

---

## 📝 论文图表

本项目支撑毕业论文的 8 张 300DPI SCI 风格图表（生成脚本：`gen_thesis_figures_v2.py`）：

| 图号 | 文件名 | 内容 |
|:---|:---|:---|
| 图2-1 | `fig2-1_technical_roadmap.png` | 技术路线图 |
| 图3-1 | `fig3-1_topic_distribution.png` | 委员会主题分布（文档数+主题数+离群率） |
| 图3-2 | `fig3-2_mepc_top10.png` | MEPC 前10活跃实体+多样性指数 |
| 图3-3 | `fig3-3_china_crosscommittee.png` | 中国跨委员会提案数+多样性 |
| 图3-4 | `fig3-4_mepc_alliance.png` | MEPC 共同提案 Top15 对（社群着色） |
| 图3-5 | `fig3-5_iswg_stance.png` | ISWG-GHG 政策立场分布+会议间演变 |
| 图3-6 | `fig3-6_dynamic_trends.png` | 上升/下降主题趋势 |
| 图3-7 | `fig3-7_cross_committee_heatmap.png` | 跨委员会实体参与重叠热力图 |

---

## 常见问题

### 停用词调整
若分析结果中包含过多无意义词汇（如 "document", "page"），请编辑 `src/stopword.py` 中的 `additional_stopwords` 集合。

### API 错误处理
若 `api_clean.py` 报错 `RateLimitError` 或连接超时，请检查 API Key 是否有效，或在代码中增加重试逻辑。

### PDF 解析乱码
部分旧版 PDF 可能无法通过 fitz 完美提取文本。可尝试 pdfplumber 作为备选方案。

### BERTopic 内存不足
MSC 委员会（837 篇）的 UMAP 降维和 HDBSCAN 聚类需要较大内存，建议至少 16GB RAM。

---

## License

本项目仅用于学术研究。IMO 会议文档版权归国际海事组织所有。
