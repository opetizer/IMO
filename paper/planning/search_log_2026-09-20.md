# IMO 论文定位调研：检索记录与边界

_检索日期：2026-09-20。目的：选刊、定位邻近研究、设计可检验的创新路线。_

## 🔎 检索范围

采用公开网络的主题检索、题名精确检索和出版社域名限定检索，随后访问原始来源。重点覆盖 2024–2026 年，并纳入对研究设计有直接意义的较早文献。时间截止为检索日，不纳入尚未发生的会议结果。

本记录整理检索主题和关键复核式，不是原始搜索引擎全部返回结果的逐条导出。23 项是最终纳入本轮工作矩阵的来源数，不是系统综述的检出数、筛选数或全部相关文献数。

| 检索组 | 主题词或检索式组合 | 用途 |
| --- | --- | --- |
| 期刊范围 | Marine Policy / Maritime Policy & Management / Transport Policy + aims and scope | 确认政策、管理与方法的适配关系 |
| 环境与海事期刊 | Transportation Research Part D / Ocean & Coastal Management / WMU Journal of Maritime Affairs / Journal of Shipping and Trade + scope | 建立有条件备选 |
| IMO 治理 | IMO + decarbonisation + negotiations / coalition / influence / policy instruments | 定位制度过程及政策组合文献 |
| 政策一致性 | maritime decarbonization + policy coherence / design framing | 检验政策一致性方向是否已有研究 |
| 海事文本方法 | IMO + text classification / topic model / BERT | 检验主题与分类研究的新颖性 |
| 海事 LLM | maritime + large language model + governance / regulations / registration / RAG | 查找领域适配、检索和治理任务的进展 |
| 立场测量 | LLM + actor stance / codebook / external information | 设计立场编码与方法验证 |
| 提案处理 | stakeholder influence + automated / manual / text reuse | 定位主张吸收与影响测量的近邻研究 |
| 模型深化 | policy text + active learning / fine tuning；GraphRAG；political coalition LLM agents | 判断先进方法的适用条件 |
| 最新制度背景 | site:imo.org + MEPC 83 / ISWG-GHG 22 / Net-Zero Framework | 区分草案批准、正式通过与生效 |

最终复核包括以下精确检索：

```text
"Codebook LLMs" "2026"
site.tandfonline.com/doi "13501763.2026.2697930"
"Large language model enhanced maritime ship registration" site:sciencedirect.com
"From Local to Global: A Graph RAG" proceedings 2025
```

## 📚 来源处理

1. 期刊适配判断主要依据出版社范围页面，并以近期已发表研究佐证。共比较 9 个候选期刊；另说明 Ocean Engineering 与 Transportation Research Part E 当前不宜优先。
2. 文献优先采用出版社、ACL Anthology、作者机构库和 arXiv 原始页面。搜索中出现的聚合站仅用作线索，不承担本报告的方法结论依据。
3. 对 20 项已识别 DOI 查询 Crossref，核对题名、作者、刊物及登记日期；原始返回字段保存在 JSON 的 crossref 键下。遇到 HTTP 429 后降低请求频率并重试，20 项最终均取得元数据。
4. Crossref 的 published 可能是在线日期，也可能是卷期日期。矩阵 publication_date 显式保留已经核对的差别，避免把 2024 年在线、2026 年编入卷期当成两项新研究。
5. 另 3 项按本次读取的 arXiv 版本记录，作者和版本日期来自原始摘要页。没有把 arXiv 的收录当作同行评审证明；这也不等于断言作者此后未正式发表。
6. reading_level 逐项记录读到摘要、索引选段还是部分正文。没有将摘要阅读表述为全文精读，也没有据此比较未读实验的精确性能。

## ⚠️ 覆盖限制

- 未检索订阅制 Web of Science/Scopus 的完整索引，没有执行可重放的数据库系统综述流程。
- ScienceDirect 和 Taylor & Francis 部分页面直接打开返回 403；可访问的官方搜索索引选段与作者机构库用于补充，访问深度保留在矩阵中。
- 对作者机构库摘要可确认的研究对象、方法和数量作概括；未据此推断未读正文的识别强度。
- 搜索结果时间标签不作为正式发表日期。以出版社、版本记录和 Crossref 的明确字段核对。
- 没有核验最新 JCR / 中科院分区、期刊录用率或统一口径 APC；本轮排序只表示研究匹配判断。
- 未检出完全同构论文不能支持“全球首次”。下一阶段应优先精读 L05、L10–L13、L21 的完整正文与附录，再最终收敛贡献表述。

## 📂 可复核文件

- [研究报告](D:/university/2025-2026-2/Research/IMO/paper/journal_literature_innovation_review.md)
- [23 项文献矩阵 CSV](D:/university/2025-2026-2/Research/IMO/paper/planning/literature_sources_2026-09-20.csv)
- [来源及 Crossref 元数据 JSON](D:/university/2025-2026-2/Research/IMO/paper/planning/literature_sources_2026-09-20.json)
- [矩阵与参考资料生成脚本](D:/university/2025-2026-2/Research/IMO/paper/planning/finalize_literature.py)

脚本只整理已经核验的书目信息，不重新检索网络，不修改原始 IMO 文件或既有分析结果。
