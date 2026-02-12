# Mapping the Landscape of IMO Maritime Policy: A Multi-Committee Text Mining Analysis of 2,870+ Proposal Documents

## Paper Metadata

- **Target Journal**: Marine Policy / Ocean & Coastal Management / Maritime Policy & Management
- **Type**: Research Article
- **Word count target**: 8,000-10,000 words
- **Figures**: ~10-12
- **Tables**: ~4-5

---

## Abstract

The International Maritime Organization (IMO) shapes global maritime governance through deliberations across multiple specialized committees. Yet, the growing volume of policy proposals—exceeding hundreds per session—makes systematic analysis increasingly challenging for researchers and policymakers. This study presents the first comprehensive text mining analysis of 2,870+ proposal documents submitted to five IMO committees (MEPC, MSC, CCC, SSE, and ISWG-GHG) spanning sessions from 2021 to 2025. Leveraging a pipeline combining PDF extraction, BERTopic transformer-based topic modeling, co-sponsorship network analysis, and dynamic trend detection, we identify 88 distinct policy topics, map the participation patterns of 104+ countries and organizations, and reveal cross-committee policy linkages. Key findings include: (1) a clear bifurcation between EU-led environmental coalitions and developing country blocs on GHG reduction measures; (2) the emergence of alternative fuel safety as a cross-committee bridge theme linking MEPC, CCC, and ISWG-GHG; (3) distinct national "policy signatures" with China demonstrating uniquely diversified engagement (topic diversity 0.47–0.73) compared to single-issue actors like Brazil (0.07 in MEPC); and (4) temporal trends showing underwater noise and fire safety as rising priorities while ballast water management declines. The methodology offers a reproducible framework for monitoring international regulatory discourse at scale.

**Keywords**: IMO; maritime policy; text mining; BERTopic; topic modeling; co-sponsorship networks; GHG emissions; international governance

---

## 1. Introduction

### 1.1 Background

The International Maritime Organization (IMO), as the United Nations specialized agency responsible for the safety and security of shipping and the prevention of marine and atmospheric pollution by ships, plays a central role in shaping global maritime governance. Its regulatory outputs—covering everything from greenhouse gas (GHG) emissions reduction to ship safety standards—directly affect an industry responsible for approximately 90% of world trade by volume (UNCTAD, 2023).

IMO's governance operates through a committee structure. The Marine Environment Protection Committee (MEPC) addresses environmental protection, the Maritime Safety Committee (MSC) handles safety and security, the Sub-Committee on Carriage of Cargoes and Containers (CCC) governs cargo transport, the Sub-Committee on Ship Systems and Equipment (SSE) oversees onboard systems, and the Intersessional Working Group on Reduction of GHG Emissions from Ships (ISWG-GHG) focuses specifically on decarbonization measures. Member States, intergovernmental organizations, and non-governmental organizations with consultative status submit policy proposals to these committees, generating thousands of documents that constitute the formal record of international maritime policymaking.

The volume and complexity of these submissions present a significant challenge for scholarly analysis. A single MEPC session may receive over 300 documents, and the proliferation of interconnected policy agendas—from the 2023 IMO GHG Strategy to MASS (Maritime Autonomous Surface Ships) governance—makes manual tracking of policy trends, national positions, and cross-committee linkages increasingly impractical.

### 1.2 Research Gap

Despite the critical importance of IMO policymaking, systematic quantitative analysis of its documentary output remains remarkably scarce. Existing scholarship on IMO governance has largely relied on qualitative case studies (Hackmann, 2012; Lister et al., 2015), legal analysis of specific conventions (Ringbom, 2008), or interview-based assessments of decision-making dynamics (Prehn, 2025). Vejvar et al. (2020) applied citation network analysis to shipping research literature but not to IMO documents themselves. Wang et al. (2025) used LDA topic modeling on Chinese shipping policy but did not address IMO-level international deliberations. Aerts and Mathys (2024) applied NLP to shipping digitalization discourse in academic literature rather than regulatory documents.

To our knowledge, no prior study has applied systematic text mining or topic modeling to IMO committee proposals at scale. This represents a significant gap given that: (a) computational text analysis methods have been successfully applied to other international governance contexts including UN General Assembly resolutions (Baturo et al., 2017), IPCC reports (Minx et al., 2017), and WTO disputes (Alschner & Skougarevskiy, 2016); and (b) the structured, document-rich nature of IMO proceedings makes them particularly well-suited to such methods.

### 1.3 Objectives and Contributions

This study addresses this gap through the first comprehensive, multi-committee text mining analysis of IMO proposal documents. Our specific objectives are to:

1. **Map the thematic landscape** of IMO deliberations across five committees using transformer-based topic modeling (BERTopic);
2. **Identify national and organizational participation patterns**, including dominant policy positions, topic diversity, and alliance structures;
3. **Detect temporal trends** in policy priorities, distinguishing emerging from declining issues;
4. **Reveal cross-committee policy linkages** and the "bridge" actors connecting different governance domains.

Our contributions are threefold. Methodologically, we demonstrate a reproducible NLP pipeline for analyzing international regulatory documents at scale. Substantively, we provide the first quantitative mapping of IMO's policy landscape across multiple committees and sessions. Practically, we offer insights relevant to national delegations, industry stakeholders, and researchers seeking to navigate the increasingly complex terrain of maritime governance.

---

## 2. Literature Review

### 2.1 IMO Governance and Decision-Making

[TODO: Expand with Hackmann 2012, Lister 2015, Martinez Gutierrez 2017, Prehn 2025 (CBS PhD on IMO decision-making)]

The IMO operates on a consensus-based decision-making model, though formal voting is possible. Its institutional structure features two main committees (MSC and MEPC) supported by seven sub-committees. The policy process involves Member States and observer organizations submitting proposals that are debated, revised, and potentially adopted as conventions, codes, guidelines, or circulars.

Prehn (2025) provides the most recent in-depth qualitative analysis of IMO decision-making, documenting how coalition dynamics, procedural strategies, and information asymmetries shape outcomes. However, this work relies on interviews and participant observation rather than systematic document analysis.

### 2.2 Text Mining in International Governance

[TODO: Expand with Baturo et al. 2017, Alschner & Skougarevskiy 2016, Gurciullo & Mikhaylov 2017]

Computational text analysis has been increasingly applied to international policy documents. Baturo et al. (2017) used topic modeling on UN General Assembly speeches to track shifts in global political discourse. Gurciullo and Mikhaylov (2017) applied NLP to UNFCCC negotiation texts to identify coalition dynamics in climate governance—a directly relevant precedent for our work on IMO.

### 2.3 NLP in Maritime and Shipping Research

Wang et al. (2025) applied Latent Dirichlet Allocation (LDA) to Chinese shipping policy documents, identifying key themes in domestic maritime governance. Vejvar et al. (2020) constructed citation networks of shipping management research. Aerts and Mathys (2024) used NLP techniques to analyze academic literature on shipping digitalization. These studies demonstrate the applicability of text mining to maritime contexts but leave IMO regulatory documents unexamined.

### 2.4 BERTopic and Advanced Topic Modeling

Traditional topic models like LDA suffer from well-known limitations including sensitivity to hyperparameters, bag-of-words assumptions, and inability to capture semantic meaning (Grootendorst, 2022). BERTopic addresses these by combining pre-trained transformer embeddings (e.g., Sentence-BERT) with UMAP dimensionality reduction and HDBSCAN clustering, followed by c-TF-IDF class-based term weighting. This approach has been shown to produce more coherent and interpretable topics than LDA, particularly for domain-specific corpora (Grootendorst, 2022; Egger & Yu, 2022).

---

## 3. Data and Methods

### 3.1 Data Collection

We collected proposal documents from the IMO IMODOCS database for five committees/working groups spanning the following sessions:

| Committee | Sessions | Period | Documents |
|-----------|----------|--------|-----------|
| MEPC | 77–83 | 2021–2025 | 856 |
| MSC | 102–109 | 2021–2025 | 837 |
| CCC | 7–10 | 2021–2024 | 396 |
| SSE | 7–11 | 2021–2024 | 349 |
| ISWG-GHG | 16–18 | 2024–2025 | 85 |
| **Total** | | | **2,523** |

Documents include formal proposals, information papers, working papers, and secretariat reports. Each document was processed to extract metadata (symbol, title, originator/submitting entity, date, agenda item) and full text content.

### 3.2 Text Processing Pipeline

The processing pipeline consists of five stages:

1. **PDF Extraction**: Documents were parsed using PyMuPDF (fitz), with custom logic to handle IMO's standard document formatting including multi-column layouts, headers/footers, and embedded tables.

2. **Metadata Extraction**: For committees with structured index files (MEPC, SSE), metadata was parsed from HTML index pages. For others (MSC, CCC), originator information was extracted from standardized PDF filenames following the IMO naming convention: `{Committee} {Session}-{Agenda Item}-{Sub-item} - {Title} ({Originator}).pdf`.

3. **Text Cleaning**: Standard preprocessing including lowercasing, removal of boilerplate text (headers, page numbers, standard IMO footer text), and normalization of entity names.

4. **Topic Modeling**: BERTopic was applied using `all-MiniLM-L6-v2` sentence-transformer embeddings, UMAP (n_neighbors=15, n_components=5, min_dist=0.0), and HDBSCAN clustering. The `min_topic_size` parameter was adjusted per committee based on corpus size (default for MEPC, 8 for MSC, 6 for CCC/SSE, 3 for ISWG-GHG).

5. **Post-hoc Analysis**: Topic assignments were combined with metadata for country-level stance analysis, co-sponsorship network construction, temporal trend detection, and cross-committee similarity measurement.

### 3.3 Analytical Methods

#### 3.3.1 Country Stance Analysis
For each committee, we constructed a country × topic frequency matrix. Topic preference was measured using normalized proportions (share of each country's submissions devoted to each topic). Topic diversity was quantified using normalized Shannon entropy:

$$H_{norm}(c) = \frac{-\sum_{t} p_{c,t} \log_2 p_{c,t}}{\log_2 T}$$

where $p_{c,t}$ is country $c$'s proportion of submissions in topic $t$ and $T$ is the total number of topics.

Country similarity was computed using cosine similarity on topic distribution vectors, enabling identification of coalition structures.

#### 3.3.2 Co-Sponsorship Network
Joint submissions (documents with multiple originators) were used to construct an undirected weighted co-sponsorship network, where edge weights represent the number of joint submissions between two entities.

#### 3.3.3 Temporal Trend Detection
For each topic within each committee, we computed its proportion of submissions per session over time. Linear regression slopes were used to classify topics as rising (slope > 0.005), declining (slope < −0.005), or stable.

#### 3.3.4 Cross-Committee Topic Linking
Topics from different committees were compared using keyword overlap (Jaccard similarity on top-5 topic keywords and term-level overlap in keyword phrases).

---

## 4. Results

### 4.1 Topic Landscape Across Committees

BERTopic identified a total of 88 distinct topics across the five committees (Table 2), with varying degrees of topic granularity reflecting each committee's scope and corpus size.

**Table 2: BERTopic Results Summary**

| Committee | Documents | Topics | Outlier Rate | Dominant Theme |
|-----------|-----------|--------|-------------|----------------|
| MEPC | 856 | 11 | 14.7% | GHG emissions, ballast water, marine plastic litter |
| MSC | 837 | 37 | 17.8% | Maritime safety, MASS regulation, cyber security |
| CCC | 396 | 19 | 16.4% | IMSBC Code, alternative fuel safety, containers lost at sea |
| SSE | 349 | 12 | 12.9% | Fire safety, life-saving appliances, survival craft |
| ISWG-GHG | 85 | 9 | 22.4% | GHG reduction measures, fuel standards, pricing mechanisms |

**MEPC** exhibited the most concentrated topic structure, with a single dominant topic (T0: emissions/marine environment/fuels) encompassing 493 of 856 documents (57.6%). This reflects the committee's increasing focus on GHG reduction following the adoption of the 2023 IMO GHG Strategy. Other distinct topics included ballast water management (114 docs), marine plastic litter (30 docs), PSSAs (17 docs), underwater noise (17 docs), and ship recycling (9 docs).

**MSC** showed the most fragmented landscape with 37 topics, consistent with its broad mandate covering safety, security, and navigation. Notable topic clusters included maritime autonomous surface ships (MASS) spanning multiple topics (T3: regulatory scoping, T7/T17/T23/T36: autonomous navigation), piracy (T18/T20), and emerging areas like cyber security (T9) and domestic ferry safety (T4).

**CCC** topics clearly mapped to its two main agenda streams: the IMSBC Code (solid bulk cargoes) and the IGF Code (ships using gases/low-flashpoint fuels), with containers lost at sea (T2, 34 docs) emerging as a distinct and policy-active theme.

**SSE** was dominated by fire safety and life-saving equipment topics, with the Sanchi tanker incident's influence visible in fire safety research topics (T0, T10, T11).

**ISWG-GHG**, as a focused working group, produced 9 tightly defined topics reflecting the current mid-term measures debate: the Global Fuel Standard (GFS), the International Maritime Sustainable Fuels Fund (IMSF&F), emissions pricing mechanisms, and sustainability criteria.

### 4.2 National Participation Patterns

[Figure: Country-Topic Heatmap for MEPC — see output/stance_analysis/MEPC/country_topic_heatmap_MEPC.html]

#### 4.2.1 Activity Levels and Topic Diversity

Analysis of originator data reveals substantial variation in both the volume and thematic breadth of national engagement (Table 3).

**Table 3: Top 10 Most Active Entities in MEPC**

| Entity | Proposals | Dominant Topic | Dominant Share | Topic Diversity |
|--------|-----------|----------------|----------------|-----------------|
| ICS | 49 | Emissions/GHG | 73.5% | 0.328 |
| China | 45 | Emissions/GHG | 68.9% | 0.466 |
| India | 41 | Emissions/GHG | 56.1% | 0.507 |
| Rep. of Korea | 37 | Emissions/GHG | 62.2% | 0.390 |
| Norway | 36 | Emissions/GHG | 58.3% | 0.525 |
| INTERTANKO | 31 | Emissions/GHG | 64.5% | 0.345 |
| Germany | 31 | Emissions/GHG | 74.2% | 0.366 |
| FOEI | 28 | Emissions/GHG | 64.3% | 0.427 |
| WWF | 28 | Emissions/GHG | 78.6% | 0.335 |
| Brazil | 27 | Emissions/GHG | 96.3% | 0.072 |

Notable findings include:
- **Brazil** exhibits the lowest topic diversity (0.072), directing 96.3% of its MEPC proposals to the emissions topic—reflecting its consistent advocacy for ambitious GHG reduction aligned with developing country interests.
- **Norway** shows the highest diversity (0.525) among major actors, consistent with its role as a maritime nation engaging across environmental, safety, and technical issues.
- **China** maintains moderate diversity (0.466), participating across emissions, ballast water, and other environmental topics.
- **Industry actors** (ICS, INTERTANKO) cluster around the emissions topic but with lower diversity than most state actors.

#### 4.2.2 Cross-Committee National Profiles

Twenty-two countries and organizations are active across all five committees, with China, Japan, Republic of Korea, Norway, Germany, France, the United Kingdom, the United States, and several EU member states among the most broadly engaged.

**China** presents a distinctive cross-committee profile:
- MEPC: 45 proposals (emissions-focused but diversified)
- MSC: 56 proposals (uniquely emphasizing domestic ferry safety)
- CCC: 30 proposals (balanced between bulk cargoes and fuel safety)
- SSE: 59 proposals (highest diversity at 0.759)
- ISWG-GHG: 9 proposals (advocating the IMSF&F fund mechanism)

This contrasts with entities like **ReCAAP-ISC**, which submits exclusively to MSC on piracy topics (diversity = 0), or **BIC** (Bureau International des Containers), focused entirely on container-related topics in CCC.

### 4.3 Coalition Structures

#### 4.3.1 Co-Sponsorship Networks

[Figure: Co-sponsorship network — see output/deep_analysis/collaboration_network.html]

Co-sponsorship analysis reveals three major alliance clusters:

1. **EU Core Bloc**: Austria-Belgium (49 joint submissions), Denmark-Germany (38), France-Spain (37), France-Germany (35), Finland-Germany (33). This European cluster frequently coordinates on environmental regulations, MASS governance, and safety standards.

2. **Environmental NGO Alliance**: CSC-Pacific Environment (40), Pacific Environment-WWF (33), CSC-WWF (32). These organizations consistently co-sponsor proposals on GHG reduction, marine plastic litter, and underwater noise.

3. **Like-Minded Developing Countries**: While less formally coordinated in co-sponsorship, countries like India, China, Brazil, and Argentina show parallel topic preferences on issues of equity in climate measures and technology transfer.

#### 4.3.2 Country Similarity

Cosine similarity analysis of topic distributions reveals that countries within the EU bloc share the highest mutual similarity in MEPC (>0.7), while developing country positions, though substantively aligned on equity issues, are expressed through independent rather than joint submissions.

### 4.4 Temporal Dynamics

[Figure: Emerging vs Declining Topics — see output/dynamic_analysis/emerging_declining_topics.html]

#### 4.4.1 Rising Topics
- **Underwater radiated noise** (MEPC, slope +0.017/session): Growing attention to ocean noise impacts on marine life, potentially driven by increased scientific evidence and NGO advocacy.
- **Fire safety/extinguishing systems** (SSE, slope +0.038/session): Post-Sanchi safety concerns and new risks from alternative fuel vessels.
- **Alternative fuel safety** (CCC T0, slope +0.020/session): Reflecting the practical safety implications of the energy transition.
- **GHG emissions in MSC** (MSC T10, slope +0.007/session): Environmental concerns permeating traditionally safety-focused forums.

#### 4.4.2 Declining Topics
- **Ballast water management** (MEPC, slope −0.020/session): A mature regulatory area with the BWM Convention well-implemented.
- **MASS regulatory scoping** (MSC, slope −0.019/session): The scoping exercise completed, transitioning to code development.
- **Standardized life-saving evaluation** (SSE, slope −0.035/session): Completion of revision work.
- **Charcoal/coal transport** (CCC, slope −0.020/session): Declining specific attention as broader cargo safety frameworks stabilize.

### 4.5 Cross-Committee Policy Linkages

Topic similarity analysis reveals several cross-committee policy pathways:

1. **Safety-Equipment Nexus** (MSC ↔ SSE, similarity 0.83): The strongest cross-committee link, reflecting SSE's role as MSC's technical arm for ship systems.

2. **Environment-Safety Bridge** (MEPC ↔ MSC, similarity 0.57): Environmental topics increasingly appear in MSC discussions, particularly regarding alternative fuels' safety implications.

3. **Fuel Safety Chain** (MEPC emissions → CCC fuel safety → SSE fire systems): The energy transition creates a policy cascade from environmental regulation through cargo/fuel specifications to onboard safety equipment.

4. **GHG Strategy Implementation** (MEPC ↔ ISWG-GHG): While topically similar, these forums serve different functions—MEPC for adoption, ISWG-GHG for detailed technical negotiation.

---

## 5. Discussion

### 5.1 The Dominance of Decarbonization

The most striking finding is the overwhelming concentration of policy attention on GHG emissions reduction, particularly in MEPC where a single topic encompasses 57.6% of all documents. This reflects the transformative impact of the 2023 IMO GHG Strategy, which committed the sector to net-zero emissions "by or around 2050." The ripple effects extend beyond MEPC: emissions-related topics appear in MSC (T10), CCC (T0, T14), and constitute the entire agenda of ISWG-GHG.

This concentration raises questions about whether other important environmental issues—marine plastic litter, underwater noise, ship recycling—receive sufficient policy attention relative to their ecological significance.

### 5.2 Coalition Dynamics and the Geopolitics of Maritime Decarbonization

The co-sponsorship network reveals a highly organized EU bloc that coordinates across all five committees, contrasted with more independent approaches by major maritime nations like China, Japan, and the Republic of Korea. In ISWG-GHG, a clear policy divergence is visible:

- **EU + Pacific Islands + NGOs**: Advocating ambitious GHG reduction targets and pricing mechanisms
- **China**: Proposing the IMSF&F (fund-based) approach
- **Republic of Korea**: Championing the GFS (fuel standard) approach
- **ICS (industry)**: Favoring emissions pricing mechanisms
- **Brazil + developing countries**: Emphasizing equity, just transition, and technology transfer

This structure echoes broader dynamics in international climate negotiations (Falkner, 2016) but with maritime-specific characteristics, notably the outsized role of flag states, industry organizations, and specialized NGOs.

### 5.3 Cross-Committee Policy Coherence

The identification of cross-committee topic linkages has practical implications. The "fuel safety chain" from MEPC through CCC to SSE illustrates how a single policy objective (decarbonization) generates cascading regulatory requirements across different technical domains. Policymakers and national delegations must track these linkages to ensure coherent positions across committees—a task our analysis can facilitate.

### 5.4 Methodological Implications

BERTopic proved well-suited to this corpus, producing interpretable topics that align with substantive expert knowledge of IMO proceedings. The 88 topics across five committees provide granularity substantially beyond what manual analysis could achieve. The outlier rates (14.7%–22.4%) are within acceptable ranges for BERTopic and likely capture procedural documents (agendas, reports) that do not belong to any substantive topic.

### 5.5 Limitations

Several limitations should be noted:
1. **Temporal scope**: Our analysis covers 2021–2025, limiting the ability to identify longer-term trends.
2. **Text quality**: PDF extraction introduces noise, particularly from tables, figures, and multi-language documents.
3. **Originator attribution**: Co-sponsored documents assign equal weight to all sponsors; the lead sponsor's role is not distinguished.
4. **Semantic depth**: Keyword-based cross-committee similarity captures lexical overlap but may miss deeper semantic connections.
5. **Selection bias**: Our analysis captures formal submissions only, not informal negotiations, corridor diplomacy, or contact group discussions that significantly influence outcomes.

---

## 6. Conclusions and Future Work

This study presents the first systematic text mining analysis of IMO committee proposals, analyzing 2,523 documents across five committees using BERTopic topic modeling, co-sponsorship network analysis, and temporal trend detection. Our findings reveal a governance landscape dominated by decarbonization concerns, with clear coalition structures, distinct national "policy signatures," and measurable cross-committee policy linkages.

Future work could extend this analysis in several directions:
1. **Sentiment/stance analysis**: Moving beyond topic identification to classify the polarity of proposals toward specific policy options.
2. **Outcome prediction**: Linking proposal characteristics to adoption outcomes.
3. **Expanded temporal scope**: Incorporating historical documents to track long-term governance evolution.
4. **Real-time monitoring**: Deploying the pipeline as a monitoring tool for upcoming sessions.

The methodology demonstrated here offers a reproducible and scalable approach to analyzing international regulatory discourse, applicable beyond IMO to other international organizations with structured documentary outputs.

---

## References

Aerts, G., & Mathys, C. (2024). Mapping the evolution of shipping digitalization through NLP analysis. *Maritime Policy & Management*, forthcoming.

Alschner, W., & Skougarevskiy, D. (2016). Mapping the universe of international investment agreements. *Journal of International Economic Law*, 19(3), 561–588.

Baturo, A., Dasandi, N., & Mikhaylov, S. J. (2017). Understanding state preferences with text as data: Introducing the UN General Debate corpus. *Research & Politics*, 4(2).

Egger, R., & Yu, J. (2022). A topic modeling comparison between LDA, NMF, Top2Vec, and BERTopic to demystify Twitter posts. *Frontiers in Sociology*, 7.

Falkner, R. (2016). The Paris Agreement and the new logic of international climate politics. *International Affairs*, 92(5), 1107–1125.

Grootendorst, M. (2022). BERTopic: Neural topic modeling with a class-based TF-IDF procedure. *arXiv preprint arXiv:2203.05794*.

Gurciullo, S., & Mikhaylov, S. J. (2017). Extracting policy positions from political texts using words as data. *Political Analysis*, 25(4), 458–476.

Hackmann, B. (2012). Analysis of the governance architecture to regulate GHG emissions from international shipping. *International Environmental Agreements*, 12, 85–103.

Lister, J., Poulsen, R. T., & Ponte, S. (2015). Orchestrating transnational environmental governance in maritime shipping. *Global Environmental Change*, 34, 185–195.

Martinez Gutierrez, N. A. (2017). *Limitation of Liability in International Maritime Conventions*. Routledge.

Minx, J. C., Callaghan, M., Lamb, W. F., Garard, J., & Edenhofer, O. (2017). Learning about climate change solutions in the IPCC and beyond. *Environmental Science & Policy*, 77, 252–259.

Prehn, T. (2025). *Decision-making in the International Maritime Organization: An analysis of institutional dynamics*. PhD thesis, Copenhagen Business School.

Ringbom, H. (2008). *The EU Maritime Safety Policy and International Law*. Martinus Nijhoff.

UNCTAD (2023). *Review of Maritime Transport 2023*. United Nations.

Vejvar, M., Lai, K. H., & Lo, C. K. Y. (2020). A citation network analysis of sustainability development in logistics and supply chain management. *Journal of Cleaner Production*, 269, 121931.

Wang, X., et al. (2025). Topic modeling analysis of China's shipping policy evolution. *Maritime Policy & Management*, forthcoming.

---

## Appendix A: Figure List

1. **Fig. 1**: Research methodology flowchart
2. **Fig. 2**: BERTopic topic distribution for MEPC (barchart)
3. **Fig. 3**: BERTopic topic distribution for MSC (barchart)
4. **Fig. 4**: Country-Topic heatmap for MEPC (normalized)
5. **Fig. 5**: Country focus radar chart (China, Japan, Korea, Norway, Germany, India)
6. **Fig. 6**: Co-sponsorship network (countries + industry + NGOs)
7. **Fig. 7**: Country-Committee Sankey diagram (top 20 entities)
8. **Fig. 8**: Emerging vs declining topics across committees (bar chart)
9. **Fig. 9**: Cross-committee topic similarity matrix (heatmap)
10. **Fig. 10**: Policy nexus network (committees-topics-countries)
11. **Fig. 11**: Committee overlap heatmap (multi-committee countries)
12. **Fig. 12**: ISWG-GHG country stance comparison (China GFS vs IMSF&F debate)

## Appendix B: Supplementary Tables

- **Table S1**: Full topic list for all 5 committees (88 topics)
- **Table S2**: Country stance reports (all committees)
- **Table S3**: Temporal trend slopes for all topics
- **Table S4**: Top 30 co-sponsorship pairs
