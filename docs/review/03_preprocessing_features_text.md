# Pre-processing, features and text

## 1. What we built

| Step | Logic | Code | Output |
|---|---|---|---|
| HTML to text | Strip markup, flatten tables to rows, normalise whitespace | `parse/html.py`, `parse/worker.py` | 2,550,579 tables flattened |
| Section extraction | Heading detection, longest span, title-only fallbacks, 8-K body to signature | `parse/sections.py` | 121,398 sections |
| Masking | Replace firm names, former names and tickers with `[FIRM]` and `[TICKER]` | `parse/anonymize.py` | 881,908 name and 249,224 ticker replacements |
| Numeric-line drop | Remove lines dominated by numbers before modelling | `parse/html.py:drop_numeric_lines`, used at `features/text.py:80` and `models/embeddings.py:136` | Narrative-only text |
| Chunking | 350-word windows, 50-word overlap, at most 24 chunks per section selected at even spacing | `models/embeddings.py:15-27` | About 900,694 chunks, 474 million tokens |
| Lexical features | Loughran-McDonald category shares for the whole document and the main section | `features/text.py:36-49` | 18 features |
| Change features | Hashing-vector cosine, sentence-set Jaccard, length change, tone change against the firm's previous same-type filing | `features/text.py:84-131` | 11 features |
| Novelty | Share of sentences not seen in the firm's earlier filings | `features/text.py:124` | 1 feature |
| TF-IDF | Fitted inside each training fold, 20,000 terms, truncated SVD to 64 | `models/ladder.py:120-133` | 64 factors in L4, L6, L7 |
| Embeddings | Frozen ChronoBERT (yearly vintage Y-1) and BGE-base, mean (or CLS) pooling per chunk, event vector = mean and max over chunks, PCA to 64 inside each fold | `models/embeddings.py`, `ladder.py:135-143` | 1,536 numbers per event before PCA |
| Price features | Returns over 1, 5, 20, 60, 250 days; momentum 12-2; volatility; dollar volume; Amihud; distance from 52-week high; beta; idiosyncratic volatility; market state | `features/price.py` | 19 features |
| Accounting features | Latest point-in-time values for assets, leverage, ROA, revenue, EPS, market cap, book-to-market; SUE | `features/accounting.py` | 10 features |
| Metadata | Form, 8-K categories, sections present, industry, calendar, session, filing lag | `events/build.py`, `features/registry.py` | 68 features |

## 2. What is good

1. Every feature has an availability rule asserted per row (`feature_registry.csv`).
2. Change features use a fixed hashing space, so nothing is learned from the future (`text.py` docstring).
3. TF-IDF and PCA are fitted on training rows only (`ladder.py:120-143`), and ties are processed in batches so simultaneous events never see each other (`text.py:92-130`).
4. Price features use only closes at or before the decision time (`price.py:33-34`).

## 3. Weaknesses, with evidence

### 3.1 Feature hygiene (W11)

| Finding | Evidence | Fix |
|---|---|---|
| `px_rev_1m` is exactly `px_ret_20` | `price.py:48` assigns one from the other; Spearman 1.000 | Drop it or define reversal on a different window |
| 18 feature pairs with abs(rho) above 0.95 | `review_facts.json`: `meta_has_10K_7` with `meta_type_periodic_10K` (0.997), `lm_n_words` with `txt_doc_chars` (0.999), `boilerplate_share` with `chg_jaccard_sent` (0.979) | Prune by clustering on rank correlation; keep one per cluster |
| 2 zero-variance columns | `meta_cat_abs`, `meta_cat_shell_status` | Drop |
| `meta_year`, `meta_month`, `meta_dow` are model features | Feature set; `meta_year` has univariate rank correlation 0.047 with magnitude | Drop `meta_year` (file 12, N2) |
| Accounting availability rule counts `sue_*` columns, so pre-2009 events pass with no accounting values | `ladder.py:72`; 65% pass in 2000-2008, 0% real values | Real-value rule and per-rung subsets (file 12, N1) |
| Missing data by family | accounting 42.1%, change 17.5%, earnings-reaction 6.8%, others about 1% | Add missingness indicators for accounting (elastic net adds them, trees handle them) and fix coverage upstream |
| Raw levels drift | Adversarial validation top drivers: `acc_log_assets`, `acc_roa`, `tfidf_svd1-5`, `px_ret_250` (AUC 0.9999 to 1.000) | Use cross-sectional ranks or z-scores within date, ratios to history, and drop pure levels |
| No cross-sectional normalisation | `ladder.py:160-193` builds raw matrices | Per-date rank-gauss or z-score for price, accounting and tone features |

### 3.2 Text design (W5)

| Finding | Evidence | Why it matters |
|---|---|---|
| **Numeric lines are deleted** before any text model | `default.yaml:76`; `text.py:80` | On an earnings 8-K, the surprise, guidance range and margin sit in the numbers. The text model sees the commentary around them |
| **Whole-filing mean pooling** | `embeddings.py:143-161` | Averages hundreds of chunks. One sentence that moves the stock is a tiny share of the vector |
| **Chunk selection by even spacing** | `select_chunks` | Drops the middle of long sections without regard to content |
| **No target awareness** | Frozen encoders; PCA to 64 | Compression keeps variance, not return-relevant variance |
| **No passage-level change against the prior filing** | Change features are document-level cosine and Jaccard | The informative signal is which passages are new or deleted (Cohen, Malloy and Nguyen 2020 use this idea) |
| **Dictionary tone is old-fashioned** | Loughran-McDonald counts | Fine as a baseline, weak as the main text signal |
| **Two of four planned encoders dropped** | `fold_provenance.csv`, D-074 | No finance-specific encoder in the comparison |
| **Embeddings only for 60,030 events** | `s14.json` | 231 events have none |

### 3.3 Masking (W14)

| Finding | Evidence |
|---|---|
| Single-word firm names masked only when Title or UPPER case | `anonymize.py` single-word branch |
| Product, brand, subsidiary and executive names are unmasked | No such lists in the code |
| NER masking is optional and not run | Docstring: "applied on GPU before embedding when enabled" |
| Masking quality is untested | No firm-identification probe; firm-held-out folds never scored |

### 3.4 Earnings surprise and accounting (W4)

| Finding | Evidence | Fix |
|---|---|---|
| SUE is a seasonal random walk | `accounting.py:76-101` | Replace with consensus-based surprise; keep SUE as a baseline |
| SUE exists for 31% of events | `review_facts.json` | Raise coverage with quarterly XBRL and release tables |
| Ten fact tags only | `default.yaml:84-86` | Add revenue surprise, gross margin, operating cash flow, accruals, inventory, deferred revenue, backlog |
| No growth or trend features | Levels only | Year-over-year and sequential changes, standardised |
| No quality or accrual features | None | Sloan accruals, Piotroski-type scores, working-capital changes |

### 3.5 Other

| Finding | Evidence |
|---|---|
| Event vector averages across sections and filings in a same-day 8-K group | `pool_events` |
| LM tone computed on the whole document and the main section separately but not by item | `text.py:104-107` |
| No readability, length, or complexity features (You and Zhang 2009 use these) | None |
| No interaction features between event type and content | Left to the tree |

## 4. World-class target state

| Component | Target |
|---|---|
| Normalisation | Per-date cross-sectional ranks for continuous features; ratios and changes in place of levels; documented per feature |
| Feature governance | A feature store with schema, availability time, owner, version and a drift monitor; automatic pruning of duplicates and zero variance |
| Text units | Item-level and passage-level units; numeric content parsed into typed fields (EPS, revenue, guidance low and high, buyback size) next to the sentence that carries it |
| Change | Passage alignment between this filing and the last same-type filing: added, deleted and changed passages scored separately |
| Text models | A tournament of (a) dictionary, (b) frozen embeddings with attention pooling, (c) fine-tuned encoder on return-relevant targets, (d) LLM structured extraction with scored accuracy |
| Masking | Names, products, subsidiaries, executives and addresses masked; a firm-identification probe must score near chance |
| Surprise | Consensus-based surprise, revisions, guidance versus consensus, and a multi-metric surprise score |

## 5. Additions, ranked

| Rank | Addition | Gain expected | Effort |
|---|---|---|---|
| 1 | Cross-sectional normalisation and level-to-ratio rewrite of accounting and tone features | Lower covariate shift; steadier linear fits | 1 week |
| 2 | Duplicate, collinearity and zero-variance pruning plus a feature-registry test | Hygiene | 2 days |
| 3 | Keep numerics: parse EPS, revenue, guidance, buyback and dividend amounts into typed fields; drop only table noise | Gives text its missing half | 3 to 4 weeks |
| 4 | Passage-level "what changed" against the prior same-type filing (alignment by embedding similarity, then added and deleted passage scores) | The best-supported text anomaly (Cohen, Malloy and Nguyen 2020) | 3 weeks |
| 5 | Item-level segmentation of 8-Ks and one model input per item | Separates earnings from officer change from agreement | 2 weeks |
| 6 | Attention or multiple-instance pooling over chunks, trained against abnormal-return targets | Lets the model find the sentence that matters | 3 weeks |
| 7 | Fine-tune ChronoBERT and a finance encoder on a return-relevant objective with a nested walk-forward grid | Gives the gate a real test | 4 weeks plus GPU |
| 8 | Complete masking and add a firm-identification probe | Controls memorisation | 1 week |
| 9 | Readability, complexity and length features, and risk-factor-section change | Documented signals | 1 week |
| 10 | Peer-text features: similarity-weighted average of peer filings published in the previous 5 days | Spillover signal | 3 weeks |

## 6. Experiments

| ID | Question | Design | Accept or kill |
|---|---|---|---|
| EXP-F1 | How much of the L4 and L6 gain comes from level features that drift? | Rebuild L6 with per-date ranks and no pure levels; compare macro-F1, IC, and adversarial AUC | Keep if IC is no worse and adversarial AUC falls below 0.9 |
| EXP-F2 | Does keeping numerics help text models? | Parse numerics into typed fields; add to L6; measure lift on the earnings subset only | Worth continuing if the earnings-subset direction rank IC reaches 0.02 or better with a lower bound above 0 |
| EXP-F3 | Does passage-level change beat document-level change? | Passage alignment features versus `chg_*` | Keep if the lift over L6 has a lower bound above 0 on walk-forward and development |
| EXP-F4 | Is firm identity recoverable from masked text? | Train a classifier to predict firm from masked chunks; compare accuracy with chance | Pass if accuracy is within 2 points of the majority class; otherwise extend masking |
| EXP-F5 | Is dictionary tone alone any use at long horizons? | Spearman of tone with CAR over days +2 to +60 using alpha-free returns, by event type | Report as a baseline either way |
| EXP-F6 | Does readability or complexity add to L6? | Add `You and Zhang` style measures | Keep if the lower bound of the lift exceeds 0 |
