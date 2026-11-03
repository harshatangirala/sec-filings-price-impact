# Modelling, training and model selection

## 1. What we built

| Element | Logic | Code | Setting |
|---|---|---|---|
| Ladder | L0 prior, L1 price, L2 metadata, L3 accounting, L4 classical text, L5 embeddings alone, L6 all non-neural, L7 L6 plus embeddings, L8 plus extraction (not run) | `models/ladder.py:27-34` | Same events across rungs ("common cohort") |
| Heads | LightGBM, CatBoost, elastic-net logistic regression | `models/heads.py` | Multiclass, 3 classes |
| Loss | Class-balanced weights times label-uniqueness weights, cross-entropy | `heads.py:105-109` | Huber for regression heads |
| Early stopping | Last 15% of each training fold in time order; patience 100; up to 3,000 trees | `heads.py:114-137`, `default.yaml:160` | LightGBM, CatBoost |
| Calibration | Prior correction (undo balanced weights), then temperature scaling fitted on the same inner validation slice | `heads.py:43-48, 65-87, 110-112` | |
| Decision rule | `argmax p / prior` with a Laplace-smoothed training prior | `metrics.py:68-78` | Forces balanced predictions |
| Tuning | Optuna TPE; 15, 6, 8 trials for LightGBM, CatBoost, elastic net; last 3 of 6 folds; 30,000-row time-stratified subsample; metric = mean fold macro-F1 | `ladder.py:280-310`, `default.yaml:151-160` | Embedding stage: 8 and 4 trials |
| Selection | Highest mean walk-forward macro-F1 freezes the primary model | `stages_model.py:872-` (S18) | L7 (LightGBM plus BGE) frozen; tied with L6 |
| Stacking | Caruana greedy ensemble and a logistic stack on out-of-fold predictions | `extensions/stacking.py` | No gain |
| Fine-tuning | Gate G3: lift at least 0.01 with a CI excluding zero | `default.yaml:174`, `models/finetune.py` | Gate failed; never run |
| RAG extraction | BM25 plus dense retrieval, reciprocal rank fusion, cross-encoder rerank, Qwen2.5-7B with a JSON schema and guards | `rag/` | 149 filings run; unscored |

## 2. Strengths

1. The ladder holds the event set fixed, so differences are attributable.
2. Heads and calibration are conventional and defensible; prior correction before the decision rule is correct.
3. The elastic net is certified at its global optimum (`enet_certificate.json`, spread 1.7e-9).
4. Fit diagnostics are recorded (`fit_diagnosis.csv`): overfit gaps are visible (LightGBM and CatBoost train macro-F1 0.51 to 0.65 against 0.36 to 0.39 validation).
5. Gates and a pre-registered hypothesis list limit post-hoc selection.

## 3. Weaknesses, with evidence

### 3.1 Objective and decision rule (W7)

| Finding | Evidence | Consequence |
|---|---|---|
| Macro-F1 is the target; the decision rule forces balanced classes | `metrics.py:68-78` | Rewards predicting classes in equal proportions, not predicting returns |
| Direction rank IC is approximately zero | Locked-test rank IC of `p_up - p_down` with `ar_mm`: L6 -0.0077, L7 -0.0021; monthly IC t-values -1.3 to +0.7 across models and periods (`rank_ic.csv`) | The model has no ranking skill for direction |
| Magnitude rank IC is large | `1 - p_side` versus abs(AR): 0.205 (development), 0.279 (locked) for L6 | Real, but a single trailing-volatility feature scores 0.16 to 0.26 and L1 (price only, which contains those features) scores about 0 because the classifier does not use them for magnitude (file 12) |
| Up-class recall is 21.5% on the locked test | `locked_confusion.csv`: Up predicted Down 1,121 times and Up 543 times | The Up class is below its 30% base rate; the model is not finding Up |
| Brier, log-loss and ECE improvements are slight | Locked log-loss 1.071 versus 1.096 prior | Probabilities carry little information beyond the prior |

### 3.2 One pooled model (W6)

A single 3-class head sees earnings releases, officer changes, Reg FD, agreements, periodic filings and 10-Ks. Their volatility differs (earnings mean absolute AR 2.37% versus 1.36% for the rest), their information content differs, and the trees must learn event-type interactions from 128 features. There is no hierarchical structure, no per-type calibration, and no shared-encoder multi-head design.

### 3.3 Tuning (W8)

| Finding | Evidence |
|---|---|
| Budgets of 6 to 15 trials, below Optuna's 10 random start trials for CatBoost and elastic net | `default.yaml:155`; `ladder.py:287` (`n_startup_trials=3` for pruner only) |
| The median tuning gain is exactly zero; 44% of 32 configurations improve; the mean gain is +0.003 | `tuning.csv` |
| Tuning folds are reported folds | `ladder.py:294-297`: last 3 of 6 folds tune, and all 6 are reported |
| 30,000-row subsample for search; final fits use all rows | `default.yaml:157` |
| Search space is narrow and fixed (e.g. `num_leaves` 8 to 128) | `heads.py:21-40` |
| Default elastic net is poor (0.322 against 0.389 tuned on L6) | `tuning.csv` |
| Elastic net `saga` with `max_iter=400, tol=1e-3` | `heads.py:172`; the convergence flag is recorded but not enforced |
| Temperature scaling and early stopping share one validation slice | `heads.py:110-112, 128`; small optimism |

### 3.4 Determinism and selection (W8, W15)

| Finding | Evidence |
|---|---|
| Same model and data on two machines: LightGBM 80.2% identical classes, macro-F1 0.3898 versus 0.3943; elastic net 98.1% | `reproducibility.csv` |
| L6 versus L7 compared across two runs; reversed within one run | `meta.json`: L6 0.3909 vs L7 0.3898 in S15; frozen L7 by +0.0010 |
| Rescoring noise (0.0045) exceeds the selection margin (0.0010) | `report` section 5a |
| No tie-break toward the strict-leaderboard model | D-078 |
| 237 configurations tried | `meta.json` |

### 3.5 Text models and RAG

| Finding | Evidence |
|---|---|
| L7 minus L6 compares LightGBM plus embeddings with an elastic net without them | Paper Section VI.B |
| L6 contains word counts and TF-IDF, so the contrast is embeddings over classical text | `ladder.py:27-34`; D-079 |
| Embedding PCA to 64 dimensions per fold | `ladder.py:135-143` |
| Two of four encoders dropped (fast mode) | D-074 |
| RAG ran on 149 filings with no human labels | `rag.json`: G3b failed, student not trained |
| Extraction guards check quotes and numbers but not correctness | `rag/extract.py` |
| Stacking tried; no gain | `dev_lifts.csv` H5 |

### 3.6 Missing model families

No ranking objective, no quantile or heavy-tailed loss, no multi-task magnitude-and-direction head, no monotone constraints, no sequence model over a firm's filing history, no cross-sectional (date-grouped) training, no conformal abstention, no neural tabular baseline, no seed bagging.

## 4. World-class target state

| Component | Target |
|---|---|
| Objective | Primary: rank IC or a decile-spread utility on a continuous target. Secondary: calibrated magnitude. Classification only for reporting |
| Architecture | Shared feature trunk with event-type heads; separate magnitude and direction heads; hierarchical shrinkage for rare types |
| Training | Date-grouped cross-sectional objectives (LambdaRank or pairwise), per-date feature normalisation, monotone constraints on well-understood features |
| Tuning | Nested walk-forward: an inner time-ordered loop for hyperparameters inside each outer fold; at least 60 trials with a broad space; report results only from outer folds |
| Determinism | Fixed thread count, deterministic mode, 10 to 20 seed bags averaged; report the seed-to-seed standard deviation |
| Selection | Rules written down, including a tie-break and a margin tied to measured noise; one run produces all contenders |
| Uncertainty | Conformal or quantile prediction; trade only when the interval excludes no-trade |
| Text | Tournament with fixed budget per arm; extraction scored on human labels before use |
| Governance | A model card per frozen model: data cut, features, hyperparameters, seeds, metrics with intervals, known failure modes |

## 5. Additions, ranked

| Rank | Addition | Effort | Expected effect |
|---|---|---|---|
| 1 | **Report rank IC, ICIR, decile spread and hit rate** for every model | 3 days | Makes results readable to PMs; exposes the direction null directly |
| 2 | **Seed bagging and deterministic mode** | 3 days | Removes the 20% class flip noise |
| 3 | **Nested walk-forward tuning with 60 or more trials** | 1 to 2 weeks | Removes tuning optimism; tests whether tuning can ever help |
| 4 | **Rank and regression objectives**, with cross-sectional normalisation | 1 to 2 weeks | Aligns training with ranking |
| 5 | **Event-type heads** on a shared trunk, or separate models for earnings, officer change and Reg FD | 2 weeks | Reduces dilution |
| 6 | **Separate magnitude and direction heads**; use magnitude for sizing or options | 1 week | Matches where skill is |
| 7 | **A magnitude baseline** (event type times trailing volatility) and incremental IC of L6 over it | 2 days | Tests whether the magnitude signal is more than metadata |
| 8 | **Conformal abstention** with coverage-versus-IC curves | 1 week | Selective prediction; already partly in `selective prediction` tables |
| 9 | **Monotone constraints** on price momentum, size and tone | 2 days | Stability |
| 10 | **Neural tabular baseline** (FT-Transformer or TabM) and a gradient-boosted ranker | 1 week | Diversity |
| 11 | **Firm-history sequence model** over the last 8 filings | 3 to 4 weeks | Captures trends in language |
| 12 | **Model cards and a frozen model registry** | 1 week | Governance |

## 6. Experiments

| ID | Question | Design | Accept or kill |
|---|---|---|---|
| EXP-M1 | Is there any ranking skill in direction? | Compute rank IC and monthly ICIR on pooled walk-forward, development and locked predictions for all stored models | Already measured: +0.018 on 2008-2019 walk-forward and about zero since 2020 (file 12). Use as the baseline for every later model; require direction rank IC at or above 0.02 with a lower bound above 0 |
| EXP-M2 | Is magnitude skill more than metadata? | Compare L6 with a baseline using event type, session bucket and trailing volatility only | Treat as alpha only if incremental rank IC is above 0.05 and its lower bound is above 0 |
| EXP-M3 | Does seed bagging remove the noise? | 20 seeds on L6 and L7 under deterministic settings; compare class stability and macro-F1 spread | Adopt if class flips fall below 5% |
| EXP-M4 | How much of validation is tuning optimism? | Nested versus current tuning | If nested scores are lower by more than 0.005, restate validation |
| EXP-M5 | Does an event-type structure help? | Per-type models versus pooled; compare per-type rank IC | Keep if at least two types gain with lower bounds above 0 |
| EXP-M6 | Do rank objectives beat classification? | LambdaRank or pairwise loss on `ar_mm` percentiles | See EXP-T2 |
| EXP-M7 | Does conformal abstention raise IC? | Trade only when the conformal set is a singleton | Continue if IC at 25% coverage exceeds IC at 100% by 0.01 |
| EXP-M8 | Does a firm-history model beat flat features? | Sequence encoder over prior filings versus flat change features | Continue if lift lower bound exceeds 0 |

## 7. Text and RAG sub-programme (summary; detail in 10)

1. Label 300 to 500 filings by hand (the templates are ready) so extraction can be scored (`gold_filings_template.xlsx`).
2. Gate extraction on precision and recall by field; fields that pass become typed features with numeric magnitudes.
3. Run SEC-BERT and a finance encoder with the same pooling, then attention pooling, then fine-tuning in a nested walk-forward.
4. Use a chronologically consistent LLM (ChronoGPT or a dated checkpoint) for any generative step, and test contamination with the Glasserman-Lin and Sarkar-Vafa protocols.
5. Compare against a baseline of dictionary tone plus numeric surprise on the earnings subset, which is where the text should matter.
