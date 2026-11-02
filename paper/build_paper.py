"""Builds paper/paper.html from paper/tables, dashboard/data and paper/references.json. Numbers in tables and most in-text numbers are read from files."""
import json
import re
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
P, T, D, FIG = ROOT / "paper", ROOT / "paper" / "tables", ROOT / "dashboard" / "data", ROOT / "paper" / "figures"
GH = "https://github.com/harshatangirala/sec-filings-price-impact"
TITLE = "Do the Words in SEC Filings Predict the Next Trading Session? A Point-in-Time Test"
csv = lambda n, base=T: pd.read_csv(base / f"{n}.csv")
facts = json.load(open(T / "facts.json"))

PERS = ["Walk-forward 2008-2019", "Development 2020-2022", "Locked test 2023-2026"]
perf, lifts, sh, shs, ss, ti = csv("performance"), csv("lifts"), csv("same_head_lifts"), csv("same_head_by_session"), csv("text_lift_by_subset"), csv("tinfo_vs_ttrade")
dm, cal, rb, fma, q0, qa, deci, sret = csv("drift_by_measure"), csv("calendar_time"), csv("drift_noalpha_robustness"), csv("fama_macbeth_alpha"), csv("drift_quintiles"), csv("drift_quintiles_alpha"), csv("move_deciles"), csv("sample_returns")
dro, eo, corr, frm = csv("drift_robustness"), csv("earnings_only"), csv("text_reaction_corr"), csv("form_drift_alpha")
lad, tune, bt, ga, h4 = csv("ladder", D), csv("tuning", D), csv("backtest", D), csv("track_a_groups", D), csv("h4", D)
dev_r, lock_r, rep, perm, ht, lk_aud = csv("dev_results", D), csv("locked_results", D), csv("reproducibility", D), csv("permutation", D), csv("hypothesis_tests", D), csv("leakage_audit", D)
g3 = json.load(open(D / "g3.json")); mcs = json.load(open(D / "mcs_pbo.json")); cert = json.load(open(D / "enet_certificate.json")); meta = json.load(open(D / "meta.json")); E = meta["eda"]; rag = json.load(open(D / "rag.json"))


def L(per, cmp_, met="macro_f1"):
    return lifts[(lifts.period.str.startswith(per)) & (lifts.comparison.str.startswith(cmp_)) & (lifts.metric == met)].iloc[0]


def ci(r, d=3, lo="lo", hi="hi", val="lift", sign=True):
    f = f"{{:+.{d}f}}" if sign else f"{{:.{d}f}}"
    return f"{f.format(r[val])} [{f.format(r[lo])}, {f.format(r[hi])}]".replace("-", "&minus;")


def num(x, d=3, sign=False):
    return (f"{x:+.{d}f}" if sign else f"{x:.{d}f}").replace("-", "&minus;")


def pc(x, d=2, sign=True):
    return (f"{x * 100:+.{d}f}%" if sign else f"{x * 100:.{d}f}%").replace("-", "&minus;")


def PF(per, model):
    return perf[(perf.period.str.startswith(per)) & (perf.model == model)].iloc[0]


def DM(col):
    return dm[dm["col"] == col].iloc[0]


TXW, TXD, TXL = L("Walk", "Text"), L("Dev", "Text"), L("Locked", "Text")
L6L, L7L, L1L = PF("Locked", "L6"), PF("Locked", "L7"), PF("Locked", "L1")
L6W, L6D = PF("Walk", "L6"), PF("Dev", "L6")
C4A, C4N, MMA, MMN, C4P, MKT, RAW = DM("c4_alpha_replica"), DM("c4_noalpha"), DM("mm_alpha_replica"), DM("mm_noalpha"), DM("c4_post"), DM("mkt_adj"), DM("raw")
CTA = cal[cal["sample"] == "All reports"].iloc[0]
Q = qa.set_index("q")
aq = Q.loc[5, "alpha_term_c4"] - Q.loc[1, "alpha_term_c4"]
d6, d1 = deci[(deci.model == "L6") & (deci.decile == 10)].iloc[0], deci[(deci.model == "L6") & (deci.decile == 1)].iloc[0]
bt_ = lambda block, leg, lag=0, cost=10: bt[(bt.block == block) & (bt.leg == leg) & (bt.lag == lag) & (bt.cost_bps == cost) & (bt.sp_bucket == "Combined")].iloc[0]
FM = lambda dep, spec, var: fma[(fma.dep == dep) & (fma.spec == spec) & (fma["var"] == var)].iloc[0]

NT = csv("no_text_lifts"); NL_ = csv("no_text_levels")
N = lambda per, c, met="macro_f1": NT[(NT.period.str.startswith(per)) & (NT.comparison.str.startswith(c)) & (NT.metric == met)].iloc[0]
A_LG, A_EN = "Classical text + embeddings (L7 - numbers only, both", "All classical text"
# ---------------------------------------------------------------------------------------------- numbering
TAB = ["sources", "sample", "features", "ladder", "perf", "lifts", "samehead", "notext", "session", "tinfo", "evstudy", "quint", "measure", "fm", "caltime", "backtest", "diag", "hyp"]
ROM = "I II III IV V VI VII VIII IX X XI XII XIII XIV XV XVI XVII XVIII".split()
tn = {k: f"Table {ROM[i]}" for i, k in enumerate(TAB)}
FIGS = ["sample", "target", "clock", "timeline", "ladder", "performance", "auc", "move_deciles", "confusion", "reliability", "by_year", "lifts", "no_text", "subsets", "tinfo", "car_paths", "sue_quintiles", "alpha_term", "drift_robustness", "survivorship_bound", "backtest"]
fn = {k: f"Figure {i + 1}" for i, k in enumerate(FIGS)}
H = []; a = H.append


def p(*xs):
    for x in xs: a(f"<p>{x}</p>")


def h1(t): a(f"<h2>{t}</h2>")
def h2(t): a(f"<h3>{t}</h3>")
def eq(body, n): a(f"<div class='eq'><span class='eqb'>{body}</span><span class='eqn'>({n})</span></div>")


def fig(name, cap, width="100%"):
    a(f"<figure><img style='width:{width}' src='figures/{name}.svg'/><figcaption><b>{fn[name]}.</b> {cap}</figcaption></figure>")


def table(key, title, note, head, rows, cls="", align=None):
    al = align or ["l"] + ["c"] * (len(head) - 1)
    th = "".join(f"<th class='{x}'>{h}</th>" for h, x in zip(head, al))
    body = ""
    for r in rows:
        if isinstance(r, str):
            body += f"<tr class='sec'><td colspan='{len(head)}'>{r}</td></tr>"; continue
        body += "<tr>" + "".join(f"<td class='{x}'>{v}</td>" for v, x in zip(r, al)) + "</tr>"
    a(f"<div class='tab {cls}'><div class='tt'>{tn[key]}</div><div class='ts'>{title}</div><div class='tn'>{note}</div><table><thead><tr>{th}</tr></thead><tbody>{body}</tbody></table></div>")


# ---------------------------------------------------------------------------------------------- title page
ABSTRACT = (f"I test whether the text of 10-K, 10-Q, and 8-K filings predicts the abnormal return available after an investor reads them. For {E['events']:,} events at 150 S&amp;P 500 firms, 2000-2026, with a single-use test period, language-model embeddings add nothing to price, metadata, accounting, and dictionary features (macro-F1 change {num(TXL.lift, 3)}; 95% interval {num(TXL.lo, 3)} to {num(TXL.hi, 3)}), and a model without text does about as well. Skill concerns the size of the move, not its direction. A pre-specified test also shows a large post-earnings reversal, which I trace to subtracting pre-event alpha; in calendar-time tests the spread is indistinguishable from zero.")
a(f"""<div class='titlepage'><h1>{TITLE}</h1>
<div class='author'>Krishna Harsha Tangirala</div><div class='date'>Working paper draft, November 2026</div>
<div class='abstract'><b>Abstract</b><br/>{ABSTRACT}</div>
<div class='kw'><b>JEL classification:</b> G12, G14, C52, C53, M41<br/><b>Keywords:</b> SEC filings, textual analysis, language models, look-ahead bias, out-of-sample prediction, post-earnings announcement drift, event-study methodology</div>
<div class='thanks'>Code, configuration, decision log, and result tables are available at <a href='{GH}'>{GH}</a>. Raw prices and the Loughran-McDonald dictionary are not redistributed; Section IX explains how to rebuild them.
<b>Disclosure of AI use:</b> an AI assistant (Claude, Anthropic) was used to write code, carry out analyses, and draft this text. The author reviewed the work and takes responsibility for its content. All errors are the author's.</div></div>""")

# ---------------------------------------------------------------------------------------------- I
h1("I. Introduction")
p(f"Financial economists have spent two decades showing that the words firms and journalists write are related to stock returns. {{tetlock07}} finds that negative media language predicts market prices, {{lm11}} show that finance needs its own word lists, and {{cohen20}} find that firms that change the language of their annual reports earn lower returns in the following months. Language models have raised expectations further. {{huang23}} show that a BERT model trained on finance text labels sentiment better than earlier tools, and {{lopezlira}} report that ChatGPT scores of news headlines predict next-day returns.",
  f"Two practical questions remain open for regulatory filings. First, can an investor who reads a filing as soon as it is public still capture a return, given that the price may already have moved by the first available trade? Second, does the text add anything once the forecaster already knows the event type, the time of day, recent prices, and the accounting numbers? Answering them needs care with timing, because several recent papers show that pretrained models can carry knowledge of the future into a backtest ({{glasserman23}}; {{sarkar24}}; {{he26}}).",
  f"This paper runs that test. I assemble {E['filings']:,} filings from EDGAR for 150 firms drawn from the S&amp;P 500, merge them into {E['events']:,} events from January 2000 to July 2026, and give each event four timestamps: SEC acceptance, assumed public availability, the decision time, and the next market open. The target is the abnormal open-to-close return in the first session an investor could trade after the decision time. Feature sources are compared alone and together, the comparisons use identical events, and a 2023-2026 test period was scored once, after the final model and its settings had been hashed.",
  "I report three results.")
p(f"First, filing text does not reliably improve the forecast. The pre-specified test adds frozen embeddings, from a general encoder (BGE) or a chronologically consistent one (ChronoBERT), to a model that already holds price, event metadata, accounting data, dictionary counts, language-change measures, and TF-IDF factors. Macro-F1 changes by {ci(TXW)} on the 2008-2019 validation folds, {ci(TXD)} on the 2020-2022 development period, and {ci(TXL)} on the locked test. The gate I set in advance for fine-tuning the encoders, a lift of 0.01 with an interval excluding zero, failed, so no encoder was fine-tuned. Because that baseline contains classical text, I also fit a numbers-only model after the lock. With LightGBM on both sides, all text together changes macro-F1 by {ci(N('Walk', A_LG))}, {ci(N('Dev', A_LG))}, and {ci(N('Locked', A_LG))}. With an elastic net, classical text shows a small gain of about 0.01 in the last two periods, which does not survive the change of classifier. An apparent gain on pre-market filings also comes from the choice of classifier, since it appears without any text.",
  f"Second, what skill exists concerns the size of the move. The best model without embeddings scores a locked-test macro-F1 of {num(L6L.macro_f1)}, against about {num(L6L.chance_shuffled, 2)} for a shuffled version of its own predictions. Its AUC for telling movers from non-movers is {num(L6L.move_auc)} [{num(L6L.move_auc_lo)}, {num(L6L.move_auc_hi)}], and among stocks that moved its AUC for direction is {num(L6L.direction_auc)} [{num(L6L.direction_auc_lo)}, {num(L6L.direction_auc_hi)}]. The top decile of predicted movers has a mean absolute abnormal return of {pc(d6.mean_abs_ar, 2, False)}, about twice the {pc(d1.mean_abs_ar, 2, False)} in the bottom decile. A long-short backtest built on the frozen model loses money on the locked test at realistic costs.",
  f"Third, I document a methodological trap, and I fell into it myself. The study's pre-specified test of post-earnings announcement drift rejected the null with the wrong sign: high-surprise reports underperformed low-surprise reports by {pc(abs(C4A.estimate), 2, False)} over days +2 to +60, with a t-statistic of {num(C4A.t, 1)}. The result held in every subperiod, among index members and non-members, and with a one-session delay. It is an artifact. The abnormal returns subtract a pre-event alpha, estimated over days &minus;250 to &minus;30, for each of the 59 days in the window. Good earnings tend to follow good years, so that subtraction is {pc(aq, 2, False)} larger for the top quintile than for the bottom one, which accounts for the entire spread. Without the alpha term the spread is {pc(C4N.estimate)} [{pc(C4N.ci_low)}, {pc(C4N.ci_high)}], and a calendar-time portfolio, which estimates no pre-event parameters, gives a Carhart alpha of {pc(CTA.alpha_59d)} per 59 days (t = {num(CTA.alpha_t, 2)}).",
  "The paper contributes in three ways. It adds a tradable-target test to the literature on filing text, which mostly studies returns measured from before or at the filing date. It adds evidence on the size of look-ahead bias in embeddings for a new text source, comparing yearly ChronoBERT vintages with a static encoder on identical events and finding no detectable difference, consistent with what {he26} find for news. And it shows, with a worked example, how an ordinary event-study specification can produce a highly significant long-horizon result that passes standard robustness checks. That example supports the warnings of {kothari97} and {fama98} about long-horizon inference, and it suggests that robustness checks sharing the same bias offer little protection against it.",
  "The sample limits what can be claimed. It contains only firms that survived to 2026, it covers 150 large firms, it uses free price data and a time-series measure of earnings surprise, and two of the four planned encoders were dropped to save compute. Section VIII sets these limits out. The structure of the paper follows the order of the argument: Section II reviews related work and states the hypotheses, Section III describes the data, Section IV the design, Section V the forecast results, Section VI the tests of text, Section VII the drift analysis, Section VIII the economic test and limits, and Section IX concludes.")

# ---------------------------------------------------------------------------------------------- II
h1("II. Related Literature and Hypotheses")
h2("A. Text and returns")
p("{lm16} and {bochkay23} survey the use of text in accounting and finance. Early work counted words: {tetlock08} show that negative words in firm-specific news predict earnings and returns, and {lm11} show that general-purpose dictionaries misclassify finance terms. Other papers study the information in filings directly. {li10} finds that the tone of forward-looking statements in MD&amp;A predicts future earnings, {you_zhang09} find larger underreaction to more complex 10-Ks, {lerman10} find that 8-K items carry significant information after the 2004 rule change, {dyer17} document how 10-K text has grown and changed, and {cohen20} show that year-to-year changes in filings predict returns. {frankel21} compare machine-learning and dictionary sentiment measures of disclosure. Two recent papers illustrate the current direction: {magner25} relate the tone of 10-K risk factors to weekly index returns, and {fengler25} use word embeddings to find MD&amp;A topics and relate topic sentiment to returns at several horizons.",
  "A newer group of papers applies neural text models. {ke_kelly_xiu} build a supervised sentiment measure from news and find strong return predictability. {lopezlira} use ChatGPT on headlines. {kim_llm} ask a language model to analyze anonymized financial statements. {gu20} and {jiang23} show that machine learning can extract return signals from characteristics and price charts. My design differs from these papers in two respects. The target excludes any move that occurs before an investor could act, and the benchmark is a rich set of features measured on the same events, so the question is about the increment from text.")
h2("B. Look-ahead bias in pretrained models")
p("A pretrained model may have read text from after the event it is asked to evaluate. {glasserman23} and {sarkar24} develop tests for this bias. {he26} respond by training ChronoBERT and ChronoGPT on text available at each date and report that look-ahead bias is modest in news-based return forecasts. I use their yearly vintages, so that an event in year Y is encoded by the model trained through year Y&minus;1, and I compare them with a static encoder on identical events.")
h2("C. Post-earnings announcement drift")
p("{ball_brown} and {bernard_thomas89} {bernard_thomas90} established that prices keep moving in the direction of an earnings surprise. Later work asks when and why: {livnat06} find stronger drift with analyst-based surprises than with time-series surprises, and {hirshleifer09} and {dellavigna09} link it to limited attention. {karolyi25} attribute part of it to how investors process large numbers, {fink24} reproduce it in laboratory markets, {zhang24} study its asymmetry and its relation to order flow, and {lan24} propose an overnight-return measure of surprise in Chinese stocks. Several papers find that anomalies weaken after publication or as liquidity improves ({mclean16}; {chordia14}), and {martineau22} finds that this one has gone in large US stocks. {hou20} show that many anomalies fail to replicate, and {harvey16} argue for higher hurdles. On measurement, {kothari97} and {fama98} warn that long-horizon abnormal returns are sensitive to the benchmark model, which turns out to be decisive here.")
h2("D. Hypotheses")
p(f"I wrote down the following hypotheses, with the primary metric and pass-or-fail rules, in a configuration file that was hashed before any development-period scoring. {tn['hyp']} gives the outcome of each. <i>H1</i>: filing text adds out-of-sample value beyond price, metadata, and accounting data. <i>H2</i>: changes in filing language add more than levels. <i>H3</i>: validated structured extraction adds value beyond embeddings. <i>H4</i>: point-in-time encoders differ from static ones. <i>H5</i>: a stacked ensemble beats the best single model. <i>H6</i>: richer retrieval designs beat a hybrid baseline. <i>H8</i>: gains shrink under strict timing, costs, and the locked test. (H7, on analogue-retrieval and time-series extensions, was not run.) Separately, I specified a test of earnings drift within an event study.")

# ---------------------------------------------------------------------------------------------- III
h1("III. Data")
h2("A. Sources")
table("sources", "Data sources", "All inputs are free and public. Versions are pinned where the source allows it.",
      ["Source", "Use", "Coverage in this study", "Access"], [
      ["SEC EDGAR", "10-K, 10-K405, 10-Q, 8-K text and exhibits; SGML acceptance times", f"{E['filings']:,} filings", "Public"],
      ["EDGAR XBRL", "Balance-sheet and income data as first reported", f"{E['xbrl_facts']:,} facts", "Public"],
      ["Press-release exhibits (EX-99)", "EPS figures parsed from earnings releases", f"{E['results_releases']:,} results releases", "Parsed from EDGAR"],
      ["Daily prices", "Returns, volatility, benchmark", f"{E['price_bars']:,} daily bars", "Local OHLCV files, Yahoo Finance"],
      ["Kenneth French data library", "Market, size, value, momentum factors", "Daily", "Public"],
      ["fja05680/sp500 (GitHub)", "S&amp;P 500 membership history", "Pinned commit", "Public"],
      ["Loughran-McDonald dictionary", "Word-tone counts", "1993-2025 edition", "Free for academic use"],
      ["JanosAudran/financial-reports-sec", "Firm overlap and parser checks only, never as text", "10-Ks through 2020", "Hugging Face, pinned"]], align=["l", "l", "l", "l"])
h2("B. Sample construction")
p(f"Of {E['candidate_firms']} candidate firms with price data, {E['eligible_firms']} also appear in the Hugging Face 10-K corpus. From the members of the S&amp;P 500 among them, I draw 150 with a fixed seed, stratified across the 12 Fama-French industries. An earlier plan used 500 firms (353 S&amp;P members and 147 others). I cut it to these 150 to finish a complete run, and all 150 are in the larger universe. {facts['member_at_event_share'] * 100:.1f}% of events occur while the firm is in the index.",
  f"I download all 10-K, 10-K405, 10-Q, and 8-K filings for these firms. Same-day 8-Ks from one firm merge into a single event, giving {E['events']:,} events: {E['events_by_type']['8K']:,} 8-Ks, {E['events_by_type']['periodic_10Q']:,} 10-Qs, and {E['events_by_type']['periodic_10K']:,} 10-Ks. Of the filings, {E['filings_parsed_ok']:,} yield at least one target section. I keep Items 1, 1A, 7, and 7A of the 10-K; the MD&amp;A, market-risk, and risk-factor sections of the 10-Q; and the item text and press-release exhibit of the 8-K, for {E['sections']:,} sections in total. I replace firm names and tickers with placeholders ({E['firm_name_masks']:,} name and {E['ticker_masks']:,} ticker replacements), so that a model cannot learn firm identity from text. Sections are split into 350-word chunks overlapping by 50 words, about {E['chunks_est']:,} chunks.",
  f"{fn['sample']} shows the sample through time and when filings reach EDGAR. More than half arrive after the close, which is why the target is measured from the next open. {tn['sample']} reports the target by form and session. The mean abnormal return is close to zero in every group, and its standard deviation is about 2.6%.")
fig("sample", "<i>Sample.</i> Panel A shows events per year by form. Same-day 8-Ks from one firm count as one event, and 2026 runs through July. Panel B shows the share of events by session of EDGAR acceptance. Events on non-trading days, under 1% of the sample, are omitted.")


def sr(x):
    return [x["sample"], f"{int(x.n):,}", num(x.mean_ar * 100, 2), num(x.sd_ar * 100, 2), num(x.p05 * 100, 2), num(x.p95 * 100, 2), num(x.mean_abs * 100, 2), f"{x.down * 100:.1f}", f"{x.side * 100:.1f}", f"{x.up * 100:.1f}"]


table("sample", "The target by form and session", "Abnormal open-to-close return in percent, in the first session whose open is at or after the decision time. Class shares use &plusmn;0.5 times the stock's own daily volatility.",
      ["", "N", "Mean", "SD", "P5", "P95", "Mean |AR|", "Down %", "Side %", "Up %"], ["By form"] + [sr(sret.iloc[i]) for i in range(0, 4)] + ["By session of acceptance"] + [sr(sret.iloc[i]) for i in range(4, 7)])
p(f"<i>Survivorship.</i> Every firm in the sample was listed in 2026. Firms that were delisted, acquired, or failed are missing, and the eligible data cover about 353 of the 1,094 tickers that were index members during the period. Prices come from free sources, which lack delisting returns. This limit affects the long-horizon results most, and Section VII quantifies its potential effect on the drift result.")
fig("target", "<i>The target.</i> Panel A shows the distribution of the abnormal open-to-close return, truncated at &plusmn;10%. Panel B shows the three class shares by calendar year.")

# ---------------------------------------------------------------------------------------------- IV
h1("IV. Empirical Design")
h2("A. Timing")
p(f"Each event carries four timestamps ({fn['clock']}). Acceptance comes from the SGML header for {E['ts_quality']['A'] / sum(E['ts_quality'].values()) * 100:.0f}% of events; the rest are date-only or uncertain and are flagged. The JSON feed from EDGAR turned out to mix two time conventions, with a {0.2975 * 100:.0f}% mismatch against the headers on the sample checked, so I use the headers. I assume a filing is public five minutes after acceptance and that an investor can decide thirty minutes after that. Those two lags are assumptions, so the results are a point-in-time evaluation under an assumed availability and processing-latency policy. I do not report sensitivity to them. Same-day 8-K groups are clocked at their last acceptance. Every feature records the time of its latest input, and an assertion checks each row against the decision time; it found {E['availability_violations']} violations in the final run.")
fig("clock", "<i>Event clock.</i> Features may use information up to the decision time. $T_{trade}$, the primary target, starts at the first open at or after the decision time. $T_{info}$ starts at the last close before acceptance, so it contains the first reaction.".replace("$T_{trade}$", "<i>T</i><sub>trade</sub>").replace("$T_{info}$", "<i>T</i><sub>info</sub>"))
h2("B. Target")
p("For event <i>i</i> of stock <i>s</i>, the primary target is the abnormal return over the first tradable session,")
eq("<i>AR</i><sub><i>i</i></sub> = <i>R</i><sub><i>s</i>,open&rarr;close</sub> &minus; <i>&beta;&#770;</i><sub><i>s</i></sub> <i>R</i><sub>SPY,open&rarr;close</sub>,", 1)
p("where the stock and the benchmark are measured over the identical interval, and <i>&beta;&#770;</i> comes from a market model estimated on sessions &minus;250 to &minus;30 with at least 120 observations. No alpha is subtracted from this one-session return. The class label is Up if <i>AR</i> exceeds <i>k&sigma;&#770;</i><sub><i>s</i></sub>, Down if it is below &minus;<i>k&sigma;&#770;</i><sub><i>s</i></sub>, and Sideways otherwise, with <i>k</i> = 0.5 and <i>&sigma;&#770;</i> the stock's estimation-window volatility. This scaling makes utilities and technology stocks comparable. A complementary target, <i>T</i><sub>info</sub>, runs close to close from before acceptance and includes the first reaction; it excludes events accepted during the session.")
h2("C. Features and the ablation ladder")
fam = E["feature_families"]
table("features", "Feature families", f"{sum(fam.values())} structured features. Each carries an availability rule that is asserted row by row.",
      ["Family", "Features", "Examples", "Availability rule"], [
      ["Metadata", fam["metadata"], "Form, 8-K item categories, sections present, industry, calendar, session", "Own filing metadata at decision time"],
      ["Price and market", fam["price"], "Returns over 1 to 250 days, volatility, volume, beta, market state", "Closes at or before decision time"],
      ["Word counts", fam["lexical"], "Loughran-McDonald tone categories, length, boilerplate", "Own filing text"],
      ["Language change", fam["change"], "Cosine and Jaccard change from the firm's prior filing", "Prior documents public before decision time"],
      ["Accounting", fam["accounting"], "Assets, leverage, ROA, revenue, book-to-market, earnings surprise", "Filed before decision time; never a later filing of the same cluster"],
      ["Reaction and novelty", fam["ear"] + fam["novelty"], "Last earnings reaction, share of new text", "Reaction window closed before decision time"]], align=["l", "c", "l", "l"])
p(f"Rungs are compared on identical events. L0 is the class prior. L1 (price), L2 (metadata), L3 (accounting), L4 (classical text: word counts, language change, novelty, and up to 64 TF-IDF factors fitted in each training fold), and L5 (embeddings) each use one source on its own. L6 combines price, metadata, accounting, and classical text, and L7 adds the embedding vectors to L6. L6 without the change features is the control for H2. Note what this design tests: L7 minus L6 measures the increment from language-model embeddings over a baseline that already contains classical text. It does not measure the value of text as such, and the pipeline had no rung free of text. Section VI.C adds one. Metadata includes 8-K item categories, which are themselves a coded summary of what the filing reports.")
p("<i>Earnings surprise.</i> I have no analyst forecasts. The surprise of report <i>q</i> is a seasonal random walk scaled by its own history,")
eq("<i>SUE</i><sub><i>q</i></sub> = (<i>EPS</i><sub><i>q</i></sub> &minus; <i>EPS</i><sub><i>q</i>&minus;4</sub>) / <i>sd</i>(past seasonal differences),", 2)
p(f"using first-reported values only. Current EPS comes from the press-release exhibit, parsed with a set of rules whose agreement with XBRL I checked on a replay sample (about 84%), and history comes from filings public before the decision time. {facts['verified_releases']:,} events are verified earnings releases, and {facts['sue_nonmissing']:,} have a surprise score.")
h2("D. Encoders and classifiers")
p(f"I compare two frozen encoders. ChronoBERT ({{he26}}) comes as 26 yearly vintages, each trained on text through its date, and has about 150 million parameters. BGE-base ({{xiao24}}) is a general encoder of about 110 million parameters with no stated training cutoff, so results using it carry a retrospective label. Each chunk becomes 768 numbers. I take the mean and maximum over a section's chunks and combine sections into a 1,536-dimensional event vector, then reduce it with PCA fitted inside each training fold. I planned two more encoders, SEC-BERT and FinBERT-tone, and dropped them to save GPU time. I have no evidence on how they would have performed.",
  f"The classifiers are LightGBM ({{ke17}}), CatBoost ({{catboost}}), and an elastic-net logistic regression ({{zou05}}). The elastic net's objective is convex, and repeated random starts reach the same value to within {cert['spread']:.1e}. Hyperparameters come from Optuna ({{optuna}}) on the three latest folds, with budgets of 15, 6, and 8 trials in the structured stage and 8 and 4 in the embedding stage. These budgets are small: across 32 rung-and-classifier combinations, the median change in validation score from tuning is zero, and only 44% improve. No model should be regarded as fully tuned. Class weights balance the classes and down-weight overlapping events. Predicted probabilities are corrected to true frequencies, and the predicted class maximizes the probability relative to the class prior.")
h2("E. Splits and the locked test")
p(f"{fn['timeline']} shows the splits. Training expands from 2000 with at least eight years of data, followed by six two-year validation blocks from 2008 to 2019. The 2020-2022 development period chooses the final model, and the period from January 2023 through July 2026 is the locked test. Events whose outcome window reaches the next block are purged and a 90-day embargo separates blocks, following {{lopezdeprado}}. Fold-level preprocessing is fitted on training rows only.")
fig("timeline", "<i>Sample splits.</i> Numbers are events in each evaluation block. Training windows expand, and a 90-day embargo separates training from evaluation. The locked test was scored once.")
p(f"The rule for the final model was the highest mean walk-forward macro-F1. It chose L7 with LightGBM and BGE. The architecture was hashed and the locked test was scored once, on {pd.Timestamp('2026-10-25').strftime('%d %B %Y').lstrip('0')}, with a software guard against reuse. That guard held, but the procedure fell short in two respects described in Appendix A: the choice compared scores from two different runs, and the lock was opened minutes after the freeze, without the review of development results that I had planned.")
h2("F. Metrics and inference")
p(f"The primary metric is macro-F1, the unweighted mean of the F1 scores of the three classes. Intervals come from a month-block bootstrap with 2,000 replications, since events in the same month share market shocks. Paired comparisons resample the same months for both models. Scores in this paper are pooled over all events in a period; the pipeline's gate used the mean across folds, which gives a lift of {num(g3['lift']['value'], 3, True)} [{num(g3['lift']['ci_low'], 3, True)}, {num(g3['lift']['ci_high'], 3, True)}].",
  f"The relevant benchmark needs care. A classifier that always picks the largest class scores {num(meta['modelling']['majority_baseline_f1'], 2)}, and a within-month label permutation of the full pipeline averages {num(perm[perm.kind == 'permuted'].macro_f1.mean(), 2)}, but both understate chance for a model with balanced predictions. Shuffling a model's own predictions against the labels keeps its class mix and removes its information, and it gives about {num(L6L.chance_shuffled, 2)}. I read every score against that line.",
  "To separate magnitude from direction I define two AUCs: the move AUC scores one minus P(Sideways) against whether the stock left the Sideways band, and the direction AUC scores P(Up) minus P(Down) against Up, among events that moved. Drift tests use two-way clustered standard errors by firm and date ({petersen09}; {cameron11}) and Benjamini-Hochberg control within test families ({bh95}). Calendar-time tests use Newey-West errors ({newey_west}).")

# ---------------------------------------------------------------------------------------------- V
h1("V. What Can Be Predicted")
h2("A. The ladder")


def lad_row(lab, rung, cohort):
    x = lad[(lad.rung == rung) & (lad.cohort == cohort)]
    out = [lab]
    for hd in ("lightgbm", "catboost", "enet"):
        y = x[(x["head"] == hd) & (x.tuned == "tuned")]
        if rung in ("L5", "L7"):
            y = y[y.encoder == "bge_base"]
        out.append(f"{y.iloc[0].macro_f1:.3f}" if len(y) else "&ndash;")
    return out


table("ladder", "Walk-forward results by rung and classifier", "Pooled macro-F1 over 28,533 events, 2008-2019, identical across rungs. L5 and L7 use BGE; CatBoost was not run on embeddings. The class prior (L0) scores 0.191; shuffled-prediction chance is about 0.33.",
      ["Rung", "LightGBM", "CatBoost", "Elastic net"], [lad_row("L1 Price", "L1", "common"), lad_row("L2 Metadata", "L2", "common"), lad_row("L3 Accounting", "L3", "common"), lad_row("L4 Classical text", "L4", "common"), lad_row("L5 Embeddings", "L5", "common_emb"), lad_row("L6 All non-neural", "L6", "common"), lad_row("L7 L6 + embeddings", "L7", "common_emb")])
e7 = lad[(lad.rung == "L7") & (lad["head"] == "enet") & (lad.encoder == "bge_base") & (lad.tuned == "tuned")].iloc[0]; e6 = lad[(lad.rung == "L6") & (lad["head"] == "enet") & (lad.cohort == "common") & (lad.tuned == "tuned")].iloc[0]
p(f"{tn['ladder']} and {fn['ladder']} give the walk-forward results. Metadata is the strongest single source and accounting data the weakest. Embeddings alone do better than chance, close to classical text and a little below it, so they contain real signal. Yet that signal is already available from other features. L7 ties L6 under LightGBM. Under the elastic net it falls from {e6.macro_f1:.3f} to {e7.macro_f1:.3f}, which I attribute to many correlated columns swamping a regularized linear model.")
fig("ladder", "<i>The ablation ladder on the validation folds.</i> Dots show pooled macro-F1 and bars show 95% month-block intervals. Open markers use ChronoBERT and filled markers use BGE. The dashed line is shuffled-prediction chance.", "88%")
h2("B. Out-of-sample performance")


def pf_row(r, name):
    e = {"Walk": lad[(lad.run_id == {"L1": "common|L1|enet|-|T_trade|h1|tuned", "L6": "common|L6|enet|-|T_trade|h1|tuned", "L7": "common_emb|L7|lightgbm|bge_base|T_trade|h1|tuned"}[r.model])].ece.iloc[0] if r.period.startswith("Walk") else None,
         "Dev": dev_r[dev_r.run_id == f"dev|{r.model}"].ece.iloc[0] if r.period.startswith("Dev") else None, "Lock": lock_r[lock_r.run_id == f"locked|{r.model}"].ece.iloc[0] if r.period.startswith("Lock") else None}
    ece = [v for v in e.values() if v is not None][0]
    return [name, f"{r.accuracy:.3f}", f"{r.macro_f1:.3f} [{r.macro_f1_lo:.3f}, {r.macro_f1_hi:.3f}]", f"{r.chance_shuffled:.3f}", f"{r.logloss:.3f}", f"{ece:.3f}", f"{r.move_auc:.3f}", f"{r.direction_auc:.3f}"]


rows = []
for per in PERS:
    rows.append(f"{per} (N = {int(perf[perf.period == per].n.iloc[0]):,})")
    for m, nm in (("L1", "Price only (L1)"), ("L6", "All non-neural (L6)"), ("L7", "L6 + embeddings (L7)")):
        rows.append(pf_row(perf[(perf.period == per) & (perf.model == m)].iloc[0], nm))
table("perf", "Out-of-sample performance", "Each model uses the settings chosen by validation: L1 and L6 are elastic nets, and L7 is LightGBM with BGE. Brackets are 95% month-block bootstrap intervals. Chance shuffles the model's own predictions. ECE is the pipeline's expected calibration error.",
      ["Model", "Accuracy", "Macro-F1", "Chance", "Log-loss", "ECE", "Move AUC", "Dir. AUC"], rows)
fig("performance", "<i>Macro-F1 by period.</i> Bars show point estimates and whiskers show 95% month-block intervals. The dashed line is shuffled-prediction chance, averaged over L6 and L7.", "86%")
p(f"{tn['perf']} and {fn['performance']} report the main results. L6 scores {num(L6W.macro_f1)}, {num(L6D.macro_f1)}, and {num(L6L.macro_f1)} in the three periods, so skill holds up over time, including on data that played no part in model selection. Its lead over the price-only model is {ci(L('Walk', 'Structured'))}, {ci(L('Dev', 'Structured'))}, and {ci(L('Locked', 'Structured'))}. The magnitude is modest. Locked-test log-loss is {num(L6L.logloss)} against {num(lock_r.logloss_prior.iloc[0])} for the class prior, and accuracy is {num(L6L.accuracy)} against {num(meta['modelling']['class_shares']['locked']['majority_class_accuracy'])} for always predicting Sideways. On the locked test L6 is well calibrated, as {fn['reliability']} shows.")
h2("C. Magnitude, not direction")
p(f"{fn['auc']} shows the central pattern. The move AUC of L6 is well above 0.5 in every period. The direction AUC is {num(L6W.direction_auc)} [{num(L6W.direction_auc_lo)}, {num(L6W.direction_auc_hi)}] on the validation folds, barely above a coin flip, and it falls to {num(L6D.direction_auc)} and {num(L6L.direction_auc)} afterwards, with intervals covering 0.5. The two locked-test values of L6 and L7 happen to match to six decimals, but the models differ: their directional signals correlate only {num(facts['locked_L6L7_signal_corr'], 2)} and they predict the same class for {facts['locked_L6L7_same_class'] * 100:.0f}% of events. Restricting to verified earnings releases, where news should be densest, does not help: L6's direction AUC is {num(eo[(eo.period.str.startswith('Walk')) & (eo.model == 'L6')].direction_auc.iloc[0])}, {num(eo[(eo.period.str.startswith('Dev')) & (eo.model == 'L6')].direction_auc.iloc[0])}, and {num(eo[(eo.period.str.startswith('Lock')) & (eo.model == 'L6')].direction_auc.iloc[0])}.",
  f"{fn['move_deciles']} expresses the magnitude result in returns. Sorting locked-test events on L6's predicted probability of a move, mean absolute abnormal return rises from {pc(d1.mean_abs_ar, 2, False)} in the lowest decile to {pc(d6.mean_abs_ar, 2, False)} in the highest, and the share of events leaving the Sideways band rises from {d1.share_moved * 100:.0f}% to {d6.share_moved * 100:.0f}%. The price-only model shows much less of this, which points to metadata (an earnings release after the close is likely to produce a large move at the next open, whatever it says). The confusion matrix in {fn['confusion']} shows the same thing from another angle: the model finds Sideways events best and often confuses Up with Down.")
fig("auc", "<i>Move and direction skill.</i> Move AUC is computed on all events and direction AUC on events that left the Sideways band. Bars are 95% month-block intervals, and the dashed line marks no information.")
fig("move_deciles", "<i>Predicted move and realized size, locked test.</i> Events sorted into deciles of predicted probability of leaving the Sideways band, 1 minus P(Sideways).")
fig("confusion", "<i>Confusion matrix of L6, locked test.</i> Shares are by realized class, with event counts in parentheses.", "48%")
fig("reliability", "<i>Calibration on the locked test.</i> Each marker is a decile of predicted probability. Points on the dashed line are perfectly calibrated.")
fig("by_year", "<i>Stability.</i> Metrics by calendar year, using validation predictions through 2019, then the development (yellow) and locked (red) periods. Years with fewer than 200 events are omitted. Dashed lines mark chance and no information.")
p(f"{fn['by_year']} shows these metrics year by year. L6 and L7 stay above chance in most years. The price-only model is much less stable and falls well below chance in 2020-2022, when market volatility shifted sharply, which explains the unusually large gain of L6 over L1 in the development period.")

# ---------------------------------------------------------------------------------------------- VI
h1("VI. Does the Text Add Anything?")
h2("A. The main test")


def lift_rows(c):
    out = []
    for lab, m in (("Macro-F1", "macro_f1"), ("Log-loss", "logloss"), ("Multiclass AUC", "auc_ovr"), ("Move AUC", "move_auc"), ("Direction AUC", "direction_auc")):
        out.append([lab] + [ci(L(per, c, m)) for per in ("Walk", "Dev", "Locked")])
    return out


table("lifts", "Paired comparisons on identical events", "Differences between models, with 95% month-block bootstrap intervals from 2,000 draws. For log-loss, lower is better. L6 is an elastic net and L7 is LightGBM with BGE. Macro-F1 is pooled within period.",
      ["Metric", "Walk-forward 2008-2019", "Development 2020-2022", "Locked test 2023-2026"], ["Panel A. Embeddings over all non-neural features (L7 &minus; L6)"] + lift_rows("Text") + ["Panel B. Change features (L6 &minus; L6 without change features)"] + lift_rows("Change")[:1] + ["Panel C. All non-neural features over price (L6 &minus; L1)"] + lift_rows("Structured")[:1])
fig("lifts", "<i>Paired changes in macro-F1.</i> The dashed line in the left panel is the 0.01 gate set in advance for fine-tuning the encoders.")
p(f"{tn['lifts']} and {fn['lifts']} give the answer to H1. All three macro-F1 intervals cover zero, and the point estimates do not share a sign. The gate failed, so no encoder was fine-tuned, and the study's language side stays at frozen embeddings. Other metrics agree: on the locked test, the changes in log-loss, multiclass AUC, move AUC, and direction AUC are all within about 0.002 of zero, with intervals covering it. The intervals are tight enough to carry information. Lifts of 0.01 or more occur in {TXW.share_ge_margin * 100:.1f}% of bootstrap draws on the validation folds and {TXL.share_ge_margin * 100:.1f}% on the locked test, so a gain of the size the gate required is unlikely in those periods, although the development period, at {TXD.share_ge_margin * 100:.0f}%, does not rule it out. The change features add nothing either (H2).")
h2("B. Same classifier, two encoders")


def sh_row(i):
    r = sh.iloc[i]
    return [r.comparison, ci(r), f"{r.p_two_sided:.3f}"]


table("samehead", "Embedding lift with the classifier held fixed", "Walk-forward folds only, 28,533 events, because the locked test scored only the frozen finalists. Paired month-block intervals, scored within one run. L6 with LightGBM scores 0.394 in that run and 0.390 in the structured stage of Table IV, which shows the size of cross-machine noise.",
      ["Comparison (macro-F1)", "Difference", "p-value"], [sh_row(i) for i in range(5)])
p(f"The main comparison has a flaw, because L6 is an elastic net and L7 is LightGBM. {tn['samehead']} holds the classifier fixed. With LightGBM the change is {ci(sh.iloc[0])} for BGE and {ci(sh.iloc[1])} for ChronoBERT. The vintage encoder and the static encoder do not differ detectably, {ci(sh.iloc[4])}, which is the answer to H4, and it matches the modest look-ahead bias found by {{he26}} for news. Since this is a retrospective comparison on a 2008-2019 sample, it does not establish that the static encoder is clean, only that any bias it carries does not produce a detectable gain here. A stacked ensemble does not beat L7 in the development period (H5: {num(csv('dev_lifts', D).query('hypothesis == \"H5\"').value.iloc[0], 3, True)}, with an interval covering zero).")
h2("C. A model with no text")
table("notext", "What numbers alone achieve", "A rung fitted after the locked test with price, metadata, and accounting features only, using L6's chosen settings on the same folds. Paired changes in macro-F1 on identical events with 95% month-block intervals. L6 predictions come from the stored run, so small differences also reflect machine noise for the LightGBM row (Section VIII.B).",
      ["", "Walk-forward 2008-2019", "Development 2020-2022", "Locked test 2023-2026"], ["Levels of macro-F1"] + [[m] + [f"{NL_[(NL_.model == k) & (NL_.period == per)].macro_f1.iloc[0]:.3f}" for per in PERS] for m, k in (("Numbers only, elastic net", "Numbers only, elastic net"), ("Numbers only, LightGBM", "Numbers only, LightGBM"))] + ["Changes in macro-F1"] + [[m] + [ci(N(per, c)) for per in ("Walk", "Dev", "Locked")] for m, c in (("All text, both LightGBM (L7 &minus; numbers)", A_LG), ("Classical text, both elastic net (L6 &minus; numbers)", A_EN), ("TF-IDF factors, elastic net (L6 &minus; numbers + word features)", "TF-IDF"), ("Metadata and accounting, elastic net (numbers &minus; L1)", "Metadata"))])
fig("no_text", "<i>Text relative to a numbers-only model.</i> Exploratory, fitted after the locked test. The dashed line marks the 0.01 gate.")
p(f"Since L6 holds classical text, the comparison above cannot say whether text helps at all. To ask that directly I fit a rung that the pipeline did not have, with price, metadata, and accounting features only, using L6's settings without further tuning. {tn['notext']} and {fn['no_text']} show the results. Numbers alone score {NL_[(NL_.model == 'Numbers only, LightGBM') & (NL_.period == PERS[2])].macro_f1.iloc[0]:.3f} on the locked period with LightGBM, against {num(L6L.macro_f1)} for L6 and {num(L7L.macro_f1)} for L7. With LightGBM on both sides, all text together changes macro-F1 by {ci(N('Walk', A_LG))}, {ci(N('Dev', A_LG))}, and {ci(N('Locked', A_LG))}, all covering zero. The elastic net gives a somewhat different picture: classical text adds {ci(N('Walk', A_EN))}, {ci(N('Dev', A_EN))}, and {ci(N('Locked', A_EN))}, a gain of about the size of the gate in the two later periods, though the locked-period interval just reaches zero. I read this as weak evidence of a small gain that depends on the classifier. A linear model may benefit from text features that a tree model can replace with interactions among the numbers. Direction does not improve either: L7 minus numbers on direction AUC is {ci(N('Locked', A_LG, 'direction_auc'))} on the locked period, and the numbers-only LightGBM's own direction AUC there is {NL_[(NL_.model == 'Numbers only, LightGBM') & (NL_.period == PERS[2])].direction_auc.iloc[0]:.3f}, the highest of any locked-period model and still small. Because I fit this rung after seeing the locked results, its locked-period column is descriptive and has none of the protection of the original test.")
h2("D. Where does a gain appear? Subsets and one false lead")
fig("subsets", "<i>Text lift by subset.</i> Change in macro-F1 from L6 to L7 within each subset, pooled over 2008-2026 and on the locked test. Exploratory, chosen after the locked test.", "88%")
pre = ss[(ss.subset == "Before the open") & (ss.scope.str.startswith("Pooled"))].iloc[0]; prel = ss[(ss.subset == "Before the open") & (ss.scope == "Locked test")].iloc[0]; q10 = ss[(ss.subset == "10-Q") & (ss.scope.str.startswith("Pooled"))].iloc[0]; ne = ss[(ss.subset == "Non-earnings events") & (ss.scope == "Locked test")].iloc[0]
S = lambda c, b: shs[(shs.comparison.str.startswith(c)) & (shs.bucket == b)].iloc[0]
p(f"{fn['subsets']} breaks the lift down by form, earnings status, and session. These cuts are exploratory, and I chose them after the locked test had been scored. One stands out: filings accepted before the open show a lift of {ci(pre)} pooled and {ci(prel)} on the locked test. Taken at face value, that would suggest text helps exactly when an investor has the whole trading day ahead. 10-Qs and the locked-test non-earnings events go the other way, at {ci(q10)} and {ci(ne)}.")
table("session", "The pre-market lift comes from the classifier", "Walk-forward folds, by session of acceptance. Change in macro-F1 with 95% month-block intervals.",
      ["Comparison", "Before the open", "During the session", "After the close"], [[c] + [ci(S(k, b)) for b in ("Before the open", "During the session", "After the close")] for c, k in (("LightGBM with BGE vs. elastic net (L7 vs. L6)", "LightGBM+BGE vs elastic"), ("LightGBM vs. elastic net, no text", "LightGBM vs elastic"), ("LightGBM with BGE vs. LightGBM (text only)", "LightGBM+BGE vs LightGBM"), ("LightGBM with ChronoBERT vs. LightGBM (text only)", "LightGBM+ChronoBERT"))])
p(f"{tn['session']} shows that this lead is false. The L7-versus-L6 comparison changes two things at once. Without any text, moving from the elastic net to LightGBM already raises pre-market macro-F1 by {ci(S('LightGBM vs elastic', 'Before the open'))}. With LightGBM on both sides, adding text changes it by {ci(S('LightGBM+BGE vs LightGBM', 'Before the open'))}. The pre-market gain belongs to the tree model's ability to use structured features in that session. The locked test scored only the frozen pair, so I cannot repeat this decomposition there.")
h2("E. Does the timing of the target explain the null?")
g_ = lambda rg, t: ti[(ti.rung == rg) & (ti.target == t)].iloc[0]
p(f"One explanation for the null is that the news is priced before the first open, leaving nothing for a tradable target. <i>T</i><sub>info</sub>, which includes that first reaction, lets me check this, though only for the elastic-net models of L1, L4, and L6. {tn['tinfo']} and {fn['tinfo']} compare the two targets on the same {int(g_('L6', 'T_info').n):,} events. Move AUC rises under <i>T</i><sub>info</sub>, from {num(g_('L6', 'T_trade').move_auc)} to {num(g_('L6', 'T_info').move_auc)} for L6, as one would expect of a window that contains the reaction. Direction AUC barely changes, at {num(g_('L6', 'T_trade').direction_auc)} and {num(g_('L6', 'T_info').direction_auc)}. So even with the first reaction included, these features say little about direction, and pricing before the open is at most part of the story. Dictionary tone correlates weakly with the reaction: across earnings releases, the Spearman correlation of net tone with CAR[&minus;1,+1] is {num(corr[(corr['var'] == 'main_lm_net_tone') & (corr['sample'] == 'Earnings releases') & (corr.window == 'car_m1_1')].spearman.iloc[0], 3)}, and SUE's is {num(corr[(corr['var'] == 'sue_event') & (corr['sample'] == 'Earnings releases') & (corr.window == 'car_m1_1')].spearman.iloc[0], 3)}. I did not run embeddings on this target, so the test covers classical text, and the embedding question stays open for <i>T</i><sub>info</sub>.")
table("tinfo", "Tradable and information targets", "Elastic-net models on events eligible for both targets, walk-forward folds. Brackets are 95% month-block intervals from 1,000 draws.",
      ["Rung", "Target", "Macro-F1", "Move AUC", "Direction AUC"], [[{"L1": "L1 Price", "L4": "L4 Classical text", "L6": "L6 All non-neural"}[r.rung], "<i>T</i><sub>" + r.target[2:] + "</sub>", f"{r.macro_f1:.3f}", f"{r.move_auc:.3f} [{r.move_lo:.3f}, {r.move_hi:.3f}]", f"{r.direction_auc:.3f} [{r.dir_lo:.3f}, {r.dir_hi:.3f}]"] for _, r in ti.iterrows()])
fig("tinfo", "<i>Tradable and information targets.</i> Same events and classifiers for both targets.")

# ---------------------------------------------------------------------------------------------- VII
h1("VII. Post-Earnings Drift and a Measurement Trap")
h2("A. Short-window reactions")
ev_r = []
for grp, nm in (("8K", "8-K"), ("periodic_10Q", "10-Q"), ("periodic_10K", "10-K")):
    x = ga[(ga.group_by == "event_type") & (ga.group == grp) & (ga.event_set == "all")]
    a1, a2 = x[x.measure == "car_m1_1"].iloc[0], x[x.measure == "car_0_1"].iloc[0]; f1, f2, f3 = (frm[(frm.form == nm) & (frm.measure == k)].iloc[0] for k in ("mm_alpha_replica", "mm_noalpha", "mkt_adj"))
    ev_r.append([nm, f"{int(a1.n):,}", f"{num(a1['mean'] * 100, 2, True)} ({num(a1.t, 2)})", f"{num(a2['mean'] * 100, 2, True)} ({num(a2.t, 2)})", f"{num(f1['mean'] * 100, 2, True)} ({num(f1.t, 2)})", f"{num(f2['mean'] * 100, 2, True)} ({num(f2.t, 2)})", f"{num(f3['mean'] * 100, 2, True)} ({num(f3.t, 2)})"])
table("evstudy", "Event-study returns by form", "Mean returns in percent with t-statistics in parentheses, two-way clustered by firm and date. Short windows use the market model. The three right-hand columns differ only in how expected return is measured over days [+2,+60]; the sample contains only survivors.",
      ["Form", "N", "CAR[&minus;1,+1]", "CAR[0,+1]", "[+2,+60], alpha subtracted", "[+2,+60], no alpha", "[+2,+60], minus S&amp;P 500"], ev_r)
p(f"{tn['evstudy']} reports event-study returns. Over short windows the choice of expected-return model hardly matters, and reactions are small on average. Over the longer window it matters a great deal: the mean for 8-Ks is {num(frm[(frm.form == '8-K') & (frm.measure == 'mm_alpha_replica')]['mean'].iloc[0] * 100, 2, True)}% with pre-event alpha subtracted and {num(frm[(frm.form == '8-K') & (frm.measure == 'mm_noalpha')]['mean'].iloc[0] * 100, 2, True)}% without it. {fn['car_paths']} draws the two versions. The upward slope in Panel B is what a survivor sample should show, and the flat line in Panel A comes from removing each firm's own past outperformance. The two give upper and lower reference points, and neither is a clean estimate of drift in levels. Only the spreads in the next subsections are informative.")
fig("car_paths", "<i>Abnormal-return paths by form, with two expected-return models.</i> Paths are normalized to zero at day &minus;1. Panel A subtracts a market-model alpha and beta estimated on days [&minus;250, &minus;30]. Panel B subtracts only the beta term.")
h2("B. The pre-specified test")
p(f"I sort earnings releases into quintiles of SUE using breakpoints from the prior four quarters, so the sort uses no future data, and compare the top and bottom quintiles over days +2 to +60. The pre-specified measure is the cumulative Carhart ({{carhart97}}) abnormal return, with alpha and factor loadings estimated on days &minus;250 to &minus;30:")
eq("<i>CAR</i><sub><i>i</i></sub> = &Sigma;<sub><i>t</i>=+2</sub><sup>+60</sup> ( <i>R</i><sub><i>st</i></sub> &minus; <i>R</i><sub><i>ft</i></sub> &minus; <i>&alpha;&#770;</i><sub><i>i</i></sub> &minus; <i>&beta;&#770;</i>&prime;<sub><i>i</i></sub> <i>F</i><sub><i>t</i></sub> ).", 3)
s0 = ht[ht.test.str.startswith("SUE-conditioned post-earnings drift (Q5-Q1, SUE_release)")].iloc[0]
p(f"The test gave {pc(s0.estimate)} [{pc(s0.ci_low)}, {pc(s0.ci_high)}] on {int(s0.n):,} reports in the two quintiles, a reversal rather than the classic continuation. The sign of the surprise itself is right: the three-day reaction rises from {pc(q0.ann_c4.iloc[0])} in the lowest quintile to {pc(q0.ann_c4.iloc[4])} in the highest ({fn['sue_quintiles']}, Panel A). The reversal then passed every check I applied, shown in Appendix B: three subperiods, a one-session delay, index members and non-members, size and volatility splits, winsorizing, excluding crisis years, and quarter-by-quarter spreads (mean {pc(facts['quarterly_spread_mean'])}, t = {num(facts['quarterly_spread_t'], 2)}, negative in {facts['quarterly_spread_negative_share'] * 100:.0f}% of {facts['quarterly_spread_n']} quarters). An earlier audit in the project had accepted it as a feature of the sample and blamed survivorship. Earlier write-ups of this study, including the project report and dashboard, described it as a puzzling finding before the problem below was found.")
fig("sue_quintiles", "<i>Earnings surprise and returns.</i> Panel A shows the three-day reaction. Panel B shows drift with and without the pre-event alpha term, on the same reports.")
h2("C. Diagnosis")
fm1, fm2 = FM("c4_alpha_replica", "(1)", "sue_rank"), FM("c4_alpha_replica", "(2)", "sue_rank"); fmm = FM("c4_alpha_replica", "(2)", "px_mom_12_2"); fn1, fn2 = FM("c4_noalpha", "(1)", "sue_rank"), FM("c4_noalpha", "(2)", "sue_rank"); fnm = FM("c4_noalpha", "(2)", "px_mom_12_2")
p(f"A Fama-MacBeth regression ({{fama_macbeth}}) gave the first clue. With the pre-specified measure as the dependent variable, the coefficient on the within-quarter SUE rank is {num(fm1.coef, 3)} (t = {num(fm1.t, 2)}). Adding controls takes it to {num(fm2.coef, 3)} (t = {num(fm2.t, 2)}), and the coefficient on past 12-month return is {num(fmm.coef, 3)} per standard deviation (t = {num(fmm.t, 1)}). A t-statistic of that size for a momentum variable, inside a measure already adjusted for momentum, suggests an effect built into the left-hand side.",
  f"It is. Equation (3) subtracts <i>&alpha;&#770;</i> for each of 59 days, and <i>&alpha;&#770;</i> is the stock's average abnormal return over roughly the prior year. The Spearman correlation between this alpha term and past return is {num(facts['spearman_mom_alpha_c4'], 2)}, and past return correlates {num(facts['spearman_sue_mom'], 2)} with SUE. High-surprise firms have had good years: mean past return rises from {Q.loc[1, 'mom'] * 100:.1f}% in the bottom quintile to {Q.loc[5, 'mom'] * 100:.1f}% in the top one. So the amount subtracted rises from {pc(Q.loc[1, 'alpha_term_c4'])} to {pc(Q.loc[5, 'alpha_term_c4'])} ({fn['alpha_term']}). The difference, {pc(aq, 2, False)}, is the whole spread: it is the gap between the estimate with alpha, {pc(C4A.estimate)}, and the estimate without it, {pc(C4N.estimate)}. Because past alpha is largely noise and a sample artifact rather than a persistent expected return, projecting it forward manufactures a negative relation with anything that correlates with past performance. Every check in Appendix B shares this term, which is why all of them agreed.")
fig("alpha_term", "<i>Why the spread appeared.</i> Panel A shows mean past return by surprise quintile. Panel B shows the amount the pre-specified measure subtracts from each report's return over days [+2,+60] for its pre-event alpha, from a Carhart regression on days [&minus;250, &minus;30].")
h2("D. Re-measuring drift")
table("quint", "Returns by earnings-surprise quintile", "Means in percent over days [+2,+60], except the reaction column. The last columns isolate the alpha term. Reports sorted on breakpoints from the prior four quarters.",
      ["Quintile", "Reports", "Mean SUE", "Reaction [&minus;1,+1]", "Carhart with alpha", "Alpha term", "Carhart, no alpha", "Stock minus S&amp;P 500"],
      [[f"Q{int(r.q)}" + {1: " (worst)", 5: " (best)"}.get(int(r.q), ""), f"{int(r.reports):,}", num(q0.sue.iloc[i], 2), num(q0.ann_c4.iloc[i] * 100, 2, True), num(r.c4_alpha * 100, 2, True), num(r.alpha_term_c4 * 100, 2, True), num(r.c4_noalpha * 100, 2, True), num(r.mkt_adj * 100, 2, True)] for i, r in qa.iterrows()])
table("measure", "The Q5 minus Q1 spread under different return measures", f"Days [+2,+60], in percent. Standard errors are two-way clustered by firm and date; intervals come from a month-block bootstrap. My reconstruction of the pipeline's measures correlates {facts['replica_corr_c4']:.3f} with the stored values; it keeps a few more reports, so the first two rows differ slightly from the {pc(s0.estimate)} reported by the pipeline.",
      ["Measure", "Q1", "Q5", "Q5 &minus; Q1", "95% interval", "t", "N"], [[r.measure, num(r.q1 * 100, 2, True), num(r.q5 * 100, 2, True), num(r.estimate * 100, 2, True), f"[{num(r.ci_low * 100, 2, True)}, {num(r.ci_high * 100, 2, True)}]", num(r.t, 2), f"{int(r.n):,}"] for _, r in dm.iterrows()], align=["l", "c", "c", "c", "c", "c", "c"])
p(f"{tn['quint']} and {tn['measure']} measure the spread several ways. Dropping the alpha term gives {pc(C4N.estimate)} [{pc(C4N.ci_low)}, {pc(C4N.ci_high)}] for the Carhart model and {pc(MMN.estimate)} [{pc(MMN.ci_low)}, {pc(MMN.ci_high)}] for the market model. Estimating alpha and loadings after the window instead, on days +61 to +280, gives {pc(C4P.estimate)} [{pc(C4P.ci_low)}, {pc(C4P.ci_high)}]. Simple stock-minus-index returns give {pc(MKT.estimate)} and raw returns give {pc(RAW.estimate)}. In the Fama-MacBeth regressions the SUE coefficient without the alpha term is {num(fn1.coef, 3)} (t = {num(fn1.t, 2)}), and with controls it is {num(fn2.coef, 3)} (t = {num(fn2.t, 2)}), while the coefficient on past return collapses to {num(fnm.coef, 3)} (t = {num(fnm.t, 2)}), as {tn['fm']} shows.")


def fm_cell(dep, sp, var):
    x = fma[(fma.dep == dep) & (fma.spec == sp) & (fma["var"] == var)]
    return f"{num(x.coef.iloc[0], 4)} ({num(x.t.iloc[0], 2)})" if len(x) else ""


vars_ = (("sue_rank", "SUE rank (0 to 1)"), ("car_m1_1", "Reaction [&minus;1,+1]"), ("px_mom_12_2", "Past return, months &minus;12 to &minus;2"), ("acc_log_mcap", "Log market cap"), ("acc_book_to_market", "Book-to-market"), ("px_vol_60", "60-day volatility"))
table("fm", "Fama-MacBeth regressions of drift on earnings surprise", f"Quarterly cross-sections, {int(fma.quarters.iloc[0])} quarters with enough reports. The dependent variable is the return over days [+2,+60], winsorized at 1% and 99%. Controls are standardized within quarter, so coefficients are per standard deviation. t-statistics in parentheses use Newey-West errors with four lags.",
      ["", "Carhart with alpha (1)", "Carhart with alpha (2)", "Carhart, no alpha (1)", "Carhart, no alpha (2)", "Minus S&amp;P 500 (2)"], [[lab] + [fm_cell(d, s, v) for d, s in (("c4_alpha_replica", "(1)"), ("c4_alpha_replica", "(2)"), ("c4_noalpha", "(1)"), ("c4_noalpha", "(2)"), ("mkt_adj", "(2)"))] for v, lab in vars_])
p(f"The cleanest evidence comes from calendar-time portfolios, which need no event-level estimation. Each day I hold, equally weighted, all stocks in days +2 to +60 after a report in the top quintile, do the same for the bottom quintile, and regress the daily spread on the Carhart factors. {tn['caltime']} shows the results. The spread's alpha is {pc(CTA.alpha_59d)} per 59 days (t = {num(CTA.alpha_t, 2)}), with an interval of {pc(CTA.alpha_lo)} to {pc(CTA.alpha_hi)}. Its raw mean is {pc(CTA.raw_59d)}. It is the momentum loading that stands out, at {num(CTA.b_mom, 2)} (t = {num(CTA.b_mom_t, 1)}), confirming that the top-surprise portfolio tilts toward past winners. {fn['drift_robustness']} collects these estimates.")
table("caltime", "Calendar-time portfolios", "Daily long Q5, short Q1 spread, holding each report for days [+2,+60], equal weighted, at least five names per side. Alpha and raw mean are scaled to 59 days, in percent. t-statistics use Newey-West errors with five lags.",
      ["Sample", "Days", "Names per side", "Raw mean", "t", "Carhart alpha", "t", "Momentum loading", "t"], [[r["sample"], f"{int(r.days):,}", f"{r.mean_names_q5:.0f} / {r.mean_names_q1:.0f}", num(r.raw_59d * 100, 2, True), num(r.raw_t, 2), num(r.alpha_59d * 100, 2, True), num(r.alpha_t, 2), num(r.b_mom, 2), num(r.b_mom_t, 1)] for _, r in cal.iterrows()])
fig("drift_robustness", "<i>The Q5 minus Q1 spread across measures and samples.</i> Squares subtract a pre-event alpha. Circles do not. The diamond is the calendar-time Carhart alpha.", "90%")
h2("E. Interpretation")
sv = lambda d, L_: (Q.loc[5, 'c4_noalpha'] - ((1 - d) * Q.loc[1, 'c4_noalpha'] + d * L_))
p(f"So this sample shows neither a reversal nor detectable continuation in large firms over days +2 to +60, using a time-series surprise. An absence of drift fits the evidence that the anomaly has faded in large liquid stocks ({{chordia14}}; {{martineau22}}), although two features of the design bias against finding it. A seasonal random-walk surprise is a weaker signal than an analyst-based one ({{livnat06}}). And survivorship can hide continuation, since firms that kept falling after bad news are the ones most likely to be missing ({{shumway97}}; {{brown92}}). {fn['survivorship_bound']} gives a rough sense of scale: if 5% of low-surprise reports were missing and those averaged a 30% loss, a figure taken from the delisting returns reported by {{shumway97}} for performance-related delistings in a much broader sample, the true spread would be about {pc(sv(.05, -.3), 1)} rather than {pc(C4N.estimate, 1)}. These are illustrative inputs, and for large firms the missing share is probably smaller, but the direction of the bias is clear. The interval around my estimate should be read as a statement about survivors.",
  "The broader lesson concerns specification. Subtracting a pre-event alpha is common in short-window event studies, where it does little harm because the window is short. Over 59 days it can dominate. The bias here does not require any error in the code, and it survives subperiods, sample splits, clustered errors, multiple-testing control, and a one-session delay, since all of them share the term. It would appear in any sort on a variable correlated with past performance. Calendar-time portfolios, or alpha-free measures, avoid it. A simple diagnostic is to regress the drift measure on past return: a large negative coefficient, from a measure already adjusted for momentum, is the warning sign.")
fig("survivorship_bound", "<i>How much could survivorship hide?</i> Implied Q5 minus Q1 spread if a share of low-surprise reports is missing and those missing reports have the mean loss shown at the right of each line. Starts from the Carhart, no-alpha estimate. Illustrative.", "68%")

# ---------------------------------------------------------------------------------------------- VIII
h1("VIII. Economic Significance and Limits")
h2("A. A backtest")
ld, dd = bt_("dev", "long_only"), bt_("locked", "long_only")


def bt_row(block, lag, cost):
    x = [bt_(block, leg, lag, cost) for leg in ("long_only", "long_minus_spy_matched", "long_short_hypothetical")]
    return [{"dev": "Development", "locked": "Locked test"}[block], lag, cost] + [pc(y.ann_return, 1) for y in x] + [num(x[0].sharpe, 2), num(x[2].sharpe, 2), num(x[2].deflated_sharpe_prob, 3)]


p(f"As an economic check, I trade the frozen model, refitted on a five-session target to match the holding period. The signal is P(Up) minus P(Down). Thresholds come from out-of-fold training signals and mark the top and bottom fifths. The strategy enters at the first open after the decision time, holds for five sessions, caps weights at 5%, charges 10 or 20 basis points per side, and assumes 50 basis points a year to borrow. {tn['backtest']} and {fn['backtest']} show the results. Under the base setting (no delay, 10 basis points) the long book earns {pc(ld.ann_return, 1)} a year in the development period, {pc(bt_('dev', 'long_minus_spy_matched').ann_return, 1)} against the S&amp;P 500 on the same days, and {pc(bt_('locked', 'long_only').ann_return, 1)} a year on the locked test, {pc(bt_('locked', 'long_minus_spy_matched').ann_return, 1)} against the index. The hypothetical long-short portfolio loses {pc(abs(bt_('locked', 'long_short_hypothetical').ann_return), 1, False)} a year on the locked test. The delayed version looks better in development and also loses on the locked test. The long book holds few names ({dd.avg_names_long:.0f} on average) and sits in cash on {dd.share_days_long_in_cash * 100:.0f}% of locked-test days, so index comparisons use only the days it is invested. Deflated Sharpe probabilities ({{bailey14}}) are all low. Since the model has no directional skill, these outcomes are what one should expect.")
table("backtest", "Backtest of the frozen model", "Annualized returns. An economic-relevance check only: shorts are hypothetical, capacity is not modeled, and the universe contains no delisted firms. DSR is the deflated Sharpe probability of the long-short portfolio.",
      ["Period", "Delay", "Cost (bps)", "Long", "Long &minus; S&amp;P", "Long-short", "Sharpe, long", "Sharpe, L-S", "DSR"], [bt_row(b, l, c) for b in ("dev", "locked") for l in (0, 1) for c in (10, 20)])
fig("backtest", "<i>Backtest.</i> Cumulative returns with no delay and 10 basis points per side. The S&amp;P 500 line is matched to the days the long book is invested.")
h2("B. Validity checks")
ad = csv("adversarial", D)
table("diag", "Validity checks", "Diagnostics from the pipeline. The full 24-item leakage checklist is in Appendix C.",
      ["Check", "Result", "Reading"], [
      ["Availability assertion", f"{E['availability_violations']} violations in {E['events']:,} rows", "No feature is dated after the decision time"],
      ["Planted leaks", f"{E['planted_leaks_caught']} of 4 caught", "The audit detects a future-return feature, a same-event reaction, a later-cluster surprise, and a future document in retrieval"],
      ["Permutation of labels", f"Observed {num(perm[perm.kind == 'observed'].macro_f1.iloc[0])}; null mean {num(perm[perm.kind == 'permuted'].macro_f1.mean())}, max {num(perm[perm.kind == 'permuted'].macro_f1.max())}", "Ten permutations, so the p-value is coarse"],
      ["Model Confidence Set, development log-loss", ", ".join(mcs['mcs']['included']).replace("L6_nochange", "L6 without change features").replace("STACK", "stack"), "L1 and L6 excluded; {hansen11}. A diagnostic only"],
      ["Probability of backtest overfitting", f"{mcs['pbo']['pbo']:.2f} over {mcs['pbo']['n_configs']} configurations", "{bailey17}. Mixes rungs of very different content, so it cannot speak to L6 versus L7"],
      ["Adversarial validation", f"AUC {ad.auc.mean():.3f}", "Training and validation periods are separable: covariate shift, not evidence of leakage"],
      ["Cross-machine repeatability", f"LightGBM {rep.iloc[0].same_class_share * 100:.0f}% same class; elastic net {rep.iloc[1].same_class_share * 100:.0f}%", "Same model and data on two machines. Macro-F1 moves by up to 0.0045, more than the gap between L6 and L7"]], align=["l", "l", "l"])
h2("C. Limits")
p("The results are conditional on several choices that a reader should weigh.",
  "<i>Universe.</i> The firms are large survivors. Results may not extend to small firms, where filings may matter more because other information is scarce. The planned 500-firm study, which includes non-index firms, was not completed on corrected code.",
  "<i>Encoders and tuning.</i> The null covers two frozen encoders with small tuning budgets. The way I pool chunks and compress vectors discards detail, and a model fine-tuned on returns, a finance-specific encoder such as SEC-BERT, or a model that reads the whole document could do better. The gate rule meant fine-tuning never ran, so the paper cannot speak to it.",
  "<i>Structured extraction.</i> I built a retrieval and extraction system (BM25 and dense retrieval, reciprocal rank fusion, reranking, and Qwen2.5-7B with schema-constrained output; {cormack09}; {lewis20}), and ran it on 149 filings. No human labels exist, so it was never scored, and H3 and H6 are untested. No result in this paper relies on it.",
  "<i>Data.</i> Prices are from free sources and only daily bars are available, so the open price is the first executable price I can observe, and I cannot measure intraday reactions after the decision time. The surprise measure is time series, and EPS parsing is rule-based. Latency assumptions were not varied.",
  "<i>Stress folds.</i> Firm-held-out and sector-held-out folds were built but not scored, so firm-level memorization by the models is controlled by masking alone.",
  "<i>Post-lock analysis.</i> Everything in Sections VI.B to VI.E and VII.C to VII.E was written after the locked test had been scored. Apart from the numbers-only rung in Section VI.C, which is a new fit, these analyses read stored predictions and refit nothing. All of them are exploratory and should be read that way. The headline locked-test comparison of L6 and L7 is not.")

# ---------------------------------------------------------------------------------------------- IX
h1("IX. Conclusion")
p(f"For 150 large US firms over 26 years, frozen embeddings of their SEC filings do not improve a forecast of the next tradable session's abnormal return once price, event metadata, accounting data, and classical text measures are known, and a model with no text at all performs about as well. A model can predict how much a stock will move after a filing. Its ability to predict which way is slight on the validation folds and undetectable on the locked test. Yearly point-in-time encoders and a static encoder perform alike, and a tradable backtest does not earn its costs. Post-earnings drift is not detectable in this survivor sample without the alpha term, and the large reversal the pre-specified test produced came from the expected-return model.",
  "Three extensions would test these conclusions directly: the same design on a universe that includes delisted and non-index firms, analyst-based surprises in place of the time-series measure, and human labels for the extraction system, which would allow testing whether specific facts add what embeddings do not.",
  f"<i>Replication.</i> The repository at <a href='{GH}'>{GH}</a> contains the 25-stage pipeline with its tests, configuration, pre-registration file, decision log, all result tables, out-of-sample predictions and event-level drift measures in <code>paper/replication</code>, and the scripts in <code>paper/analysis</code> that produce every table and figure here. Filings can be re-downloaded from EDGAR. Prices and the dictionary are not redistributed, and the pinned sources in {tn['sources']} rebuild them; results that depend on Yahoo Finance data may change slightly with revisions.")

# ---------------------------------------------------------------------------------------------- references
refs = json.load(open(P / "references.json", encoding="utf-8"))


def tc(s):
    s = s.strip()
    return s.title() if s.isupper() else s


KEEP = {"SEC", "CRSP", "NLP", "GPT", "ChatGPT", "BERT", "FinBERT", "LightGBM", "CatBoost", "Optuna", "C-Pack", "Chinese", "Form", "8-K", "10-K", "10-Ks", "CAPM", "U.S.", "Latent", "Dirichlet", "Allocation", "Friday", "CRSP's", "Sharpe", "Non-Normality", "Multiway", "NLP", "ACM"}


def clean_title(t):
    t = re.sub(r"\s+", " ", re.sub(r"<[^>]+>", "", t)).strip().replace("\u2010", "-")
    if "<i>" in t or t.startswith("Advances"):
        return t
    out, start = [], True
    for w in t.split(" "):
        core = w.strip(":?,.()")
        if start or core in KEEP or core.isupper() or any(c.isupper() for c in core[1:]):
            out.append(w)
        else:
            out.append(w[0].lower() + w[1:] if w else w)
        start = w.endswith(":") or w.endswith("?")
    return " ".join(out)


def fmt_auth(au):
    names = []
    for i, (g, f) in enumerate(au):
        g, f = tc(g), tc(f)
        names.append(f"{f}, {g}" if i == 0 and g else (f"{g} {f}".strip()))
    if len(names) == 1: return names[0]
    return ", ".join(names[:-1]) + ", and " + names[-1]


def cite_name(r):
    fam = [tc(a[1]).replace("‐", "-") for a in r["authors"]]
    y = r["year"]
    if len(fam) == 1: s = fam[0]
    elif len(fam) == 2: s = f"{fam[0]} and {fam[1]}"
    else: s = f"{fam[0]} et al."
    return s, y


used = set()


def cite_sub(m):
    k = m.group(1)
    if k not in refs: raise KeyError(k)
    used.add(k); s, y = cite_name(refs[k])
    return f"{s} ({y})"


body = "\n".join(H)
# parenthetical groups: ({a}; {b}) -> (A 2000; B 2001)
def group_sub(m):
    parts = re.findall(r"\{(\w+)\}", m.group(1)); out = []
    for k in parts:
        used.add(k); s, y = cite_name(refs[k]); out.append(f"{s}, {y}")
    return "(" + "; ".join(out) + ")"


body = re.sub(r"\(((?:\{\w+\}(?:; )?)+)\)", group_sub, body)
body = re.sub(r"\{bernard_thomas89\} \{bernard_thomas90\}", lambda m: (used.update(["bernard_thomas89", "bernard_thomas90"]) or "Bernard and Thomas (1989, 1990)"), body)
body = re.sub(r"\{(\w+)\}", cite_sub, body)
items = []
for k in sorted(used, key=lambda k: (fmt_auth(refs[k]["authors"]).lower(), str(refs[k]["year"]))):
    r = refs[k]; t = r["title"] if r["title"].startswith("<i>") else clean_title(r["title"]); j = r.get("journal") or ""; vol = r.get("volume"); pg = (r.get("page") or "").replace("--", "-")
    tail = f"<i>{j}</i>" if j else ""
    if vol: tail += f" {vol}"
    if pg: tail += f", {pg}"
    if r.get("note"): tail = (tail + ", " if tail else "") + r["note"]
    doi = f" <a href='https://doi.org/{r['doi']}'>doi:{r['doi']}</a>" if r.get("doi") else ""
    items.append(f"<div class='ref'>{fmt_auth(r['authors'])}, {r['year']}, {t}{', ' + tail if tail else ''}.{doi}</div>")
refs_html = "<h2>References</h2>" + "".join(items)

# ---------------------------------------------------------------------------------------------- appendices
A = []
A.append("<h2 class='pb'>Appendix A. Deviations from the Plan</h2><p>The plan was recorded in a pre-registration file and a design document, with its hash stored in the run manifest. The file is internal: it was not filed with a registry, and the repository became public only after the study ended, so its timing cannot be verified by third parties. Registered designs ({nosek18}) would give stronger assurance. The project's decision log records each change. The deviations are listed here.</p><ol class='dev'>"
         "<li><b>Sample size.</b> 150 firms instead of 500, to finish a full run on available compute. Intervals are wider and the index versus non-index comparison is not possible.</li>"
         "<li><b>Encoders and search.</b> Two of four encoders, no CatBoost on embeddings, and smaller trial budgets in the GPU stage. Fine-tuning was disabled in this mode, and the gate also failed.</li>"
         "<li><b>Model selection across runs.</b> The freeze rule compared L6, scored on one machine, with L7, scored on another. It selected L7 by 0.0010. Scored in the same run the order reverses (L6 0.3909, L7 0.3898), and rescoring noise exceeds the margin. L6 and L7 should be regarded as tied. The frozen model has a retrospective label, because BGE states no training cutoff.</li>"
         "<li><b>Different classifiers in the main contrast.</b> L6 is an elastic net and L7 is LightGBM, as selection dictated. Section VI.B and VI.C address the consequences on the validation folds, where same-classifier predictions exist.</li>"
         "<li><b>Lock timing.</b> The locked test was scored 2.6 minutes after the freeze, in the same run, without the planned review of development results. The freeze did precede scoring.</li>"
         "<li><b>Baseline contains text.</b> H1 was written as text over structured data, but L6 includes classical text, and no rung without text existed. The numbers-only rung in Section VI.C was added after the lock.</li><li><b>Locked-test lifts.</b> The pipeline's scoring stage does not compute paired lifts, so these come from stored predictions, using the same method as the development period.</li>"
         "<li><b>Stress folds and Track C.</b> Firm-held-out and sector-held-out folds were built but not scored. No human labels exist, so H3 and H6 are untested.</li>"
         "<li><b>Earlier runs.</b> Before this sample I ran a pilot and part of the 500-firm study. They shaped the code, and the 500-firm event study had already shown the drift reversal. That earlier run never scored the locked prediction test, which the guard log confirms was scored once.</li>"
         "<li><b>Drift measure.</b> The pre-specified Carhart measure is reported in Section VII.B. Everything after that was added on finding the artifact.</li>"
         "<li><b>Bugs found and fixed before the final run.</b> Same-day 8-K groups were clocked at the first filing, so later content leaked (5,189 events in the larger sample). A 10-Q filed minutes after a release fed that release's surprise history. An accounting lookup returned the prior-year figure in about half of events, found by comparing two runs on the same firms. Regression tests cover these. The drift problem in Section VII was not a coding bug and is not fixed in the pipeline's stored outputs, which keep the pre-specified definition.</li></ol>")
dr_rows = [[r.spec, num(r.q1_mean * 100, 2, True), num(r.q5_mean * 100, 2, True), num(r.estimate * 100, 2, True), f"[{num(r.ci_low * 100, 2, True)}, {num(r.ci_high * 100, 2, True)}]", num(r.t, 2), f"{int(r.n):,}"] for _, r in dro.iterrows()]
th_ = "".join(f"<th class='{'l' if i == 0 else 'c'}'>{x}</th>" for i, x in enumerate(["Specification", "Q1", "Q5", "Q5 &minus; Q1", "95% interval", "t", "N"]))
A.append(f"<h2>Appendix B. Checks on the Pre-Specified Drift Measure</h2><p>This table shows why the artifact was convincing. All rows use a pre-event alpha, so every row inherits the bias described in Section VII.C, and none of them distinguishes a real reversal from a mechanical one. Returns are in percent over days [+2,+60], except where stated.</p><div class='tab'><table><thead><tr>{th_}</tr></thead><tbody>" + "".join("<tr>" + "".join(f"<td class='{'l' if i == 0 else 'c'}'>{v}</td>" for i, v in enumerate(r)) + "</tr>" for r in dr_rows) + "</tbody></table></div>")
au = lk_aud.copy(); au.columns = [c.lower() for c in au.columns]
A.append("<h2>Appendix C. Leakage Checklist</h2><p>Each item maps to an assertion or a stated report item in the pipeline. Item 4 was relabelled because stress-fold scores were not produced.</p><div class='tab small'><table><thead><tr><th class='l'>#</th><th class='l'>Check</th><th class='l'>Status</th></tr></thead><tbody>" + "".join(f"<tr><td class='l'>{r['item']}</td><td class='l'>{str(r['check']).replace('->', '&rarr;')}</td><td class='l'>{'built, not scored' if int(r['item']) == 4 else r['status']}</td></tr>" for _, r in au.iterrows()) + "</tbody></table></div>")
hyp = [("H1", "Text adds value beyond price, metadata, and accounting", f"As run, embeddings over L6, which already holds classical text: validation {ci(TXW)}; development {ci(TXD)}; locked {ci(TXL)}. Numbers-only comparison in Section VI.C", "Not supported"),
       ("H2", "Language change adds more than levels", f"Validation {ci(L('Walk', 'Change'))}; development {ci(L('Dev', 'Change'))}; locked {ci(L('Locked', 'Change'))}", "Not supported"),
       ("H3", "Validated extraction adds value", "Not run: no human labels", "Untested"),
       ("H4", "Vintage encoders differ from static ones", f"LightGBM, L7: {ci(sh.iloc[4])}", "No detectable difference"),
       ("H5", "Stacking beats the best single model", f"Development: {num(csv('dev_lifts', D).query('hypothesis == \"H5\"').value.iloc[0], 3, True)} [{num(csv('dev_lifts', D).query('hypothesis == \"H5\"').ci_low.iloc[0], 3, True)}, {num(csv('dev_lifts', D).query('hypothesis == \"H5\"').ci_high.iloc[0], 3, True)}]", "Not supported"),
       ("H6", "Richer retrieval beats hybrid retrieval", "Not run: no human labels", "Untested"),
       ("H7", "Analogue or time-series extensions add value", "Extension disabled", "Not run"),
       ("H8", "Gains shrink under timing, costs, and the lock", f"L6 &minus; L1: validation {ci(L('Walk', 'Structured'))}; locked {ci(L('Locked', 'Structured'))}. Long book on locked test {pc(bt_('locked', 'long_minus_spy_matched').ann_return, 1)} a year against the index", "Classification gain persists; no economic gain"),
       ("Drift", "SUE-conditioned drift", f"Pre-specified measure {pc(s0.estimate)} [{pc(s0.ci_low)}, {pc(s0.ci_high)}]; without alpha {pc(C4N.estimate)} [{pc(C4N.ci_low)}, {pc(C4N.ci_high)}]; calendar time {pc(CTA.alpha_59d)} (t = {num(CTA.alpha_t, 2)})", "No detectable drift; reversal is an artifact")]
tab_h = f"<div class='tab'><div class='tt'>{tn['hyp']}</div><div class='ts'>Hypotheses and outcomes</div><div class='tn'>Changes in macro-F1 for H1, H2, H4, H5, H8, with 95% intervals.</div><table><thead><tr><th class='l'>Id</th><th class='l'>Claim</th><th class='l'>Evidence</th><th class='l'>Outcome</th></tr></thead><tbody>" + "".join(f"<tr>" + "".join(f"<td class='l'>{v}</td>" for v in r) + "</tr>" for r in hyp) + "</tbody></table></div>"
concl_marker = "<h2>IX. Conclusion</h2>"
assert concl_marker in body
body = body.replace(concl_marker, tab_h + concl_marker)
apx = "".join(A)
apx = re.sub(r"\(\{(\w+)\}\)", lambda m: (used.add(m.group(1)) or "(" + ", ".join(map(str, cite_name(refs[m.group(1)]))) + ")"), apx)
if "nosek18" in used and "nosek" not in refs_html.lower():
    r = refs["nosek18"]; refs_html += f"<div class='ref'>{fmt_auth(r['authors'])}, {r['year']}, {clean_title(r['title'])}, <i>{r['journal']}</i> {r['volume']}, {r['page']}. <a href='https://doi.org/{r['doi']}'>doi:{r['doi']}</a></div>"

css = """
@page { size: Letter; margin: 1in 1in 1in 1in; }
body { font-family: 'Times New Roman', Times, serif; font-size: 11.5pt; line-height: 1.42; color: #000; }
p { margin: 0 0 8pt 0; text-align: justify; text-indent: 0; hyphens: auto; orphans: 3; widows: 3; }
a { color: #123a7a; text-decoration: none; word-break: break-word; }
h1 { font-size: 18pt; font-weight: bold; text-align: center; line-height: 1.2; margin: 34pt 0 16pt 0; }
h2 { font-size: 12.5pt; text-align: center; font-weight: bold; margin: 20pt 0 8pt 0; page-break-after: avoid; }
h2.pb { page-break-before: always; }
h3 { font-size: 11.5pt; font-style: italic; font-weight: bold; text-align: left; margin: 12pt 0 5pt 0; page-break-after: avoid; }
.titlepage { page-break-after: always; }
.author { text-align: center; font-size: 13pt; margin-bottom: 4pt; } .date { text-align: center; color: #333; margin-bottom: 24pt; }
.abstract { margin: 0 34pt 18pt 34pt; text-align: justify; } .kw { margin: 0 34pt 22pt 34pt; font-size: 10.5pt; } .thanks { border-top: 0.6pt solid #666; padding-top: 7pt; font-size: 9.5pt; line-height: 1.3; margin-top: 40pt; text-align: justify; }
.eq { display: flex; justify-content: center; position: relative; margin: 8pt 0 10pt 0; } .eqb { font-size: 11.5pt; } .eqn { position: absolute; right: 0; }
figure { margin: 10pt 0 13pt 0; text-align: center; page-break-inside: avoid; } figure img { max-width: 100%; }
figcaption { text-align: justify; font-size: 9.6pt; line-height: 1.28; margin-top: 4pt; }
.tab { margin: 11pt 0 14pt 0; page-break-inside: avoid; font-size: 9.4pt; line-height: 1.22; } .tab.small { font-size: 8.7pt; page-break-inside: auto; }
.tt { text-align: center; font-weight: bold; font-size: 10.4pt; } .ts { text-align: center; font-weight: bold; font-size: 10.4pt; margin-bottom: 3pt; } .tn { text-align: justify; font-size: 8.9pt; margin: 0 0 4pt 0; }
table { border-collapse: collapse; width: 100%; border-top: 1pt solid #000; border-bottom: 1pt solid #000; }
thead tr { border-bottom: 0.6pt solid #000; } th { font-weight: bold; padding: 2.6pt 4pt; vertical-align: bottom; } td { padding: 1.9pt 4pt; vertical-align: top; }
th.c, td.c { text-align: center; } th.l, td.l { text-align: left; } tr.sec td { font-style: italic; padding-top: 5pt; border-bottom: 0.3pt solid #999; }
.ref { padding-left: 18pt; text-indent: -18pt; margin-bottom: 5pt; font-size: 10.4pt; line-height: 1.26; text-align: left; }
ol.dev li { margin-bottom: 5pt; text-align: justify; } code { font-size: 10pt; }
"""
html = f"<!doctype html><html lang='en'><head><meta charset='utf-8'><title>{TITLE}</title><style>{css}</style></head><body>{body}{refs_html}{apx}</body></html>"
left = re.findall(r"\{[a-z_0-9]+\}", re.sub(r"<style>.*?</style>", "", html, flags=re.S))
assert not left, left[:10]
open(P / "paper.html", "w", encoding="utf-8").write(html)
txt = re.sub(r"<[^>]+>", " ", re.sub(r"<style>.*?</style>", "", body, flags=re.S)); print("body words:", len(txt.split()), "| references:", len(items), "| abstract words:", len(re.sub(r"<[^>]+>|&\w+;", " ", ABSTRACT).split()))
print("unused refs:", sorted(set(refs) - used))
