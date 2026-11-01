"""Builds the written report (HTML). Numbers in tables and charts come from the dashboard data snapshot; the prose is the narrative."""
import base64
import json

import pandas as pd

D = r"C:\Users\harsh\Downloads\sec-filings-price-impact\dashboard\data"
IMG = r"C:\Users\harsh\Downloads\sec_report_build\img"
GH = "https://github.com/harshatangirala/sec-filings-price-impact"
APP = "https://sec-filings-study.streamlit.app/"
csv = lambda n: pd.read_csv(f"{D}/{n}")  # noqa: E731
meta = json.load(open(f"{D}/meta.json"))
E = meta["eda"]

_fig = [0]


def fig(name, caption, width="100%"):
    _fig[0] += 1
    b = base64.b64encode(open(f"{IMG}/{name}.png", "rb").read()).decode()
    return (f"<figure style='width:{width}'><img src='data:image/png;base64,{b}'/>"
            f"<figcaption><b>Figure {_fig[0]}.</b> {caption}</figcaption></figure>")


def para(*t):
    return "".join(f"<p>{x}</p>" for x in t)


def link(url, text=None):
    return f"<a href='{url}'>{text or url}</a>"


# ----------------------------------------------------------------------------------------------------------- tables
def metrics_table():
    lad = csv("ladder.csv").set_index("run_id")
    dev, lock = csv("dev_results.csv"), csv("locked_results.csv")
    rows = []
    wf = {"L1": "common|L1|enet|-|T_trade|h1|tuned", "L6": "common|L6|enet|-|T_trade|h1|tuned", "L7": "common_emb|L7|lightgbm|bge_base|T_trade|h1|tuned"}
    nm = {"L1": "price only", "L6": "all the numbers", "L7": "numbers + text"}
    for m in ("L1", "L6", "L7"):
        r = lad.loc[wf[m]]
        rows.append((nm[m], "validation 2008–19", r.n_events, r.accuracy, r.macro_f1, r.ci_low, r.ci_high, r.auc_macro_ovr, r.logloss, r.ece))
    for tag, df in (("development 2020–22", dev), ("locked test 2023–26", lock)):
        for m in ("L1", "L6", "L7"):
            r = df[(df.run_id == ("dev|" if "dev" in tag else "locked|") + m)].iloc[0]
            rows.append((nm[m], tag, r.n_events, r.accuracy, r.macro_f1, r.ci_low, r.ci_high, r.auc_macro_ovr, r.logloss, r.ece))
    h = "<table><tr><th>model</th><th>period</th><th>events</th><th>accuracy</th><th>macro-F1 (95% interval)</th><th>AUC</th><th>log-loss</th><th>calibration error</th></tr>"
    for r in rows:
        h += (f"<tr><td>{r[0]}</td><td>{r[1]}</td><td>{int(r[2]):,}</td><td>{r[3]:.3f}</td><td>{r[4]:.3f} [{r[5]:.3f}, {r[6]:.3f}]</td>"
              f"<td>{r[7]:.3f}</td><td>{r[8]:.3f}</td><td>{r[9]:.3f}</td></tr>")
    return h + "</table>"


def models_table():
    rows = [
        ("LightGBM", "trained here", "tree ensemble; the all-numbers model used about 25 leaves per tree", "fast on tables with lots of mixed features", "—"),
        ("CatBoost", "trained here", "tree ensemble", "a second, different tree method as a cross-check", "—"),
        ("Elastic net", "trained here", "about 400 weights", "simple and convex, so it can't hide overfitting", "—"),
        ("ChronoBERT (26 yearly versions)", "borrowed, frozen", "about 150 million each", "each version only saw text up to its own year", "https://huggingface.co/collections/manelalab/chronobert"),
        ("BGE-base-en-v1.5", "borrowed, frozen", "about 110 million", "strong general text encoder; also does the passage search", "https://huggingface.co/BAAI/bge-base-en-v1.5"),
        ("BGE reranker-base", "borrowed, frozen", "about 280 million", "puts the best retrieved passages first", "https://huggingface.co/BAAI/bge-reranker-base"),
        ("Qwen2.5-7B-Instruct", "borrowed, frozen", "about 7.6 billion", "open model I can run myself, no paid API", "https://huggingface.co/Qwen/Qwen2.5-7B-Instruct"),
        ("SEC-BERT", "planned, dropped", "about 110 million", "trained on 10-Ks; left out to save GPU time", "https://huggingface.co/nlpaueb/sec-bert-base"),
        ("FinBERT-tone", "planned, dropped", "about 110 million", "finance tone classifier; left out to save GPU time", "https://huggingface.co/yiyanghkust/finbert-tone"),
    ]
    h = "<table><tr><th>model</th><th>status</th><th>size (approximate)</th><th>why I picked it</th><th>where to find it</th></tr>"
    for a, b, c, d, u in rows:
        h += f"<tr><td><b>{a}</b></td><td>{b}</td><td>{c}</td><td>{d}</td><td>{('<a href=' + repr(u).replace(chr(39), chr(34)) + '>Hugging Face</a>') if u != '—' else '—'}</td></tr>"
    return h + "</table>"


def sources_table():
    rows = [
        ("SEC EDGAR", "all 10-K, 10-Q and 8-K filings for the 150 firms, plus exhibits", f"{E['filings']:,} filings", "public, free"),
        ("EDGAR XBRL", "balance-sheet and income numbers, as first reported", f"{E['xbrl_facts']:,} facts", "public, free"),
        ("Earnings press releases", "the EPS figure in each release, to build an earnings surprise", f"{E['results_releases']:,} results releases", "parsed from EDGAR exhibits"),
        ("Daily prices", "returns, volatility, the benchmark, the backtest", f"{E['price_bars']:,} daily bars", "my own price files, filled in with Yahoo Finance"),
        ("Fama-French factors", "to measure what is 'abnormal', and for the alpha test", "daily factors", "Kenneth French's data library"),
        ("S&P 500 membership list", "to choose the firms", "history since 1996", "a public GitHub list, pinned to one commit"),
        ("Loughran-McDonald dictionary", "counts of positive and negative finance words", "1993–2025 edition", "free for academic use"),
        ("Hugging Face SEC corpus", "only to pick firms and to check my parser (never as study text)", "10-Ks up to 2020", "JanosAudran/financial-reports-sec"),
    ]
    h = "<table><tr><th>source</th><th>what I used it for</th><th>how much</th><th>access</th></tr>"
    for r in rows:
        h += "<tr>" + "".join(f"<td>{x}</td>" for x in r) + "</tr>"
    return h + "</table>"


# ----------------------------------------------------------------------------------------------------------- numbers used in text
bd = pd.read_parquet(f"{D}/backtest_daily.parquet")
x = bd[(bd.block == "locked") & (bd.lag == 0) & (bd.cost_bps == 10)].sort_values("date")
cum = lambda c: ((1 + x[c].fillna(0)).cumprod().iloc[-1] - 1) * 100  # noqa: E731
BT_LONG, BT_SPY, BT_EXC = cum("long_ret"), cum("spy_matched"), cum("long_excess_spy")
sk = csv("skill.csv")
L6L = sk[(sk.period == "locked") & (sk.model == "L6")].iloc[0]
li = csv("lifts.csv")
TL = li[(li.period == "locked") & li.comparison.str.startswith("L7")].iloc[0]
tun = csv("tuning.csv").dropna(subset=["default_mean_f1", "tuned_mean_f1"])
gain = tun.tuned_mean_f1 - tun.default_mean_f1
rep = csv("reproducibility.csv")
lg = rep[rep["head"] == "lightgbm"].iloc[0]
bt = csv("backtest.csv")


def btrow(block, leg, lag, cost=10):
    r = bt[(bt.sp_bucket == "Combined") & (bt.block == block) & (bt.leg == leg) & (bt.lag == lag) & (bt.cost_bps == cost)]
    return r.iloc[0]


lk_long, lk_exc, lk_ls = btrow("locked", "long_only", 0), btrow("locked", "long_minus_spy_matched", 0), btrow("locked", "long_short_hypothetical", 0)
dv_long, dv_exc, dv_ls = btrow("dev", "long_only", 1), btrow("dev", "long_minus_spy_matched", 1), btrow("dev", "long_short_hypothetical", 1)

# ----------------------------------------------------------------------------------------------------------- the report text
H = []
a = H.append

a(f"""
<div class='cover'>
<div class='kicker'>A project write-up · November 2026</div>
<h1>Can a company's own filing tell you what its stock will do next?</h1>
<p class='sub'>What I built, what I found, and what went wrong on the way. A study of 150 S&amp;P 500 stocks, 2000 to 2026, using 64,213 SEC filings.</p>
<p class='by'>Krishna Harsha Tangirala</p>
<table class='links'>
<tr><td>Code and data</td><td>{link(GH)}</td></tr>
<tr><td>Interactive dashboard</td><td>{link(APP)} <span class='small'>(free apps go to sleep; if you see a "wake this app up" button, press it and wait a minute)</span></td></tr>
<tr><td>Models used</td><td>{link('https://huggingface.co/collections/manelalab/chronobert', 'ChronoBERT')}, {link('https://huggingface.co/BAAI/bge-base-en-v1.5', 'BGE-base')},
{link('https://huggingface.co/BAAI/bge-reranker-base', 'BGE reranker')}, {link('https://huggingface.co/Qwen/Qwen2.5-7B-Instruct', 'Qwen2.5-7B')} (all on Hugging Face)</td></tr>
</table>
</div>
""")

a("<h2>The short version</h2>")
a(para(
    "Public companies in the US have to file reports with the SEC. There are yearly ones (10-K), quarterly ones (10-Q), and short \"something just happened\" ones (8-K) that cover things like earnings, a merger or a CEO leaving. "
    "I wanted to know whether you could read one of these the moment it appears and say anything useful about how the stock will react. "
    "And a second question sat underneath that one: do the <i>words</i> in a filing add anything, once you already know the numbers?",
    f"The short answers are these. The reaction is partly predictable, but only the <i>size</i> of it. On the final test period the best model scored a macro-F1 of {L6L.macro_f1:.3f}, against about 0.33 for pure luck. "
    f"It can tell which filings will move a stock (AUC {L6L.move_auc:.2f}) but it is a coin flip on <i>which way</i> (AUC {L6L.direction_auc:.2f}). "
    f"The text did not help at all: adding language models changed the score by {TL.lift:+.3f}, with an interval that sits comfortably around zero. "
    "And one result surprised me. After very good earnings, stocks did slightly <i>worse</i> over the following months, not better. I spent a long time checking whether that was a mistake, and I explain below why I think it is mostly real in this sample, and why I still wouldn't trust it too much.",
    "Most of what follows is about how I got there, because that is where the actual work was. I found several bugs that would have made the results look better than they were, and I'd rather show those than hide them."))

# --------------------------------------------------------------------------------------------------------------
a("<h2>1. The idea, and the one rule I kept to</h2>")
a(para(
    "There is an obvious way to do this badly. You take every filing, look up what the stock did around it, fit a model, and get a lovely score. The problem is that you have quietly used information that nobody had at the time. "
    "If a filing was published at 4:30 in the afternoon, you can't use that day's closing price as an input, and you can't trade at that close either. So the one rule I kept to is this: <b>act like a trader sitting at the screen the moment the filing becomes public</b>. "
    "Every input has to be something that trader could have known, and the thing I predict has to be something they could have actually captured.",
    "I split the work into three tracks and tried hard not to mix them. Track A is an event study: how do prices move around filings, nothing more. That is description, not prediction. "
    "Track B is the prediction problem. Track C asks whether the facts a language model pulls out of a filing match what a human would pull out, and it is judged on the text, never on prices. "
    "Track C turned out to need hand-labelled examples that I don't have yet, so it only half exists. I'll be clear about that when we get there.",
    "The questions, in the order I wanted to answer them:"))
a("<ol><li>Can we predict the next reaction at all?</li><li>Does the text add anything beyond numbers that are already public (price history, company size, accounting)?</li>"
  "<li>Does the well-known post-earnings drift show up in this data?</li><li>Could any of it make money?</li></ol>")
a(para("I also wrote these down, along with the metric and the pass or fail rules, before looking at the final test. That is called pre-registering, and it is boring, and it is the thing that stops you from changing the question after you've seen the answer."))
a(fig("A01", "The whole pipeline: 25 stages in three parts. Part 1 ran on my laptop, Part 2 on a Colab GPU, and Part 3 back on CPU. The two orange boxes ran but couldn't produce a result because they need human-labelled data I don't have."))

# --------------------------------------------------------------------------------------------------------------
a("<h2>2. The data</h2>")
a(para(
    f"Everything comes from free, public sources, and I pinned the version of each so someone else can rerun it. The table lists what I used.",
))
a(sources_table())
a(para(
    f"I picked 150 companies that were members of the S&amp;P 500 at some point since 2000, using a fixed random seed and spreading them across 12 industries. They came from {E['eligible_firms']} firms that had prices in my files and at least one 10-K in the Hugging Face corpus. "
    f"That gave {E['events']:,} <b>events</b>: {E['events_by_type']['8K']:,} 8-Ks, {E['events_by_type']['periodic_10Q']:,} 10-Qs and {E['events_by_type']['periodic_10K']:,} 10-Ks. If a firm filed several 8-Ks on the same day I merged them into one event. "
    f"Events run from January {E['event_first'][:4]} to July 2026.",
    "I started out planning 500 stocks, 353 from the S&amp;P 500 and 147 that were not. The first half of that run finished. Then, while checking it, I found a bug in how I looked up accounting numbers (more on that later), and rather than re-run everything at the larger size I dropped to the 150-stock version, which I could finish properly. "
    "That is why the final study is 150 S&amp;P stocks only, and why I don't compare S&amp;P and non-S&amp;P firms in the end."))
a(fig("filings_by_year", "Filings downloaded per year. Quarterly and yearly reports take off in the early 2000s, and most of the volume is 8-Ks."))
a(fig("sectors", "The 150 firms by Fama-French industry. \"BusEq\" is business equipment (software, chips, hardware) and \"Other\" is a grab bag.", "62%"))
a(para(
    "<b>The weakness I have to flag right here.</b> All 150 firms still exist today. Companies that were delisted, bought, or went bust are not in the sample. That is survivorship bias, and it matters most for the earnings-drift result later. "
    "My data covers about a third of the 1,094 tickers that were ever in the index during these years, and I couldn't fix that with free price data."))

# --------------------------------------------------------------------------------------------------------------
a("<h2>3. Turning filings into text a model can read</h2>")
a(para(
    f"The raw filings are messy HTML, sometimes ancient SGML, full of tables and page numbers. Getting from there to clean text was a large part of the first week. I'll go through what happens, in order.",
    f"<b>Download and clean.</b> I downloaded {E['filings']:,} filings from EDGAR politely (a few requests a second, with my contact details in the header, as the SEC asks). {E['filings_parsed_ok']:,} parsed cleanly, and 280 had none of the sections I wanted. I turned HTML into plain text and flattened tables into rows, since the earnings numbers I wanted are in tables. That meant flattening {E['tables_flattened']:,} of them.",
    "<b>Keep only the parts that carry news.</b> A 10-K is mostly boilerplate. I kept the business description, risk factors, MD&amp;A and market-risk sections from 10-Ks; MD&amp;A, market-risk and risk-factor sections from 10-Qs; and the item text plus the press-release exhibit from 8-Ks. "
    "Finding these sections is harder than it sounds, because the table of contents repeats every heading. My rule was to take the candidate start that has the longest span before the next heading. That left "
    f"{E['sections']:,} sections.",
    f"<b>Hide the company's name.</b> I replaced the firm's own names, former names and ticker with a placeholder, {E['firm_name_masks']:,} name mentions and {E['ticker_masks']:,} ticker mentions in all. Without this, a model can quietly learn \"this is Apple\" instead of \"this text sounds bad\", and that is a form of leakage.",
    f"<b>Drop number-heavy lines, cap the length, and cut into chunks.</b> Each section was chopped into 350-word windows that overlap by 50 words, giving about {E['chunks_est']:,} chunks, roughly {E['tokens_est'] / 1e6:,.0f} million tokens. The same chunks went to every model."))
a(fig("sections", "How many sections of each type I kept, and how long a typical one is. 10-K sections are long (thousands of tokens), 8-K item text is short."))
a(fig("A02", "From sources to features. The blue boxes ran on my laptop, the yellow one needed the GPU."))
a(para("I also built some old-fashioned word features, because I wanted a fair baseline for the fancy models to beat. These are counts of positive and negative finance words (Loughran-McDonald), how much a filing's language changed since the same firm's previous one, and how much of it was new. "
       "None of these learn a vocabulary from the future. The change features use a fixed hashing trick, and where I did fit something like TF-IDF, I refitted it inside each training fold only."))

# --------------------------------------------------------------------------------------------------------------
a("<h2>4. What counts as &quot;the stock reacted&quot;</h2>")
a(para(
    "Every event carries four timestamps: when the SEC accepted the filing, when I assume it was public (five minutes later), when a trader could decide (thirty minutes after that), and the next time the market opens. "
    f"Exact acceptance times come from the filing header for {E['ts_quality']['A'] / sum(E['ts_quality'].values()):.0%} of events. The SEC's own JSON feed turned out to follow two different time conventions mixed together, which I only noticed because I checked it against the headers on a sample. "
    "The five and thirty minute gaps are assumptions, and I say so wherever the results appear.",
    "The thing I predict is the stock's <b>abnormal return</b> in the first tradable session: the stock's return from the open to the close, minus what its market sensitivity says it should have done given what the S&amp;P 500 did over exactly the same hours. "
    "I turn that into three classes. A move bigger than half the stock's own typical daily swing is Up or Down, and anything smaller is Sideways. Using each stock's own volatility keeps quiet utilities and wild tech names comparable. "
    "The result is roughly 29% Down, 41% Sideways and 30% Up."))
a(fig("target", "Left: abnormal returns on the next session cluster tightly around zero, with fat tails. Right: how the three classes come out."))
a(fig("when_filed", "More than half of filings land after the market closes. That is why the trade happens at the next open rather than at the filing time.", "70%"))

# --------------------------------------------------------------------------------------------------------------
a("<h2>5. The features, and the leaks I went looking for</h2>")
a(para(
    "Each event gets 128 features in seven groups: 19 about price and market state, 68 metadata (what type of event, which 8-K categories, which sections exist, the industry, the calendar, the time of day), 10 accounting numbers including an earnings-surprise score, "
    "18 word-tone counts, 11 change-since-last-filing measures, and one novelty score. Each feature carries the latest time of any information it used, and a check runs on every single row that this time is no later than the decision time. "
    f"The count of violations was {E['availability_violations']}.",
    "The earnings surprise (SUE) deserves a sentence. I don't have analyst forecasts, so I defined \"surprise\" as this quarter's earnings per share minus the same quarter last year, divided by how variable that gap has been in the past. It is a crude version of the real thing, and I'll come back to why that matters.",
    "Then I tried to catch myself cheating. I wrote a list of 24 leakage checks, each with an assertion in the code, and planted four deliberate leaks (a feature built from future returns, an earnings reaction used for its own event, a surprise taken from a later filing, and a future document slipped into retrieval) to see whether the audit would notice. All four were caught. "
    "More usefully, the checks caught three real problems of my own while I was building:"))
a("<ul><li><b>Same-day 8-Ks.</b> When several 8-Ks arrived on one day, I had clocked the merged event at the first filing, so text and numbers from later filings leaked in. An assertion failed, and I moved the clock to the last acceptance.</li>"
  "<li><b>A 10-Q hiding in an earnings surprise.</b> A quarterly report filed minutes after the earnings release was feeding that release's own surprise history. Another assertion caught it.</li>"
  "<li><b>The stale balance sheet.</b> The worst one, and it deserves its own paragraph below.</li></ul>")
a(para(
    "A filing reports this year's balance sheet and last year's at the same moment. My lookup for \"the latest total assets known at decision time\" sorted by publication time only, so it picked between the two more or less by accident. "
    "It didn't look into the future, so no alarm went off. But I found it by comparing the 150-stock run with the earlier 500-stock run on the same firms: events and returns matched exactly while six accounting features did not, in a third to a half of rows. "
    "Checked against the raw data, the old code returned last year's figure in 52% of a random sample of events. After the fix it was wrong in 0 of 1,500. This is what made me discard the earlier 500-stock results."))

# --------------------------------------------------------------------------------------------------------------
a("<h2>6. How I tested it</h2>")
a(para(
    "A random split would be wrong here, because market conditions in 2015 tell you about 2014 and 2016. So I only ever train on the past. The pipeline trains on everything up to 2008 and tests on 2008-2009, then trains up to 2010 and tests on 2010-2011, and so on for six two-year folds ending in 2019. "
    "The 2020 to 2022 period is a separate development set that I used to choose the final model. The 2023 to July 2026 period is the locked test. I scored it once, after the final model was frozen and its settings were hashed. Events whose outcome window crossed a boundary are dropped, and there is a 90-day gap between blocks."))
a(fig("splits", "How the 26 years were used. The first folds start in 2008 because the model needs at least eight years of history to learn from.", "92%"))
a(para(
    "<b>The metric.</b> My main number is macro-F1: the average of how well the model does on Down, Sideways and Up, weighting the three equally. It stops a model scoring well just by predicting the biggest class. I also report accuracy, AUC (how well the probabilities rank events), log-loss and calibration error (do the stated probabilities match reality), and every interval comes from resampling whole months, since events in the same month share market shocks.",
    f"<b>What a score of 0.38 means.</b> This tripped me up at first. The report originally compared scores to a \"majority class\" baseline of 0.19, which makes everything look great. But a model with balanced predictions gets about 0.33 by luck on three roughly equal classes. I measured that directly by shuffling a model's own predictions against the true labels (0.33), so that is the line to read scores against."))
a(fig("chance", "Macro-F1 for three models in three periods. The dashed line is what luck gives you."))

# --------------------------------------------------------------------------------------------------------------
a("<h2>7. The models: what I trained, what I borrowed</h2>")
a(para(
    "There are two quite different kinds of models in this project, and I don't want to blur them. The ones that make the predictions are small, classical models that I trained on my own data. The big neural models are borrowed and frozen: I use them only to turn text into numbers or to answer questions, and I never changed their weights."))
a(models_table())
a(para(
    "For the trained models I used LightGBM, CatBoost and an elastic-net logistic regression, because they are the standard tools for tables of mixed features and they are quick to tune. Using three different ones was a cross-check as much as anything else: if all three agree, the result is not an accident of one library. "
    "The elastic net was a good conservative baseline. Its optimisation problem is convex, so I could even verify it found the global optimum, which you can't say for trees. It ended up being the best model on the all-numbers rung.",
    "I tuned settings with Optuna on the three most recent validation folds, then refit on every fold. The class weights balance the three classes and also down-weight events that overlap in time. Predicted probabilities are corrected back to the real class frequencies, and the final class is whichever has the highest probability relative to its frequency.",
    f"One honest observation about tuning. I expected it to matter more than it did. Across 32 combinations of rung and model, the median change in the validation score from tuning was {gain.median():+.3f}, and only {(gain > 0).mean():.0%} improved at all. The few big improvements were the elastic net, whose default settings were bad. "
    "The search budgets were also tiny (6 to 15 trials), smaller than the ten random starts Optuna uses by default, so several rungs ended up with identical settings. I would not call any of these models \"fully tuned\"."))
a(fig("tuning", "Almost nothing changed when I tuned. A few elastic-net models improved a lot, which is the right-hand bump.", "62%"))
a(para("The experiment is built as a ladder, so I can see exactly what each kind of information adds. Each rung gets the one before plus something new, always on identical events."))
a(fig("A03", "The ablation ladder. Green rungs only use information that was clearly available at the time, red ones use a model whose training cutoff is unknown."))
a(fig("ladder", "Validation results for every rung and model. Metadata (L2) is the strongest single group. Accounting alone (L3) is the weakest, and putting everything together (L6) is best."))
a(para("Two rungs surprised me. Text vectors on their own (L5) come close to the word features, a little below them, so the language models do contain some real signal. And the elastic net got much <i>worse</i> when I added the text vectors to everything else (0.39 down to 0.35). A regularised linear model drowning in hundreds of correlated extra columns is not a shock in hindsight, but it was a good reminder that more features is not more information."))

# --------------------------------------------------------------------------------------------------------------
a("<h2>8. The pre-trained models, and where the embeddings come from</h2>")
a(para(
    "People sometimes ask where the embeddings were downloaded from. They weren't. I downloaded the <i>models</i> from Hugging Face, at fixed versions, and ran my own filing text through them on a Colab GPU. The embeddings exist only for my 60,030 filing events and never saw a price.",
    "<b>ChronoBERT</b> is the one I'm happiest with. It comes as 26 yearly versions, and each was trained only on text written up to its date. For an event in year Y I use the version trained to the end of year Y-1, so the encoder cannot know anything from after the filing. If a model has read the future, it can look clever for the wrong reason, and this removes that worry. "
    "<b>BGE-base</b> is a strong general-purpose encoder with no stated cutoff, so it only counts on the less strict leaderboard. It also drives the passage search I describe later.",
    "Each of the 900,000 chunks becomes 768 numbers. I averaged and took the maximum over a section's chunks, then combined sections into one vector of 1,536 numbers per event, and also stored how far the vector moved compared with the same firm's previous filing of the same type. Before using them I shrank the vectors with PCA, fitted inside each training fold. "
    "Embedding took 56 minutes with ChronoBERT and 39 minutes with BGE on the Colab GPU.",
    "Two other encoders, SEC-BERT and FinBERT, were in my original plan. I dropped them to fit the GPU budget. That was a time decision, not a result: I have no evidence they would have done worse, and a domain model like SEC-BERT might have done better."))

# --------------------------------------------------------------------------------------------------------------
a("<h2>9. Fine-tuning: what I planned and why I didn't do it</h2>")
a(para(
    "I had a fine-tuning stage in the plan. The idea was to take the two best encoders and train them end to end on my labels, with a grid over learning rate, batch size and weight decay. But I made it conditional. Fine-tuning is expensive, and it multiplies the number of configurations you try, which is how people fool themselves. "
    "So the rule was: fine-tune only if adding text to the numbers improves the score by at least 0.01, with an interval that excludes zero.",
    f"It didn't come close. Across the validation folds the change was −0.001 with an interval from −0.007 to +0.005, and on the development and locked periods the estimates were +0.007 and −0.001, neither with an interval that excluded zero. So the gate stayed closed and <b>no encoder was fine-tuned</b>. The honest label for the language side of this project is \"frozen embeddings, plus a retrieval and extraction pipeline\", not \"fine-tuned models\". "
    "If you do want to see what fine-tuning would have looked like, the code is in the repository and runs on a smoke test, but it has never run on the real data."))

# --------------------------------------------------------------------------------------------------------------
a("<h2>10. The RAG experiment, and why its results are not in the results</h2>")
a(para(
    "The other use of a language model was to pull specific events out of a filing: did the company give guidance, record an impairment, announce a restructuring, a buyback, a dividend, a merger, a lawsuit, a change of management. Embeddings blur all of that into one vector, and I wanted facts I could check.",
    "The pipeline works like this. A filing's text is cut into chunks, and I search them two ways at once, by keywords and by embedding similarity, then merge the two rankings and let a reranking model sort the top thirty. The best eight passages go to Qwen2.5-7B, which answers twelve questions in a fixed JSON format with a quote and a chunk id for every answer. "
    "Then a set of guards checks the answer. Every quote has to appear word-for-word in the chunk it cites, any number has to appear in its evidence, and anything that fails is blanked out.",
    "It ran on 149 filings from the development period, at about 47 seconds each, two hours in total. And that is where the story stops. To say whether it is any good I need human-labelled examples to compare against, and I haven't made them. The templates are pre-filled and ready, but I haven't annotated them. "
    "So I can't report accuracy for the extraction, I couldn't test whether the extracted events improve the predictions, and I could not train the smaller student model I'd planned. The chart below only shows how often the model said each event was present. Treat it as a description of what the system produces, not as evidence that it works."))
a(fig("A04", "The retrieval and extraction system. Solid boxes ran. Dashed grey boxes are built but waiting on human labels."))
a(fig("rag", "How often Qwen said each field was present in 141 filings that returned a valid answer (8 returned nothing usable). Unchecked output.", "78%"))

# --------------------------------------------------------------------------------------------------------------
a("<h2>11. Results</h2>")
a(para(
    "This is the main table. \"All the numbers\" is the best model with no text, and \"numbers + text\" adds the embeddings. The development and locked periods are the ones where the model never saw the data it was scored on."))
a(metrics_table())
a(para(
    f"A few things to take from it. The scores are modest but real, and they hold up: validation, development and the final test all land around 0.38 to 0.40. "
    f"The calibration error is small for the all-numbers model ({lock_ece:.3f} on the locked test), so when it says 40%, something like 40% happens. And the log-loss is only slightly better than just using the class frequencies (about 1.096), which is a fair summary of how much there is to find here."
    if (lock_ece := csv('locked_results.csv').query("run_id == 'locked|L6'").iloc[0].ece) is not None else ""))
a(fig("move_direction", "The most important chart in the project. The model is good at telling which events will produce a big move, and no better than a coin at telling which direction. An AUC of 0.5 means no information."))
a(para(
    "That split explains what the model is actually doing. Metadata like event type, time of day, and the stock's recent volatility predict how much the stock will react. An earnings release after the close for a volatile stock is going to move, whatever the content. "
    "Whether the move is up or down depends on whether the news is better or worse than expected, and my numbers contain almost nothing about that. The accounting-based surprise is the nearest thing, and as the earnings section shows, it behaves strangely."))
a(fig("confusion", "The all-numbers model on the locked test. It finds the Sideways events best (53%) and does little better than chance on telling Down from Up.", "52%"))
a(para(
    f"<b>Did the text help?</b> No. The comparison I care about is the same model with and without the language-model embeddings, scored on identical events. On the locked test, adding text changed macro-F1 by {TL.lift:+.3f} (95% interval {TL.ci_low:+.3f} to {TL.ci_high:+.3f}). "
    "On the earlier folds and the development period it was equally flat. The yearly ChronoBERT versions did not behave differently from the single BGE model either, and a stacked ensemble did not beat the best single model. "
    "The numbers beat price alone by about 0.03 in the validation and locked periods, so the structured features are doing real work. The text is not."))
a(fig("lifts", "Every comparison I pre-registered, with its interval. Where the bar crosses the vertical line, I can't tell the difference from zero."))

# --------------------------------------------------------------------------------------------------------------
a("<h2>12. The earnings drift puzzle</h2>")
a(para(
    "The classic finding in this area is that stocks keep drifting in the direction of an earnings surprise for weeks afterwards. Good news, then more gains. So I sorted earnings reports into five groups by the size of the surprise and tracked the average abnormal return for the next 2 to 60 trading days.",
    "It went the other way. On announcement day it behaves the way you would hope: the best-news group rose 0.81% and the worst-news group fell 0.54%. Then, over the following months, the best-news group <i>lost</i> 1.85% and the worst-news group <i>gained</i> 1.40%."))
a(fig("sue", "Left: the immediate reaction goes in the direction of the news. Right: the later drift goes the other way, and it gets steadily more negative as the news gets better."))
a(para(
    "My first reaction was that I had got a sign wrong somewhere, so I tested it. The announcement-day move has the right sign, so the surprise is not flipped. The reversal is there in 2000 to 2009 (−4.4% between best and worst news), 2010 to 2019 (−2.0%) and 2020 to 2026 (−4.3%), and it survives when I keep only events with no other filing nearby (−2.1%). "
    "It rises smoothly from group to group, which is not what noise looks like. The formal test, which uses a stricter surprise measure built from the press-release figures (3,825 reports), gives −2.6% with an interval from −3.5% to −1.7%. The chart above uses a broader sample of 9,359 reports, which is why its gap is larger, about −3.3%.",
    "A fair objection is that good news should push a stock up, not down. It did, on the day. Here is how the best-news reports break down over the next two months:"))
a(fig("twobytwo", "The 1,908 best-news earnings reports, split by what the stock did on the day and what it did over the next two months. 55% fell afterwards, and 31% rose first and then gave back about 10%.", "72%"))
a(para(
    "So it is a tilt in the averages, not a rule: 45% of those stocks went on to rise. Why would it tilt this way? I have four candidate reasons and I did not test any of them. "
    "First, my surprise compares earnings with the same quarter last year, not with what analysts expected, so a result can beat last year and still disappoint. Second, prices can overreact to a headline and settle back, and the worst-news group shows the mirror image. Third, these are large, closely followed companies where good news is priced in within hours. "
    "Fourth, survivorship. Companies that got bad news and kept falling were often delisted, and they aren't in my data. That alone makes the worst-news group look better afterwards and could manufacture part of this pattern.",
    "The wording I'm comfortable with is this. I did not find classic post-earnings drift in this survivor-only sample, and the data point the other way. That is a finding about this sample, not a refutation of the theory. To settle it I'd add delisted firms and use analyst forecasts for the surprise."))

# --------------------------------------------------------------------------------------------------------------
a("<h2>13. Could you trade it?</h2>")
a(para(
    "I added a simple backtest as a sanity check, not as a strategy. The signal is the probability of Up minus the probability of Down from the frozen model. Go long the top fifth, short the bottom fifth, enter at the first open after the decision time, and charge 10 basis points a side (20 in a second run), plus a borrow cost on shorts. I also ran a version with a one-session delay.",
    f"On the development years it looked pleasant: with the one-session delay the long book returned {dv_long.ann_return:.1%} a year, {dv_exc.ann_return:+.1%} a year ahead of the S&amp;P 500 on the same days. On the locked test it fell apart. "
    f"The long book earned {lk_long.ann_return:.1%} a year, the long-minus-index line {lk_exc.ann_return:+.1%} a year, and the hypothetical long-short {lk_ls.ann_return:+.1%}. Over the 3.6 years the long book gained about {BT_LONG:.0f}% while the index, on the same days, gained about {BT_SPY:.0f}%.",
    "That is a useful lesson in itself. A good-looking result on the set you used to choose the model is not the same thing as a result on data you haven't touched, and the locked test did what it was there to do."))
a(fig("backtest", "The locked test period, with no delay and 10 bps per side. Shorts are hypothetical, there is no capacity or delisting model, and this is a sanity check rather than a strategy."))

# --------------------------------------------------------------------------------------------------------------
a("<h2>14. Things that went wrong</h2>")
a(para(
    "I want this part to be useful, not apologetic, because it is where I learned the most. Besides the three data bugs in section 5, these are the ones I remember:"))
a("<ul>"
  "<li><b>My own benchmark.</b> The first time I ran the GPU part on Colab it refused to start, saying the embeddings would take 19 GPU-hours against a 10-hour budget. The cause was my speed test: it timed a cold start, including the one-off cost of waking the GPU up. After adding a warm-up the projection was under an hour (the real run took 1.65 hours).</li>"
  "<li><b>A regular expression that never finished.</b> A pattern for table rows had a nasty backtracking case, and three worker processes sat at full CPU for half an hour on one filing. I now run an adversarial timing test against every parsing pattern.</li>"
  "<li><b>63 filings lost to server errors.</b> A burst of HTTP 503 responses from the SEC dropped 63 filings quietly. I added a retry pass that recovers them on the next run, and all 63 came back.</li>"
  "<li><b>A p-value of zero.</b> In the drift-robustness check a p-value of exactly zero was being read as missing. That quietly marked a real effect as non-robust. Fixed, with a test.</li>"
  "<li><b>The laptop went to sleep.</b> A long end-to-end test hung for two hours. Windows had put the machine into standby, and when it woke up the timeout fired. The test was fine, the setup was not.</li>"
  "<li><b>Drive, and Streamlit.</b> Google Drive on Windows would not sync thousands of small files, so I uploaded zips through the browser. Getting the dashboard online took a second attempt: the first deployment stalled, so I added a requirements file listing only the six packages the dashboard needs instead of the whole project's.</li>"
  "</ul>")
a(fig("compute", "Wall-clock time per stage. The long bars are the tuning of the tree models (S11, two hours, on my laptop), the embeddings (S14) and the extraction run (S16).", "88%"))
a(para("The pipeline has 114 automated tests, and each of the problems above has one that fails on the old code."))

# --------------------------------------------------------------------------------------------------------------
a("<h2>15. What the results do and don't show</h2>")
a(para(
    "<b>What I'm comfortable claiming.</b> On 150 surviving S&amp;P 500 stocks, with careful point-in-time controls, adding language-model embeddings of filing text to price, metadata and accounting features gave no out-of-sample gain. The models beat chance mostly by predicting <i>whether</i> a stock moves, not which way. "
    "The post-earnings drift reverses in this sample, with a survivorship caveat I can't remove.",
    "<b>What I'm not claiming.</b> That text never helps. I only tested two encoders with small tuning budgets, on 150 big companies, with daily prices. I'm not claiming the reversal is a real market effect, or that anyone could trade on it, and the backtest says otherwise. And I'm not claiming the language-model extraction works, because I haven't scored it.",
    f"<b>Known limits.</b> The universe is survivors only, with daily bars only. The choice between the two top models was a statistical tie, though the rule froze the one with text: scored in the same run the model without text was slightly ahead, and LightGBM's predictions changed for about {1 - lg.same_class_share:.0%} of events when I re-scored the same model on a different machine. "
    "The firm-held-out and sector-held-out stress tests are built but I never ran their scores. The GPU model used on Colab was not recorded in the manifest. And everything about Track C is waiting for human labels."))

a("<h2>16. What I'd do next</h2>")
a("<ol><li>Run the 500-stock version, including non-S&amp;P firms, on the corrected accounting code. Tighter intervals and a chance to compare index and non-index stocks.</li>"
  "<li>Add delisted companies, which is the only real fix for the survivorship problem.</li>"
  "<li>Label a few hundred filings by hand so the extraction can be scored, and then see whether extracted events beat plain embeddings.</li>"
  "<li>Make LightGBM deterministic and compare the final candidates inside one run, preferring the strict model on a tie.</li>"
  "<li>Run all four encoders, and consider real analyst forecasts for the earnings surprise.</li></ol>")

a("<h2>17. Look at it yourself</h2>")
a(para(
    f"Everything is public. The repository ({link(GH)}) has the pipeline, the configuration, the decision log (78 entries explaining each choice and bug), the notebook for Colab, the dashboard and its data, the automated tests, the slide deck, this report, and the analysis scripts behind the earnings-drift numbers. "
    f"The interactive dashboard is at {link(APP)}. It walks the same story as a clickable flowchart of 16 steps with the charts behind each one, plus an 18-slide deck. If the app has gone to sleep, press the button and give it a minute.",
    "To rerun it: open the notebook in Colab, set the profile and stock count in the first cell, and run all. Finished stages are skipped, so a disconnect costs nothing. You do not need any paid API or key, only a contact email for the SEC if you re-download."))

a("<h2>Appendix: where each thing is</h2>")
a("<table><tr><th>what</th><th>where</th></tr>"
  f"<tr><td>pipeline code</td><td>{link(GH + '/tree/main/src/secprice')}</td></tr>"
  f"<tr><td>decision log (every choice and bug)</td><td>{link(GH + '/blob/main/docs/decision_log.md')}</td></tr>"
  f"<tr><td>Colab notebook</td><td>{link(GH + '/tree/main/notebooks')}</td></tr>"
  f"<tr><td>dashboard code and data</td><td>{link(GH + '/tree/main/dashboard')}</td></tr>"
  f"<tr><td>automated report, figures, results tables</td><td>{link(GH + '/tree/main/reports')}</td></tr>"
  f"<tr><td>this report, the slide deck, and how I built them</td><td>{link(GH + '/tree/main/docs/write-up')}</td></tr>"
  f"<tr><td>earnings-drift analysis scripts</td><td>{link(GH + '/tree/main/analysis')}</td></tr></table>")
a("<p class='small'>Parameter counts are approximate, taken from model cards and file sizes. Nothing here is investment advice.</p>")

REFS = [
    ("He, S., Lv, L., Manela, A., &amp; Wu, J. (2026). Chronologically consistent large language models. <i>Journal of Financial Economics</i>, 185, 104374.",
     "https://doi.org/10.1016/j.jfineco.2026.104374",
     "The ChronoBERT paper. The models are trained only on text that existed at each date, and the authors find look-ahead bias in news-based return forecasts to be modest. It is the reason I used a different yearly vintage for each event year (section 8)."),
    ("Fink, J., Palan, S., &amp; Theissen, E. (2024). Earnings autocorrelation and the post-earnings-announcement drift: Experimental evidence. <i>Journal of Financial and Quantitative Analysis</i>, 59(6), 2799-2837.",
     "https://doi.org/10.1017/S0022109023000881",
     "A lab experiment showing the drift appears whether or not earnings surprises are correlated, and is stronger when they are. Useful background for the drift puzzle in section 12."),
    ("Karolyi, S., Ruchti, T., &amp; Truong, P. (2025). Number processing constraints and earnings news. <i>Management Science</i>, 71(3), 2413-2442.",
     "https://doi.org/10.1287/mnsc.2023.01722",
     "Finds investors react less in the short run to earnings news for stocks with a high earnings-per-share level, which strengthens the later drift. A reminder that the drift depends on how the surprise is measured and who the firms are."),
    ("Zhang, S., Gregoriou, A., &amp; Wu, H. (2024). Asymmetric post earnings announcement drift and order flow imbalance: The impact on stock market returns. <i>International Review of Financial Analysis</i>, 94, 103316.",
     "https://doi.org/10.1016/j.irfa.2024.103316",
     "An event study of earnings announcements in which the drift differs between good-news and bad-news cases and is linked to order flow. Close in spirit to my good-news-then-falls count in section 12."),
    ("Lan, Q., Xie, Y., Mi, X., &amp; Zhang, C. (2024). Post earnings announcement drift: A simple earnings surprise measure, the medium effect of investor attention and investing strategy. <i>International Review of Financial Analysis</i>, 95(PB), 103460.",
     "https://doi.org/10.1016/j.irfa.2024.103460",
     "Builds a surprise measure from the overnight return around the announcement, tested on Chinese stocks. Similar in idea to the price-based announcement-return feature in my model, though the market is different."),
    ("Magner, N., Henríquez, P. A., &amp; Sanhueza, A. (2025). Decoding risk sentiment in 10-K filings: Predictability for U.S. stock indices. <i>Finance Research Letters</i>, 81, 107472.",
     "https://doi.org/10.1016/j.frl.2025.107472",
     "Reports that the tone of the risk-factors section of 10-Ks predicts weekly returns on major U.S. stock indices. It works at the index level, whereas my study is firm by firm and next day."),
    ("Fengler, M. R., &amp; Phan, T. M. (2025). Unveiling themes in 10-K disclosures: A new topic modeling perspective. <i>International Review of Financial Analysis</i>, 103, 104121.",
     "https://doi.org/10.1016/j.irfa.2025.104121",
     "Uses word embeddings to find topics in the MD&amp;A section of 10-Ks and relates topic-level sentiment to returns over different horizons. Related to how I turned filing sections into embeddings."),
]
a("<h2 style='page-break-before: always;'>References</h2>")
a("<div class='refnote'><p>Seven peer-reviewed articles from 2024 to 2026 that sit closest to this project. I checked the authors, journal, volume and DOI of each against the publisher's or an indexing page. I read abstracts and the parts available online, not every paper end to end, so the notes say what each paper is about and why I list it, not more.</p></div>")
a("<ol class='refs'>" + "".join(f"<li>{c} {link(u)}<br/><span class='why'>{w}</span></li>" for c, u, w in REFS) + "</ol>")
a("<div class='refnote'><p>Several of these report that filing text or earnings news carries information about returns, while I found no added value from text over the structured features. That is not necessarily a conflict: the studies differ in level (index versus single stock), horizon, market and controls, and I did not test whether those differences explain the gap.</p>"
  "<p class='small'>Quartile: International Review of Financial Analysis and Finance Research Letters are listed as best quartile Q1 on Scimago (SJR 2025). Quartiles change between years and ranking systems, so check the one your institution uses.</p></div>")

body = "\n".join(H)
css = """
@page { size: A4; margin: 20mm 20mm 22mm 20mm; }
body { font-family: Georgia, 'Times New Roman', serif; font-size: 10.6pt; line-height: 1.55; color: #1f2937; }
h1, h2, .kicker, figcaption, th, .small, .by, table.links td:first-child { font-family: 'Segoe UI', Calibri, Arial, sans-serif; }
h1 { font-size: 25pt; line-height: 1.15; margin: 4pt 0 10pt 0; color: #14304d; }
h2 { font-size: 15pt; color: #14304d; margin: 26pt 0 6pt 0; page-break-after: avoid; }
p { margin: 0 0 8pt 0; text-align: left; orphans: 3; widows: 3; }
ul, ol { margin: 0 0 8pt 0; padding-left: 20pt; } li { margin-bottom: 4pt; }
.cover { padding: 14pt 0 10pt 0; border-bottom: 1px solid #cbd5e1; margin-bottom: 8pt; }
.kicker { font-size: 9pt; letter-spacing: .08em; text-transform: uppercase; color: #8a5a2b; }
.sub { font-size: 12pt; color: #374151; margin-bottom: 6pt; } .by { font-size: 10.5pt; color: #374151; margin-bottom: 10pt; }
a { color: #1d4ed8; text-decoration: none; word-break: break-all; }
table { border-collapse: collapse; width: 100%; margin: 6pt 0 12pt 0; font-size: 8.8pt; page-break-inside: avoid; }
th { background: #eef2f7; text-align: left; font-weight: 600; } td, th { border: 1px solid #d5dbe3; padding: 3.2pt 5pt; vertical-align: top; }
table.links td { border: none; padding: 2pt 8pt 2pt 0; font-size: 9.6pt; } table.links td:first-child { color: #6b7280; width: 22%; white-space: nowrap; }
figure { margin: 8pt auto 12pt auto; page-break-inside: avoid; text-align: center; } figure img { max-width: 100%; }
figcaption { font-size: 8.6pt; color: #4b5563; text-align: left; margin-top: 3pt; line-height: 1.4; }
.small { font-size: 8.8pt; color: #6b7280; }
ol.refs li { margin-bottom: 5pt; font-size: 9.1pt; line-height: 1.42; page-break-inside: avoid; } ol.refs a { font-size: 8.2pt; } .why { color: #4b5563; font-size: 8.6pt; }
.refnote p { font-size: 9.3pt; line-height: 1.45; margin-bottom: 5pt; }
"""
open(r"C:\Users\harsh\Downloads\sec_report_build\report.html", "w", encoding="utf-8").write(
    f"<!doctype html><html><head><meta charset='utf-8'><title>Can a company's own filing tell you what its stock will do next?</title><style>{css}</style></head><body>{body}</body></html>")
print("figures:", _fig[0], "| chars:", len(body))
