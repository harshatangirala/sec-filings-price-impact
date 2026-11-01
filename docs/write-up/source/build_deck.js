// 10-slide deck: "Do SEC filings predict the next stock move?"  Numbers come from the finished 150-stock study (report v2).
const pptxgen = require("pptxgenjs");
const React = require("react");
const ReactDOMServer = require("react-dom/server");
const sharp = require("sharp");
const fa = require("react-icons/fa");
const { applyTheme } = require("C:/Users/harsh/.claude/skills/synced/7202cbe6-6536-49d0-9951-a72f02f486ff_3f60d2a8-5158-4890-a774-496494f5c86d/pptx/scripts/apply_theme.js");

const THEME = {
  name: "Market Teal",
  headFontFace: "Cambria",
  bodyFontFace: "Calibri",
  colors: { dk1: "10242B", lt1: "FFFFFF", dk2: "0F4C5C", lt2: "EAF2F1", accent1: "0F4C5C", accent2: "F2A541", accent3: "E4572E",
            accent4: "2E9E8F", accent5: "5C6B73", accent6: "8FB8B0", hlink: "0F4C5C", folHlink: "5C6B73" },
};
const HEX = THEME.colors;

const pres = new pptxgen();
pres.layout = "LAYOUT_16x9"; // 10 x 5.625 in
pres.theme = { headFontFace: THEME.headFontFace, bodyFontFace: THEME.bodyFontFace };
pres.title = "Do SEC filings predict the next stock move?";
pres.subject = "Point-in-time NLP study of 150 S&P 500 stocks, 2000-2026";
const C = pres.SchemeColor;

// ------------------------------------------------------------------ icons
async function icon(name, color, px = 256) {
  const Comp = fa[name] || fa.FaCircle;
  const svg = ReactDOMServer.renderToStaticMarkup(React.createElement(Comp, { color: "#" + color, size: String(px) }));
  const buf = await sharp(Buffer.from(svg)).png().toBuffer();
  return "image/png;base64," + buf.toString("base64");
}

// ------------------------------------------------------------------ layouts
const W = 10, H = 5.625;
pres.defineSlideMaster({
  title: "TITLE", background: { color: HEX.dk2 },
  objects: [
    { placeholder: { options: { name: "title", type: "title", x: 0.6, y: 0.95, w: 8.8, h: 1.5, fontSize: 40, bold: true, color: C.background1, align: "left", valign: "top", margin: 0 }, text: "" } },
    { placeholder: { options: { name: "body", type: "body", x: 0.6, y: 2.5, w: 8.8, h: 0.6, fontSize: 18, color: C.background2, align: "left", valign: "top", margin: 0 }, text: "" } },
  ],
});
pres.defineSlideMaster({
  title: "CONTENT", background: { color: HEX.lt1 },
  objects: [
    { placeholder: { options: { name: "title", type: "title", x: 0.5, y: 0.3, w: 9.0, h: 0.75, fontSize: 32, bold: true, color: C.text2, align: "left", valign: "middle", margin: 0 }, text: "" } },
  ],
  slideNumber: { x: 9.2, y: 5.2, w: 0.5, h: 0.3, fontSize: 10, color: C.accent5, align: "right" },
});
pres.defineSlideMaster({
  title: "CLOSE", background: { color: HEX.dk2 },
  objects: [
    { placeholder: { options: { name: "title", type: "title", x: 0.5, y: 0.3, w: 9.0, h: 0.75, fontSize: 32, bold: true, color: C.background1, align: "left", valign: "middle", margin: 0 }, text: "" } },
  ],
  slideNumber: { x: 9.2, y: 5.2, w: 0.5, h: 0.3, fontSize: 10, color: C.background2, align: "right" },
});

// ------------------------------------------------------------------ helpers
const shadow = () => ({ type: "outer", color: HEX.dk1, opacity: 0.12, blur: 6, offset: 2, angle: 90 });
function card(s, x, y, w, h, name, fill = C.background2) {
  s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x, y, w, h, rectRadius: 0.08, fill: { color: fill }, line: { color: fill, width: 0 }, objectName: name });
}
function badge(s, x, y, d, color, ic, name) {
  s.addShape(pres.shapes.OVAL, { x, y, w: d, h: d, fill: { color }, line: { color, width: 0 }, objectName: name + " circle" });
  const p = d * 0.26;
  s.addImage({ data: ic, x: x + p, y: y + p, w: d - 2 * p, h: d - 2 * p, objectName: name + " icon", altText: name });
}
function text(s, t, o) { s.addText(t, Object.assign({ isTextBox: true, margin: 0, fontFace: "Calibri", color: C.text1, valign: "top" }, o)); }
function source(s, t, dark = false) {
  text(s, t, { x: 0.5, y: 5.2, w: 8.4, h: 0.3, fontSize: 10, color: dark ? C.background2 : C.accent5, objectName: "Source" });
}
const bulletRuns = (items, opts = {}) => items.map((t, i) => ({ text: t, options: Object.assign({ bullet: true, breakLine: i < items.length - 1, paraSpaceAfter: 6 }, opts) }));

async function main() {
  const ic = {};
  const need = { q: "FaQuestionCircle", file: "FaFileAlt", db: "FaDatabase", chart: "FaChartLine", broom: "FaBroom", cut: "FaCut", mask: "FaUserSecret",
    puzzle: "FaPuzzlePiece", brain: "FaBrain", tags: "FaTags", clock: "FaClock", shield: "FaShieldAlt", scale: "FaBalanceScale", bulb: "FaLightbulb",
    tree: "FaTree", robot: "FaRobot", road: "FaRoad", flask: "FaFlask", building: "FaBuilding", coins: "FaCoins", search: "FaSearch", check: "FaCheckCircle",
    sitemap: "FaSitemap", download: "FaDownload", rocket: "FaRocket", book: "FaBook", lock: "FaLock", cogs: "FaCogs" };
  for (const [k, v] of Object.entries(need)) { ic[k] = await icon(v, "FFFFFF"); ic[k + "_d"] = await icon(v, HEX.dk2); ic[k + "_a"] = await icon(v, HEX.accent2); }

  // ---------------------------------------------------------------- sections
  pres.addSection({ title: "Idea and data" });
  // ================================================================ 1 TITLE
  let s = pres.addSlide({ masterName: "TITLE", sectionTitle: "Idea and data" });
  s.addText("Do SEC filings predict the next stock move?", { placeholder: "title" });
  s.addText("A point-in-time NLP study of 150 S&P 500 stocks, 2000 to 2026", { placeholder: "body" });
  const stats = [["150", "S&P 500 firms"], ["64,213", "SEC filings read"], ["60,261", "events studied"], ["25", "pipeline stages"]];
  stats.forEach(([n, l], i) => {
    const x = 0.6 + i * 2.25;
    s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x, y: 3.55, w: 2.05, h: 1.25, rectRadius: 0.08, fill: { color: C.text1 }, line: { color: C.text1, width: 0 }, shadow: shadow(), objectName: "Stat tile " + (i + 1) });
    text(s, n, { x, y: 3.68, w: 2.05, h: 0.6, fontSize: 32, bold: true, color: C.accent2, align: "center", fontFace: "Cambria", objectName: "Stat value " + (i + 1) });
    text(s, l, { x, y: 4.3, w: 2.05, h: 0.35, fontSize: 14, color: C.background2, align: "center", objectName: "Stat label " + (i + 1) });
  });
  s.addNotes("Open with the question: if you read a company's SEC filing the moment it becomes public, can you predict how the stock reacts next? Then the scale: 150 S&P 500 firms, 64,213 filings, 60,261 events, 26 years, a 25-stage pipeline. Stress that everything was evaluated the way a trader would have experienced it, with no peeking at the future.");

  // ================================================================ 2 IDEA + QUESTIONS
  s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "Idea and data" });
  s.addText("The idea and the questions", { placeholder: "title" });
  // flow row
  const flow = [["file", "Filing goes public", "10-K yearly, 10-Q quarterly, 8-K for news"], ["clock", "Trader decides", "35 minutes after the SEC accepts it"], ["chart", "Price reacts", "next trading session: Up, Sideways or Down"]];
  flow.forEach(([k, h, d], i) => {
    const x = 0.5 + i * 3.1;
    card(s, x, 1.3, 2.8, 1.25, "Flow card " + (i + 1));
    badge(s, x + 0.15, 1.45, 0.55, HEX.dk2, ic[k], "Flow " + (i + 1));
    text(s, h, { x: x + 0.8, y: 1.42, w: 1.9, h: 0.55, fontSize: 16, bold: true, color: C.text2, objectName: "Flow heading " + (i + 1) });
    text(s, d, { x: x + 0.8, y: 1.95, w: 1.9, h: 0.55, fontSize: 12, color: C.accent5, objectName: "Flow text " + (i + 1) });
    if (i < 2) text(s, ">", { x: x + 2.8, y: 1.6, w: 0.3, h: 0.5, fontSize: 24, bold: true, color: C.accent2, align: "center", valign: "middle", objectName: "Flow arrow " + (i + 1) });
  });
  text(s, "Questions we wanted to answer", { x: 0.5, y: 2.85, w: 9, h: 0.4, fontSize: 20, bold: true, color: C.text2, fontFace: "Cambria", objectName: "Questions heading" });
  const qs = [["1", "Can we predict the next-day reaction at all?"], ["2", "Does the text add anything beyond numbers already known?"],
              ["3", "Does post-earnings drift (PEAD) show up in this data?"], ["4", "Could any of it be traded for profit?"]];
  qs.forEach(([n, t], i) => {
    const col = i % 2, row = Math.floor(i / 2);
    const x = 0.5 + col * 4.6, y = 3.35 + row * 0.8;
    s.addShape(pres.shapes.OVAL, { x, y, w: 0.5, h: 0.5, fill: { color: C.accent2 }, line: { color: C.accent2, width: 0 }, objectName: "Question number " + n });
    text(s, n, { x, y: y + 0.02, w: 0.5, h: 0.46, fontSize: 18, bold: true, color: C.text1, align: "center", valign: "middle", objectName: "Question digit " + n });
    text(s, t, { x: x + 0.65, y: y - 0.02, w: 3.75, h: 0.6, fontSize: 15, valign: "middle", objectName: "Question " + n });
  });
  source(s, "Three separate tracks: event study (association), prediction (what a trader could know), content check (needs human labels)");
  s.addNotes("Explain the three filing types in one breath. The key design rule: we pretend to be a trader at the moment the filing becomes public, so every input must be known by then. Four questions, in order: can we predict, does text help, does the classic post-earnings drift appear, can it be traded. We also kept three tracks separate: event study is association only, prediction is what a trader could know, and the content check needs human labels we do not have yet.");

  // ================================================================ 3 DATA
  s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "Idea and data" });
  s.addText("The data: free, public and point-in-time", { placeholder: "title" });
  const data = [
    ["db", "64,213 SEC filings", "10-K, 10-Q, 8-K plus exhibits from EDGAR, with exact acceptance times"],
    ["coins", "304,348 accounting facts", "XBRL values as first reported, plus 14,981 earnings releases"],
    ["chart", "1.0M daily price bars", "Your price files plus Yahoo Finance, 1998 to Oct 2026"],
    ["building", "150 S&P 500 firms", "Seeded, spread across 12 industries, all still listed today"],
    ["scale", "Factors and index list", "Fama-French factors; public S&P 500 membership history"],
    ["book", "Finance word list", "Loughran-McDonald dictionary of positive and negative terms"],
  ];
  data.forEach(([k, h, d], i) => {
    const col = i % 3, row = Math.floor(i / 3);
    const x = 0.5 + col * 3.07, y = 1.3 + row * 1.5;
    card(s, x, y, 2.85, 1.3, "Data card " + (i + 1));
    badge(s, x + 0.15, y + 0.15, 0.5, HEX.accent4, ic[k], "Data " + (i + 1));
    text(s, h, { x: x + 0.78, y: y + 0.13, w: 2.0, h: 0.55, fontSize: 15, bold: true, color: C.text2, valign: "middle", objectName: "Data heading " + (i + 1) });
    text(s, d, { x: x + 0.15, y: y + 0.75, w: 2.6, h: 0.5, fontSize: 12, color: C.accent5, objectName: "Data text " + (i + 1) });
  });
  card(s, 0.5, 4.35, 9.0, 0.7, "Timeline strip", C.text2);
  text(s, [{ text: "Timeline:  ", options: { bold: true, color: C.accent2 } },
           { text: "events Jan 2000 to Jul 2026  |  prices from 1998  |  SEC filings only, no news", options: { color: C.background1 } }],
       { x: 0.7, y: 4.35, w: 8.6, h: 0.7, fontSize: 14, valign: "middle", objectName: "Timeline text" });
  s.addNotes("Every source is free and pinned to a version. Filing times come from the SEC header because the JSON feed mixes two time conventions. The 150 firms were drawn by a fixed seed from S&P 500 members, spread across industries. A caution to state early: all 150 still exist today, so the sample is survivor-only. No news or social media text is used anywhere.");

  // ---------------------------------------------------------------- Method
  pres.addSection({ title: "Method" });
  // ================================================================ 4 PREPROCESSING
  s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "Method" });
  s.addText("Pre-processing: filing to model input", { placeholder: "title" });
  const steps = [
    ["download", "1  Download and clean", "64,213 filings; HTML turned into plain text, tables flattened"],
    ["cut", "2  Keep the useful parts", "121,398 sections: risk factors, MD&A, 8-K items, press releases"],
    ["mask", "3  Hide the company name", "Names and tickers masked so models cannot memorise firms"],
    ["puzzle", "4  Cut into chunks", "About 900,000 chunks of 350 words with 50 words overlap"],
    ["brain", "5  Turn text into numbers", "ChronoBERT and BGE read each chunk into 768 numbers"],
    ["tags", "6  Add simple word features", "Positive/negative words, change since last filing, novelty"],
  ];
  steps.forEach(([k, h, d], i) => {
    const col = i % 3, row = Math.floor(i / 3);
    const x = 0.5 + col * 3.07, y = 1.3 + row * 1.9;
    card(s, x, y, 2.85, 1.7, "Step card " + (i + 1));
    badge(s, x + 0.15, y + 0.15, 0.55, HEX.dk2, ic[k], "Step " + (i + 1));
    text(s, h, { x: x + 0.82, y: y + 0.15, w: 1.95, h: 0.55, fontSize: 15, bold: true, color: C.text2, valign: "middle", objectName: "Step heading " + (i + 1) });
    text(s, d, { x: x + 0.15, y: y + 0.85, w: 2.55, h: 0.8, fontSize: 14, objectName: "Step text " + (i + 1) });
  });
  source(s, "Nothing is learned from the future: word-change features use a fixed hashing space, and TF-IDF is refitted inside each training fold");
  s.addNotes("Walk the six steps left to right, top to bottom. The masking step is a leakage control: without it a model could learn which company it is reading rather than what the text says. The chunks are shared by every encoder and by the retrieval system. Nothing in this pipeline is fitted on future data.");

  // ================================================================ 5 EVENTS / FEATURES / LEAKAGE
  s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "Method" });
  s.addText("Events, features and leakage control", { placeholder: "title" });
  const clock = [["SEC accepts", "filing time from the SGML header"], ["Public", "+5 min (assumed)"], ["Decision", "+30 min"], ["Trade", "next market open"]];
  clock.forEach(([h, d], i) => {
    const x = 0.5 + i * 2.28;
    s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x, y: 1.25, w: 2.05, h: 0.95, rectRadius: 0.08, fill: { color: i === 3 ? C.accent2 : C.text2 }, line: { color: i === 3 ? C.accent2 : C.text2, width: 0 }, objectName: "Clock box " + (i + 1) });
    text(s, h, { x, y: 1.33, w: 2.05, h: 0.4, fontSize: 16, bold: true, color: i === 3 ? C.text1 : C.background1, align: "center", objectName: "Clock heading " + (i + 1) });
    text(s, d, { x: x + 0.05, y: 1.73, w: 1.95, h: 0.4, fontSize: 12, color: i === 3 ? C.text1 : C.background2, align: "center", objectName: "Clock text " + (i + 1) });
    if (i < 3) s.addText(">", { x: x + 2.05, y: 1.45, w: 0.23, h: 0.5, fontSize: 20, bold: true, color: C.accent5, align: "center", valign: "middle", isTextBox: true, margin: 0, objectName: "Clock arrow " + (i + 1) });
  });
  text(s, "Features may only use what is known at the decision time. Target: Up, Down or Sideways, judged against the stock's own volatility.", { x: 0.5, y: 2.3, w: 9, h: 0.5, fontSize: 14, color: C.accent5, objectName: "Clock caption" });
  s.addChart(pres.charts.BAR, [{ name: "Features", labels: ["Metadata (type, timing, industry)", "Price and market state", "Word tone (finance list)", "Change vs last filing", "Accounting and surprise", "Novelty"], values: [68, 19, 18, 11, 10, 1] }], {
    x: 0.5, y: 2.85, w: 5.6, h: 2.3, barDir: "bar", chartColors: [HEX.accent4], showTitle: true, title: "128 features, by family", titleFontSize: 14, titleColor: HEX.dk2, titleFontFace: "+mn-lt",
    catAxisOrientation: "maxMin", showValue: true, dataLabelPosition: "outEnd", dataLabelFontSize: 11, dataLabelColor: HEX.dk1, dataLabelFontFace: "+mn-lt",
    catAxisLabelColor: HEX.dk1, catAxisLabelFontSize: 11, catAxisLabelFontFace: "+mn-lt", valAxisHidden: true, valGridLine: { style: "none" }, catGridLine: { style: "none" }, showLegend: false, objectName: "Feature family chart",
  });
  const audit = [["24", "leak checks, each with an assertion"], ["4 / 4", "deliberately planted leaks caught"], ["0", "availability violations on 60,261 events"]];
  audit.forEach(([n, l], i) => {
    const y = 2.95 + i * 0.75;
    card(s, 6.4, y, 3.1, 0.65, "Audit card " + (i + 1));
    text(s, n, { x: 6.5, y, w: 0.95, h: 0.65, fontSize: 22, bold: true, color: C.accent3, fontFace: "Cambria", valign: "middle", objectName: "Audit value " + (i + 1) });
    text(s, l, { x: 7.5, y, w: 1.95, h: 0.65, fontSize: 12, valign: "middle", objectName: "Audit label " + (i + 1) });
  });
  s.addNotes("Four timestamps per event; the trade happens at the next market open after the decision time, and the abnormal return subtracts beta times the S&P benchmark on exactly the same interval. Labels are Up or Down beyond half the stock's own volatility, otherwise Sideways. 128 features in seven families, each with an availability timestamp that is asserted on every row. The audit had 24 checks and four planted leaks; all four were caught. The audit also caught two real look-ahead problems and an accounting bug during the build.");

  // ================================================================ 6 MODELS
  s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "Method" });
  s.addText("Models: what we trained, what we borrowed", { placeholder: "title" });
  card(s, 0.5, 1.25, 4.4, 2.95, "Trained card");
  badge(s, 0.65, 1.38, 0.5, HEX.dk2, ic.tree, "Trained");
  text(s, "Trained on our data", { x: 1.3, y: 1.38, w: 3.4, h: 0.5, fontSize: 18, bold: true, color: C.text2, valign: "middle", fontFace: "Cambria", objectName: "Trained heading" });
  text(s, bulletRuns(["LightGBM and CatBoost: gradient-boosted trees, at most 25 leaves per tree, 3 trees per round", "Elastic-net regression: about 400 weights", "Tuned with Optuna on 2008-2019 folds", "Output: probabilities for Down, Sideways, Up"], { fontSize: 14 }),
       { x: 0.7, y: 2.05, w: 4.0, h: 2.05, objectName: "Trained list" });
  card(s, 5.1, 1.25, 4.4, 2.95, "Pretrained card");
  badge(s, 5.25, 1.38, 0.5, HEX.accent4, ic.brain, "Pretrained");
  text(s, "Pre-trained, kept frozen", { x: 5.9, y: 1.38, w: 3.4, h: 0.5, fontSize: 18, bold: true, color: C.text2, valign: "middle", fontFace: "Cambria", objectName: "Pretrained heading" });
  text(s, bulletRuns(["ChronoBERT, about 150M parameters, 26 yearly versions with no look-ahead", "BGE-base, about 110M: embeddings and passage search", "BGE reranker, about 280M, and Qwen2.5-7B, about 7.6B: RAG extraction", "SEC-BERT and FinBERT dropped for time"], { fontSize: 14 }),
       { x: 5.3, y: 2.05, w: 4.0, h: 2.05, objectName: "Pretrained list" });
  text(s, "Ablation ladder: each rung adds one kind of information", { x: 0.5, y: 4.3, w: 9, h: 0.3, fontSize: 14, bold: true, color: C.text2, objectName: "Ladder heading" });
  const ladder = ["L1 price", "L2 metadata", "L3 accounting", "L4 word features", "L6 all numbers", "L7 + text vectors"];
  ladder.forEach((t, i) => {
    const x = 0.5 + i * 1.5;
    s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x, y: 4.65, w: 1.4, h: 0.45, rectRadius: 0.06, fill: { color: i >= 4 ? C.accent2 : C.text2 }, line: { color: i >= 4 ? C.accent2 : C.text2, width: 0 }, objectName: "Ladder rung " + (i + 1) });
    text(s, t, { x, y: 4.65, w: 1.4, h: 0.45, fontSize: 12, bold: true, color: i >= 4 ? C.text1 : C.background1, align: "center", valign: "middle", objectName: "Ladder label " + (i + 1) });
  });
  s.addNotes("Parameter counts for the pre-trained models are approximate, derived from model file sizes; the exact number of trees in the boosted models was not recorded, but each tree has at most 25 leaves. The pre-trained encoders are never retrained. ChronoBERT matters because an event in year Y uses the version trained only on text up to the end of year Y-1. SEC-BERT and FinBERT were dropped to fit the compute budget, a time decision rather than a measured result. The ladder adds one kind of information at a time on identical events.");


  // ================================================================ 6b WHY THESE MODELS
  s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "Method" });
  s.addText("Why these models?", { placeholder: "title" });
  const whys = [
    ["tree", "LightGBM", "trained", "Fast, strong default for mixed tabular data and interactions", false],
    ["tree", "CatBoost", "trained", "A different tree algorithm as a cross-check; robust untuned", false],
    ["scale", "Elastic net", "trained", "Simple, convex, certifiable baseline; guards against overfitting", false],
    ["clock", "ChronoBERT", "frozen", "Each yearly version only saw earlier text: no look-ahead", true],
    ["brain", "BGE-base", "frozen", "Strong general text encoder; also powers passage search", true],
    ["search", "BGE reranker", "frozen", "Re-orders passages so the best evidence reaches the LLM", true],
    ["robot", "Qwen2.5-7B", "frozen", "Open model that runs locally on one GPU and follows JSON format", true],
  ];
  whys.forEach(([k, name, tag, reason, pre], i) => {
    const y = 1.2 + i * 0.54;
    card(s, 0.5, y, 9.0, 0.47, "Model row " + (i + 1));
    badge(s, 0.58, y + 0.04, 0.39, pre ? HEX.accent4 : HEX.dk2, ic[k], "Model " + (i + 1));
    text(s, name, { x: 1.1, y, w: 1.75, h: 0.47, fontSize: 14, bold: true, color: C.text2, valign: "middle", objectName: "Model name " + (i + 1) });
    s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x: 2.85, y: y + 0.1, w: 0.85, h: 0.27, rectRadius: 0.08, fill: { color: pre ? C.accent2 : C.text2 }, line: { color: pre ? C.accent2 : C.text2, width: 0 }, objectName: "Model tag " + (i + 1) });
    text(s, tag, { x: 2.85, y: y + 0.1, w: 0.85, h: 0.27, fontSize: 10, bold: true, color: pre ? C.text1 : C.background1, align: "center", valign: "middle", objectName: "Model tag text " + (i + 1) });
    text(s, reason, { x: 3.9, y, w: 5.5, h: 0.47, fontSize: 13, valign: "middle", objectName: "Model reason " + (i + 1) });
  });
  source(s, "Trained = fitted on our data. Frozen = used as is. SEC-BERT and FinBERT dropped for time, not because they scored worse.");
  s.addNotes("One line per model. LightGBM: gradient-boosted trees are the standard strong choice for tabular data with many mixed features and interactions, and they are fast to tune. CatBoost: a different boosting algorithm, so if both agree the result is not an artefact of one library; it also needs little tuning. Elastic net: a simple regularised regression whose optimum can be certified, a guard against overfitting with many correlated features; it was actually the best L6 head. ChronoBERT: each yearly version was trained only on text up to its date, so an event is embedded with a model that cannot know the future. BGE-base: a strong general-purpose text encoder that also provides the similarity search in the RAG system. BGE reranker: re-orders the retrieved passages so the best evidence reaches the language model. Qwen2.5-7B: an open model that runs locally on a single GPU, which keeps the study free of paid APIs, and it follows a JSON output format. SEC-BERT and FinBERT were in the plan but dropped to fit the compute budget; that was a time decision, not a measured result.");


  // ================================================================ 7b EMBEDDINGS
  s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "Method" });
  s.addText("Where do the embeddings come from?", { placeholder: "title" });
  const emb = [
    ["download", "Borrow the models", "ChronoBERT and BGE-base from Hugging Face, pinned versions"],
    ["puzzle", "Feed in our text", "About 900,000 chunks of masked filing text, same for both"],
    ["brain", "Get 768 numbers", "Each chunk becomes a vector; no prices are involved"],
    ["sitemap", "Combine per filing", "Average and max of chunks give 1,536 numbers per event"],
    ["flask", "Add to the model", "Shrunk with PCA inside each fold, then added: L7"],
  ];
  emb.forEach(([k, h, d], i) => {
    const x = 0.5 + i * 1.83;
    card(s, x, 1.25, 1.68, 1.85, "Embedding step " + (i + 1));
    badge(s, x + 0.55, 1.35, 0.58, HEX.accent4, ic[k], "Embedding " + (i + 1));
    text(s, h, { x: x + 0.08, y: 2.0, w: 1.52, h: 0.4, fontSize: 13, bold: true, color: C.text2, align: "center", objectName: "Embedding heading " + (i + 1) });
    text(s, d, { x: x + 0.1, y: 2.4, w: 1.48, h: 0.7, fontSize: 11, align: "center", objectName: "Embedding text " + (i + 1) });
    if (i < 4) text(s, ">", { x: x + 1.68, y: 1.85, w: 0.15, h: 0.4, fontSize: 18, bold: true, color: C.accent2, align: "center", valign: "middle", objectName: "Embedding arrow " + (i + 1) });
  });
  card(s, 0.5, 3.25, 9.0, 0.6, "No-look-ahead strip", C.text2);
  text(s, [{ text: "No look-ahead: ", options: { bold: true, color: C.accent2 } }, { text: "each event uses the ChronoBERT version trained on text up to the previous year-end.", options: { color: C.background1 } }],
       { x: 0.7, y: 3.25, w: 8.6, h: 0.6, fontSize: 14, valign: "middle", objectName: "No-look-ahead text" });
  const embStats = [["60,030", "events embedded"], ["56 min", "ChronoBERT, Colab GPU"], ["39 min", "BGE-base, Colab GPU"], ["184 MB", "vectors per encoder"]];
  embStats.forEach(([n, l], i) => {
    const x = 0.5 + i * 2.28;
    card(s, x, 4.05, 2.1, 1.0, "Embedding stat " + (i + 1));
    text(s, n, { x, y: 4.1, w: 2.1, h: 0.55, fontSize: 24, bold: true, color: C.accent3, fontFace: "Cambria", align: "center", objectName: "Embedding stat value " + (i + 1) });
    text(s, l, { x, y: 4.65, w: 2.1, h: 0.35, fontSize: 12, align: "center", color: C.accent5, objectName: "Embedding stat label " + (i + 1) });
  });
  s.addNotes("The models come from Hugging Face, but the embeddings are ours: we ran our own filing text through the pre-trained models on a Colab GPU. We did not download ready-made embeddings. The input is about 900,000 masked text chunks of 350 words, identical for both encoders. Each chunk becomes 768 numbers. Chunk vectors are averaged and max-pooled to one vector per section and then one per filing event, 1,536 numbers, and we also store a change vector against the firm's previous filing. ChronoBERT took about 56 minutes and BGE about 39 minutes; the saved vectors are about 184 megabytes per encoder. For the model they are shrunk with PCA inside each training fold and added to the number features, which gives rung L7; L5 uses the embeddings alone. Embeddings exist only for our 60,030 filing events and never look at prices.");

  // ================================================================ 7 EVALUATION DESIGN
  s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "Method" });
  s.addText("How we tested it", { placeholder: "title" });
  const x0 = 0.6, span = 8.8, yr0 = 2000, yr1 = 2026.58, sc = span / (yr1 - yr0);
  const blocks = [["Train (expanding)", 2000, 2008, C.accent5, C.background1], ["6 validation folds", 2008, 2020, C.accent4, C.background1], ["Dev", 2020, 2023, C.accent2, C.text1], ["Locked", 2023, 2026.58, C.accent3, C.background1]];
  blocks.forEach(([t, a, b, fill, tc], i) => {
    const x = x0 + (a - yr0) * sc, w = (b - a) * sc;
    s.addShape(pres.shapes.RECTANGLE, { x, y: 1.55, w, h: 0.8, fill: { color: fill }, line: { color: C.background1, width: 2 }, objectName: "Split block " + (i + 1) });
    text(s, t, { x, y: 1.55, w, h: 0.8, fontSize: 13, bold: true, color: tc, align: "center", valign: "middle", objectName: "Split label " + (i + 1) });
  });
  [2000, 2008, 2020, 2023].forEach((yv) => text(s, String(yv), { x: x0 + (yv - yr0) * sc - 0.3, y: 2.4, w: 0.6, h: 0.25, fontSize: 11, color: C.accent5, align: "center", objectName: "Year " + yv }));
  text(s, "Jul 2026", { x: x0 + span - 0.5, y: 2.4, w: 0.7, h: 0.25, fontSize: 11, color: C.accent5, align: "right", objectName: "Year end" });
  const rules = [["Walk-forward", "Train on the past, test on the next 2 years, six times (2008-2019)"], ["Dev holdout 2020-22", "Used to choose the final model; 7,455 events"], ["Locked test 2023-26", "Scored once, after the model was frozen and hashed; 8,213 events"], ["Purge + embargo", "Overlapping outcome windows removed, 90-day gap between blocks"]];
  rules.forEach(([h, d], i) => {
    const x = 0.5 + i * 2.28;
    card(s, x, 2.85, 2.1, 1.35, "Rule card " + (i + 1));
    text(s, h, { x: x + 0.12, y: 2.93, w: 1.9, h: 0.4, fontSize: 14, bold: true, color: C.text2, objectName: "Rule heading " + (i + 1) });
    text(s, d, { x: x + 0.12, y: 3.33, w: 1.9, h: 0.85, fontSize: 12, objectName: "Rule text " + (i + 1) });
  });
  text(s, [{ text: "Metrics:  ", options: { bold: true, color: C.accent3 } }, { text: "macro-F1 (main: Down, Sideways, Up weighted equally), accuracy, AUC, log-loss, calibration error, and 95% intervals from resampling whole months", options: {} }],
       { x: 0.5, y: 4.4, w: 9, h: 0.7, fontSize: 14, objectName: "Metrics line" });
  s.addNotes("Random splits would leak the future in time-series data, so training always uses only the past. Six walk-forward folds choose settings, the 2020-22 holdout chooses the final model, and the 2023-26 period is scored exactly once after the choice is frozen and hashed. Macro-F1 is the main metric because it weights the three classes equally and cannot be inflated by predicting the biggest class. Intervals resample whole months because events in one month share market shocks.");

  // ---------------------------------------------------------------- Results
  pres.addSection({ title: "Results" });
  // ================================================================ 8 RESULTS
  s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "Results" });
  s.addText("Accuracy: modest, real and stable", { placeholder: "title" });
  const periods = ["Walk-forward 2008-19", "Development 2020-22", "Locked test 2023-26"];
  s.addChart([
    { type: pres.charts.BAR, data: [{ name: "L1 price only", labels: periods, values: [0.360, 0.305, 0.351] }, { name: "L6 all numbers", labels: periods, values: [0.392, 0.396, 0.384] }, { name: "L7 numbers + text", labels: periods, values: [0.393, 0.404, 0.383] }],
      options: { chartColors: [HEX.accent5, HEX.accent4, HEX.accent2], barGrouping: "clustered", barGapWidthPct: 60, showValue: true, dataLabelPosition: "outEnd", dataLabelFormatCode: "0.000", dataLabelFontSize: 10, dataLabelColor: HEX.dk1, dataLabelFontFace: "+mn-lt" } },
    { type: pres.charts.LINE, data: [{ name: "Real chance level 0.33", labels: periods, values: [0.328, 0.328, 0.328] }],
      options: { chartColors: [HEX.accent3], lineSize: 2, lineDash: "dash", lineDataSymbol: "none" } },
  ], {
    x: 0.5, y: 1.2, w: 5.9, h: 3.9, showTitle: true, title: "Macro-F1 by period", titleFontSize: 14, titleColor: HEX.dk2, titleFontFace: "+mn-lt", valAxisMinVal: 0.2, valAxisMaxVal: 0.45, valAxisMajorUnit: 0.05,
    valAxisLabelFormatCode: "0.00", valAxisLabelColor: HEX.accent5, valAxisLabelFontSize: 11, valAxisLabelFontFace: "+mn-lt", catAxisLabelColor: HEX.dk1, catAxisLabelFontSize: 11, catAxisLabelFontFace: "+mn-lt",
    valGridLine: { color: "D9E3E1", size: 0.5 }, catGridLine: { style: "none" }, showLegend: true, legendPos: "b", legendFontSize: 11, legendColor: HEX.dk1, legendFontFace: "+mn-lt", objectName: "Macro-F1 chart",
  });
  const big = [["0.384", "locked macro-F1", "vs ~0.33 for luck", C.text2], ["0.635", "move AUC", "will it move? above chance", C.accent4], ["0.494", "direction AUC", "which way? coin flip", C.accent3]];
  big.forEach(([n, l, d, col], i) => {
    const y = 1.25 + i * 1.3;
    card(s, 6.65, y, 2.85, 1.15, "Result card " + (i + 1));
    text(s, n, { x: 6.8, y: y + 0.08, w: 1.25, h: 0.99, fontSize: 28, bold: true, color: col, fontFace: "Cambria", valign: "middle", objectName: "Result value " + (i + 1) });
    text(s, [{ text: l, options: { bold: true, breakLine: true, fontSize: 13 } }, { text: d, options: { fontSize: 11, color: C.accent5 } }], { x: 8.0, y: y + 0.08, w: 1.45, h: 0.99, valign: "middle", objectName: "Result label " + (i + 1) });
  });
  source(s, "Source: study report, 150 S&P 500 stocks. Locked test n = 8,199 events; chance = macro-F1 of shuffled predictions");
  s.addNotes("Read the bars against the dashed line at about 0.33, not against the 0.19 majority-class baseline, which understates luck. L6 scores 0.392, 0.396 and 0.384 across the three periods, so performance is stable out of sample. The key nuance: the model knows whether a stock will move (AUC 0.635) but not which way (AUC 0.494, interval 0.477 to 0.511). Most of the signal is event type, timing and volatility, not direction. On the locked test accuracy is 0.401, AUC 0.590, calibration error 0.009.");

  // ================================================================ 9 TEXT + PEAD
  s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "Results" });
  s.addText("Text adds nothing; earnings drift reverses", { placeholder: "title" });
  card(s, 0.5, 1.25, 4.2, 3.85, "Text card");
  text(s, "Does filing text help?", { x: 0.7, y: 1.35, w: 3.8, h: 0.4, fontSize: 18, bold: true, color: C.text2, fontFace: "Cambria", objectName: "Text heading" });
  text(s, "-0.001", { x: 0.7, y: 1.85, w: 3.8, h: 0.8, fontSize: 48, bold: true, color: C.accent3, fontFace: "Cambria", objectName: "Text lift value" });
  text(s, "lift in macro-F1 from adding text, locked test, 95% interval [-0.012, +0.010]. Gate to fine-tune needed +0.01.", { x: 0.7, y: 2.7, w: 3.8, h: 0.85, fontSize: 14, objectName: "Text lift caption" });
  text(s, bulletRuns(["Same answer in all three periods", "Yearly (no look-ahead) vs static encoder: no difference", "Numbers beat price-only by +0.033; text beats nothing"], { fontSize: 14 }), { x: 0.7, y: 3.6, w: 3.8, h: 1.4, objectName: "Text bullets" });
  s.addChart(pres.charts.BAR, [{ name: "Drift", labels: ["Q1 worst news", "Q2", "Q3", "Q4", "Q5 best news"], values: [1.40, 0.78, -0.92, -0.94, -1.85] }], {
    x: 4.9, y: 1.25, w: 4.6, h: 2.65, barDir: "col", chartColors: [HEX.accent4, HEX.accent4, HEX.accent3, HEX.accent3, HEX.accent3], showTitle: true, title: "Return over days +2 to +60, by earnings-surprise group (%)", titleFontSize: 13, titleColor: HEX.dk2, titleFontFace: "+mn-lt",
    showValue: true, dataLabelFormatCode: '+0.00;-0.00', dataLabelPosition: "outEnd", dataLabelFontSize: 11, dataLabelColor: HEX.dk1, dataLabelFontFace: "+mn-lt", catAxisLabelColor: HEX.dk1, catAxisLabelFontSize: 10, catAxisLabelFontFace: "+mn-lt",
    valAxisHidden: true, valAxisMinVal: -2.6, valAxisMaxVal: 2.2, valGridLine: { style: "none" }, catGridLine: { style: "none" }, showLegend: false, objectName: "SUE drift chart",
  });
  card(s, 4.9, 3.95, 4.6, 1.15, "PEAD note", C.text2);
  text(s, [
    { text: "Q5 minus Q1: -3.3% here (-2.6% in the stricter formal test).", options: { bold: true, color: C.accent2, breakLine: true } },
    { text: "55% of best-news reports fell afterwards; 31% first rose (+4.8%) then fell (-10%).", options: { color: C.background1, breakLine: true } },
    { text: "Opposite of classic drift; survivor-only data can fake it.", options: { color: C.background1 } },
  ], { x: 5.05, y: 3.95, w: 4.35, h: 1.15, fontSize: 12, valign: "middle", objectName: "PEAD note text" });
  s.addNotes("Text: the paired lift of the best text model over the best numbers-only model is minus 0.001 on the locked test, with an interval that includes zero; the gate to fine-tune required plus 0.01, so no encoder was fine-tuned. Post-earnings drift: classic PEAD says good surprises keep rising; here the best-news group underperformed the worst-news group by about 3.3 percent over days 2 to 60 in this chart, which uses 9,359 earnings reports; the stricter formal test, with 3,825 reports, gives 2.6 percent. We checked the surprise sign (good news does raise the announcement-day price) and the pattern holds in 2000-09, 2010-19 and 2020-26. It could still be survivorship: all 150 firms exist today, so firms that kept falling and were delisted are missing. Fair wording: we did not find classic PEAD in this survivor-only sample. Instance counts, from 9,359 earnings reports: of the 1,908 best-news reports, 56 percent rose on announcement (average +0.81 percent), yet 1,047 (55 percent) fell over days 2 to 60; 587 (31 percent) rose first, gaining about 4.8 percent, then gave back about 10 percent; 831 (44 percent) fell on announcement despite good news. The worst-news group mirrors it: it fell on announcement (-0.54 percent) and then drifted up 1.40 percent. Why it might happen, hypotheses not tests: our surprise is measured against the same quarter last year, not against analyst expectations; prices may overreact and settle back; these are closely watched large companies; and survivor bias, because delisted firms are missing. The backtest also shows no edge: long minus the S&P 500 was minus 5.6 percent a year on the locked test.");


  // ================================================================ 10 WHY GOOD NEWS FALLS
  s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "Results" });
  s.addText("Why can good news be followed by a fall?", { placeholder: "title" });
  s.addChart(pres.charts.BAR, [{ name: "Reports", labels: ["Rose, then rose (26%)", "Rose, then fell (31%)", "Fell, then rose (19%)", "Fell, then fell (24%)"], values: [490, 587, 371, 460] }], {
    x: 0.5, y: 1.2, w: 4.9, h: 3.1, barDir: "col", chartColors: [HEX.accent4, HEX.accent3, HEX.accent6, HEX.accent5], showTitle: true,
    title: "The 1,908 best-news earnings reports: announcement move, then later drift", titleFontSize: 13, titleColor: HEX.dk2, titleFontFace: "+mn-lt",
    showValue: true, dataLabelPosition: "outEnd", dataLabelFontSize: 12, dataLabelColor: HEX.dk1, dataLabelFontFace: "+mn-lt", dataLabelFormatCode: "#,##0",
    catAxisLabelColor: HEX.dk1, catAxisLabelFontSize: 10, catAxisLabelFontFace: "+mn-lt", valAxisHidden: true, valAxisMinVal: 0, valAxisMaxVal: 700,
    valGridLine: { style: "none" }, catGridLine: { style: "none" }, showLegend: false, objectName: "Best-news outcomes chart",
  });
  card(s, 0.5, 4.4, 4.9, 0.7, "Outcome note", C.text2);
  text(s, [{ text: "1,047 of 1,908 (55%) fell over days +2 to +60. ", options: { bold: true, color: C.accent2 } }, { text: "Rose-then-fell: +4.8%, then -10%.", options: { color: C.background1 } }],
       { x: 0.65, y: 4.4, w: 4.6, h: 0.7, fontSize: 12, valign: "middle", objectName: "Outcome note text" });
  const why = [
    ["bulb", "Beat last year, not forecasts", "Our surprise compares with the same quarter a year ago, not with analyst forecasts."],
    ["chart", "Overreaction, then giving back", "Prices can jump too far on the news and settle. Bad-news stocks bounced (+1.4%)."],
    ["building", "Large, closely watched firms", "S&P 500 news may be fully priced in within hours."],
    ["search", "Survivor bias", "Firms that kept falling were delisted and are missing, which flatters the bad-news group."],
  ];
  why.forEach(([k, h, d], i) => {
    const y = 1.25 + i * 0.97;
    card(s, 5.6, y, 3.9, 0.87, "Reason card " + (i + 1));
    badge(s, 5.7, y + 0.2, 0.47, HEX.dk2, ic[k], "Reason " + (i + 1));
    text(s, [{ text: h, options: { bold: true, color: C.text2, fontSize: 13, breakLine: true } }, { text: d, options: { fontSize: 11 } }], { x: 6.28, y: y + 0.04, w: 3.15, h: 0.79, valign: "middle", objectName: "Reason text " + (i + 1) });
  });
  source(s, "Hypotheses, not tests: an average tendency across 9,359 earnings reports, not a rule for any single stock. Source: study report, event study");
  s.addNotes("This answers the obvious question: shouldn't good news lift a stock? It did, on announcement day: 56 percent of the 1,908 best-news reports rose, by 0.81 percent on average. The surprise is the later drift over days 2 to 60: 1,047 of the 1,908 (55 percent) fell. The chart splits them four ways. 587 (31 percent) rose first, gaining 4.8 percent, then gave back about 10 percent. 490 (26 percent) kept rising. 460 (24 percent) fell both times, and 371 (19 percent) fell first and then recovered. So it is a tilt in the averages, not a rule. The four reasons are hypotheses we did not test: our surprise is measured against the same quarter last year, not analyst expectations; prices may overreact and settle back, and the worst-news group shows the mirror image; these are large, closely followed companies where news is priced in quickly; and survivor bias, because firms that kept falling were delisted and are not in our data. The backtest confirms there was no tradable edge.");

  // ---------------------------------------------------------------- Wrap-up
  pres.addSection({ title: "Wrap-up" });
  // ================================================================ 10 CLOSE
  s = pres.addSlide({ masterName: "CLOSE", sectionTitle: "Wrap-up" });
  s.addText("What we can claim, and what is next", { placeholder: "title" });
  card(s, 0.5, 1.25, 4.4, 3.15, "Claim card", C.text1);
  badge(s, 0.65, 1.4, 0.5, HEX.accent2, ic.check, "Claim");
  text(s, "Smallest defensible claim", { x: 1.3, y: 1.4, w: 3.5, h: 0.5, fontSize: 18, bold: true, color: C.accent2, valign: "middle", fontFace: "Cambria", objectName: "Claim heading" });
  text(s, "On 150 surviving S&P 500 stocks, adding filing-text embeddings to price, metadata and accounting features gave no out-of-sample gain. The models beat chance only by predicting whether a stock moves, not which way.", { x: 0.7, y: 2.0, w: 4.0, h: 1.4, fontSize: 14, color: C.background1, objectName: "Claim text" });
  text(s, [{ text: "Limits: ", options: { bold: true, color: C.accent2 } }, { text: "survivor-only, daily bars, 150 firms, 2 of 4 encoders, no human labels for the RAG check.", options: { color: C.background2 } }], { x: 0.7, y: 3.45, w: 4.0, h: 0.9, fontSize: 13, objectName: "Limits text" });
  card(s, 5.1, 1.25, 4.4, 3.15, "Next card", C.text1);
  badge(s, 5.25, 1.4, 0.5, HEX.accent4, ic.road, "Next");
  text(s, "Next steps", { x: 5.9, y: 1.4, w: 3.4, h: 0.5, fontSize: 18, bold: true, color: C.accent2, valign: "middle", fontFace: "Cambria", objectName: "Next heading" });
  text(s, bulletRuns(["Expand to 500 stocks, including non-S&P firms", "Add delisted firms to remove survivor bias", "Hand-label filings to test the RAG extraction", "Make training reproducible across machines", "Run all four encoders"], { fontSize: 14, color: C.background1 }), { x: 5.3, y: 2.0, w: 4.0, h: 2.3, objectName: "Next list" });
  s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x: 0.5, y: 4.55, w: 9.0, h: 0.55, rectRadius: 0.08, fill: { color: C.accent2 }, line: { color: C.accent2, width: 0 }, objectName: "Reproducibility strip" });
  text(s, "Reproducible end to end: one notebook, pinned data, 25 resumable stages, 114 automated tests", { x: 0.7, y: 4.55, w: 8.6, h: 0.55, fontSize: 14, bold: true, color: C.text1, valign: "middle", objectName: "Reproducibility text" });
  s.addNotes("Close on the claim: a credible null result for text, obtained with a pre-registered design, a single-use locked test and an audited pipeline. Be upfront about limits: survivor-only universe, daily bars, 150 firms, two of four encoders, and no human-labelled data to validate the RAG extraction. The roadmap is the 500-stock study, delisted firms, gold labels, deterministic training and the full encoder set.");

  await pres.writeFile({ fileName: "SEC_Filings_Study_13_Slides.pptx" });
  await applyTheme("SEC_Filings_Study_13_Slides.pptx", THEME);
  console.log("written");
}
main().catch((e) => { console.error(e); process.exit(1); });
