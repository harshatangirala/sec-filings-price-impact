"""2 November 2026: corrects slides 11 and 12 of the built deck for decision D-079 (drift artifact; embeddings versus text)."""
from pptx import Presentation
from pptx.chart.data import CategoryChartData

F = r"C:\Users\harsh\Downloads\sec_deck_build\SEC_Filings_Study_13_Slides.pptx"
p = Presentation(F)
s11, s12 = p.slides[10], p.slides[11]
by = lambda s: {x.name: x for x in s.shapes}


def set_paras(shape, lines):
    tf = shape.text_frame
    ps = list(tf.paragraphs)
    assert len(ps) >= len(lines), (shape.name, len(ps), len(lines))
    for q, t in zip(ps, lines):
        q.runs[0].text = t
        for r in q.runs[1:]:
            r.text = ""
    for q in ps[len(lines):]:
        q._p.getparent().remove(q._p)


a = by(s11)
a["Text 0"].text_frame.paragraphs[0].runs[0].text = "Language models add nothing; no detectable drift"
set_paras(a["Text heading"], ["Do language models help?"])
set_paras(a["Text lift caption"], ["change in macro-F1 from adding embeddings, locked test, 95% interval [-0.012, +0.009]. Gate to fine-tune needed +0.01."])
print([q.text for q in a["Text bullets"].text_frame.paragraphs])
ch = a["SUE drift chart"].chart
print([x.name for x in ch.plots[0].series], list(ch.plots[0].categories))
d = CategoryChartData(); d.categories = list(ch.plots[0].categories); d.add_series("Without pre-event alpha", (0.43, 0.52, 0.27, 1.31, 0.56)); ch.replace_data(d)
set_paras(a["PEAD note text"], ["Q5 minus Q1: +0.12% [-0.80, +0.88] without pre-event alpha.", "The pre-specified -2.6% came from subtracting 59 days of past alpha."])
s11.notes_slide.notes_text_frame.text = ("Language models: adding embeddings to a model that already holds price, metadata, accounting, word counts and TF-IDF changes macro-F1 by minus 0.001 on the locked test, with an interval that includes zero. A model with no text at all does about as well, though that check came after the locked test. "
                                         "Earnings: the pre-specified drift test showed a reversal of 2.6 percent, but that measure subtracts each stock's past-year alpha for 59 days. Without that term the spread is plus 0.12 percent, and a calendar-time portfolio gives plus 0.25 percent, t = 0.46. The survivor-only sample can still hide real drift.")

b = by(s12)
b["Text 0"].text_frame.paragraphs[0].runs[0].text = "Why the \"good news falls\" result was wrong"
c2 = b["Best-news outcomes chart"].chart
print(c2.chart_type, [x.name for x in c2.plots[0].series], list(c2.plots[0].categories), c2.has_title and c2.chart_title.text_frame.text)
d = CategoryChartData(); d.categories = ["Q1 worst", "Q2", "Q3", "Q4", "Q5 best"]; d.add_series("Subtracted as pre-event alpha, %", (-0.38, 0.04, 1.10, 2.06, 2.40)); c2.replace_data(d)
if c2.has_title:
    c2.chart_title.text_frame.text = "How much the old measure subtracted, by surprise group (%)"
set_paras(b["Outcome note text"], ["The gap between Q5 and Q1, 2.78%, is the whole reversal."])
for n, (h, t) in enumerate((("Past alpha, 59 times", "The measure subtracted each stock's past-year average outperformance on all 59 days."), ("Good news follows good years", "Mean past return: 5% in the worst-news group, 27% in the best."), ("Remove it: no reversal", "Q5 minus Q1 becomes +0.12%. Calendar-time alpha: +0.25%, t = 0.46."), ("Survivor bias remains", "Firms that kept falling are missing, so real drift could still be hidden.")), 1):
    set_paras(b[f"Reason text {n}"], [h, t])
set_paras(b["Source"], ["Found while writing the paper, after the locked test. Every robustness check shared the same term, which is why they all agreed. Source: paper/analysis/drift_alpha.py."])
s12.notes_slide.notes_text_frame.text = ("An earlier version of this slide tried to explain why good news was followed by a fall. The fall was not real. The abnormal return subtracted a pre-event alpha over the whole 59-day window. Good news tends to follow a good year, so the subtraction was largest for the best-news group. "
                                         "That one term accounts for the entire spread. It passed every robustness check because all of them used the same term. A calendar-time test, which estimates nothing before the event, finds no spread.")
p.save(F)
print("saved")
