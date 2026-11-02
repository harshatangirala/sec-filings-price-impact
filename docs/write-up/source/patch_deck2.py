from pptx import Presentation

F = r"C:\Users\harsh\Downloads\sec_deck_build\SEC_Filings_Study_13_Slides.pptx"
p = Presentation(F)
for sh in p.slides[10].shapes:
    if sh.name == "Text bullets":
        q = sh.text_frame.paragraphs[2]; q.runs[0].text = "Numbers beat price-only by +0.033; embeddings add nothing"
        for r in q.runs[1:]: r.text = ""
    if getattr(sh, "has_chart", False) and sh.has_chart and sh.chart.has_title:
        print("chart title:", sh.chart.chart_title.text_frame.text)
for i, s in enumerate(p.slides, 1):
    for sh in s.shapes:
        if sh.has_text_frame:
            t = sh.text_frame.text
            if any(w in t.lower() for w in ("revers", "drift", "good news", "text adds", "text did", "good-news", "best-news", "pead")):
                print(i, sh.name, "|", t[:170].replace("\n", " / "))
    if s.has_notes_slide:
        t = s.notes_slide.notes_text_frame.text
        if any(w in t.lower() for w in ("revers", "drift", "good news", "pead")) and i not in (11, 12):
            print(i, "NOTES |", t[:400])
p.save(F)
