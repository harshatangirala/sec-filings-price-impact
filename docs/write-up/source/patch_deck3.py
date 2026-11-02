from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.util import Pt

F = r"C:\Users\harsh\Downloads\sec_deck_build\SEC_Filings_Study_13_Slides.pptx"
p = Presentation(F)
for slide, name, lo, hi in ((p.slides[10], "SUE drift chart", 0, 1.7), (p.slides[11], "Best-news outcomes chart", -1, 3)):
    ch = [s for s in slide.shapes if s.name == name][0].chart
    va = ch.value_axis; va.minimum_scale, va.maximum_scale = lo, hi
    pl = ch.plots[0]; pl.has_data_labels = True; dl = pl.data_labels; dl.number_format = '+0.00;-0.00'; dl.number_format_is_linked = False; dl.show_value = True; dl.font.size = Pt(12)
    if name.startswith("Best"):
        r = ch.chart_title.text_frame.paragraphs[0].runs[0]; r.font.size = Pt(13); r.font.color.rgb = RGBColor(0x12, 0x4E, 0x5C)
        pl.gap_width = 60
p.save(F)
