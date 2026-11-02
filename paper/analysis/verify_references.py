"""Builds paper/references.json from a scholarly index (OpenAlex), by DOI, so bibliographic details are looked up rather than typed.
Items with no DOI (conference proceedings, a book, a working paper) are listed by hand in MANUAL."""
import json
import urllib.parse
import urllib.request
from pathlib import Path

DOIS = dict(ball_brown="10.2307/2490232", bernard_thomas89="10.2307/2491062", bernard_thomas90="10.1016/0165-4101(90)90008-R",
            tetlock07="10.1111/j.1540-6261.2007.01232.x", tetlock08="10.1111/j.1540-6261.2008.01362.x", lm11="10.1111/j.1540-6261.2010.01625.x", lm16="10.1111/1475-679X.12123",
            li10="10.1111/j.1475-679X.2010.00382.x", cohen20="10.1111/jofi.12885", ke_kelly_xiu="10.3386/w26186", huang23="10.1111/1911-3846.12832", gu20="10.1093/rfs/hhaa009",
            jiang23="10.1111/jofi.13268", chordia14="10.1016/j.jacceco.2014.06.001", martineau22="10.1561/104.00000122", mclean16="10.1111/jofi.12365", hou20="10.1093/rfs/hhy131",
            harvey16="10.1093/rfs/hhv059", shumway97="10.1111/j.1540-6261.1997.tb03818.x", carhart97="10.1111/j.1540-6261.1997.tb03808.x",
            fama_macbeth="10.1086/260061", newey_west="10.2307/1913610", petersen09="10.1093/rfs/hhn053", cameron11="10.1198/jbes.2010.07136",
            bh95="10.1111/j.2517-6161.1995.tb02031.x", hansen11="10.3982/ECTA5771", bailey14="10.3905/jpm.2014.40.5.094", bailey17="10.21314/JCF.2016.322", zou05="10.1111/j.1467-9868.2005.00503.x",
            he26="10.1016/j.jfineco.2026.104374", fink24="10.1017/S0022109023000881", karolyi25="10.1287/mnsc.2023.01722", zhang24="10.1016/j.irfa.2024.103316", lan24="10.1016/j.irfa.2024.103460",
            magner25="10.1016/j.frl.2025.107472", fengler25="10.1016/j.irfa.2025.104121", glasserman23="10.2139/ssrn.4586726", lopezlira="10.2139/ssrn.4412788", kim_llm="10.2139/ssrn.4835311",
            lerman10="10.1007/s11142-009-9115-1", you_zhang09="10.1007/s11142-008-9083-x", hirshleifer09="10.1111/j.1540-6261.2009.01501.x",
            dellavigna09="10.1111/j.1540-6261.2009.01433.x", livnat06="10.1111/j.1475-679X.2006.00196.x", cormack09="10.1145/1571941.1572114", xiao24="10.1145/3626772.3657878",
            optuna="10.1145/3292500.3330701", dyer17="10.1016/j.jacceco.2017.07.002", frankel21="10.1287/mnsc.2021.4156", bochkay23="10.1111/1911-3846.12825",
            fama98="10.1016/S0304-405X(98)00026-9", kothari97="10.1016/S0304-405X(96)00899-9", brown92="10.1093/rfs/5.4.553", nosek18="10.1073/pnas.1708274114")
MANUAL = {
    "ke17": dict(authors=[["Guolin", "Ke"], ["Qi", "Meng"], ["Thomas", "Finley"], ["Taifeng", "Wang"], ["Wei", "Chen"], ["Weidong", "Ma"], ["Qiwei", "Ye"], ["Tie-Yan", "Liu"]], year=2017, title="LightGBM: A highly efficient gradient boosting decision tree", journal="Advances in Neural Information Processing Systems", volume="30", page=None),
    "catboost": dict(authors=[["Liudmila", "Prokhorenkova"], ["Gleb", "Gusev"], ["Aleksandr", "Vorobev"], ["Anna Veronika", "Dorogush"], ["Andrey", "Gulin"]], year=2018, title="CatBoost: Unbiased boosting with categorical features", journal="Advances in Neural Information Processing Systems", volume="31", page=None),
    "lewis20": dict(authors=[["Patrick", "Lewis"], ["Ethan", "Perez"], ["Aleksandra", "Piktus"], ["Fabio", "Petroni"], ["Vladimir", "Karpukhin"], ["Naman", "Goyal"], ["Heinrich", "Küttler"], ["Mike", "Lewis"], ["Wen-tau", "Yih"], ["Tim", "Rocktäschel"], ["Sebastian", "Riedel"], ["Douwe", "Kiela"]], year=2020, title="Retrieval-augmented generation for knowledge-intensive NLP tasks", journal="Advances in Neural Information Processing Systems", volume="33", page=None),
    "sarkar24": dict(authors=[["Suproteem", "Sarkar"], ["Keyon", "Vafa"]], year=2024, title="Lookahead bias in pretrained language models", journal="", volume=None, page=None, note="Working paper"),
    "lopezdeprado": dict(authors=[["Marcos", "López de Prado"]], year=2018, title="<i>Advances in Financial Machine Learning</i>", journal="", volume=None, page=None, note="Wiley, Hoboken, NJ"),
}
# name splits an index cannot infer
FAMILY_FIX = {"Marcos López de Prado": ("Marcos", "López de Prado"), "Marcos Lopez de Prado": ("Marcos", "López de Prado"), "David H. Bailey": ("David H.", "Bailey"), "Pablo A. Henríquez": ("Pablo A.", "Henríquez"), "Colin Cameron": ("A. Colin", "Cameron"), "A. Colin Cameron": ("A. Colin", "Cameron")}

keys = list(DOIS)
out = {}
for i in range(0, len(keys), 40):
    batch = keys[i:i + 40]
    u = "https://api.openalex.org/works?per-page=50&filter=doi:" + urllib.parse.quote("|".join(DOIS[k] for k in batch), safe="|:/()")
    res = json.load(urllib.request.urlopen(urllib.request.Request(u, headers={"User-Agent": "citation-check"}), timeout=60))["results"]
    by = {r["doi"].replace("https://doi.org/", "").lower(): r for r in res if r.get("doi")}
    for k in batch:
        r = by.get(DOIS[k].lower())
        if not r:
            print("NOT FOUND", k, DOIS[k]); continue
        au = []
        for a in r["authorships"]:
            nm = a["author"]["display_name"]
            if nm in FAMILY_FIX:
                au.append(list(FAMILY_FIX[nm])); continue
            parts = nm.split(); au.append([" ".join(parts[:-1]), parts[-1]])
        b = r.get("biblio") or {}; src = ((r.get("primary_location") or {}).get("source") or {})
        pg = b.get("first_page"); pg = f"{pg}-{b['last_page']}" if pg and b.get("last_page") and b["last_page"] != pg else pg
        j = (src.get("display_name") or "")
        if j.startswith("The "): j = j[4:]
        out[k] = dict(doi=DOIS[k], title=r["title"], authors=au, journal=j, volume=b.get("volume"), page=pg, year=r["publication_year"])
out.update(MANUAL)
FIX = {
    "lm11": dict(authors=[["Tim", "Loughran"], ["Bill", "McDonald"]]), "lm16": dict(authors=[["Tim", "Loughran"], ["Bill", "McDonald"]]),
    "hansen11": dict(authors=[["Peter R.", "Hansen"], ["Asger", "Lunde"], ["James M.", "Nason"]]), "hou20": dict(authors=[["Kewei", "Hou"], ["Chen", "Xue"], ["Lu", "Zhang"]], year=2020),
    "zhang24": dict(authors=[["Sijia", "Zhang"], ["Andros", "Gregoriou"], ["He", "Wu"]]), "karolyi25": dict(authors=[["Stephen", "Karolyi"], ["Thomas", "Ruchti"], ["Phong", "Truong"]], year=2025),
    "bochkay23": dict(year=2023, title="Textual analysis in accounting: What's next?"), "fink24": dict(year=2024), "frankel21": dict(year=2022), "huang23": dict(year=2023, title="FinBERT: A large language model for extracting information from financial text"),
    "mclean16": dict(year=2016), "petersen09": dict(year=2009), "bailey17": dict(year=2017, volume="20", page="39-69"), "harvey16": dict(title="... and the cross-section of expected returns"),
    "fama98": dict(title="Market efficiency, long-term returns, and behavioral finance"), "optuna": dict(title="Optuna: A next-generation hyperparameter optimization framework", journal="Proceedings of the 25th ACM SIGKDD International Conference on Knowledge Discovery and Data Mining"),
    "cormack09": dict(journal="Proceedings of the 32nd International ACM SIGIR Conference on Research and Development in Information Retrieval"), "xiao24": dict(journal="Proceedings of the 47th International ACM SIGIR Conference on Research and Development in Information Retrieval"),
    "carhart97": dict(page="57-82"), "ball_brown": dict(page="159-178"), "bernard_thomas89": dict(page="1-36"), "newey_west": dict(page="703-708"), "magner25": dict(journal="Finance Research Letters"),
    "ke_kelly_xiu": dict(journal="", note="NBER Working Paper 26186"), "glasserman23": dict(journal="", note="Working paper, SSRN"), "kim_llm": dict(journal="", note="Working paper, SSRN"),
    "lopezlira": dict(doi=None, authors=[["Alejandro", "Lopez-Lira"], ["Yuehua", "Tang"]], year=2026, title="Can ChatGPT forecast stock price movements? Return predictability and large language models", journal="Journal of Financial Economics", volume="184", page="104335"),
    "lerman10": dict(doi="10.1007/s11142-009-9115-1", authors=[["Alon", "Lerman"], ["Joshua", "Livnat"]], year=2010, title="The new Form 8-K disclosures", journal="Review of Accounting Studies", volume="15", page="752-778"),
    "you_zhang09": dict(doi="10.1007/s11142-008-9083-x", authors=[["Haifeng", "You"], ["Xiao-jun", "Zhang"]], year=2009, title="Financial reporting complexity and investor underreaction to 10-K information", journal="Review of Accounting Studies", volume="14", page="559-586"),
    "dellavigna09": dict(doi="10.1111/j.1540-6261.2009.01433.x", authors=[["Stefano", "DellaVigna"], ["Joshua M.", "Pollet"]], year=2009, title="Investor inattention and Friday earnings announcements", journal="Journal of Finance", volume="64", page="709-749"),
}
FIX.update({k: {**FIX.get(k, {}), "authors": v} for k, v in {
    "bernard_thomas89": [["Victor L.", "Bernard"], ["Jacob K.", "Thomas"]], "bernard_thomas90": [["Victor L.", "Bernard"], ["Jacob K.", "Thomas"]], "ball_brown": [["Ray", "Ball"], ["Philip", "Brown"]],
    "harvey16": [["Campbell R.", "Harvey"], ["Yan", "Liu"], ["Heqing", "Zhu"]], "cohen20": [["Lauren", "Cohen"], ["Christopher", "Malloy"], ["Quoc", "Nguyen"]], "fengler25": [["Matthias R.", "Fengler"], ["Tri Minh", "Phan"]],
    "frankel21": [["Richard", "Frankel"], ["Jared", "Jennings"], ["Joshua", "Lee"]], "hirshleifer09": [["David", "Hirshleifer"], ["Sonya Seongyeon", "Lim"], ["Siew Hong", "Teoh"]],
    "bailey17": [["David H.", "Bailey"], ["Jonathan M.", "Borwein"], ["Marcos", "L\u00f3pez de Prado"], ["Qiji Jim", "Zhu"]], "huang23": [["Allen H.", "Huang"], ["Hui", "Wang"], ["Yi", "Yang"]],
    "magner25": [["Nicol\u00e1s", "Magner"], ["Pablo A.", "Henr\u00edquez"], ["Aliro", "Sanhueza"]], "nosek18": [["Brian A.", "Nosek"], ["Charles R.", "Ebersole"], ["Alexander C.", "DeHaven"], ["David T.", "Mellor"]],
    "kim_llm": [["Alex", "Kim"], ["Maximilian", "Muhn"], ["Valeri", "Nikolaev"]], "kothari97": [["S. P.", "Kothari"], ["Jerold B.", "Warner"]], "zou05": [["Hui", "Zou"], ["Trevor", "Hastie"]],
    "dyer17": [["Travis", "Dyer"], ["Mark", "Lang"], ["Lorien", "Stice-Lawrence"]], "tetlock08": [["Paul C.", "Tetlock"], ["Maytal", "Saar-Tsechansky"], ["Sofus", "Macskassy"]], "xiao24": [["Shitao", "Xiao"], ["Zheng", "Liu"], ["Peitian", "Zhang"], ["Niklas", "Muennighoff"], ["Defu", "Lian"], ["Jian-Yun", "Nie"]],
}.items()})
for k, v in FIX.items():
    out.setdefault(k, {}).update(v)
Path(__file__).resolve().parents[1].joinpath("references.json").write_text(json.dumps(out, indent=0, ensure_ascii=False), encoding="utf-8")
for k, r in sorted(out.items()):
    print(f"{k:17s}|{'; '.join(a[1] for a in r['authors'])[:36]:36s}|{r['year']}|{(r['title'] or '')[:66]:66s}|{(r['journal'] or '')[:30]:30s}|{r['volume']}|{r['page']}")
print(len(out), "of", len(keys) + len(MANUAL))
