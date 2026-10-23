"""Parsing, sections, 8-K taxonomy (SPEC.md §18.1 #11) and coverage reason codes (#12)."""
import pandas as pd
import pytest

from secprice.config import load_yaml, repo_root
from secprice.parse.html import drop_numeric_lines, to_text
from secprice.parse.sections import eightk_items_text, extract_sections
from secprice.parse.taxonomy import Taxonomy8K
from secprice.sec.archives import parse_sgml


@pytest.fixture(scope="module")
def tax():
    return Taxonomy8K(load_yaml(repo_root() / "configs" / "taxonomy_8k_v1.yaml"))


def test_taxonomy_legacy_modern_and_catch_all(tax):
    assert tax.map_item("2.02", "2010-01-01") == "earnings"
    assert tax.map_item("2.2", "2010-01-01") == "earnings"                 # zero-padding
    assert tax.map_item("12", "2003-06-01") == "earnings"                  # legacy item 12 after 2003-03-28
    assert tax.map_item("12", "2002-06-01") == tax.catch_all               # item 12 did not exist yet
    assert tax.map_item("5", "2001-01-01") == "other_events"
    assert tax.map_item("5", "2010-01-01") == tax.catch_all                # legacy codes are not valid after 2004-08-23
    assert tax.map_item("9.99", "2010-01-01") == tax.catch_all
    assert tax.categories([], "2001-01-01", ["Results of Operations and Financial Condition"]) == ["earnings"]
    modern, legacy = tax.earnings_candidate(["5", "7"], "2001-05-01")
    assert not modern and legacy


def test_html_tables_and_hidden_xbrl():
    html = ("<html><body><div style='display:none'><ix:header>hidden 123</ix:header></div><p>Item 7. MD&amp;A</p>"
            "<table><tr><td>Diluted</td><td>$</td><td>1.23</td></tr></table><p>Results improved &#146;s</p></body></html>")
    t, n, meth = to_text(html)
    assert "hidden" not in t and "Diluted | $ | 1.23" in t and n == 1 and meth == "html"
    assert "’" in t                                                      # C1 code points mapped via cp1252
    assert "1.23" not in drop_numeric_lines(t)


def test_10k_sections_skip_toc():
    body = ("TABLE OF CONTENTS\nItem 1. Business\nItem 1A. Risk Factors\nItem 7. Management's Discussion and Analysis\n"
            "PART I\nItem 1. Business\n" + "We make widgets. " * 100 + "\nItem 1A. Risk Factors\n" + "Risks exist. " * 200 +
            "\nItem 2. Properties\nOffices.\nItem 7. Management's Discussion and Analysis of Financial Condition\n" +
            "Revenue grew. " * 300 + "\nItem 7A. Quantitative and Qualitative Disclosures About Market Risk\n" + "Rates. " * 50 +
            "\nItem 8. Financial Statements\nTables.")
    s = extract_sections(body, "10-K")
    assert set(s) == {"10K_1", "10K_1A", "10K_7", "10K_7A"}
    assert s["10K_7"]["text"].count("Revenue grew.") == 300 and "Risks exist" not in s["10K_7"]["text"]


def test_10q_parts():
    body = ("PART I - FINANCIAL INFORMATION\nItem 1. Financial Statements\nTables\nItem 2. Management's Discussion and Analysis\n" +
            "Sales rose. " * 100 + "\nItem 3. Quantitative and Qualitative Disclosures About Market Risk\n" + "FX. " * 30 +
            "\nItem 4. Controls\nOK\nPART II - OTHER INFORMATION\nItem 1A. Risk Factors\n" + "New risk. " * 40 + "\nItem 6. Exhibits\n")
    s = extract_sections(body, "10-Q")
    assert {"10Q_MDA", "10Q_MR", "10Q_RF"} <= set(s)


def test_sgml_parse_real_fixture():
    from pathlib import Path
    raw = (Path(__file__).resolve().parents[1] / "fixtures" / "aapl_8k_2002_full.txt").read_bytes()
    sub = parse_sgml(raw)
    assert sub.acceptance_raw == "20020813164730"
    assert any(d.type.upper().startswith("8-K") for d in sub.documents)
    assert "Other events" in " ".join(sub.item_titles)
    main = [d for d in sub.documents if d.type.upper().startswith("8-K")][0]
    t, _, _ = to_text(main.text)
    assert eightk_items_text(t)["text"]


def test_coverage_reason_codes_no_8k_expectation():
    from secprice.stages_data import periodic_expectations
    per = pd.DataFrame({"cik": [1] * 5, "ry": [2020, 2020, 2020, 2020, 2021],
                        "base_form": ["10-K", "10-Q", "10-Q", "10-Q", "10-Q"]})
    cov = periodic_expectations(per)
    assert cov.set_index("year").loc[2020, "reason_code"] == "OK"
    assert cov.set_index("year").loc[2021, "reason_code"] == "MISSING_10K"
    assert not any("8-K" in c for c in cov.columns)


def test_ex13_mdna_found_under_varied_annual_report_titles():
    """Real EX-13s head the MD&A 'FINANCIAL REVIEW' (AXP, PFE) or 'Management Discussion' (IBM); found on 1999-2004 filings."""
    from secprice.parse.sections import mdna_from_ex13
    body = "Revenues rose on higher volume and pricing across all segments of the business. " * 60
    for title in ("FINANCIAL REVIEW", "Management Discussion", "MANAGEMENT'S DISCUSSION AND ANALYSIS OF FINANCIAL CONDITION AND RESULTS"):
        text = f"Letter to shareholders\n\n{title}\n\n{body}\n\nConsolidated Statements of Income\n\nnumbers"
        out = mdna_from_ex13(text)
        assert out is not None and out["method"] == "ex13_title" and len(out["text"]) > 3000, title
    assert mdna_from_ex13("Letter\n\nFinancial review of the year was positive.\n\n" + body) is None   # not a heading line
