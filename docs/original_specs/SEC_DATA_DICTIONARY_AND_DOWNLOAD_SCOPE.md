# SEC Data Dictionary and Download Scope

This file is the companion reference to `CLAUDE_CODE_SEC_DATA_ACQUISITION_PROMPT.md`. It defines the recommended entities, fields, source tiers and scope boundaries for the SEC data layer that will connect to the existing 979-stock OHLCV and three-timeframe image project.

## 1. Download tiers

| Tier | What to collect | Why |
|---|---|---|
| A | Complete metadata for every form type filed by every mapped target issuer CIK | Gives complete filing census and preserves ability to study new types later |
| B1 | Full raw documents for 8-K/8-K/A, 10-K/10-K/A, 10-Q/10-Q/A, plus foreign issuer equivalents and amendments | Core NLP and event study corpus |
| B2 | Prospectus/registration, proxy/governance, insider ownership, beneficial ownership/activism, listing/delisting and selected special reports | Financing, dilution, governance, insider and special-situation research |
| C | Company Facts JSON and filing-level XBRL/taxonomy artifacts where relevant | Financial facts and quantitative disclosure extraction |
| D | SEC Financial Statement Data Sets and Financial Statement and Notes Data Sets (2009 onward) | Optional bulk analytical cross-check, not a substitute for original filings |
| E | Third-party ownership/13D/13G matching and optionally 13F institutional filings | Separate ownership study; not fully captured by querying issuer CIK alone |
| F | Official macroeconomic releases/series and release timestamps/vintages | Later macro-regime interactions; independent module |

## 2. Core normalized tables

### `security_identity`

| Field | Meaning |
|---|---|
| `project_universe_id` | Stable row ID from user's own canonical stock universe |
| `project_universe_group` | `primary_979`, `delisted_stress_23`, or `rejected_4` |
| `ticker` | Ticker at the relevant time, not permanent key |
| `issuer_cik` | SEC registrant CIK when matched |
| `security_id` | Internal stable security identifier; do not invent CUSIP/FIGI if unavailable |
| `company_name` | Current or historically applicable name |
| `former_names_tickers` | Known aliases with validity periods/source |
| `exchange`, `share_class`, `security_type` | Distinguishes multiple listed securities and ADRs |
| `sic`, `industry`, `fiscal_year_end` | Context fields, source and timestamp preserved |
| `listing_start`, `listing_end` | Verified interval or null |
| `ohlcv_first_date`, `ohlcv_last_date` | Data coverage from user's actual files |
| `mapping_confidence`, `mapping_source`, `manual_review_status` | Prevents silent ticker/CIK guesses |
| `predecessor_cik`, `successor_cik` | Optional relation for mergers/reorganizations |

### `sec_filings`

| Field | Meaning |
|---|---|
| `accession_number` | Unique filing identifier; primary key for accession-level metadata |
| `filer_cik` | CIK that submitted this filing |
| `subject_issuer_cik` | Target issuer discussed, when different from filer; may be null |
| `form_type` | Exact SEC form label including `/A` amendments |
| `filing_date` | SEC filing date; not a substitute for intraday acceptance time |
| `acceptance_datetime_et` | Acceptance time where available from authoritative metadata/header |
| `acceptance_datetime_utc` | Normalized UTC version, preserving raw original too |
| `report_period_date` | Reporting period/period of report, not information-availability date |
| `earliest_public_datetime` | Earliest public release time only if verified |
| `earliest_public_source` | Attachment, press release, official source or unknown |
| `earliest_public_confidence` | High/medium/low/unknown |
| `items` | Original 8-K items or source metadata; keep raw and parsed versions |
| `primary_document` | Main filing filename |
| `primary_document_description` | SEC-provided description if present |
| `is_xbrl`, `is_inline_xbrl` | Flags from filing metadata where available |
| `amendment_flag`, `amends_accession` | Version relationship when resolved |
| `document_count`, `filing_size` | Source metadata or manifest-derived values |
| `selected_tier`, `selection_reason` | Scope decision with provenance |
| `retrieved_at_utc`, `source_url`, `raw_json_path` | Retrieval provenance |

### `filing_document_manifest`

Required fields include `document_id`, `accession_number`, `filer_cik`, `sequence`, `filename`, `document_type`, `description`, `archive_url`, `local_path`, `content_type`, `byte_size`, `sha256`, `retrieved_at_utc`, `http_status`, `attempt_count`, `download_status`, `parser_status`, `character_count`, `is_primary`, `is_exhibit`, `is_xbrl`, `is_pdf`, `is_graphic`, `duplicate_of`, `skip_reason`, and `validation_message`.

### `xbrl_facts_raw`

Required fields include `filer_cik`, `accession_number`, `form_type`, `filing_date`, `concept_namespace`, `concept_name`, `concept_label`, `standard_or_custom`, `unit`, `currency`, `value_raw`, `value_numeric_decimal`, `period_start`, `period_end`, `instant_date`, `fiscal_year`, `fiscal_period`, `frame`, `dimension_context`, `segment_member`, `source_document`, `source_url`, `extraction_source`, `available_at`, `parser_version`, `duplicate_or_conflict_flag`, and `quality_status`.

If a source does not provide a field, leave it null and record why/which source was used. Never manufacture a missing context or timestamp.

### `sec_sections`

Fields: `section_id`, `accession_number`, `document_id`, `parent_event_id`, `issuer_cik`, `form_type`, `section_name_raw`, `section_name_normalized`, `item_number`, `section_order`, `text_raw_or_path`, `text_hash`, `start_offset`, `end_offset`, `table_count`, `html_source`, `extraction_method`, `parser_version`, `manual_validation_flag`, `quality_status`.

### `sec_events`

Fields: `event_id`, `accession_number`, `issuer_cik`, `security_id`, `event_category`, `event_subcategory`, `form_type`, `item_number`, `source_section_id`, `source_exhibit_id`, `event_time_public`, `event_time_sec_acceptance`, `event_time_confidence`, `event_description`, `entity_names`, `financial_amounts`, `guidance_change`, `novelty_features`, `risk_features`, `sentiment_features`, `extraction_confidence`, `manual_review_status`, `overlap_group_id`, `amendment_relationship`, `feature_version`.

### `sec_event_market_alignment`

Fields: `event_id`, `security_id`, `ticker_at_event`, `ohlcv_source_hash`, `bar_frequency`, `price_convention`, `latest_pre_event_bar_end`, `first_eligible_post_event_bar`, `market_session`, `timezone`, `exchange_calendar_version`, `timestamp_quality`, `event_contamination_flag`, `concurrent_earnings_flag`, `overlap_event_count`, `benchmark_id`, `estimation_window`, `expected_return_model`, `estimation_n`, `alpha_beta_or_factor_loadings`, `abnormal_return_1d`, `car_1d`, `car_5d`, `car_20d`, `abnormal_volume`, `volatility_change`, `downside_tail_label`, `corporate_action_flag`, `dividend_adjustment_status`, `exclusion_reason`, `calculation_version`.

## 3. Forms registry

The registry should support exact form codes and prefix patterns without baking assumptions into parser code.

### Core financial/current disclosures
`8-K`, `8-K/A`, `10-K`, `10-K/A`, `10-KT`, `10-KT/A`, `10-Q`, `10-Q/A`, `10-QT`, `10-QT/A`, `20-F`, `20-F/A`, `6-K`, `6-K/A`, `40-F`, `40-F/A`.

### Financing/registration/prospectus
`S-1`, `S-1/A`, `S-3`, `S-3/A`, `S-4`, `S-4/A`, `S-8`, `F-1`, `F-1/A`, `F-3`, `F-3/A`, `F-4`, all relevant `424B*`, `EFFECT`, `8-A` variants.

### Governance/proxy
`DEF 14A`, `DEFA14A`, `PRE 14A`, `DEF 14C`, `DEFC14A`, and other discovered proxy/proposal variants in the registry.

### Ownership/insider
Forms `3`, `3/A`, `4`, `4/A`, `5`, `5/A`, `144`, `SC 13D`, `SC 13D/A`, `SC 13G`, `SC 13G/A`.

### Additional forms to inventory and review
`11-K`, `SD`, `25`, `25-NSE`, `15-12B`, `15-12G` and every other observed issuer CIK filing form. Keep all metadata even when full text is not selected.

**Important:** `SC 13D/G`, Forms 3/4/5 and 13F do not always appear in the target issuer's own submission history because they may be submitted under an investor/insider CIK. A separate relation-discovery workflow is required to find filings about target securities. 13F is optional and delayed; it is not a direct measure of intraday institutional transactions.

## 4. Key data features for research (do not substitute for raw downloads)

### Event characteristics
- filing form/item and taxonomy
- reporting period versus filing/public-release date
- amendment/revision status
- disclosure timing relative to regular trading hours
- earnings-event/concurrent news flags
- event novelty and similarity to previous reports
- source evidence and extraction confidence

### Financial metrics from XBRL/narrative
- sales/revenue, cost of revenue, gross profit, operating income, net income, EPS and share counts
- cash, current assets/liabilities, receivables/inventory, operating cash flows and capex
- debt, maturities, interest expense, leverage, lease liabilities, credit facilities/covenants where available
- share issuance/buybacks, dividends, conversions/warrants, equity/debt raises and dilution
- impairments, restructuring, stock-based compensation, tax items, restatements, going-concern statements, internal-control weaknesses
- segment/geographic values, backlog/bookings/customer concentration or other company-specific custom facts
- management guidance and changes versus earlier guidance; consensus surprise only if reliable expectations data exist

### Text/NLP variables
- financial sentiment/tone (as one feature, not a market-impact label)
- event/topic class; conditionality/uncertainty; risk language
- forward-looking/guidance text
- semantic embeddings and novelty relative to prior company disclosures
- changed risk-factor sentences/paragraphs and repeated boilerplate
- extracted entities, amounts, dates, conditions and source offsets
- text/feature model versions and reproducible hashes

## 5. Historical and source limitations to report

- EDGAR electronic availability starts in 1994/1995; not all older records are fully electronic or consistently structured.
- SEC full-text search coverage and machine-readable structured-data coverage are not identical. Do not assume every archive document can be downloaded or parsed uniformly.
- SEC financial statement datasets and structured XBRL coverage begin in 2009; pre-XBRL financial features require different extraction and will not be uniformly comparable.
- Company Facts omits some custom tags and dimensional detail; filing-level XBRL is necessary for some facts.
- Current ticker/CIK files may not describe historical ticker assignments completely.
- Filing acceptance time may not be the earliest public dissemination time.
- The existing price file is split-adjusted but not dividend-adjusted; total-return event analysis needs separate dividend treatment or a plainly stated limitation.
- Delisted/security-history coverage is currently a stress-test set, not a verified complete historical stock universe.
- Consensus analyst expectations, intraday newswire timestamps, detailed ownership relations and institutional holdings may need separate data acquisition and validation.

## 6. Source registry

| Data | Official source |
|---|---|
| SEC API docs and bulk sources | https://www.sec.gov/search-filings/edgar-application-programming-interfaces |
| EDGAR access, CIK/ticker notes, historical indexes, request policy | https://www.sec.gov/search-filings/edgar-search-assistance/accessing-edgar-data |
| Company Facts bulk ZIP | https://www.sec.gov/Archives/edgar/daily-index/xbrl/companyfacts.zip |
| Submissions bulk ZIP | https://www.sec.gov/Archives/edgar/daily-index/bulkdata/submissions.zip |
| Financial Statement Data Sets | https://www.sec.gov/dera/data/financial-statement-data-sets.html |
| Financial Statement and Notes Data Sets | https://www.sec.gov/data-research/sec-markets-data/financial-statement-notes-data-sets |
| FRED/ALFRED API docs | https://fred.stlouisfed.org/docs/api/fred/ |
| FRED historical observation vintages | https://fred.stlouisfed.org/docs/api/fred/series_observations.html |

**Design rule:** keep raw ingestion, normalized metadata, parsed text, model-derived features, event-study outcomes, and training samples as separate versioned layers. Every result must be traceable back to raw EDGAR source documents and the precise data versions used.
