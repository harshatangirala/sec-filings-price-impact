# Annotation guide (gold set v1)

Label each filing x field from the filing text only. Never look at prices or returns: price outcomes never validate semantic labels (SPEC.md §12).

- `present`: TRUE / FALSE.
- `direction`: one of the listed values, or blank.
- `magnitude` + `unit`: the stated number and unit (e.g. 250, USD millions), or blank.
- `certainty`: low / medium / high.
- `evidence_quote`: a verbatim quote of <= 30 words supporting the label.

**gold-test is labelled blind**: its template has no LLM pre-fill. gold-dev and gold-val may show a pre-fill column; treat it as a suggestion and correct it (anchoring risk noted in the report).

A second annotator should label a stratified >= 30% of gold-test independently (columns `annotator`), so Cohen's kappa can be reported. Without one, all content-validity results are labelled single-rater, provisional.

## Fields
- **guidance** - Does the filing initiate, raise, lower, withdraw or reaffirm forward-looking guidance? Directions: initiated, raised, lowered, withdrawn, reaffirmed.
- **results_tone** - Does the filing describe results as better (beat) or worse (miss) than the prior year or expectations? Directions: beat, miss, mixed.
- **impairment** - Does the filing report an impairment or write-down? Give the amount if stated. Directions: recorded, reversed.
- **restructuring** - Does the filing announce restructuring, layoffs or exit/disposal activities? Directions: announced, expanded, completed.
- **liquidity_stress** - Does the filing disclose liquidity stress or going-concern doubt? Directions: going_concern, liquidity_stress.
- **financing_covenant** - Does the filing disclose new financing, a covenant issue, waiver or default? Directions: new_financing, covenant_breach, waiver, default.
- **mna** - Does the filing announce a merger, acquisition or disposition? Directions: acquisition, disposition, merger.
- **dividend** - Does the filing announce a dividend change? Give the type and amount if stated. Directions: initiated, increased, decreased, suspended, special, maintained.
- **buyback** - Does the filing announce a share repurchase authorization or change? Directions: authorized, increased, suspended.
- **litigation_regulatory** - Does the filing disclose new litigation or regulatory action against the company? Directions: new, settled, adverse_ruling.
- **management_change** - Does the filing announce a change of CEO, CFO or board chair? Directions: departure, appointment.
- **risk_factor_change** - Compared with the firm's prior filings, are risk factors new or removed? Directions: new, removed.

Return the completed CSV files to `DATA_ROOT/inputs/gold/` with the same file names.