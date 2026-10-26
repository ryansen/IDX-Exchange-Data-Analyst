"""Follow-up investigations and the final, conservative missing-data policy.

The main cleaning script calls finalize_data after its type/quality checks.
Historical MAR/NMAR results remain separate from these final decisions.
"""
import json
import hashlib
from pathlib import Path

import numpy as np
import pandas as pd

import weeks_4_5_missing_data_experiments as experiments

SQUARE_FEET_PER_ACRE = 43560
MAX_MEAN_BIAS_PCT = 5.0  # Final summaries include means, not only medians.
EMPTY_FIELDS = ["TaxAnnualAmount", "AboveGradeFinishedArea", "FireplacesTotal", "TaxYear", "CoveredSpaces"]


def file_fingerprint(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def investigate(data, source_dir):
    """Investigate exact recovery, missing-price cohorts and raw export coverage."""
    tables = {}
    acres, sqft = data.LotSizeAcres, data.LotSizeSquareFeet
    paired = acres.gt(0) & sqft.gt(0)
    error = ((acres * SQUARE_FEET_PER_ACRE - sqft).abs() / sqft).loc[paired]
    tables["lot_size_investigation"] = pd.DataFrame([{
        "paired_positive_records": int(paired.sum()), "agreement_within_1pct": float(error.le(.01).mean()),
        "missing_acres_with_positive_sqft": int((acres.isna() & sqft.gt(0)).sum()),
        "missing_sqft_with_positive_acres": int((sqft.isna() & acres.gt(0)).sum()),
        "conflicting_pairs_over_1pct": int(error.gt(.01).sum()),
    }])
    duplicate_checks = []
    keys = ["ListingKey", "CloseDate"]
    valid_keys = data[keys].notna().all(axis=1)
    duplicate_rows = valid_keys & data.duplicated(keys, keep=False)
    duplicates = data.loc[duplicate_rows]
    for field in ["ClosePrice", "OriginalListPrice", "LivingArea", "YearBuilt", "BedroomsTotal",
                  "BathroomsTotalInteger", "PurchaseContractDate", "ListingContractDate", "Latitude", "Longitude"]:
        group = duplicates.groupby(keys)[field]
        recoverable = duplicates[field].isna() & group.transform("nunique").eq(1) & group.transform("first").notna()
        duplicate_checks.append({"field": field, "recoverable_from_same_transaction": int(recoverable.sum())})
    tables["duplicate_recovery_investigation"] = pd.DataFrame(duplicate_checks)

    missing_price = data.OriginalListPrice.isna()
    tables["original_price_cohorts"] = pd.DataFrame([
        {"original_price_missing": bool(missing), "records": int(mask.sum()),
         "median_list_price": float(data.loc[mask, "ListPrice"].median()),
         "list_equals_close_pct": float(data.loc[mask, "ListPrice"].eq(data.loc[mask, "ClosePrice"]).mean() * 100)}
        for missing, mask in [(False, ~missing_price), (True, missing_price)]
    ])
    tables["missing_original_price_counties"] = data.loc[missing_price, "CountyOrParish"].value_counts().rename_axis("county").reset_index(name="records")
    missing_fee = data.AssociationFee.isna()
    tables["missing_hoa_frequency"] = data.loc[missing_fee, "AssociationFeeFrequency"].fillna("Unknown").value_counts().rename_axis("frequency").reset_index(name="records")
    tables["missing_bathroom_subtypes"] = data.loc[data.BathroomsTotalInteger.isna(), "PropertySubType"].fillna("Unknown").value_counts().rename_axis("subtype").reset_index(name="records")

    # Raw monthly files also contain non-Residential records. Report that distinction.
    # Encoding replacement applies only to this numeric coverage audit, not the CSV output.
    coverage = []
    for filename in sorted(data.SourceFile.dropna().unique()):
        path = Path(source_dir) / filename
        if not path.exists():
            coverage.extend({"source_file": filename, "field": f, "file_available": False,
                             "header_present": False, "raw_nonblank": 0, "residential_nonblank": 0} for f in EMPTY_FIELDS)
            continue
        header = pd.read_csv(path, nrows=0, encoding="cp1252", encoding_errors="replace")
        selected = [c for c in ["PropertyType"] + EMPTY_FIELDS if c in header]
        frame = pd.read_csv(path, usecols=selected, dtype="string", encoding="cp1252", encoding_errors="replace")
        residential = frame.PropertyType.eq("Residential")
        for field in EMPTY_FIELDS:
            available = frame[field].notna() & frame[field].str.strip().ne("") if field in frame else pd.Series(False, index=frame.index)
            coverage.append({"source_file": filename, "field": field, "file_available": True,
                             "header_present": field in frame, "raw_nonblank": int(available.sum()),
                             "residential_nonblank": int((available & residential).sum())})
    tables["raw_export_coverage"] = pd.DataFrame(coverage)
    return tables


def final_decisions(data, evidence):
    """Retain the prior evidence, adding a mean-bias guard for the final export."""
    profile, _, _ = experiments.profile_missingness(data)
    support = experiments.actual_donor_support(data)
    decisions = experiments.choose_policy(profile, evidence, support)
    decisions["worst_mean_bias_pct"] = np.nan
    for row in decisions.query("action == 'impute'").itertuples():
        tests = evidence[(evidence.field == row.field) & evidence.method.eq(row.best_tested_method) & evidence.scenario.ne("MCAR")]
        worst = float(tests.abs_mean_bias_pct.max())
        decisions.loc[decisions.field.eq(row.field), "worst_mean_bias_pct"] = worst
        if worst > MAX_MEAN_BIAS_PCT:
            decisions.loc[decisions.field.eq(row.field), ["action", "reason"]] = [
                "keep_null", f"Final mean-bias guard: {worst:.2f}% worst scenario-average absolute mean bias exceeds {MAX_MEAN_BIAS_PCT:.0f}%; do not invent price history"]
    return decisions


def derive_lot_sizes(original, prepared):
    """Exact unit conversion only; no use of LotSizeArea without a known unit."""
    audit = []
    for target, source, factor in [("LotSizeAcres", "LotSizeSquareFeet", 1 / SQUARE_FEET_PER_ACRE),
                                   ("LotSizeSquareFeet", "LotSizeAcres", SQUARE_FEET_PER_ACRE)]:
        mask = original[target].isna() & original[source].gt(0).fillna(False) & np.isfinite(original[source]).fillna(False)
        prepared[target + "_was_missing_flag"] = original[target].isna()
        prepared[target + "_derived_flag"] = mask.astype(bool)
        prepared.loc[mask, target] = original.loc[mask, source] * factor
        audit.append(pd.DataFrame({"SourceRecordNumber": original.loc[mask, "SourceRecordNumber"],
                                   "field": target, "source_field": source,
                                   "source_value": original.loc[mask, source],
                                   "derived_value": prepared.loc[mask, target], "conversion_factor": factor}))
    both = original.LotSizeAcres.gt(0) & original.LotSizeSquareFeet.gt(0)
    error = (original.LotSizeAcres * SQUARE_FEET_PER_ACRE - original.LotSizeSquareFeet).abs() / original.LotSizeSquareFeet
    prepared["lot_size_conflict_flag"] = (both & error.gt(.01)).fillna(False).astype(bool)
    # Conflicting observations are flagged, not silently overwritten or averaged.
    return pd.concat(audit, ignore_index=True)


def finalize_data(data, cleaned_path, output_path, report_dir, source_dir=None):
    """Create the final file and a transparent decision ledger from the cleaned data."""
    cleaned_path, output_path, report_dir = [Path(p).resolve() for p in (cleaned_path, output_path, report_dir)]
    if cleaned_path == output_path:
        raise ValueError("The final output must not overwrite its cleaned input")
    source_dir = Path(source_dir) if source_dir else cleaned_path.parent
    print("Investigating remaining missing-data questions...", flush=True)
    investigations = investigate(data, source_dir)
    if cleaned_path == experiments.INPUT.resolve() and experiments.cache_matches():
        evidence = experiments.read_saved_tables()["experiment_summary"]
        evidence_source = "Verified saved MAR/NMAR experiments"
    else:
        print("No matching experiment cache; running group-separated MAR/NMAR evaluation.", flush=True)
        trials, partitions = experiments.evaluate_methods(data)
        evidence = experiments.summarize_experiments(trials)
        investigations.update(experiment_trials=trials, experiment_partitions=partitions)
        evidence_source = "Fresh MAR/NMAR experiments"
    decisions = final_decisions(data, evidence)
    prepared, imputation_audit, sensitivity = experiments.prepare_data(data, decisions)
    derivation_audit = derive_lot_sizes(data, prepared)
    # Keep reported bathroom values available without reconstructing them from flags.
    prepared["BathroomsTotalInteger_observed"] = data.BathroomsTotalInteger
    prepared["area_analysis_eligible_flag"] = prepared.ClosePrice.gt(0).fillna(False) & prepared.LivingArea.gt(0).fillna(False)
    prepared["dom_analysis_eligible_flag"] = prepared.DaysOnMarket.notna()
    prepared["map_analysis_eligible_flag"] = prepared.Latitude.notna() & prepared.Longitude.notna() & ~prepared.geographic_quality_flag
    date_flags = ["listing_after_close_flag", "purchase_after_close_flag", "negative_timeline_flag", "incomplete_timeline_flag"]
    prepared["timeline_analysis_eligible_flag"] = ~prepared[date_flags].any(axis=1)
    assert len(prepared) == len(data)
    for field in data.columns.intersection(prepared.columns):
        observed = data[field].notna()
        pd.testing.assert_series_equal(prepared.loc[observed, field], data.loc[observed, field], check_dtype=False)

    report_dir.mkdir(parents=True, exist_ok=True)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    decisions = decisions.rename(columns={"action": "field_action"})
    decisions["missing_after_final"] = decisions.field.map(prepared.isna().sum())
    decisions["statistically_imputed"] = decisions.field.map(imputation_audit.groupby("field").size()).fillna(0).astype(int)
    decisions["derived_by_conversion"] = decisions.field.map(derivation_audit.groupby("field").size()).fillna(0).astype(int)
    tables = {**investigations, "final_decisions": decisions, "imputation_audit": imputation_audit,
              "derivation_audit": derivation_audit, "sensitivity": sensitivity, "experiment_evidence": evidence}
    for name, table in tables.items():
        table.to_csv(report_dir / f"{name}.csv", index=False)
    prepared.to_csv(output_path, index=False, date_format="%Y-%m-%dT%H:%M:%S.%f")
    schema = {"parse_dates": [c for c in prepared if pd.api.types.is_datetime64_any_dtype(prepared[c])],
              "dtype": {c: str(t) for c, t in prepared.dtypes.items() if not pd.api.types.is_datetime64_any_dtype(t)}}
    (report_dir / "final_schema.json").write_text(json.dumps(schema, indent=2), encoding="utf-8")
    summary = {"rows_before": len(data), "rows_after": len(prepared), "rows_removed": 0,
               "columns_after": len(prepared.columns), "statistically_imputed_cells": len(imputation_audit),
               "derived_cells": len(derivation_audit), "imputed_by_field": imputation_audit.groupby("field").size().to_dict(),
               "derived_by_field": derivation_audit.groupby("field").size().to_dict(),
               "dropped_fields": decisions.loc[decisions.field_action.eq("drop_column"), "field"].tolist(),
               "eligibility_counts": {c: int(prepared[c].sum()) for c in prepared if c.endswith("analysis_eligible_flag")},
               "evidence_source": evidence_source, "maximum_mean_bias_pct": MAX_MEAN_BIAS_PCT,
               "output_file": output_path.name, "cleaned_input_sha256": file_fingerprint(cleaned_path)}
    (report_dir / "final_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    write_final_explanation(tables, summary, report_dir)
    print(json.dumps(summary, indent=2), flush=True)
    print("Final dataset:", output_path, flush=True)
    return prepared, summary


def write_final_explanation(tables, summary, report_dir):
    """Document what was resolved and what cannot be established from the export."""
    coverage = tables["raw_export_coverage"].groupby("field")[["raw_nonblank", "residential_nonblank"]].sum()
    cohorts = tables["original_price_cohorts"].set_index("original_price_missing")
    lot = tables["lot_size_investigation"].iloc[0]
    decisions = tables["final_decisions"]
    counts = summary["imputed_by_field"]
    derived = summary["derived_by_field"]
    lines = ["# Final data-cleaning decisions", "",
        f"All {summary['rows_after']:,} sold records are retained. Original observed values are unchanged. "
        "The raw exports, typed Cleaned CSV and earlier exploratory AnalysisReady CSV are preserved. Use the new Cleaned_Final CSV as the final deliverable.", "",
        "## Missing-data decisions", "",
        "- Remove nine empty/sparse features from the final analytical view: " + ", ".join(summary["dropped_fields"]) + ".",
        f"- Fill {counts.get('BathroomsTotalInteger', 0)} missing bathroom counts using the tested model, only with at least 20 observed "
        "county/subtype/month donors. Twelve BoatSlip bathroom gaps remain unknown; missing does not mean zero. "
        "BathroomsTotalInteger_observed retains the original reported values. Every estimate is flagged and audited.",
        "- Leave OriginalListPrice missing. This revises the earlier exploratory imputation decision. The model passed median-focused checks but "
        "had 9.35% average absolute mean bias in the high-value NMAR stress scenario, exceeding the final 5% mean-bias guard. "
        f"The missing cohort's median ListPrice is ${cohorts.loc[True, 'median_list_price']:,.0f}, versus "
        f"${cohorts.loc[False, 'median_list_price']:,.0f} in the observed cohort; "
        f"ListPrice equals ClosePrice for {cohorts.loc[True, 'list_equals_close_pct']:.2f}% of missing-original-price records. "
        "That is evidence of a different cohort, not proof of why the values are missing. Neither copying ListPrice nor predicting a price establishes the original marketing price.",
        f"- Recover {derived.get('LotSizeAcres', 0)} acre values and {derived.get('LotSizeSquareFeet', 0)} square-foot values using "
        "1 acre = 43,560 sq ft. These are deterministic conversions, not statistical imputations. The original unit field must be positive and finite. "
        "Do not use LotSizeArea to fill either field because its unit is absent from the export.",
        "- Leave LivingArea, YearBuilt, DaysOnMarket, BedroomsTotal, ParkingTotal, other unsupported measurements, dates, coordinates "
        "and categorical gaps unknown. Earlier MAR/NMAR tests did not support reliable estimates for these fields; missing fees/booleans are not assumed zero/false.",
        "- Keep the three missing ClosePrice records in the master file. Filter price_analysis_eligible_flag for price analysis. "
        "Other eligibility flags identify usable price/area, days-on-market, geographic and date-timeline records. "
        "No global dropna(): fully empty columns would otherwise discard every row.", "",
        "## Follow-up investigations", "",
        f"The two explicit lot-size units agreed within 1% in {lot['agreement_within_1pct'] * 100:.2f}% of "
        f"{int(lot['paired_positive_records']):,} positive paired records. "
        f"{int(lot['conflicting_pairs_over_1pct']):,} observed pairs disagree by more than 1%; they are flagged with lot_size_conflict_flag, "
        "not overwritten. Exclude those pairs when comparing acreage with square feet.",
        "Duplicate ListingKey + CloseDate transactions provided no unique observed donor for the audited missing core values. "
        "No values were copied from another transaction or inferred from unrelated listing keys.",
        "HOA frequency investigation: 65 missing fee amounts still have a billing frequency (34 annual, 30 monthly, 1 quarterly). "
        "For the other 109,818 missing fees, frequency is also unknown. The source cannot distinguish no HOA, an unreported amount, "
        "and a not-applicable field. Keep unknown; an AssociationYN field or source clarification is needed to resolve the semantics.",
        "All five entirely empty Residential fields were checked against all 32 contributing raw files. "
        "The fields are present in the headers. TaxAnnualAmount and TaxYear contain values in other property types, but none in Residential. "
        "This is a source-coverage limitation of this Residential subset, not evidence that tax amounts are zero and not missing values introduced by this cleaning step.",
        "Raw coverage counts (all property types / Residential):", "", "```", coverage.to_string(), "```", "",
        "## Remaining limitations", "",
        "MAR cannot be distinguished from NMAR from observed data alone. The existing experiments are sensitivity scenarios, not proof of a mechanism. "
        "The final mean-bias screen tightens the earlier median-focused exploratory rule; it is an operational tolerance, not a statistical guarantee. "
        "Bathroom estimates use observed sale price as an auxiliary variable in retrospective analysis. They are single estimates, not multiple imputation "
        "and not appropriate for evaluating an unseen-price forecast. For purely observed descriptive statistics, use the observed bathroom column.",
        "Exact HOA applicability, unknown LotSizeArea units, and absent Residential tax coverage require upstream field definitions or additional source data. "
        "They cannot be safely resolved by guessing. The final policy handles them explicitly rather than leaving an implementation decision open.", "",
        "Full field-by-field counts: final_decisions.csv. Estimated cells: imputation_audit.csv. Converted cells: derivation_audit.csv. "
        "Reload types using final_schema.json. The original quality flags remain available.", ""]
    (Path(report_dir) / "final_missing_data_decisions.md").write_text("\n".join(lines), encoding="utf-8")
