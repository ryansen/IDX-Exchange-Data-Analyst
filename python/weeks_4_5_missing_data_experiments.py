"""MAR/NMAR sensitivity experiments for the cleaned Residential SOLD data.

This is retrospective descriptive analysis, not a forecasting pipeline or proof
of a missingness mechanism. NMAR (also called MNAR) cannot be identified from
observed data alone. Separate original, experimental and prepared data are kept.
Run this script, or use the short experiment cells in the cleaning notebook.
"""
from __future__ import annotations

import json
import os
import hashlib
import inspect
from pathlib import Path

# Keep repeated local experiments from exhausting desktop CPU threads.
os.environ.setdefault("OMP_NUM_THREADS", "4")
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder

ROOT = Path(__file__).resolve().parent.parent
INPUT = ROOT / "csv/CRMLSSold_Combined_Residential_202401_202608_With_MortgageRates_Cleaned.csv"
REPORT_DIR = ROOT / "reports/weeks_4_5/missing_data"
OUTPUT = INPUT.with_name(INPUT.stem + "_AnalysisReady.csv")
TARGETS = ["LivingArea", "DaysOnMarket", "YearBuilt", "BedroomsTotal", "BathroomsTotalInteger",
           "ParkingTotal", "OriginalListPrice", "GarageSpaces", "Stories", "LotSizeAcres", "AssociationFee"]
NUMERIC_PREDICTORS = ["ListPrice", "ClosePrice", "LivingArea", "DaysOnMarket", "YearBuilt",
                      "BedroomsTotal", "BathroomsTotalInteger"]
CATEGORICAL_PREDICTORS = ["CountyOrParish", "PropertySubType", "SourceMonth"]
METHODS = ["delete", "median", "group_median", "model"]
SCENARIOS = ["MCAR", "MAR", "NMAR_high", "NMAR_low"]
POLICY = {"max_missing_pct_for_imputation": 1.0, "max_worst_median_error_iqr": 0.25,
          "max_worst_population_median_bias_pct": 2.0, "min_improvement_over_median_pct": 10.0,
          "min_group_donors": 20,
          "mask_fraction": 0.20, "sample_size": 80000,
          "split_seeds": [42, 123, 2026], "mask_seeds": [7, 19]}
SOURCES = {
    "missingness": "https://www.ncbi.nlm.nih.gov/books/NBK209895/",
    "leakage": "https://scikit-learn.org/stable/common_pitfalls.html",
}


def load_data():
    schema = json.loads((REPORT_DIR.parent / "schema.json").read_text(encoding="utf-8"))
    return pd.read_csv(INPUT, dtype=schema["dtype"], parse_dates=schema["parse_dates"])


def profile_missingness(data):
    """Describe original fields, complete cases and missingness by observed groups."""
    fields = [c for c in data if not c.endswith("_flag") and c != "SourceRecordNumber"]
    profile = pd.DataFrame({"field": fields, "missing_count": data[fields].isna().sum().values,
                            "observed_count": data[fields].notna().sum().values})
    profile["missing_pct"] = profile.missing_count / len(data) * 100
    profile = profile.sort_values("missing_count", ascending=False).reset_index(drop=True)
    core = ["CloseDate", "ClosePrice", "LivingArea", "DaysOnMarket", "BedroomsTotal", "BathroomsTotalInteger", "YearBuilt"]
    complete_cases = pd.DataFrame([
        {"scope": name, "required_fields": ", ".join(columns),
         "rows_retained": int(data[columns].notna().all(axis=1).sum()),
         "rows_lost": int(data[columns].isna().any(axis=1).sum())}
        for name, columns in [("all_original_fields", fields), ("core_housing_metrics", core),
                              ("core_plus_coordinates", core + ["Latitude", "Longitude"])]
    ])
    groups = []
    for group in CATEGORICAL_PREDICTORS:
        for field in [f for f in fields if data[f].isna().any()]:
            table = data[field].isna().groupby(data[group], dropna=False).agg(["sum", "size"])
            table.columns = ["missing_count", "rows"]
            table["missing_pct"] = table.missing_count / table.rows * 100
            table = table.reset_index().rename(columns={group: "group_value"})
            table.insert(0, "group_field", group)
            table.insert(0, "field", field)
            groups.append(table)
    return profile, complete_cases, pd.concat(groups, ignore_index=True)


def predictors(data, target):
    """Never use a target, its missing flag, or a redundant lot-size/coordinate alias."""
    numeric = [c for c in NUMERIC_PREDICTORS if c != target]
    # All remaining variables are observed at the time of retrospective analysis.
    x = data[numeric + CATEGORICAL_PREDICTORS].copy()
    for c in numeric:
        x[c] = pd.to_numeric(x[c], errors="coerce").astype(float).replace([np.inf, -np.inf], np.nan)
    for c in CATEGORICAL_PREDICTORS:
        x[c] = x[c].astype("string").fillna("Unknown").astype(str)
    return x, numeric


def estimator(numeric, classification=False, seed=42):
    # Each pipeline learns medians and category encodings only from its training rows.
    preparation = ColumnTransformer([
        ("numeric", SimpleImputer(strategy="median", add_indicator=True), numeric),
        ("categories", OneHotEncoder(handle_unknown="ignore", sparse_output=False,
                                     min_frequency=30, max_categories=40), CATEGORICAL_PREDICTORS),
    ])
    options = dict(max_iter=70, max_leaf_nodes=15, min_samples_leaf=40,
                   l2_regularization=10, random_state=seed, early_stopping=False)
    model = (HistGradientBoostingClassifier(class_weight="balanced", **options) if classification else
             HistGradientBoostingRegressor(loss="absolute_error", **options))
    return make_pipeline(preparation, model)


def diagnose_missingness(data):
    """Predict missingness out of sample. Association supports informative gaps, not MAR proof."""
    rows = []
    for target in TARGETS + ["Latitude", "ViewYN", "PoolPrivateYN"]:
        flag = data[target].isna()
        present = data.index[~flag]
        absent = data.index[flag]
        row = {"field": target, "missing_count": len(absent), "missing_pct": flag.mean() * 100,
               "auc": np.nan, "diagnostic": "Too few missing values for a stable held-out diagnostic"}
        if min(len(absent), len(present)) >= 30:
            rng = np.random.default_rng(42)
            indices = np.concatenate([rng.choice(absent, min(len(absent), 20000), replace=False),
                                      rng.choice(present, min(len(present), 40000), replace=False)])
            sample = data.loc[indices]
            x, numeric = predictors(sample, target)
            y = flag.loc[indices].to_numpy(dtype=int)
            train, test = next(GroupShuffleSplit(n_splits=1, test_size=0.30, random_state=42).split(
                x, groups=sample.ListingKey.fillna(sample.SourceRecordNumber.astype(str))))
            if min(y[train].sum(), y[test].sum()) >= 10 and len(np.unique(y[test])) == 2:
                model = estimator(numeric, classification=True)
                model.fit(x.iloc[train], y[train])
                row["auc"] = roc_auc_score(y[test], model.predict_proba(x.iloc[test])[:, 1])
                row["train_rows"], row["test_rows"] = len(train), len(test)
                row["diagnostic"] = ("Observed characteristics predict missingness; do not assume MCAR"
                                     if row["auc"] >= 0.65 else "Weak measured association; MCAR/MAR/NMAR remain possible")
        rows.append(row)
        print(f"Missingness diagnostic: {target}; AUC={row['auc']:.3f}", flush=True)
    return pd.DataFrame(rows)


def group_predictions(train, test, target):
    """County/subtype medians, falling back to county, then global training median."""
    donor = train[CATEGORICAL_PREDICTORS[:2] + [target]].copy()
    for column in CATEGORICAL_PREDICTORS[:2]:
        donor[column] = donor[column].astype("string").fillna("Unknown")
    keys = test[CATEGORICAL_PREDICTORS[:2]].astype("string").fillna("Unknown")
    pair = donor.groupby(CATEGORICAL_PREDICTORS[:2])[target].agg(["median", "count"])
    lookup = pd.MultiIndex.from_frame(keys)
    estimates = pair["median"].where(pair["count"] >= POLICY["min_group_donors"]).reindex(lookup).to_numpy(dtype=float)
    county = donor.groupby("CountyOrParish")[target].agg(["median", "count"])
    fallback = keys.CountyOrParish.map(county["median"].where(county["count"] >= POLICY["min_group_donors"])).to_numpy(dtype=float)
    estimates = np.where(np.isnan(estimates), fallback, estimates)
    return np.where(np.isnan(estimates), float(train[target].median()), estimates)


def mask_indices(frame, truth, scenario, seed, fraction=0.20):
    """Synthetic masks: MAR uses only observed X; NMAR explicitly uses held-out y."""
    rng = np.random.default_rng(seed)
    if scenario == "MCAR":
        score = np.zeros(len(frame))
    elif scenario == "MAR":
        price = pd.to_numeric(frame.ListPrice, errors="coerce")
        score = price.rank(pct=True).fillna(0.5).to_numpy()
        score += 0.5 * frame.PropertySubType.eq("Condominium").fillna(False).to_numpy(dtype=float)
    else:
        score = pd.Series(truth).rank(pct=True).to_numpy()
        if scenario == "NMAR_low":
            score = 1 - score
    weights = np.exp(4 * (score - np.max(score)))
    count = max(1, min(len(frame) - 1, int(round(fraction * len(frame)))))
    return rng.choice(len(frame), size=count, replace=False, p=weights / weights.sum())


def evaluate_methods(data):
    """3 group-separated splits x 2 masks x 4 scenarios x 4 methods per field.

    Hide 20% of known hold-out values (a stress test, not the actual missing rate).
    Donors, preprocessing and regressors never see hold-out targets. No listing
    key occurs in both the training and evaluation sets. Evaluate all true values,
    including outliers; do not silently trim them to flatter an imputer.
    """
    results, partitions = [], []
    for target in TARGETS:
        available = data.loc[data[target].notna() & data[target].ge(0)].copy()
        if len(available) > POLICY["sample_size"]:
            available = available.sample(POLICY["sample_size"], random_state=42)
        x, numeric = predictors(available, target)
        for split_seed in POLICY["split_seeds"]:
            train_ix, test_ix = next(GroupShuffleSplit(n_splits=1, test_size=0.30, random_state=split_seed).split(
                x, groups=available.ListingKey.fillna(available.SourceRecordNumber.astype(str))))
            train, test = available.iloc[train_ix], available.iloc[test_ix]
            assert not set(train.ListingKey.dropna()) & set(test.ListingKey.dropna())
            truth = test[target].to_numpy(dtype=float)
            donor_y = train[target].to_numpy(dtype=float)
            scale = max(float(np.quantile(donor_y, .75) - np.quantile(donor_y, .25)), 1.0)
            model = estimator(numeric, seed=split_seed)
            model.fit(x.iloc[train_ix], np.log1p(donor_y))
            predictions = {"median": np.full(len(test), np.median(donor_y)),
                           "group_median": group_predictions(train, test, target),
                           "model": np.maximum(0, np.expm1(model.predict(x.iloc[test_ix])))}
            if target in ["DaysOnMarket", "YearBuilt", "BedroomsTotal", "BathroomsTotalInteger"]:
                predictions = {name: np.rint(values) for name, values in predictions.items()}
            partitions.append({"field": target, "split_seed": split_seed,
                               "train_rows": len(train), "evaluation_rows": len(test), "overlapping_listing_keys": 0})
            for scenario in SCENARIOS:
                for mask_seed in POLICY["mask_seeds"]:
                    hidden = mask_indices(test, truth, scenario, split_seed + mask_seed, POLICY["mask_fraction"])
                    kept = np.ones(len(test), dtype=bool)
                    kept[hidden] = False
                    for method in METHODS:
                        reconstructed = truth[kept] if method == "delete" else truth.copy()
                        if method != "delete":
                            reconstructed[hidden] = predictions[method][hidden]
                            error = np.abs(predictions[method][hidden] - truth[hidden])
                        results.append({
                            "field": target, "scenario": scenario, "method": method,
                            "split_seed": split_seed, "mask_seed": mask_seed, "hidden_count": len(hidden),
                            "mae": np.nan if method == "delete" else float(error.mean()),
                            "median_absolute_error": np.nan if method == "delete" else float(np.median(error)),
                            "median_error_iqr": np.nan if method == "delete" else float(np.median(error) / scale),
                            "population_mean_bias_pct": 100 * float(np.mean(reconstructed) - np.mean(truth)) / max(abs(float(np.mean(truth))), 1),
                            "population_median_bias_pct": 100 * float(np.median(reconstructed) - np.median(truth)) / max(abs(float(np.median(truth))), 1),
                            "retained_fraction": len(reconstructed) / len(truth),
                        })
            print(f"Experiment complete: {target}, split {split_seed}", flush=True)
    return pd.DataFrame(results), pd.DataFrame(partitions)


def summarize_experiments(trials):
    table = trials.copy()
    table["absolute_mean_bias_pct"] = table.population_mean_bias_pct.abs()
    table["absolute_median_bias_pct"] = table.population_median_bias_pct.abs()
    return table.groupby(["field", "scenario", "method"], as_index=False).agg(
        repetitions=("split_seed", "size"), mae=("mae", "mean"),
        median_absolute_error=("median_absolute_error", "mean"), median_error_iqr=("median_error_iqr", "mean"),
        mean_bias_pct=("population_mean_bias_pct", "mean"), abs_mean_bias_pct=("absolute_mean_bias_pct", "mean"),
        median_bias_pct=("population_median_bias_pct", "mean"), abs_median_bias_pct=("absolute_median_bias_pct", "mean"),
        worst_abs_median_bias_pct=("absolute_median_bias_pct", "max"))


def matching_donor_counts(data, target):
    """Observed donors matching county, property subtype and source month."""
    keys = data[CATEGORICAL_PREDICTORS].astype("string").fillna("Unknown")
    counts = data[target].notna().groupby([keys[c] for c in CATEGORICAL_PREDICTORS]).sum()
    result = counts.reindex(pd.MultiIndex.from_frame(keys)).fillna(0).to_numpy(dtype=int)
    return pd.Series(result, index=data.index)


def actual_donor_support(data):
    rows = []
    for field in TARGETS:
        missing = data[field].isna()
        counts = matching_donor_counts(data, field)[missing]
        rows.append({"field": field, "actual_missing_count": len(counts),
                     "supported_missing_count": int(counts.ge(POLICY["min_group_donors"]).sum()),
                     "zero_donor_count": int(counts.eq(0).sum()),
                     "donor_coverage_pct": float(counts.ge(POLICY["min_group_donors"]).mean() * 100) if len(counts) else 100.0})
    return pd.DataFrame(rows)


def choose_policy(profile, experiments, support=None):
    """A predeclared, conservative screen for descriptive predictor imputation.

    These are operational tolerances, not universal statistical thresholds.
    Keep uncertainty flags. Never manufacture outcomes, dates or coordinates.
    """
    rows = []
    integer = ["YearBuilt", "BedroomsTotal", "BathroomsTotalInteger", "DaysOnMarket"]
    for row in profile.itertuples():
        action, method, reason = "keep_null", "none", "Keep unavailable values explicit; no validated imputation rule"
        coverage = (float(support.set_index("field").loc[row.field, "donor_coverage_pct"])
                    if support is not None and row.field in TARGETS else np.nan)
        evidence = {"worst_median_error_iqr": np.nan, "worst_median_bias_pct": np.nan,
                    "improvement_pct": np.nan, "donor_coverage_pct": coverage}
        if row.missing_count == 0:
            action, reason = "keep", "No missing values"
        elif row.missing_pct == 100:
            action, reason = "drop_column", "Entirely empty: no observed values or local donors"
        elif row.missing_pct >= 90 and row.field not in ["latfilled", "lonfilled"]:
            action, reason = "drop_column", "At least 90% missing; cannot justify reconstructing a mostly unobserved feature"
        elif row.field in TARGETS:
            candidate = experiments[(experiments.field == row.field) & experiments.scenario.ne("MCAR") & experiments.method.ne("delete")]
            ranked = candidate.groupby("method").agg(score=("median_error_iqr", "mean"),
                                                      worst_error=("median_error_iqr", "max"),
                                                      worst_bias=("worst_abs_median_bias_pct", "max"))
            best = ranked.score.idxmin()
            baseline = float(ranked.loc["median", "score"])
            improvement = 100 * (baseline - float(ranked.loc[best, "score"])) / max(baseline, 1e-12)
            evidence = {"worst_median_error_iqr": float(ranked.loc[best, "worst_error"]),
                        "worst_median_bias_pct": float(ranked.loc[best, "worst_bias"]),
                        "improvement_pct": improvement, "donor_coverage_pct": coverage}
            method = best
            passed = (row.missing_pct <= POLICY["max_missing_pct_for_imputation"] and
                      evidence["worst_median_error_iqr"] <= POLICY["max_worst_median_error_iqr"] and
                      evidence["worst_median_bias_pct"] <= POLICY["max_worst_population_median_bias_pct"] and
                      improvement >= POLICY["min_improvement_over_median_pct"])
            if passed:
                action, reason = "impute", "Passes typical-error and median-bias screens; fill ONLY records with 20 matching county/subtype/month donors, leaving unsupported records null"
            elif row.missing_pct > POLICY["max_missing_pct_for_imputation"]:
                reason = "More than 1% missing; structural/NMAR ambiguity remains, so retain null plus missingness flag"
            else:
                reason = "Imputation failed at least one predeclared accuracy/bias/improvement tolerance"
        elif row.field in ["ClosePrice", "CloseDate"]:
            reason = "Never impute analysis outcomes; exclude unavailable rows only from analyses requiring this outcome"
        elif row.field in ["Latitude", "Longitude"]:
            reason = "Do not invent point locations; omit these rows from maps only"
        elif "Date" in row.field:
            reason = "Do not invent event dates; omit from the affected duration calculation only"
        rows.append({"field": row.field, "missing_count": row.missing_count, "missing_pct": row.missing_pct,
                     "action": action, "best_tested_method": method, "reason": reason,
                     "integer_output": row.field in integer, **evidence})
    return pd.DataFrame(rows)


def prepare_data(data, decisions):
    """Apply selected decisions to a new frame and audit every imputed cell."""
    prepared = data.copy()
    audit, sensitivities = [], []
    for field in TARGETS + ["Latitude", "Longitude", "ClosePrice"]:
        prepared[field + "_was_missing_flag"] = data[field].isna()
    for decision in decisions.query("action == 'impute'").itertuples():
        target, method = decision.field, decision.best_tested_method
        original_missing = data[target].isna()
        missing = original_missing & matching_donor_counts(data, target).ge(POLICY["min_group_donors"])
        train = data.loc[~original_missing & data[target].ge(0)]
        if method == "model" and len(train) > POLICY["sample_size"]:
            train = train.sample(POLICY["sample_size"], random_state=42)
        absent = data.loc[missing]
        if method == "model":
            x_train, numeric = predictors(train, target)
            x_missing, _ = predictors(absent, target)
            model = estimator(numeric)
            model.fit(x_train, np.log1p(train[target].to_numpy(dtype=float)))
            prediction = np.maximum(0, np.expm1(model.predict(x_missing)))
        elif method == "group_median":
            prediction = group_predictions(train, absent, target)
        else:
            prediction = np.full(len(absent), float(train[target].median()))
        if decision.integer_output:
            prediction = np.rint(prediction)
        prepared[target + "_imputed_flag"] = missing
        prepared.loc[missing, target] = prediction
        if decision.integer_output:
            prepared[target] = prepared[target].astype("Int64")
        audit.append(pd.DataFrame({"SourceRecordNumber": absent.SourceRecordNumber,
                                   "ListingKey": absent.ListingKey, "field": target,
                                   "method": method, "imputed_value": prediction}))
        # Pattern-mixture sensitivity on the actual missing cells, not observed values.
        for delta in [-1, 0, 1]:
            if target == "YearBuilt":
                shifted = prediction + delta * 10
                assumption = f"{delta * 10:+d} years on imputed cells"
            elif target in ["BedroomsTotal", "BathroomsTotalInteger"]:
                shifted = np.maximum(0, prediction + delta)
                assumption = f"{delta:+d} rooms on imputed cells"
            else:
                shifted = prediction * (1 + 0.20 * delta)
                assumption = f"{delta * 20:+d}% on imputed cells"
            complete = data[target].astype(float).copy()
            complete.loc[missing] = shifted
            sensitivities.append({"field": target, "delta": delta, "assumption": assumption,
                                  "imputed_cells": int(missing.sum()), "mean": float(complete.mean()),
                                  "median": float(complete.median()), "p25": float(complete.quantile(.25)),
                                  "p75": float(complete.quantile(.75))})
    drops = decisions.loc[decisions.action.eq("drop_column"), "field"].tolist()
    related_flags = [f + suffix for f in drops for suffix in ["_invalid_flag", "_parse_error_flag"] if f + suffix in prepared]
    prepared = prepared.drop(columns=drops + related_flags)
    prepared["price_analysis_eligible_flag"] = prepared.ClosePrice.notna() & prepared.CloseDate.notna()
    audit_table = pd.concat(audit, ignore_index=True) if audit else pd.DataFrame(columns=["SourceRecordNumber", "ListingKey", "field", "method", "imputed_value"])
    sensitivity = pd.DataFrame(sensitivities, columns=["field", "delta", "assumption", "imputed_cells", "mean", "median", "p25", "p75"])
    assert len(prepared) == len(data)
    # All original observed values in retained columns must remain unchanged.
    for column in data.columns.intersection(prepared.columns):
        observed = data[column].notna()
        pd.testing.assert_series_equal(prepared.loc[observed, column], data.loc[observed, column], check_dtype=False)
    return prepared, audit_table, sensitivity


def input_fingerprint():
    digest = hashlib.sha256()
    with INPUT.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def algorithm_fingerprint():
    code = "\n".join(inspect.getsource(function) for function in
                     [predictors, estimator, group_predictions, mask_indices, evaluate_methods])
    return hashlib.sha256(code.encode("utf-8")).hexdigest()


def cache_matches():
    path = REPORT_DIR / "experiment_config.json"
    if not path.exists():
        return False
    saved = json.loads(path.read_text(encoding="utf-8"))
    return (saved.get("input_sha256") == input_fingerprint() and
            saved.get("algorithm_sha256") == algorithm_fingerprint() and saved.get("policy") == POLICY)


def read_saved_tables():
    return {name: pd.read_csv(REPORT_DIR / f"{name}.csv") for name in
            ["missingness_profile", "complete_case_comparison", "missingness_by_group", "missingness_diagnostics",
             "experiment_trials", "experiment_partitions", "experiment_summary", "actual_donor_support",
             "decisions", "imputation_audit", "nmar_actual_missing_sensitivity"]}


def write_explanation(tables, summary):
    """Write the interpretation alongside the experimental evidence."""
    def markdown(frame):
        view = frame.copy()
        for column in view:
            if pd.api.types.is_float_dtype(view[column]):
                view[column] = view[column].map(lambda value: f"{value:,.3f}" if pd.notna(value) else "n/a")
        header = "| " + " | ".join(view.columns) + " |\n"
        separator = "| " + " | ".join(["---"] * len(view.columns)) + " |\n"
        return header + separator + "\n".join("| " + " | ".join(str(v) for v in row) + " |" for row in view.itertuples(index=False, name=None))

    profile = tables["missingness_profile"]
    fields = ["ClosePrice", "LivingArea", "DaysOnMarket", "YearBuilt", "BedroomsTotal", "BathroomsTotalInteger",
              "Latitude", "LotSizeAcres", "AssociationFee"]
    state = profile.set_index("field").loc[fields].reset_index()[["field", "missing_count", "missing_pct"]]
    action = tables["decisions"].query("action == 'impute'")[["field", "best_tested_method", "missing_count", "donor_coverage_pct"]]
    cells = tables["imputation_audit"].groupby("field").size().rename("actually_imputed").reset_index()
    action = action.merge(cells, on="field", how="left")
    comparison = tables["experiment_summary"].query("field == 'LivingArea' and method in ['delete', 'model']")
    comparison = comparison[["scenario", "method", "median_absolute_error", "abs_median_bias_pct", "abs_mean_bias_pct"]]
    core = tables["complete_case_comparison"].query("scope != 'all_original_fields'")[["scope", "rows_retained", "rows_lost"]]
    sensitivity = tables["nmar_actual_missing_sensitivity"]
    lines = [
        "# Missing data: MAR/NMAR experiments and preparation decision", "",
        "Population: 480,486 Residential **sold** records, January 2024–August 2026. Listings were not analyzed.", "",
        "## What the missing data looks like", "", markdown(state), "",
        "Five fields are completely empty: TaxAnnualAmount, AboveGradeFinishedArea, FireplacesTotal, TaxYear and CoveredSpaces. "
        "Four more fields exceed 90% missing: WaterfrontYN, BelowGradeFinishedArea, BasementYN and BuildingAreaTotal. "
        "These nine fields are removed from the new analysis view, not from the original cleaned file. The 90% cutoff is a practical scope decision, not a statistical law.", "",
        "Missingness is patterned: HOA fees are missing for 30.16% of SingleFamilyResidence records but 0.22% of Condominium records. "
        "Lot acreage is missing for 28.35% of condos but 1.73% of single-family homes. Shared land, no HOA, source coverage and non-reporting "
        "are plausible explanations, but the export alone does not establish which blank means not applicable. Do not replace these blanks with zero or false.", "",
        "## What MAR and NMAR mean", "",
        "MCAR: whether a value is missing is unrelated to observed or unobserved values. "
        "MAR: missingness can depend on observed variables (such as list price or property subtype), but not on the missing value after conditioning on them. "
        "NMAR/MNAR: missingness still depends on the unobserved value, for example unusually high fees being less likely to be reported. "
        "Missingness-prediction AUCs and group differences show association, not proof of MAR; they cannot rule out NMAR. "
        f"[Why sensitivity analysis is needed]({SOURCES['missingness']}).", "",
        "## Experiments actually run", "",
        f"Evaluated 11 numeric fields with {len(tables['experiment_trials']):,} result rows: "
        "3 training/hold-out splits × 2 masking seeds × 4 mechanisms × 4 methods per field. "
        "Up to 80,000 observed records per target were sampled reproducibly. Repeated listing keys were kept entirely on one side of each split. "
        "Model encoding, medians and donor groups were fitted on training records only. Repetitions overlap, so their ranges are not formal confidence intervals. "
        f"[Leakage guidance]({SOURCES['leakage']}).", "",
        "Each hold-out stress test hid 20% of known values. MCAR hid values uniformly; MAR hid preferentially by observed list price and condo subtype; "
        "NMAR_high and NMAR_low hid preferentially high or low target values. These are specified scenarios, not estimates of the true missingness mechanism. "
        "All observed evaluation values, including outliers, were scored. Methods were deletion, global median, county/subtype median with fallback, "
        "and a boosted-tree predictor fitted on log(1 + value). ClosePrice was never an imputation target. "
        "It is used as an observed auxiliary predictor for other fields because this is retrospective sold-data analysis. "
        "These estimates must not be treated as inputs for evaluating an unseen-price forecasting model.", "",
        "Example—LivingArea: median absolute prediction error is measured on hidden cells; aggregate biases compare the reconstructed or retained "
        "hold-out population with its known complete truth. The table reports average **absolute** biases across repetitions.", "",
        markdown(comparison), "",
        "Deletion changes the population under the MAR/NMAR scenarios. A global median can preserve the overall median almost mechanically while "
        "still badly predicting individual cells or distorting other relationships. Therefore preserving one aggregate alone did not qualify a method.", "",
        "## Decision: selective imputation, selective column removal, no blanket row deletion", "",
        f"Retain all {summary['rows_after']:,} rows; drop the nine nearly/entirely empty features. "
        "Only OriginalListPrice and BathroomsTotalInteger passed the selected typical-error/median-robustness screen. "
        "Impute those fields **only** when the missing record has at least 20 observed donors matching county, property subtype and source month. "
        "Unsupported missing records remain null. Every filled cell has an imputation flag and audit record; observed values are unchanged.", "",
        markdown(action), "",
        "The operational screen required <=1% actual missingness, worst scenario-average median absolute error <=0.25 × max(training IQR, 1), "
        "worst repetition population-median bias <=2%, and >=10% improvement over global-median typical error across MAR/NMAR scenarios. "
        "These tolerances are selected practical rules, not universal statistical guarantees. They prioritize typical records and median housing summaries; "
        "they do **not** establish unbiased means, tails or regression coefficients. In particular, the OriginalListPrice model still has about 9.35% "
        "mean bias in the synthetic 20% NMAR_high stress test. Keep its imputation flags, and use observed-only values for tail analysis or confirmatory pricing inference.", "",
        "LivingArea, YearBuilt, DaysOnMarket, BedroomsTotal and ParkingTotal did not pass the accuracy/robustness screen. "
        "The estimated LivingArea model's worst scenario median error was about 266 sq ft and YearBuilt about 13 years, so low missing percentages "
        "alone did not justify inventing those measurements. Higher-missingness lot size, HOA fees, garage spaces and stories stay null with missingness flags. "
        "Coordinates, dates, identifiers and the ClosePrice outcome are never statistically imputed.", "",
        "Do not use dropna() across the whole dataset: the five fully empty fields would discard every row. "
        "Analysis-specific complete-case counts before imputation are:", "", markdown(core), "",
        f"The {summary['price_analysis_excluded_rows']} missing-price records remain in the master view but are excluded by price_analysis_eligible_flag "
        "when calculating price outcomes. Exclude missing locations only for geographic analyses and missing dates only for the affected timelines.", "",
        "## NMAR sensitivity on the actual imputed cells", "",
        "In addition to synthetic masking, shift the actual price imputations by -20%/+20% and bathroom imputations by -1/+1. "
        "Recalculate whole-field summaries while keeping unsupported records null. This is a pattern-mixture sensitivity check, not a confidence interval.", "",
        markdown(sensitivity), "",
        "Single imputation is convenient for exploratory predictors but treats estimated values too precisely. "
        "Use missingness indicators and sensitivity checks, and retain observed-only versions for reporting. For formal inference, use a proper "
        "multiple-imputation model with uncertainty propagation and field-specific NMAR sensitivity; this export is not such a model.", "",
        "Outputs: decisions.csv documents every field; experiment_trials.csv contains all replications; actual_donor_support.csv documents coverage; "
        "imputation_audit.csv identifies each estimated cell. The new _AnalysisReady.csv preserves the original cleaned CSV.", "",
    ]
    path = REPORT_DIR / "missing_data_explanation.md"
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def save_results(data, prepared, tables):
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    for name, table in tables.items():
        table.to_csv(REPORT_DIR / f"{name}.csv", index=False)
    prepared.to_csv(OUTPUT, index=False, date_format="%Y-%m-%dT%H:%M:%S.%f")
    schema = {"parse_dates": [c for c in prepared if pd.api.types.is_datetime64_any_dtype(prepared[c])],
              "dtype": {c: str(t) for c, t in prepared.dtypes.items() if not pd.api.types.is_datetime64_any_dtype(t)}}
    (REPORT_DIR / "analysis_ready_schema.json").write_text(json.dumps(schema, indent=2), encoding="utf-8")
    (REPORT_DIR / "experiment_config.json").write_text(json.dumps({"policy": POLICY, "sources": SOURCES,
        "input_sha256": input_fingerprint(), "algorithm_sha256": algorithm_fingerprint(),
        "simulation_note": "20% synthetic masking of observed held-out values; not proof of actual MAR/NMAR",
        "population": "Residential SOLD records, 2024-01 through 2026-08; no listing dataset analyzed"}, indent=2), encoding="utf-8")
    decision_counts = tables["decisions"].action.value_counts().to_dict()
    summary = {"rows_before": len(data), "rows_after": len(prepared), "decisions": decision_counts,
               "imputed_cells": len(tables["imputation_audit"]),
               "rows_with_imputations": int(tables["imputation_audit"].SourceRecordNumber.nunique()),
               "price_analysis_excluded_rows": int((~prepared.price_analysis_eligible_flag).sum()),
               "dropped_fields": tables["decisions"].query("action == 'drop_column'").field.tolist(),
               "imputed_fields": tables["decisions"].query("action == 'impute'").field.tolist(),
               "mar_nmar_identified": False}
    (REPORT_DIR / "result_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    write_explanation(tables, summary)
    print(json.dumps(summary, indent=2), flush=True)
    return summary


def run():
    data = load_data()
    profile, complete_cases, groups = profile_missingness(data)
    diagnostics = diagnose_missingness(data)
    trials, partitions = evaluate_methods(data)
    experiments = summarize_experiments(trials)
    support = actual_donor_support(data)
    decisions = choose_policy(profile, experiments, support)
    prepared, audit, sensitivity = prepare_data(data, decisions)
    tables = {"missingness_profile": profile, "complete_case_comparison": complete_cases,
              "missingness_by_group": groups, "missingness_diagnostics": diagnostics,
              "experiment_trials": trials, "experiment_partitions": partitions,
              "experiment_summary": experiments, "actual_donor_support": support, "decisions": decisions,
              "imputation_audit": audit, "nmar_actual_missing_sensitivity": sensitivity}
    summary = save_results(data, prepared, tables)
    return data, prepared, tables, summary


if __name__ == "__main__":
    run()
