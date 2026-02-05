# src/eda.py
from __future__ import annotations

import argparse
import json
import os
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

try:
    import yaml
except Exception:
    yaml = None

try:
    from jinja2 import Template
except Exception:
    Template = None

# OpenAI SDK is optional unless --llm openai
try:
    from openai import OpenAI
except Exception:
    OpenAI = None


# ----------------------------
# Constants / heuristics
# ----------------------------

COMMON_MISSING_CODES = [
    "", " ", "NA", "N/A", "na", "n/a", "null", "NULL", "None", "none",
    "unknown", "Unknown", "UNK", "NIL",
    "-999", "-999.0", "-1", "-1.0", "9999", "9999.0"
]

SENSITIVE_NAME_PATTERNS = [
    r"\bname\b", r"\bfirst_name\b", r"\blast_name\b",
    r"\bemail\b", r"\bphone\b", r"\bmobile\b",
    r"\bssn\b", r"\bsocial\b", r"\bpassport\b",
    r"\baddress\b", r"\bstreet\b", r"\bzip\b", r"\bpostal\b",
    r"\bdob\b", r"\bbirth\b",
    r"\bip\b", r"\bmac\b",
    r"\buser\b.*\bid\b", r"\baccount\b.*\bid\b", r"\bdevice\b.*\bid\b",
]
EMAIL_RE = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")

# Used by numeric verification gate for LLM output
NUM_TOKEN_RE = re.compile(r"(?<![\w.])(-?\d+(?:\.\d+)?)(%?)(?![\w.])")


@dataclass
class ColumnSchema:
    name: str
    type: Optional[str] = None
    meaning: Optional[str] = None
    units: Optional[str] = None
    allowed_values: Optional[list[Any]] = None
    missing_codes: Optional[list[Any]] = None


# ----------------------------
# I/O helpers
# ----------------------------

def ensure_dirs(out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "plots").mkdir(parents=True, exist_ok=True)
    return out_dir


def save_json(path: Path, obj: Any) -> None:
    path.write_text(json.dumps(obj, indent=2, default=str), encoding="utf-8")


def read_csv_robust(path: str, max_rows: int, seed: int = 42) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Robust CSV read: tries common separators and encodings.
    Returns (df, ingestion_meta).
    """
    ingestion = {"source": path, "read_attempts": []}

    # Try encodings
    encodings = ["utf-8", "utf-8-sig", "latin-1"]
    seps = [",", "\t", ";", "|"]

    last_err = None
    for enc in encodings:
        for sep in seps:
            try:
                df = pd.read_csv(path, encoding=enc, sep=sep, engine="python")
                ingestion["read_attempts"].append({"encoding": enc, "sep": sep, "ok": True})
                if len(df) > max_rows:
                    df = df.sample(max_rows, random_state=seed).reset_index(drop=True)
                    ingestion["row_cap_applied"] = True
                    ingestion["row_cap"] = max_rows
                return df, ingestion
            except Exception as e:
                last_err = str(e)
                ingestion["read_attempts"].append({"encoding": enc, "sep": sep, "ok": False, "error": str(e)})

    raise RuntimeError(f"Failed to read CSV. Last error: {last_err}")


def load_schema(path: Optional[str]) -> Optional[Dict[str, ColumnSchema]]:
    if not path:
        return None
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Schema file not found: {path}")

    if p.suffix.lower() in [".yml", ".yaml"]:
        if yaml is None:
            raise RuntimeError("pyyaml not installed. Install with: pip install pyyaml")
        raw = yaml.safe_load(p.read_text(encoding="utf-8"))
    elif p.suffix.lower() == ".json":
        raw = json.loads(p.read_text(encoding="utf-8"))
    else:
        raise ValueError("Schema must be YAML or JSON")

    cols = raw.get("columns", [])
    out: Dict[str, ColumnSchema] = {}
    for c in cols:
        name = c.get("name")
        if not name:
            continue
        out[name] = ColumnSchema(
            name=name,
            type=c.get("type"),
            meaning=c.get("meaning"),
            units=c.get("units"),
            allowed_values=c.get("allowed_values"),
            missing_codes=c.get("missing_codes"),
        )
    return out


def normalize_columns(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Keep original names, but also create safe unique names for internal processing.
    Report will use original names.
    """
    original = list(df.columns)
    safe = []
    seen = set()
    for c in original:
        s = re.sub(r"\s+", "_", str(c).strip())
        s = re.sub(r"[^\w_]", "", s)
        if s == "":
            s = "col"
        base = s
        i = 1
        while s in seen:
            i += 1
            s = f"{base}_{i}"
        seen.add(s)
        safe.append(s)

    mapping = dict(zip(original, safe))
    df2 = df.copy()
    df2.columns = safe

    meta = {"original_columns": original, "safe_columns": safe, "mapping_original_to_safe": mapping}
    return df2, meta


def apply_missing_codes(df: pd.DataFrame, schema: Optional[Dict[str, ColumnSchema]], mapping_original_to_safe: Dict[str, str]) -> Dict[str, Any]:
    """
    Replace common missing codes and schema-specific missing codes with NaN.
    Apply common codes only to object columns.
    Apply schema codes to specified columns (even numeric) when provided.
    """
    meta = {"missing_codes_global": COMMON_MISSING_CODES.copy(), "missing_codes_schema": {}}

    # Global replacement on object columns
    codes = set(str(x) for x in COMMON_MISSING_CODES)
    for col in df.columns:
        if df[col].dtype == "object":
            df[col] = df[col].replace(list(codes), np.nan)

    # Schema specific codes
    if schema:
        for orig_name, col_schema in schema.items():
            if not col_schema.missing_codes:
                continue
            safe_name = mapping_original_to_safe.get(orig_name)
            if safe_name and safe_name in df.columns:
                df[safe_name] = df[safe_name].replace(col_schema.missing_codes, np.nan)
                meta["missing_codes_schema"][orig_name] = list(col_schema.missing_codes)

    return meta


# ----------------------------
# Type inference & privacy
# ----------------------------

def infer_column_type(s: pd.Series) -> Dict[str, Any]:
    """
    Returns a dict with:
      - type: numeric|categorical|datetime|text|id_like
      - confidence
      - reasons
      - parse_rates
    """
    reasons = []
    s_nonnull = s.dropna()
    nn = len(s_nonnull)

    if nn == 0:
        return {
            "type": "text",
            "confidence": 0.3,
            "reasons": ["all_missing"],
            "parse_rates": {}
        }

    # parse rates
    num = pd.to_numeric(s_nonnull, errors="coerce")
    num_rate = float(num.notna().mean())

    dt_rate = 0.0
    if s.dtype == "object":
        dt = pd.to_datetime(
            s_nonnull.astype(str),
            errors="coerce",
            infer_datetime_format=True
        )
        dt_rate = float(dt.notna().mean())

    nunique = int(s_nonnull.nunique(dropna=True))
    unique_ratio = nunique / max(nn, 1)

    # ---------- FIXED id_like LOGIC ----------
    # Treat as id_like only if:
    # - very high uniqueness
    # - sufficiently large dataset
    # - NOT clearly numeric
    if (
        unique_ratio >= 0.95
        and nunique >= 50
        and num_rate < 0.8
        and dt_rate < 0.3
    ):
        reasons.append(f"unique_ratio={unique_ratio:.2f}")
        return {
            "type": "id_like",
            "confidence": min(0.95, unique_ratio),
            "reasons": reasons,
            "parse_rates": {"numeric": num_rate, "datetime": dt_rate},
        }

    # datetime
    if dt_rate >= 0.8 and num_rate < 0.8:
        reasons.append(f"datetime_parse_rate={dt_rate:.2f}")
        return {
            "type": "datetime",
            "confidence": dt_rate,
            "reasons": reasons,
            "parse_rates": {"numeric": num_rate, "datetime": dt_rate},
        }

    # numeric
    if pd.api.types.is_numeric_dtype(s) or num_rate >= 0.85:
        reasons.append(f"numeric_parse_rate={num_rate:.2f}")
        return {
            "type": "numeric",
            "confidence": max(0.7, num_rate),
            "reasons": reasons,
            "parse_rates": {"numeric": num_rate, "datetime": dt_rate},
        }

    # categorical vs text
    if nunique <= 50 or unique_ratio <= 0.2:
        reasons.append(f"nunique={nunique}, unique_ratio={unique_ratio:.2f}")
        return {
            "type": "categorical",
            "confidence": 0.75,
            "reasons": reasons,
            "parse_rates": {"numeric": num_rate, "datetime": dt_rate},
        }

    return {
        "type": "text",
        "confidence": 0.6,
        "reasons": [f"nunique={nunique}, unique_ratio={unique_ratio:.2f}"],
        "parse_rates": {"numeric": num_rate, "datetime": dt_rate},
    }



def detect_sensitive_columns(df_safe: pd.DataFrame, safe_to_original: Dict[str, str]) -> Dict[str, Any]:
    flagged = []
    reasons: Dict[str, List[str]] = {}

    for safe_col in df_safe.columns:
        orig_col = safe_to_original.get(safe_col, safe_col)
        col_l = str(orig_col).lower()

        for pat in SENSITIVE_NAME_PATTERNS:
            if re.search(pat, col_l):
                flagged.append(orig_col)
                reasons.setdefault(orig_col, []).append(f"name_match:{pat}")
                break

        s = df_safe[safe_col]
        if s.dtype == "object":
            sample = s.dropna().astype(str).head(200)
            if len(sample) > 0 and sample.map(lambda x: bool(EMAIL_RE.match(x))).mean() > 0.2:
                flagged.append(orig_col)
                reasons.setdefault(orig_col, []).append("value_pattern:email")

    flagged = sorted(set(flagged))
    return {
        "flagged_columns": flagged,
        "reasons": reasons,
        "warning": (
            "Potential identifiers/sensitive fields detected. Report avoids showing raw example values for these columns."
            if flagged else None
        )
    }


# ----------------------------
# Descriptive statistics
# ----------------------------

def mode_safe_numeric(s: pd.Series) -> Optional[float]:
    try:
        m = s.mode(dropna=True)
        if m.empty:
            return None
        return float(m.iloc[0])
    except Exception:
        return None


def numeric_summary(s: pd.Series) -> Dict[str, Any]:
    s_num = pd.to_numeric(s, errors="coerce").dropna()
    if s_num.empty:
        return {"available": False}

    q1 = float(s_num.quantile(0.25))
    q3 = float(s_num.quantile(0.75))
    iqr = q3 - q1
    lo = q1 - 1.5 * iqr
    hi = q3 + 1.5 * iqr
    outlier_mask = (s_num < lo) | (s_num > hi)

    return {
        "available": True,
        "n_non_missing": int(len(s_num)),
        "min": float(s_num.min()),
        "max": float(s_num.max()),
        "mean": float(s_num.mean()),
        "median": float(s_num.median()),
        "mode": mode_safe_numeric(s_num),
        "std": float(s_num.std(ddof=1)) if len(s_num) > 1 else 0.0,
        "q1": q1,
        "q3": q3,
        "iqr": float(iqr),
        "outlier_rule": "1.5×IQR",
        "outlier_bounds": {"lower": float(lo), "upper": float(hi)},
        "outlier_count": int(outlier_mask.sum()),
        "outlier_fraction": float(outlier_mask.mean()),
    }


def categorical_summary(s: pd.Series, top_k: int = 12) -> Dict[str, Any]:
    s_cat = s.dropna().astype(str)
    if s_cat.empty:
        return {"available": False}

    vc = s_cat.value_counts(dropna=True)
    total = int(vc.sum())
    top = vc.head(top_k)
    items = [{"value": k, "count": int(v), "percent": float(v / total)} for k, v in top.items()]
    dominance = float((vc.iloc[0] / total)) if len(vc) > 0 else 0.0
    return {
        "available": True,
        "total_non_missing": total,
        "top_k": items,
        "dominance_top1": dominance,
        "n_unique": int(vc.shape[0]),
    }


def choose_columns_for_stats(type_map_safe: Dict[str, str]) -> Tuple[List[str], List[str]]:
    # exclude id_like from numeric/categorical stats
    numeric_cols = [c for c, t in type_map_safe.items() if t == "numeric"]
    cat_cols = [c for c, t in type_map_safe.items() if t == "categorical"]

    # If not enough numeric, we’ll still run and adapt, but aim for >=2 if present
    return numeric_cols[:2], cat_cols[:1]


# ----------------------------
# Relationships (numeric correlations)
# ----------------------------

def top_numeric_correlations(df_safe: pd.DataFrame, numeric_cols: List[str], k: int = 5) -> List[Dict[str, Any]]:
    if len(numeric_cols) < 2:
        return []
    num_df = df_safe[numeric_cols].apply(pd.to_numeric, errors="coerce")
    corr = num_df.corr(numeric_only=True)
    if corr.shape[0] < 2:
        return []

    pairs = []
    cols = list(corr.columns)
    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):
            v = corr.iloc[i, j]
            if np.isfinite(v):
                pairs.append((cols[i], cols[j], float(v)))
    pairs.sort(key=lambda t: abs(t[2]), reverse=True)
    return [{"col1": a, "col2": b, "corr": v} for a, b, v in pairs[:k]]


# ----------------------------
# Plotting (>=5 guaranteed)
# ----------------------------

def save_fig(fig: plt.Figure, path: Path) -> None:
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def make_plots(
    df_safe: pd.DataFrame,
    type_map_safe: Dict[str, str],
    safe_to_original: Dict[str, str],
    out_plots: Path
) -> List[Dict[str, str]]:
    out_plots.mkdir(parents=True, exist_ok=True)
    manifest: List[Dict[str, str]] = []

    numeric_cols = [c for c, t in type_map_safe.items() if t == "numeric"]
    cat_cols = [c for c, t in type_map_safe.items() if t == "categorical"]
    dt_cols = [c for c, t in type_map_safe.items() if t == "datetime"]

    # 1) Missingness top 20 (always)
    miss = df_safe.isna().mean().sort_values(ascending=False).head(20)
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.bar([safe_to_original.get(c, c) for c in miss.index], miss.values)
    ax.set_title("Top Missingness by Column (Top 20)")
    ax.set_xlabel("Column")
    ax.set_ylabel("Missing Fraction")
    ax.tick_params(axis="x", rotation=75)
    save_fig(fig, out_plots / "missingness_top20.png")
    manifest.append({"file": "missingness_top20.png", "type": "bar"})

    # 2) Numeric plots (hist + box)
    if numeric_cols:
        c = numeric_cols[0]
        s = pd.to_numeric(df_safe[c], errors="coerce").dropna()
        if len(s) > 0:
            orig = safe_to_original.get(c, c)

            fig = plt.figure()
            ax = fig.add_subplot(1, 1, 1)
            ax.hist(s.values, bins=30)
            ax.set_title(f"Histogram: {orig}")
            ax.set_xlabel(orig)
            ax.set_ylabel("Count")
            save_fig(fig, out_plots / f"hist_{c}.png")
            manifest.append({"file": f"hist_{c}.png", "type": "histogram"})

            fig = plt.figure()
            ax = fig.add_subplot(1, 1, 1)
            ax.boxplot(s.values, vert=True)
            ax.set_title(f"Boxplot: {orig}")
            ax.set_xlabel(orig)
            ax.set_ylabel("Value")
            save_fig(fig, out_plots / f"box_{c}.png")
            manifest.append({"file": f"box_{c}.png", "type": "boxplot"})

    # 3) Categorical bar chart
    if cat_cols:
        c = cat_cols[0]
        vc = df_safe[c].dropna().astype(str).value_counts().head(15)
        if len(vc) > 0:
            orig = safe_to_original.get(c, c)
            fig = plt.figure()
            ax = fig.add_subplot(1, 1, 1)
            ax.bar(vc.index, vc.values)
            ax.set_title(f"Top Categories: {orig} (Top 15)")
            ax.set_xlabel(orig)
            ax.set_ylabel("Count")
            ax.tick_params(axis="x", rotation=75)
            save_fig(fig, out_plots / f"bar_{c}.png")
            manifest.append({"file": f"bar_{c}.png", "type": "bar"})

    # 4/5) Scatter + correlation heatmap if >=2 numeric
    if len(numeric_cols) >= 2:
        x, y = numeric_cols[0], numeric_cols[1]
        xs = pd.to_numeric(df_safe[x], errors="coerce")
        ys = pd.to_numeric(df_safe[y], errors="coerce")
        m = xs.notna() & ys.notna()
        if int(m.sum()) > 5:
            fig = plt.figure()
            ax = fig.add_subplot(1, 1, 1)
            ax.scatter(xs[m].values, ys[m].values, s=12)
            ax.set_title(f"Scatter: {safe_to_original.get(x, x)} vs {safe_to_original.get(y, y)}")
            ax.set_xlabel(safe_to_original.get(x, x))
            ax.set_ylabel(safe_to_original.get(y, y))
            save_fig(fig, out_plots / f"scatter_{x}_vs_{y}.png")
            manifest.append({"file": f"scatter_{x}_vs_{y}.png", "type": "scatter"})

        num_df = df_safe[numeric_cols].apply(pd.to_numeric, errors="coerce")
        corr = num_df.corr(numeric_only=True)
        if corr.shape[0] >= 2:
            fig = plt.figure(figsize=(7, 6))
            ax = fig.add_subplot(1, 1, 1)
            im = ax.imshow(corr.values)
            ax.set_title("Correlation Heatmap (Numeric Columns)")
            ax.set_xticks(range(corr.shape[1]))
            ax.set_yticks(range(corr.shape[0]))
            ax.set_xticklabels([safe_to_original.get(c, c) for c in corr.columns], rotation=75)
            ax.set_yticklabels([safe_to_original.get(c, c) for c in corr.index])
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            save_fig(fig, out_plots / "corr_heatmap.png")
            manifest.append({"file": "corr_heatmap.png", "type": "heatmap"})

    # datetime plot (if present) - count over time
    if dt_cols:
        c = dt_cols[0]
        dt = pd.to_datetime(df_safe[c], errors="coerce")
        if dt.notna().sum() > 5:
            counts = dt.dropna().dt.to_period("M").value_counts().sort_index()
            fig = plt.figure()
            ax = fig.add_subplot(1, 1, 1)
            ax.plot(counts.index.astype(str), counts.values)
            ax.set_title(f"Monthly Row Counts by {safe_to_original.get(c, c)}")
            ax.set_xlabel("Month")
            ax.set_ylabel("Row count")
            ax.tick_params(axis="x", rotation=75)
            save_fig(fig, out_plots / f"time_{c}.png")
            manifest.append({"file": f"time_{c}.png", "type": "timeseries"})

    # Guarantee >=5 plots by adding more categorical bars or numeric histograms
    i = 1
    while len(manifest) < 5 and i < len(numeric_cols):
        c = numeric_cols[i]
        s = pd.to_numeric(df_safe[c], errors="coerce").dropna()
        if len(s) > 0:
            fig = plt.figure()
            ax = fig.add_subplot(1, 1, 1)
            ax.hist(s.values, bins=30)
            ax.set_title(f"Histogram: {safe_to_original.get(c, c)}")
            ax.set_xlabel(safe_to_original.get(c, c))
            ax.set_ylabel("Count")
            save_fig(fig, out_plots / f"hist_{c}.png")
            manifest.append({"file": f"hist_{c}.png", "type": "histogram"})
        i += 1

    j = 1
    while len(manifest) < 5 and j < len(cat_cols):
        c = cat_cols[j]
        vc = df_safe[c].dropna().astype(str).value_counts().head(15)
        if len(vc) > 0:
            fig = plt.figure()
            ax = fig.add_subplot(1, 1, 1)
            ax.bar(vc.index, vc.values)
            ax.set_title(f"Top Categories: {safe_to_original.get(c, c)} (Top 15)")
            ax.set_xlabel(safe_to_original.get(c, c))
            ax.set_ylabel("Count")
            ax.tick_params(axis="x", rotation=75)
            save_fig(fig, out_plots / f"bar_{c}.png")
            manifest.append({"file": f"bar_{c}.png", "type": "bar"})
        j += 1

    # Last resort: if still <5 (very tiny / empty), add a placeholder plot
    while len(manifest) < 5:
        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)
        ax.text(0.5, 0.5, "Insufficient data for more plots", ha="center", va="center")
        ax.set_title("Placeholder Plot")
        ax.set_xlabel("N/A")
        ax.set_ylabel("N/A")
        fname = f"placeholder_{len(manifest)+1}.png"
        save_fig(fig, out_plots / fname)
        manifest.append({"file": fname, "type": "placeholder"})

    return manifest


# ----------------------------
# Insights (computed)
# ----------------------------

def generate_computed_insights(facts: Dict[str, Any]) -> List[str]:
    bullets: List[str] = []

    rows = facts["dataset_overview"]["rows"]
    cols = facts["dataset_overview"]["cols"]
    bullets.append(f"Dataset has {rows} rows and {cols} columns.")

    # Missingness concentration
    top_miss = facts["missingness"]["top_missing_columns"]
    if top_miss and top_miss[0]["missing_fraction"] > 0:
        items = ", ".join([f"{x['column']} ({x['missing_fraction']:.0%})" for x in top_miss[:3] if x["missing_fraction"] > 0])
        if items:
            bullets.append(f"Missingness is concentrated in: {items}.")

    qc = facts["quality_checks"]
    if qc["duplicate_rows"] > 0:
        bullets.append(f"{qc['duplicate_rows']} duplicate rows detected; consider de-duplication.")
    if qc["single_value_columns"]:
        bullets.append(f"Single-value columns may be uninformative: {', '.join(qc['single_value_columns'][:8])}.")
    if qc["high_missing_columns"]:
        bullets.append(f"High-missing columns (>=40% missing): {', '.join(qc['high_missing_columns'][:8])}.")

    # Numeric outliers
    for col, st in facts["descriptive_stats"]["numeric"].items():
        if st.get("available") and st.get("outlier_fraction", 0) >= 0.02:
            bullets.append(
                f"'{col}' has {st['outlier_count']} outliers ({st['outlier_fraction']:.1%}) under the 1.5×IQR rule."
            )

    # Categorical dominance
    for col, st in facts["descriptive_stats"]["categorical"].items():
        if st.get("available") and st.get("top_k"):
            top = st["top_k"][0]
            if top["percent"] >= 0.5:
                bullets.append(f"'{col}' is imbalanced: '{top['value']}' accounts for {top['percent']:.0%} of non-missing rows.")

    # Correlation
    rels = facts["relationships"]["top_numeric_correlations"]
    if rels:
        r0 = rels[0]
        bullets.append(f"Strongest numeric correlation: corr({r0['col1']}, {r0['col2']}) = {r0['corr']:.2f}.")

    # Privacy
    if facts["privacy"]["flagged_columns"]:
        bullets.append("Potential sensitive/identifier columns were detected; raw example values are intentionally not displayed.")

    # Ensure 5–10
    bullets = bullets[:10]
    while len(bullets) < 5:
        bullets.append("Automated EDA summarizes patterns; domain context is required to interpret causes and implications.")
    return bullets


def limitations_note(facts: Dict[str, Any]) -> List[str]:
    notes = []
    miss_max = max([x["missing_fraction"] for x in facts["missingness"]["top_missing_columns"]] + [0.0])
    if miss_max >= 0.2:
        notes.append("Non-trivial missingness may bias summaries if missingness is not random.")
    if facts["quality_checks"]["duplicate_rows"] > 0:
        notes.append("Duplicates can inflate frequencies and distort descriptive statistics.")
    if facts["privacy"]["flagged_columns"]:
        notes.append("Potential identifiers constrain what can be safely displayed or shared.")
    if not notes:
        notes.append("Automated EDA cannot establish causality and may miss domain-specific validity issues without context.")
    return notes


# ----------------------------
# LLM additional analysis (facts-only + numeric gate)
# ----------------------------

def canonical_fact_numbers(facts: Dict[str, Any]) -> set[str]:
    allowed = set()

    def add_num(x: Any):
        if x is None:
            return
        if isinstance(x, bool):
            return
        if isinstance(x, (int, np.integer)):
            allowed.add(str(int(x)))
        elif isinstance(x, (float, np.floating)):
            f = float(x)
            allowed.add(str(f))
            allowed.add(f"{f:.2f}")
            allowed.add(f"{f:.3g}")
            allowed.add(f"{f:.0%}")
            allowed.add(f"{f:.1%}")

    add_num(facts["dataset_overview"]["rows"])
    add_num(facts["dataset_overview"]["cols"])
    add_num(facts["quality_checks"]["duplicate_rows"])

    for x in facts["missingness"]["top_missing_columns"]:
        add_num(x.get("missing_count"))
        add_num(x.get("missing_fraction"))

    for st in facts["descriptive_stats"]["numeric"].values():
        if st.get("available"):
            for k in ["min", "max", "mean", "median", "mode", "std", "iqr", "outlier_count", "outlier_fraction"]:
                add_num(st.get(k))

    for st in facts["descriptive_stats"]["categorical"].values():
        if st.get("available"):
            add_num(st.get("n_unique"))
            add_num(st.get("dominance_top1"))
            for it in st.get("top_k", [])[:5]:
                add_num(it.get("count"))
                add_num(it.get("percent"))

    for rel in facts["relationships"]["top_numeric_correlations"]:
        add_num(rel.get("corr"))

    return allowed


def verify_llm_text(text: str, allowed_nums: set[str]) -> Tuple[str, Dict[str, Any]]:
    kept = []
    dropped = []
    for line in text.splitlines():
        toks = NUM_TOKEN_RE.findall(line)
        bad = []
        for num, pct in toks:
            token = num + pct
            if token in allowed_nums:
                continue
            if pct == "%" and num in allowed_nums:
                continue
            bad.append(token)
        if bad:
            dropped.append({"line": line, "unverified_numbers": bad})
        else:
            kept.append(line)
    return "\n".join(kept).strip(), {"dropped_lines": dropped, "kept_lines": len(kept)}


def call_openai_facts_only(facts: Dict[str, Any], model: str) -> Dict[str, Any]:
    if OpenAI is None:
        raise RuntimeError("openai package not installed. Install with: pip install openai")
    if not os.environ.get("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY env var not set. Do not hardcode keys in code or GitHub.")

    client = OpenAI()

    prompt = f"""
You are a graduate-level data mining TA writing additional EDA analysis.

Rules:
- Use ONLY the facts provided in FACTS_JSON. Do not invent numbers.
- If a useful detail is not in facts, say it is not available.
- Output format:
  1) 6–10 bullet insights (non-generic, reference specific columns/patterns)
  2) 3 bullet limitations/bias notes
  3) 3 suggested next analyses (no numbers required)

FACTS_JSON:
{json.dumps(facts, indent=2)}
""".strip()

    resp = client.responses.create(
        model=model,
        input=prompt,
    )

    # SDK supports output_text() in many versions; fallback is safe
    text = ""
    if hasattr(resp, "output_text") and callable(resp.output_text):
        text = resp.output_text()
    elif hasattr(resp, "output_text"):
        text = str(resp.output_text)
    else:
        text = str(resp)

    return {"prompt": prompt, "response_text": text}


# ----------------------------
# Reporting
# ----------------------------

HTML_TEMPLATE = """
<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <title>Auto EDA Report</title>
  <style>
    body { font-family: Arial, sans-serif; margin: 24px; max-width: 1100px; }
    h1,h2 { margin-top: 22px; }
    .warn { padding: 10px; background: #fff3cd; border: 1px solid #ffeeba; border-radius: 6px; }
    .grid { display: grid; grid-template-columns: 1fr 1fr; gap: 14px; }
    img { max-width: 100%; border: 1px solid #ddd; border-radius: 6px; }
    table { border-collapse: collapse; width: 100%; }
    td, th { border: 1px solid #ddd; padding: 6px; font-size: 14px; }
    th { background: #f0f0f0; text-align: left; }
    pre { white-space: pre-wrap; background:#fafafa; border:1px solid #eee; padding:12px; border-radius:8px; }
  </style>
</head>
<body>
  <h1>Auto EDA Report</h1>

  {% if facts.privacy.warning %}
  <div class="warn"><b>Privacy Notice:</b> {{ facts.privacy.warning }}</div>
  {% endif %}

  <h2>Dataset Overview</h2>
  <p><b>Rows:</b> {{ facts.dataset_overview.rows }} &nbsp; <b>Columns:</b> {{ facts.dataset_overview.cols }}</p>

  <h3>Columns & Inferred Types</h3>
  <table>
    <tr><th>Column</th><th>Type</th><th>Missing (count)</th><th>Missing (%)</th></tr>
    {% for c in facts.columns %}
      <tr>
        <td>{{ c }}</td>
        <td>{{ facts.types[c].type }} (conf={{ facts.types[c].confidence | round(2) }})</td>
        <td>{{ facts.missingness.by_column[c].missing_count }}</td>
        <td>{{ (facts.missingness.by_column[c].missing_fraction*100) | round(1) }}%</td>
      </tr>
    {% endfor %}
  </table>

  <h3>Basic Data Quality Checks</h3>
  <ul>
    <li><b>Duplicate rows:</b> {{ facts.quality_checks.duplicate_rows }}</li>
    <li><b>Single-value columns:</b> {{ facts.quality_checks.single_value_columns }}</li>
    <li><b>High-missing columns (>=40%):</b> {{ facts.quality_checks.high_missing_columns }}</li>
  </ul>

  <h2>Descriptive Statistics</h2>

  <h3>Numeric (up to 2 columns)</h3>
  {% for col, st in facts.descriptive_stats.numeric.items() %}
    {% if st.available %}
      <h4>{{ col }}</h4>
      <ul>
        <li>min={{ st.min }}, max={{ st.max }}</li>
        <li>mean={{ st.mean }}, median={{ st.median }}, mode={{ st.mode }}</li>
        <li>std={{ st.std }}, IQR={{ st.iqr }} (Q1={{ st.q1 }}, Q3={{ st.q3 }})</li>
        <li>Outliers ({{ st.outlier_rule }}): {{ st.outlier_count }} ({{ (st.outlier_fraction*100) | round(1) }}%)</li>
      </ul>
    {% else %}
      <p><i>{{ col }}: numeric summary unavailable (all missing or non-numeric).</i></p>
    {% endif %}
  {% endfor %}

  <h3>Categorical (up to 1 column)</h3>
  {% for col, st in facts.descriptive_stats.categorical.items() %}
    {% if st.available %}
      <h4>{{ col }}</h4>
      <table>
        <tr><th>Value</th><th>Count</th><th>Percent</th></tr>
        {% for item in st.top_k %}
          <tr>
            <td>{{ item.value }}</td>
            <td>{{ item.count }}</td>
            <td>{{ (item.percent*100) | round(1) }}%</td>
          </tr>
        {% endfor %}
      </table>
    {% else %}
      <p><i>{{ col }}: categorical summary unavailable (all missing).</i></p>
    {% endif %}
  {% endfor %}

  <h2>Visualizations</h2>
  <p>Generated {{ plots|length }} plots (minimum 5 required).</p>
  <div class="grid">
    {% for p in plots %}
      <div>
        <img src="plots/{{ p.file }}" alt="{{ p.type }}">
        <div><small><b>{{ p.type }}</b> — {{ p.file }}</small></div>
      </div>
    {% endfor %}
  </div>

  <h2>Insights (Computed)</h2>
  <ul>
    {% for b in computed_insights %}
      <li>{{ b }}</li>
    {% endfor %}
  </ul>

  {% if llm_text %}
  <h2>Additional Analysis (LLM, Verified)</h2>
  <pre>{{ llm_text }}</pre>
  {% endif %}

  <h3>Limitations / Potential Bias</h3>
  <ul>
    {% for l in limitations %}
      <li>{{ l }}</li>
    {% endfor %}
  </ul>

</body>
</html>
"""


def write_reports(
    out_dir: Path,
    plots: List[Dict[str, str]],
    facts: Dict[str, Any],
    computed_insights: List[str],
    limitations: List[str],
    llm_text: Optional[str],
) -> None:
    # Markdown (concise)
    md = []
    md.append("# Auto EDA Report\n")
    if facts["privacy"]["warning"]:
        md.append(f"**Privacy Notice:** {facts['privacy']['warning']}\n")

    md.append("## Dataset Overview\n")
    md.append(f"- Rows: {facts['dataset_overview']['rows']}\n- Columns: {facts['dataset_overview']['cols']}\n")

    md.append("## Basic Data Quality Checks\n")
    qc = facts["quality_checks"]
    md.append(f"- Duplicate rows: {qc['duplicate_rows']}\n")
    md.append(f"- Single-value columns: {qc['single_value_columns']}\n")
    md.append(f"- High-missing columns (>=40%): {qc['high_missing_columns']}\n")

    md.append("## Visualizations\n")
    for p in plots:
        md.append(f"- {p['type']}: plots/{p['file']}\n")

    md.append("\n## Insights (Computed)\n")
    for b in computed_insights:
        md.append(f"- {b}\n")

    if llm_text:
        md.append("\n## Additional Analysis (LLM, Verified)\n")
        md.append(llm_text + "\n")

    md.append("\n## Limitations / Potential Bias\n")
    for l in limitations:
        md.append(f"- {l}\n")

    (out_dir / "report.md").write_text("\n".join(md), encoding="utf-8")

    # HTML
    if Template is None:
        # Fallback: write a minimal HTML wrapper if jinja2 isn't available
        html_fallback = "<html><body><pre>" + "\n".join(md) + "</pre></body></html>"
        (out_dir / "report.html").write_text(html_fallback, encoding="utf-8")
        return

    tpl = Template(HTML_TEMPLATE)
    html = tpl.render(
        facts=facts,
        plots=plots,
        computed_insights=computed_insights,
        limitations=limitations,
        llm_text=llm_text,
    )
    (out_dir / "report.html").write_text(html, encoding="utf-8")


# ----------------------------
# Main
# ----------------------------

def main() -> None:
    ap = argparse.ArgumentParser(description="Single-file Auto EDA + optional verified LLM insights.")
    ap.add_argument("--csv", required=True, help="Path to CSV file")
    ap.add_argument("--schema", default=None, help="Optional schema YAML/JSON")
    ap.add_argument("--out", required=True, help="Output folder")
    ap.add_argument("--max_rows", type=int, default=200000, help="Row cap for speed (default 200k)")
    ap.add_argument("--seed", type=int, default=42, help="Seed for reproducible sampling")
    ap.add_argument("--llm", choices=["openai", "none"], default="openai", help="Use OpenAI API or disable")
    ap.add_argument("--model", default="gpt-4.1-mini", help="OpenAI model name")
    args = ap.parse_args()

    out_dir = ensure_dirs(Path(args.out))
    plots_dir = out_dir / "plots"

    # Read + normalize
    df_raw, ingestion_read = read_csv_robust(args.csv, max_rows=args.max_rows, seed=args.seed)
    df_safe, col_meta = normalize_columns(df_raw)

    # Optional schema
    schema = load_schema(args.schema)

    # Apply missing codes
    missing_meta = apply_missing_codes(df_safe, schema, col_meta["mapping_original_to_safe"])

    # Build reverse mapping safe->original for reporting
    safe_to_original = {v: k for k, v in col_meta["mapping_original_to_safe"].items()}

    # Types (on safe df)
    type_info_safe = {c: infer_column_type(df_safe[c]) for c in df_safe.columns}
    type_map_safe = {c: type_info_safe[c]["type"] for c in df_safe.columns}

    # Privacy (uses original names)
    privacy = detect_sensitive_columns(df_safe, safe_to_original)

    # Quality checks
    duplicate_rows = int(df_safe.duplicated().sum())
    nunique = df_safe.nunique(dropna=True)
    single_value_cols = [safe_to_original.get(c, c) for c in df_safe.columns if int(nunique[c]) <= 1]
    miss_frac = df_safe.isna().mean()
    high_missing_cols = [safe_to_original.get(c, c) for c in df_safe.columns if float(miss_frac[c]) >= 0.4]

    missing_by_col = {
        safe_to_original.get(c, c): {"missing_count": int(df_safe[c].isna().sum()), "missing_fraction": float(miss_frac[c])}
        for c in df_safe.columns
    }
    top_missing = sorted(
        [{"column": k, **v} for k, v in missing_by_col.items()],
        key=lambda x: x["missing_fraction"],
        reverse=True
    )[:10]

    # Descriptive stats (choose columns based on types)
    chosen_num_safe, chosen_cat_safe = choose_columns_for_stats(type_map_safe)

    # If numeric columns are id_like misdetected as numeric, we still compute numeric stats safely.
    # If not enough numeric, numeric dict can be empty; report still runs.
    desc_numeric = {safe_to_original.get(c, c): numeric_summary(df_safe[c]) for c in chosen_num_safe}
    desc_cat = {safe_to_original.get(c, c): categorical_summary(df_safe[c]) for c in chosen_cat_safe}

    # Relationships
    numeric_cols_all = [c for c, t in type_map_safe.items() if t == "numeric"]
    rel = {"top_numeric_correlations": [
        {"col1": safe_to_original.get(x["col1"], x["col1"]), "col2": safe_to_original.get(x["col2"], x["col2"]), "corr": x["corr"]}
        for x in top_numeric_correlations(df_safe, numeric_cols_all, k=5)
    ]}

    # Plots
    plots = make_plots(df_safe, type_map_safe, safe_to_original, plots_dir)

    # Facts (single truth source)
    facts = {
        "run_meta": {
            "timestamp_utc": datetime.utcnow().isoformat() + "Z",
            "seed": args.seed,
            "max_rows": args.max_rows,
        },
        "dataset_overview": {"rows": int(df_safe.shape[0]), "cols": int(df_safe.shape[1])},
        "columns": [safe_to_original.get(c, c) for c in df_safe.columns],
        "ingestion": {
            "read": ingestion_read,
            "column_normalization": col_meta,
            "missing_codes": missing_meta,
        },
        "types": {safe_to_original.get(c, c): type_info_safe[c] for c in df_safe.columns},
        "missingness": {
            "by_column": missing_by_col,
            "top_missing_columns": top_missing,
        },
        "quality_checks": {
            "duplicate_rows": duplicate_rows,
            "single_value_columns": single_value_cols,
            "high_missing_columns": high_missing_cols,
        },
        "descriptive_stats": {
            "numeric": desc_numeric,
            "categorical": desc_cat,
        },
        "relationships": rel,
        "privacy": privacy,
    }
    save_json(out_dir / "facts.json", facts)

    computed_insights = generate_computed_insights(facts)
    limits = limitations_note(facts)

    llm_text_verified: Optional[str] = None
    llm_artifact: Optional[Dict[str, Any]] = None

    if args.llm == "openai":
        llm_artifact = call_openai_facts_only(facts=facts, model=args.model)
        raw_text = llm_artifact["response_text"]

        allowed = canonical_fact_numbers(facts)
        verified_text, audit = verify_llm_text(raw_text, allowed)

        if audit["dropped_lines"]:
            verified_text = (verified_text + "\n\n[Note] Some lines were removed because they contained numbers not present in facts.json.").strip()

        llm_text_verified = verified_text if verified_text else None

        save_json(out_dir / "llm_prompt_response.json", {"raw": llm_artifact, "verification_audit": audit})
        (out_dir / "llm_insights.md").write_text(llm_text_verified or "", encoding="utf-8")

    write_reports(
        out_dir=out_dir,
        plots=plots,
        facts=facts,
        computed_insights=computed_insights,
        limitations=limits,
        llm_text=llm_text_verified,
    )

    print(f"Done. Outputs written to: {out_dir}")
    print(f"- report.html: {out_dir/'report.html'}")
    print(f"- report.md:   {out_dir/'report.md'}")
    print(f"- facts.json:  {out_dir/'facts.json'}")
    print(f"- plots/:      {plots_dir}")


if __name__ == "__main__":
    main()

