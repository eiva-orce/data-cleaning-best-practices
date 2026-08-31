"""
cleaning_utils.py

A small, importable library of data cleaning diagnostic functions.

These are deliberately *diagnostic first*; every function here reports on
the data (counts, percentages, flagged rows) rather than silently mutating
it. The philosophy: a human should look at what a cleaning step would do
before it happens, not discover it after the fact in a bug three steps
downstream. See README.md for the full reasoning behind each function.
"""

from __future__ import annotations

import pandas as pd
import numpy as np


# ---------------------------------------------------------------------------
# 1. Missing values
# ---------------------------------------------------------------------------

def missing_value_report(df: pd.DataFrame) -> pd.DataFrame:
    """
    Report missing value counts and percentages for every column.

    A count alone doesn't tell you if a gap is a minor annoyance (2%) or a
    column you can no longer trust (40%+). This returns both, sorted so the
    worst offenders are at the top.

    Returns
    -------
    DataFrame indexed by column name, with 'missing_count' and 'missing_pct'.
    Columns with zero missing values are still included (pct = 0.0) so you
    get the full picture in one call.
    """
    counts = df.isnull().sum()
    pct = (counts / len(df) * 100).round(2)
    report = pd.DataFrame({"missing_count": counts, "missing_pct": pct})
    return report.sort_values("missing_pct", ascending=False)


def scoped_dropna(df: pd.DataFrame, subset: list[str], verbose: bool = True) -> pd.DataFrame:
    """
    Drop rows missing values in `subset` columns only but never a blanket
    `df.dropna()`, which silently drops rows for ANY missing column,
    including ones you never intended to use.

    Returns a new filtered DataFrame; the original `df` is left untouched,
    so you can keep using the full dataset for analyses that don't depend
    on `subset`.

    Parameters
    ----------
    df : the source DataFrame (not modified)
    subset : columns that must be non-null to keep a row
    verbose : if True, prints how many rows were kept vs dropped
    """
    filtered = df.dropna(subset=subset)
    if verbose:
        kept, total = len(filtered), len(df)
        pct = kept / total * 100 if total else 0
        print(f"scoped_dropna(subset={subset}): kept {kept}/{total} rows ({pct:.1f}%)")
    return filtered


def fillna_with_log(df: pd.DataFrame, column: str, value, verbose: bool = True) -> pd.DataFrame:
    """
    Fill missing values in `column` with `value`, printing how many rows
    were affected. Returns a new DataFrame (does not mutate `df` in place),
    so the fill is explicit and traceable rather than a silent side effect.
    """
    n_missing = df[column].isnull().sum()
    result = df.copy()
    result[column] = result[column].fillna(value)
    if verbose:
        pct = n_missing / len(df) * 100 if len(df) else 0
        print(f"fillna_with_log('{column}'): filled {n_missing} rows ({pct:.1f}%) with {value!r}")
    return result


# ---------------------------------------------------------------------------
# 2. Categorical sanity checks
# ---------------------------------------------------------------------------

def flag_suspicious_categorical(
    df: pd.DataFrame, column: str, max_share: float = 0.02, top_n: int = 20
) -> pd.DataFrame:
    """
    Flag values in a categorical column that appear more often than
    `max_share` of all rows but it is a common signature of a non-person/non-entity
    value hiding in the data (e.g. a company name like "BookRags" showing
    up as a book "author" far more often than any real author would).

    This doesn't tell you a value IS wrong but it tells you where to look.
    Confirm with domain judgment before treating a flagged value as junk.

    Returns
    -------
    DataFrame of the top `top_n` most frequent values in `column`, with
    their share of total rows, sorted descending. A `flagged` column marks
    anything exceeding `max_share`.
    """
    counts = df[column].value_counts(dropna=False).head(top_n)
    share = (counts / len(df)).round(4)
    result = pd.DataFrame({"count": counts, "share": share})
    result["flagged"] = result["share"] > max_share
    return result


def keyword_scan(
    df: pd.DataFrame, text_column: str, terms: list[str], case_sensitive: bool = False
) -> pd.DataFrame:
    """
    Check a list of candidate junk/keyword terms against a text column,
    one term at a time but deliberately NOT combined into one regex, so you
    can see which specific term is doing the work and how many rows each
    one catches before deciding whether to act on it.

    This is the single most important habit in this library: test each
    term narrowly first. A combined broad regex can silently produce a
    high false-positive rate (e.g. "Guide" or "Notes" catching dozens of
    perfectly legitimate book titles alongside the one real junk entry).

    Returns
    -------
    DataFrame with one row per term: term, match_count, and match_pct.
    Use `keyword_scan_details()` to see the actual matched rows for a term.
    """
    rows = []
    for term in terms:
        mask = df[text_column].str.contains(term, case=case_sensitive, na=False)
        rows.append({"term": term, "match_count": int(mask.sum()),
                      "match_pct": round(mask.sum() / len(df) * 100, 3)})
    return pd.DataFrame(rows)


def keyword_scan_details(
    df: pd.DataFrame, text_column: str, term: str, extra_columns: list[str] | None = None,
    case_sensitive: bool = False,
) -> pd.DataFrame:
    """
    Show the actual rows matched by a single keyword term, so you can
    eyeball whether they're genuinely junk before filtering them out.
    """
    mask = df[text_column].str.contains(term, case=case_sensitive, na=False)
    cols = [text_column] + (extra_columns or [])
    return df.loc[mask, cols]


# ---------------------------------------------------------------------------
# 3. Numeric outliers
# ---------------------------------------------------------------------------

def outlier_report(df: pd.DataFrame, column: str, method: str = "iqr", k: float = 1.5) -> pd.DataFrame:
    """
    Flag statistical outliers in a numeric column using IQR (default) or
    z-score. Returns the flagged rows sorted by the column value, so
    extremes are grouped together for quick visual review.

    IMPORTANT: a statistical outlier is not automatically a data error but it is
    e.g. an ancient text with a publication year of -1750 is a genuine,
    correct value (The Epic of Gilgamesh), not a typo. Always inspect
    flagged rows before dropping or correcting them.

    Parameters
    ----------
    method : 'iqr' (default) or 'zscore'
    k : threshold multiplier: 1.5 IQR is the conventional default;
        for zscore, k is the number of standard deviations (default
        interpretation: pass k=3 for a 3-sigma cutoff)
    """
    series = df[column].dropna()
    if method == "iqr":
        q1, q3 = series.quantile(0.25), series.quantile(0.75)
        iqr = q3 - q1
        lower, upper = q1 - k * iqr, q3 + k * iqr
        mask = (df[column] < lower) | (df[column] > upper)
    elif method == "zscore":
        z = (series - series.mean()) / series.std()
        mask = df[column].isin(series[z.abs() > k])
    else:
        raise ValueError("method must be 'iqr' or 'zscore'")
    return df.loc[mask.fillna(False)].sort_values(column)


# ---------------------------------------------------------------------------
# 4. Duplicates
# ---------------------------------------------------------------------------

def duplicate_report(df: pd.DataFrame, column: str) -> pd.DataFrame:
    """
    Return all rows involved in a duplicate (or near exact duplicate) value
    in `column`, grouped together for review. Not every duplicate is wrong
    (e.g. multiple editions of the same book title) but it surfaces them
    for a judgment call, not an automatic drop.
    """
    dupe_mask = df[column].duplicated(keep=False)
    return df.loc[dupe_mask].sort_values(column)


# ---------------------------------------------------------------------------
# 5. Filter precision comparison
# ---------------------------------------------------------------------------

def compare_filter_precision(
    df: pd.DataFrame, text_column: str, narrow_terms: list[str], broad_terms: list[str],
) -> dict:
    """
    Compare a narrow (specific, high-precision) filter against a broad
    (generic, high-recall) one on the same column. Prints the row counts
    caught by each and how many rows the broad filter catches that the
    narrow one doesn't but the "extra" catches are exactly the ones that
    need manual review, since that's where false positives hide.

    This function exists because of a real lesson: a filter for
    'BookRags|SparkNotes|Study Guide|CliffsNotes' caught exactly 1 genuine
    junk entry with zero false positives, while broadening it to include
    generic words like 'Guide' or 'Notes' caught 55 rows but of which 54
    were legitimate books (e.g. "The Hitchhiker's Guide to the Galaxy").
    """
    narrow_pattern = "|".join(narrow_terms)
    broad_pattern = "|".join(broad_terms)
    narrow_mask = df[text_column].str.contains(narrow_pattern, case=False, na=False)
    broad_mask = df[text_column].str.contains(broad_pattern, case=False, na=False)
    extra_mask = broad_mask & ~narrow_mask

    print(f"Narrow filter matches: {narrow_mask.sum()}")
    print(f"Broad filter matches:  {broad_mask.sum()}")
    print(f"Extra rows caught only by the broad filter (review these for false positives): {extra_mask.sum()}")

    return {
        "narrow_matches": df.loc[narrow_mask],
        "broad_matches": df.loc[broad_mask],
        "extra_only_in_broad": df.loc[extra_mask],
    }
