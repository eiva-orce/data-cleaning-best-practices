import sys
import os
import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from cleaning_utils import (
    missing_value_report,
    scoped_dropna,
    fillna_with_log,
    flag_suspicious_categorical,
    keyword_scan,
    keyword_scan_details,
    outlier_report,
    duplicate_report,
    compare_filter_precision,
)


@pytest.fixture
def sample_books():
    """A small synthetic dataset mirroring the real issues found in goodbooks-10k:
    a junk entry (BookRags), a missing language_code, an ancient-text outlier,
    and a duplicate title."""
    return pd.DataFrame({
        "title": [
            "The Hunger Games",
            "The Name of the Wind",
            "BookRags Summary: A Storm of Swords",
            "The Hitchhiker's Guide to the Galaxy",
            "The Epic of Gilgamesh",
            "Dune",
            "Dune",  # duplicate
        ],
        "authors": [
            "Suzanne Collins", "Patrick Rothfuss", "BookRags",
            "Douglas Adams", "Unknown", "Frank Herbert", "Frank Herbert",
        ],
        "language_code": ["eng", "eng", None, "eng", None, "eng", "eng"],
        "original_publication_year": [2008, 2007, 2010, 1979, -1750, 1965, 1965],
        "average_rating": [4.3, 4.5, 3.0, 4.2, 4.0, 4.25, 4.25],
    })


def test_missing_value_report(sample_books):
    report = missing_value_report(sample_books)
    assert report.loc["language_code", "missing_count"] == 2
    assert report.loc["language_code", "missing_pct"] == pytest.approx(28.57, abs=0.1)
    # a column with no missing values should still appear, at 0%
    assert report.loc["title", "missing_pct"] == 0.0


def test_scoped_dropna_only_affects_subset(sample_books):
    filtered = scoped_dropna(sample_books, subset=["language_code"], verbose=False)
    assert len(filtered) == 5  # drops the 2 rows with missing language_code
    assert filtered["language_code"].isnull().sum() == 0
    # original untouched
    assert sample_books["language_code"].isnull().sum() == 2


def test_fillna_with_log_does_not_mutate_original(sample_books):
    result = fillna_with_log(sample_books, "language_code", "unknown", verbose=False)
    assert result["language_code"].isnull().sum() == 0
    assert result.loc[2, "language_code"] == "unknown"
    # original dataframe must be untouched
    assert sample_books["language_code"].isnull().sum() == 2


def test_flag_suspicious_categorical_catches_dominant_value(sample_books):
    # "Frank Herbert" appears 2/7 times (~29%), well above a 2% threshold
    result = flag_suspicious_categorical(sample_books, "authors", max_share=0.02)
    assert result.loc["Frank Herbert", "flagged"] == True  # noqa: E712


def test_keyword_scan_counts_per_term(sample_books):
    result = keyword_scan(sample_books, "title", ["BookRags", "Dune", "Nonexistent"])
    counts = dict(zip(result["term"], result["match_count"]))
    assert counts["BookRags"] == 1
    assert counts["Dune"] == 2
    assert counts["Nonexistent"] == 0


def test_keyword_scan_details_returns_matched_rows(sample_books):
    details = keyword_scan_details(sample_books, "title", "BookRags", extra_columns=["authors"])
    assert len(details) == 1
    assert details.iloc[0]["authors"] == "BookRags"


def test_outlier_report_flags_ancient_text(sample_books):
    outliers = outlier_report(sample_books, "original_publication_year", method="iqr")
    # The Epic of Gilgamesh at -1750 should be flagged as a statistical outlier
    assert "The Epic of Gilgamesh" in outliers["title"].values


def test_duplicate_report_finds_dune(sample_books):
    dupes = duplicate_report(sample_books, "title")
    assert len(dupes) == 2
    assert set(dupes["title"]) == {"Dune"}


def test_compare_filter_precision_shows_false_positive_gap(sample_books, capsys):
    result = compare_filter_precision(
        sample_books, "title",
        narrow_terms=["BookRags"],
        broad_terms=["BookRags", "Guide"],
    )
    # narrow catches only the real junk entry
    assert len(result["narrow_matches"]) == 1
    # broad catches the junk entry AND the legitimate Hitchhiker's Guide
    assert len(result["broad_matches"]) == 2
    # exactly 1 row is a false positive introduced by broadening the filter
    assert len(result["extra_only_in_broad"]) == 1
    assert result["extra_only_in_broad"].iloc[0]["title"] == "The Hitchhiker's Guide to the Galaxy"
