# Data Cleaning Best Practices (aka. Data Quality Checker)

A small, tested library of data cleaning diagnostic functions, plus a worked
example applying them to a real, messy dataset ([goodbooks-10k](https://github.com/zygmuntz/goodbooks-10k)).

Every principle here comes from actual cleaning work, not theory, including a mistake worth learning from (see [Principle 3](#3-a-broader-filter-is-not-a-safer-filter)). The functions are small, reusable, and tested to verify they honor the contract set out in their docstrings.

## Install

```bash
pip install -r requirements.txt
```

```python
from cleaning_utils import missing_value_report, keyword_scan, outlier_report
# ... see examples/book_data_walkthrough.ipynb for a full worked example
```

## Principles

### 1. Diagnose before you mutate

Every function in `cleaning_utils.py` is diagnostic first: it reports on
the data (a count, a flagged row, a comparison) rather than silently
changing it. `missing_value_report()` doesn't fill anything. `outlier_report()`
doesn't drop anything. A human decides what to do with the information.
The function's job is only to surface it clearly.

### 2. Scope every drop and fill to the column that actually matters

`df.dropna()` with no arguments drops a row if *any* column is missing —
including columns you never use. On a real dataset this silently deleted
rows for a missing `isbn`, a field nobody downstream ever touches.
`scoped_dropna(df, subset=['original_publication_year'])` only drops rows
where that specific column (one actually used in analysis) is missing,
leaving the rest of the dataset untouched.

The same logic applies to filling: `fillna_with_log()` always reports how
many rows were affected, so a fill is a visible, intentional decision.

### 3. A broader filter is not a safer filter

This is the most important lesson in the repo, and it came from a real
mistake. Filtering book titles for known junk-service brand names
(`BookRags`, `SparkNotes`, `CliffsNotes`, ...) caught exactly one bad
entry, with zero false positives. Broadening the filter to catch more
generic "maybe junk" words like `Guide` or `Notes` increased matches from
1 to 55. However, 54 of those were **legitimate, real books**: *The
Hitchhiker's Guide to the Galaxy*, *Zero to One: Notes on Startups*,
*The Power of Now: A Guide to Spiritual Enlightenment*.

More recall isn't automatically better. `compare_filter_precision()`
exists specifically to make this tradeoff visible before you commit to a
filter. It shows you exactly which rows a broader net catches that a
narrower one doesn't, so you can eyeball the difference before deciding.

### 4. A statistical outlier is not automatically a data error

`outlier_report()` will flag *The Epic of Gilgamesh* (publication year
-1750) as an extreme outlier. It's also completely correct data. Outlier detection tells you where to look;
it doesn't tell you what's wrong. Confirming a flagged value is actually
an error, versus a genuine (if extreme) data point, requires domain
knowledge no function can fully supply.

### 5. Categorical columns hide entities that don't belong

`flag_suspicious_categorical()` looks for values that appear far more
often than a normal member of that category would (a real author writes
a handful of books, not dozens). When "BookRags" turned up appearing as an
"author" disproportionately often, that pattern, was the actual signal that something in the
dataset wasn't a real book at all.

### 6. Duplicates require judgment, not automatic deduplication

Two rows sharing a title might be different editions, different
translations, or a genuine scrape error.`duplicate_report()` surfaces
them for review rather than assuming they should be merged or dropped.
What's "correct" depends entirely on what the downstream analysis needs.

## Repo structure

```
data-cleaning-best-practices/
├── README.md                           # this file
├── cleaning_utils.py                   # the importable function library
├── requirements.txt
├── examples/
│   └── book_data_walkthrough.ipynb     # full worked example on real data
└── tests/
    └── test_cleaning_utils.py          # pytest suite, synthetic test data
```

## Running the tests

```bash
pytest tests/ -v
```

## Function reference

| Function | Purpose |
|---|---|
| `missing_value_report(df)` | Count + % missing per column |
| `scoped_dropna(df, subset)` | Drop rows missing specific columns only |
| `fillna_with_log(df, column, value)` | Fill missing values, logging how many rows changed |
| `flag_suspicious_categorical(df, column, max_share)` | Flag categorical values appearing disproportionately often |
| `keyword_scan(df, text_column, terms)` | Test multiple keyword terms individually, see match counts per term |
| `keyword_scan_details(df, text_column, term)` | See the actual rows a single term matches |
| `outlier_report(df, column, method)` | Flag numeric outliers via IQR or z-score |
| `duplicate_report(df, column)` | Surface duplicate/near-duplicate values for review |
| `compare_filter_precision(df, text_column, narrow_terms, broad_terms)` | Compare a narrow vs. broad filter, showing false-positive risk |
