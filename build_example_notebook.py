import nbformat as nbf

nb = nbf.v4.new_notebook()
cells = []

def md(text):
    cells.append(nbf.v4.new_markdown_cell(text))

def code(text):
    cells.append(nbf.v4.new_code_cell(text))

md("""# Worked Example: Cleaning the goodbooks-10k Dataset

This notebook applies the functions in `cleaning_utils.py` to a real, messy
dataset from the [goodbooks-10k](https://github.com/zygmuntz/goodbooks-10k)
book dataset and narrates the actual judgment calls made along the way.
Every issue found here was discovered by hand first; the utilities in this
repo generalize that process into reusable, tested functions.""")

md("## Setup")
code("""import sys, os
sys.path.insert(0, '..')
import pandas as pd
from cleaning_utils import (
    missing_value_report, scoped_dropna, fillna_with_log,
    flag_suspicious_categorical, keyword_scan, keyword_scan_details,
    outlier_report, duplicate_report, compare_filter_precision,
)

# Fetch the raw data directly keeps this notebook self-contained and
# reproducible without depending on another project's folder.
os.makedirs('data', exist_ok=True)
books_path = 'data/books.csv'
if not os.path.exists(books_path):
    import urllib.request
    url = 'https://raw.githubusercontent.com/zygmuntz/goodbooks-10k/master/books.csv'
    urllib.request.urlretrieve(url, books_path)

books = pd.read_csv(books_path)
print(f"{len(books):,} books loaded")
books.head(3)""")

md("""## 1. Missing values: know what's missing before deciding what to do about it""")
code("""missing_value_report(books)""")

md("""`isbn` and `isbn13` are missing for a meaningful chunk of rows but since
neither is used anywhere downstream (no lookup, no join), that's fine to
leave alone. `language_code` and `original_publication_year` are different:
both feed into real analysis later, so their gaps matter.""")

md("## 2. Categorical sanity check: does anything look like it doesn't belong?")
code("""flag_suspicious_categorical(books, 'authors', max_share=0.003)""")

md(""""BookRags" surfaces immediately as a flagged value but it is a company name, not
a person, appearing as an "author." That's the tell that something in this
dataset isn't actually a book in the way the rest of the catalog is.""")

md("## 3. Confirm narrowly before writing a broad filter")
code("""narrow_terms = ['BookRags', 'SparkNotes', 'Study Guide', 'Summary and Analysis', 'CliffsNotes']
keyword_scan(books, 'title', narrow_terms)""")

md("""Only `BookRags` catches anything. Before deciding this is "clean enough,"
it's worth testing whether a broader, more generic net would catch more junk
and checking whether that's actually a good thing.""")

code("""result = compare_filter_precision(
    books, 'title',
    narrow_terms=narrow_terms,
    broad_terms=['Guide', 'Notes', 'Companion', 'Summary'],
)
result['extra_only_in_broad'][['title', 'authors']].head(10)""")

md("""This is the core lesson of this repo: broadening the filter to catch more
"maybe-junk" terms increases matches from 1 to over 50. However the extra ~50
are almost entirely legitimate books ("The Hitchhiker's Guide to the
Galaxy", "Zero to One: Notes on Startups"). The narrow filter has higher
precision; the broad one has higher recall but a severe false-positive
problem. For a recommender system, false positives (wrongly removing real
books) are worse than false negatives (missing an occasional junk entry)
so the narrow filter is the right choice here, not a shortcut.""")

md("## 4. Handling missing values: scope every drop, log every fill")
code("""books_clean = fillna_with_log(books, 'language_code', 'unknown')""")

code("""books_with_year = scoped_dropna(books_clean, subset=['original_publication_year'])""")

md("""Note what did **not** happen: no `books.dropna()` with no arguments, which
would have dropped rows for missing `isbn` too but it is a column nobody uses.
Every drop and fill here is scoped to a column that's actually load-bearing
for some downstream analysis.""")

md("## 5. Outliers: statistical extremity isn't the same as a data error")
code("""year_outliers = outlier_report(books_with_year, 'original_publication_year', method='iqr')
year_outliers[['title', 'original_publication_year']].sort_values('original_publication_year').head(10)""")

md("""The lowest values here are genuine ancient texts — *The Epic of Gilgamesh*
(-1750), *The Iliad* (-750), *The Art of War* (-500). These are correct
data, not errors, even though they're extreme statistical outliers. The
`outlier_report()` function's docstring says this explicitly: flagging is
not the same as concluding something is wrong. A human still has to look.""")

md("## 6. Duplicates: not always a bug")
code("""dupes = duplicate_report(books_clean, 'title')
print(f"{dupes['title'].nunique()} distinct titles have more than one entry")
dupes[['title', 'authors', 'book_id']].sort_values('title').head(10)""")

md("""Some of these are genuinely different editions or translations sharing a
title but it is worth knowing about, not necessarily worth dropping. Whether to
deduplicate depends entirely on what the downstream analysis needs (e.g. a
recommender counting "distinct books" would want these collapsed; an
analysis of "ratings volume per edition" would not).""")

md("""## Summary

| Step | Function used | Decision made |
|---|---|---|
| Missing values | `missing_value_report()` | Ignored `isbn`/`isbn13` (unused); addressed `language_code`, `original_publication_year` |
| Junk detection | `flag_suspicious_categorical()` | Flagged "BookRags" as a non-author entity |
| Filter validation | `compare_filter_precision()` | Kept the narrow filter — broad version had a ~98% false-positive rate |
| Missing fill | `fillna_with_log()` | Filled `language_code` gaps with `'unknown'`, logged the count |
| Scoped drop | `scoped_dropna()` | Dropped rows missing `original_publication_year` — only for that analysis |
| Outliers | `outlier_report()` | Confirmed ancient texts are real data, not errors |
| Duplicates | `duplicate_report()` | Surfaced for review, not auto-dropped |

Every decision here is a judgment call informed by what the data will be
used for. The functions surface the information; they don't make the
decision for you.""")

nb['cells'] = cells
with open('examples/book_data_walkthrough.ipynb', 'w') as f:
    nbf.write(nb, f)

print("Notebook written.")
