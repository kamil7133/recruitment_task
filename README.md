# Virtual Column Builder

## Overview
This project implements a function `add_virtual_column(df, role, new_column)` that:
- Validates a textual arithmetic expression (`role`) over DataFrame columns,
- Computes a new column using vectorized Pandas operations (`+`, `-`, `*` with precedence),
- Returns a **new** DataFrame with the extra column,
- Returns an **empty DataFrame** if validation fails (invalid names, unsupported operators, missing columns, etc.).

The implementation avoids `eval` for safety and uses explicit tokenization and operator handling.

---

## File structure
```

RecruitmentTask/
├── virtual_column.py      # main implementation
├── test_extra.py          # extended test coverage (pytest)
├── requirements.txt       # dependencies
└── README.md              # this file

````

---

## Usage
```python
import pandas as pd
from virtual_column import add_virtual_column

df = pd.DataFrame({"a": [1, 2, 3], "b": [10, 20, 30]})
out = add_virtual_column(df, "a + b", "sum")
print(out)
````

Output:

```
   a   b  sum
0  1  10   11
1  2  20   22
2  3  30   33
```

---

## Running tests

Install dependencies and run pytest:

```bash
pip install -r requirements.txt
pytest -q
```

---

## Requirements

* Python 3.10+ (tested on 3.13)
* pandas
* numpy
* pytest

Install with:

```bash
pip install -r requirements.txt
```

---

## Notes

* Column names must match the regex: `^[A-Za-z_]+$`
* Operators allowed: `+`, `-`, `*` (with standard precedence)
* If any validation fails → empty DataFrame is returned.
* Existing column with the same name as `new_column` will be overwritten (explicit behavior).
