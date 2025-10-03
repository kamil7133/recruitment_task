from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence, Tuple, Optional
import re

import pandas as pd
from pandas import DataFrame, Series


# ---------- Compile-time constants (centralized for easy auditing/tuning) ------------------------

# Column (and new column) naming policy: letters + underscores only.
_NAME_RE = re.compile(r"^[A-Za-z_]+$")

# Lexical tokens allowed in `role`: a valid name or one of the explicit operators.
_TOKEN_RE = re.compile(r"[A-Za-z_]+|[+\-*]")

# Remove **all** whitespace kinds – not just spaces – to ensure tabs/newlines are tolerated.
_WS_RE = re.compile(r"\s+")


@dataclass(frozen=True)
class ParseResult:
    """
    Immutable container for the tokenized expression.
    The grammar enforces:
        NAME (OP NAME)*

    Example:
        tokens = ("label_one", "+", "label_two", "*", "label_three")
    """
    tokens: Tuple[str, ...]


# ---------------------------------- Public API ---------------------------------------------------

def add_virtual_column(df: DataFrame, role: str, new_column: str) -> DataFrame:
    """
    Return a new DataFrame containing `new_column` = evaluated(`role`) over `df`.

    Parameters
    ----------
    df : pandas.DataFrame
        Input data. This function NEVER mutates the input; it returns a copy on success.
    role : str
        Expression composed of valid column names and operators (+, -, *).
        - No parentheses/functions are supported.
        - Arbitrary whitespace (spaces/tabs/newlines) is allowed and ignored.
        - All referenced columns MUST exist and be numeric.
    new_column : str
        Name of the resulting column. Must satisfy the naming policy: `^[A-Za-z_]+$`.

    Returns
    -------
    pandas.DataFrame
        - A new DataFrame with the appended/overwritten `new_column` on success.
        - An **empty** DataFrame on any validation or computation error.

    Rationale
    ---------
    - Strict validation ensures the function is safe in production environments.
    - Deterministic behavior (returning empty DF on invalid inputs) simplifies calling code.
    """
    # Defensive copy so we never modify the caller's DataFrame in place.
    df_out = df.copy()

    # 1) Validate DataFrame column names and target name.
    if not _all_df_columns_are_valid(df_out):
        return _empty_df()

    if not _is_valid_name(new_column):
        return _empty_df()

    # 2) Tokenize and validate the `role` expression (structure + whitelist of tokens).
    parse = _tokenize_and_validate(role)
    if parse is None:
        return _empty_df()

    # 3) Ensure all referenced columns exist and are numeric.
    col_names = parse.tokens[::2]  # even positions: 0,2,4,... are column names
    if not all(name in df_out.columns for name in col_names):
        return _empty_df()

    # Numeric-only policy avoids ambiguous behavior with string columns.
    if not all(pd.api.types.is_numeric_dtype(df_out[name]) for name in col_names):
        return _empty_df()

    # 4) Evaluate with conventional precedence: `*` first, then `+`/`-` left-to-right.
    try:
        result = _evaluate_with_precedence(df_out, parse.tokens)
    except Exception:
        # We deliberately avoid leaking internal details (stack traces) to the caller.
        return _empty_df()

    # 5) Write the computed series (overwrite if exists) and return the new DataFrame.
    df_out[new_column] = result
    return df_out


# ------------------------------- Internal helpers: validation ------------------------------------

def _empty_df() -> DataFrame:
    """
    Consistent failure artifact.
    Returning a truly empty DataFrame (no rows, no columns) makes downstream checks trivial.
    """
    return pd.DataFrame()


def _is_valid_name(name: object) -> bool:
    """
    Enforce name policy: letters + underscores only.
    We treat non-str inputs as invalid (e.g., None, numbers).
    """
    return isinstance(name, str) and bool(_NAME_RE.fullmatch(name))


def _all_df_columns_are_valid(df: DataFrame) -> bool:
    """
    Validate the entire column index once.
    - Early detection of 'dirty' inputs (e.g., columns named 'label-1').
    - Keeps behavior predictable and error handling consistent.
    """
    try:
        return all(_is_valid_name(str(c)) for c in df.columns)
    except Exception:
        return False


def _tokenize_and_validate(role: object) -> Optional[ParseResult]:
    """
    Tokenize and validate `role`.

    Accepted grammar (no parentheses):
        ROLE := NAME (OP NAME)*
        NAME := /[A-Za-z_]+/
        OP   := '+' | '-' | '*'

    Validates:
    - role is a non-empty string,
    - only allowed tokens appear (names/operators/whitespace),
    - alternation NAME/OP/NAME/OP/.../NAME holds,
    - operators are strictly in {+, -, *}.
    """
    if not isinstance(role, str):
        return None

    # Trim outer whitespace; reject empty/whitespace-only roles early.
    stripped = role.strip()
    if not stripped:
        return None

    # Extract allowed tokens (names or operators). Whitespace remains out-of-band.
    tokens = _TOKEN_RE.findall(stripped)

    # Reconstruct the expression WITHOUT ANY WHITESPACE and compare with the original minus whitespace.
    # This rejects any disallowed character (e.g., '&', '/', '\', '.', digits in names).
    rebuilt_no_ws = "".join(tokens)
    original_no_ws = _WS_RE.sub("", stripped)
    if rebuilt_no_ws != original_no_ws:
        return None

    # Structural check: must start and end with NAME and alternate with OP in-between.
    if len(tokens) % 2 == 0:
        # Even number of tokens implies it ends with an operator or starts with one.
        return None

    for i, tok in enumerate(tokens):
        if i % 2 == 0:
            # NAME positions
            if not _is_valid_name(tok):
                return None
        else:
            # OP positions
            if tok not in {"+", "-", "*"}:
                return None

    return ParseResult(tokens=tuple(tokens))


# ------------------------------- Internal helpers: evaluation ------------------------------------

def _evaluate_with_precedence(df: DataFrame, tokens: Sequence[str]) -> Series:
    """
    Evaluate the token stream with operator precedence.

    Precedence & associativity:
      1. `*` (left-associative within its level)
      2. `+` and `-` (left-associative, evaluated after all `*` have been folded)

    Strategy:
      Phase 1 (fold '*'):
        Walk tokens left-to-right, multiplying adjacent NAMEs separated by '*', building
        a list like: [Series, '+', Series, '-', Series] (no '*' remains).
      Phase 2 (apply + / -):
        Reduce the list left-to-right.

    NaN behavior:
      Vectorized Pandas arithmetic naturally propagates NaNs. We don't mask/clean them here.
    """
    def col_series(name: str) -> Series:
        return df[name]

    # fold all multiplications into contiguous Series blocks
    values_ops: List[object] = []
    i = 0
    current = col_series(tokens[0])

    while i < len(tokens) - 1:
        op = tokens[i + 1]
        rhs = col_series(tokens[i + 2])
        if op == "*":
            current = current * rhs
            i += 2
        else:
            values_ops.append(current)
            values_ops.append(op)  # '+' or '-'
            current = rhs
            i += 2

    values_ops.append(current)

    # apply + and - left-to-right
    result = values_ops[0]
    j = 1
    while j < len(values_ops):
        op = values_ops[j]
        rhs = values_ops[j + 1]
        if op == "+":
            result = result + rhs
        elif op == "-":
            result = result - rhs
        else:
            # Defensive programming: should never happen given prior validation.
            raise ValueError(f"Unexpected operator in reduction: {op!r}")
        j += 2

    return result
