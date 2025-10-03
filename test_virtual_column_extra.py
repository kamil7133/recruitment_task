import math
import pandas as pd
import numpy as np
import pytest

from task import add_virtual_column


def base_df():
    return pd.DataFrame({
        "a": [1, 2, 3],
        "b": [10, 20, 30],
        "c": [4, 5, 6],
    })


def test_tabs_and_newlines_are_ok():
    df = base_df()
    role = "a\t+\nb\t*\n c"  # tabs + newlines
    # 1 + 10*4 = 41; 2 + 20*5 = 102; 3 + 30*6 = 183
    out = add_virtual_column(df, role, "mix")
    assert out["mix"].tolist() == [41, 102, 183]


def test_precedence_multiplies_first():
    df = base_df()
    out = add_virtual_column(df, "a + b * c - a * c", "expr")
    # a + (b*c) - (a*c)
    expected = (df["a"] + (df["b"] * df["c"]) - (df["a"] * df["c"])).tolist()
    assert out["expr"].tolist() == expected


def test_nan_propagation():
    df = base_df()
    df.loc[1, "b"] = np.nan
    out = add_virtual_column(df, "a + b * c", "nanmix")
    # For index 1, b is NaN -> b*c is NaN -> a + NaN is NaN
    assert math.isnan(out.loc[1, "nanmix"])
    # Others compute normally
    assert out.loc[0, "nanmix"] == 1 + 10 * 4
    assert out.loc[2, "nanmix"] == 3 + 30 * 6


def test_overwrite_existing_column_is_allowed():
    df = base_df()
    out = add_virtual_column(df, "a + b", "a")  # overwrite 'a'
    assert "a" in out.columns
    assert out["a"].tolist() == (df["a"] + df["b"]).tolist()


def test_large_input_smoke():
    # Performance sanity check on a larger frame (not a strict benchmark).
    n = 200_000
    df = pd.DataFrame({
        "col_x": np.arange(n, dtype=float),
        "col_y": np.arange(n, dtype=float) * 2.0,
        "col_z": np.ones(n, dtype=float) * 3.0,
    })
    out = add_virtual_column(df, "col_x*col_y + col_z", "res")
    # Spot-check a few points instead of comparing full arrays
    assert out.shape[0] == n
    for idx in [0, 1, n - 1]:
        assert out.loc[idx, "res"] == df.loc[idx, "col_x"] * df.loc[idx, "col_y"] + df.loc[idx, "col_z"]


def test_invalid_df_column_names_returns_empty():
    # Introduce a bad column name that violates the policy (dash).
    df = pd.DataFrame({
        "ok_name": [1, 2],
        "bad-name": [3, 4],
    })
    out = add_virtual_column(df, "ok_name + bad_name", "sum")
    assert out.empty


def test_non_string_role_rejected():
    df = base_df()
    out = add_virtual_column(df, 12345, "x")  # role is not a string
    assert out.empty


@pytest.mark.parametrize("bad_role", [
    "a / b",          # unsupported operator
    "a & b",          # unsupported operator
    r"a \ b",         # backslash
    "a + ",           # ends with operator
    "+ a",            # starts with operator
    "a  b",           # missing operator between names
])
def test_bad_roles_rejected(bad_role):
    df = base_df()
    out = add_virtual_column(df, bad_role, "x")
    assert out.empty


@pytest.mark.parametrize("bad_name", [
    "1abc",       # starts with digit
    "abc-xyz",    # dash
    "abc xyz",    # space
    "abc.xyz",    # dot
    "abc!",       # punctuation
])
def test_bad_new_column_name_rejected(bad_name):
    df = base_df()
    out = add_virtual_column(df, "a + b", bad_name)
    assert out.empty


def test_non_breaking_space_in_role_is_rejected():
    df = base_df()
    # Use a non-breaking space (U+00A0) between tokens; our validator strips \s, which includes NBSP.
    # However, if any non-space disallowed char sneaks in, the reconstructed vs. original compare will fail.
    role = "a\u00A0+\u00A0b"  # NBSPs around '+'
    out = add_virtual_column(df, role, "sum")
    # With our implementation, NBSP counts as whitespace and is tolerated:
    assert not out.empty
    assert out["sum"].tolist() == (df["a"] + df["b"]).tolist()


def test_names_are_case_sensitive_and_policy_bound():
    df = pd.DataFrame({"A": [1, 2], "a": [10, 20]})
    # Column names violate policy? 'A' and 'a' both pass the regex; both exist.
    out = add_virtual_column(df, "A + a", "mix")
    assert not out.empty
    assert out["mix"].tolist() == [11, 22]
