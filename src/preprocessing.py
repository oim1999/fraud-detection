"""
src/preprocessing.py
──────────────────────────────────────────────────────────────────────────────
Reusable helper functions shared across Task 1 notebooks.

Why a separate src/ file?
  Instead of copying the same code into every notebook, we write it once here
  and import it. This makes the code DRY (Don't Repeat Yourself) and easier
  to maintain — fix a bug once, it's fixed everywhere.
──────────────────────────────────────────────────────────────────────────────
"""

import numpy as np
import pandas as pd


# ─────────────────────────────────────────────────────────────────────────────
# 1. IP ADDRESS UTILITIES
# ─────────────────────────────────────────────────────────────────────────────

def ip_float_to_int(ip_series: pd.Series) -> pd.Series:
    """
    Convert an IP address stored as a float (scientific notation) to int64.

    Why is IP stored as float?
        When pandas reads a CSV, large integers like 3840542123 sometimes get
        loaded as float64 to avoid overflow. We need int64 for the range lookup.

    Example:
        3.840542e+09  →  3840542000  (as int64)
    """
    return ip_series.fillna(0).astype("int64")


def merge_ip_country(fraud_df: pd.DataFrame, ip_df: pd.DataFrame) -> pd.DataFrame:
    """
    Add a 'country' column to fraud_df by looking up each IP address
    in the ip_df range table.

    How it works (range-based join):
        The IP lookup table has columns:
            lower_bound_ip_address | upper_bound_ip_address | country
        For each IP in fraud_df we need to find the row where:
            lower_bound <= ip_int <= upper_bound

        We use pd.merge_asof (a sorted merge) which efficiently finds:
            "the largest lower_bound that is still <= ip_int"
        Then we validate that ip_int is also <= upper_bound.
        IPs that don't match any range are labelled 'Unknown'.

    Parameters:
        fraud_df : DataFrame that contains an 'ip_int' column (int64)
        ip_df    : IpAddress_to_Country DataFrame

    Returns:
        fraud_df with a new 'country' column added.
    """
    # Step 1: Cast IP bounds to int64 so they match the dtype of ip_int.
    # IpAddress_to_Country.csv loads its numeric columns as float64 by default.
    # merge_asof requires BOTH join keys to have the exact same dtype — if they
    # differ (int64 vs float64) pandas raises a MergeError.
    ip_sorted = ip_df.copy()
    ip_sorted["lower_bound_ip_address"] = ip_sorted["lower_bound_ip_address"].astype("int64")
    ip_sorted["upper_bound_ip_address"] = ip_sorted["upper_bound_ip_address"].astype("int64")
    ip_sorted    = ip_sorted.sort_values("lower_bound_ip_address").reset_index(drop=True)
    fraud_sorted = fraud_df.sort_values("ip_int").reset_index(drop=True)
    # Step 2: merge_asof finds, for each ip_int, the largest lower_bound <= ip_int
    # direction='backward' means: look backwards (i.e., find the nearest lower key)
    merged = pd.merge_asof(
        fraud_sorted,
        ip_sorted[["lower_bound_ip_address", "upper_bound_ip_address", "country"]],
        left_on="ip_int",
        right_on="lower_bound_ip_address",
        direction="backward",
    )

    # Step 3: Validate — ip_int must also be <= upper_bound to be a real match
    # If it's outside the range, label as 'Unknown'
    merged["country"] = np.where(
        merged["ip_int"] <= merged["upper_bound_ip_address"],
        merged["country"],
        "Unknown",
    )

    # Step 4: Drop helper columns we no longer need
    merged = merged.drop(
        columns=["lower_bound_ip_address", "upper_bound_ip_address"], errors="ignore"
    )

    return merged


# ─────────────────────────────────────────────────────────────────────────────
# 2. FEATURE ENGINEERING HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add hour-of-day and day-of-week features derived from purchase_time.

    Why these features matter for fraud:
        - Fraudsters often operate at unusual hours (late night / early morning).
        - Certain days of the week may have higher fraud rates (weekends, etc.).

    Parameters:
        df : DataFrame with a 'purchase_time' datetime column.

    Returns:
        df with two new columns:
            hour_of_day  : int 0–23
            day_of_week  : int 0 (Monday) – 6 (Sunday)
    """
    df = df.copy()
    df["hour_of_day"]  = df["purchase_time"].dt.hour
    df["day_of_week"]  = df["purchase_time"].dt.dayofweek  # 0 = Monday
    return df


def add_time_since_signup(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add time_since_signup_hours: how many hours passed between signup and purchase.

    Why this matters for fraud:
        Fraudsters often create accounts and immediately make purchases.
        A very small time_since_signup (e.g., < 1 hour) is a red flag.

    Parameters:
        df : DataFrame with 'signup_time' and 'purchase_time' datetime columns.

    Returns:
        df with a new 'time_since_signup_hours' float column.
    """
    df = df.copy()
    delta = df["purchase_time"] - df["signup_time"]
    # .total_seconds() / 3600 converts timedelta to decimal hours
    df["time_since_signup_hours"] = delta.dt.total_seconds() / 3600
    return df


def add_transaction_velocity(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add transaction count features per user and per device.

    Why velocity matters for fraud:
        - A user placing many orders quickly is suspicious (card testing).
        - The same device being used for many accounts is a strong fraud signal.

    user_transaction_count  : total purchases linked to this user_id
    device_transaction_count: total purchases linked to this device_id

    Parameters:
        df : DataFrame with 'user_id' and 'device_id' columns.

    Returns:
        df with two new integer columns.
    """
    df = df.copy()

    # groupby().transform('count') maps the group size back to each row
    # so every row keeps its original index
    df["user_transaction_count"]   = df.groupby("user_id")["user_id"].transform("count")
    df["device_transaction_count"] = df.groupby("device_id")["device_id"].transform("count")

    return df


# ─────────────────────────────────────────────────────────────────────────────
# 3. SUMMARY UTILITY
# ─────────────────────────────────────────────────────────────────────────────

def class_distribution(df: pd.DataFrame, target_col: str) -> pd.DataFrame:
    """
    Print and return a summary table of class distribution.

    Parameters:
        df         : DataFrame
        target_col : name of the binary target column (0/1)

    Returns:
        DataFrame with columns: count, percentage
    """
    counts = df[target_col].value_counts().rename_axis("class").reset_index(name="count")
    counts["percentage"] = (counts["count"] / len(df) * 100).round(2)
    counts["label"] = counts["class"].map({0: "Legitimate", 1: "Fraud"})
    print(counts.to_string(index=False))
    return counts
