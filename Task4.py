"""
MSE-based quantization (optimal 1D k-means-like DP) for FICO -> rating mapping.

Outputs:
 - fico_rating_map_mse.csv  (rating map)
 - prints bins, counts, defaults, PD per rating
"""
import math
import numpy as np
import pandas as pd
from pandas.api.types import is_numeric_dtype
import sys
import os

# ---------- User-configurable ----------
CSV_PATH = '/Users/arinayare/QR JPMC /Loan Example(task 3&4).csv'
NUM_BUCKETS = 5  # number of categories required by Charlie's model
FORCE_FICO_COL = None   # e.g. "fico_score"  (set to None to auto-detect)
FORCE_DEFAULT_COL = None  # e.g. "default" (set to None to auto-detect)
OUTPUT_CSV = "/Users/arinayare/Quantitative Research JP Morgan & Chase/fico_rating_map_mse.csv"
EPS = 1e-12
# ---------------------------------------

def detect_columns(df):
    # try common names first
    lower_cols = {c.lower(): c for c in df.columns}
    fico_candidates = ["fico", "fico_score", "fico score", "score", "fico_score.1", "fico1"]
    default_candidates = ["default", "is_default", "defaulted", "loan_default", "y", "target", "delinquent"]
    fico_col = None
    default_col = None

    if FORCE_FICO_COL and FORCE_FICO_COL in df.columns:
        fico_col = FORCE_FICO_COL
    else:
        for s in fico_candidates:
            if s in lower_cols:
                fico_col = lower_cols[s]
                break

    if FORCE_DEFAULT_COL and FORCE_DEFAULT_COL in df.columns:
        default_col = FORCE_DEFAULT_COL
    else:
        for s in default_candidates:
            if s in lower_cols:
                default_col = lower_cols[s]
                break

    # fallback heuristics:
    if fico_col is None:
        # choose numeric column with majority values in 300-850
        numeric_cols = [c for c in df.columns if is_numeric_dtype(df[c])]
        for c in numeric_cols:
            vals = df[c].dropna()
            if len(vals)==0:
                continue
            frac_in_range = vals.between(300, 850).mean()
            if frac_in_range > 0.6:
                fico_col = c
                break

    if default_col is None:
        # numeric 0/1 column or boolean-like
        for c in df.columns:
            vals = df[c].dropna()
            if len(vals)==0: 
                continue
            unique = set(vals.unique())
            # allow {0,1} and {"0","1","yes","no","true","false"}
            if unique <= {0,1}:
                default_col = c
                break
            if unique <= {"0","1","yes","no","y","n","true","false","t","f"}:
                default_col = c
                break

    return fico_col, default_col

def load_and_prepare(csv_path):
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV not found at {csv_path}")
    df = pd.read_csv(csv_path)
    fico_col, default_col = detect_columns(df)
    if fico_col is None or default_col is None:
        raise ValueError(f"Could not auto-detect columns. Detected fico_col={fico_col}, default_col={default_col}. CSV columns: {list(df.columns)}")

    df = df[[fico_col, default_col]].dropna(subset=[fico_col])
    df.columns = ["FICO", "Default"]
    # ensure numeric types
    df["FICO"] = pd.to_numeric(df["FICO"], errors="coerce")
    df = df.dropna(subset=["FICO"])
    # make Default numeric 0/1
    try:
        df["Default"] = pd.to_numeric(df["Default"], errors="coerce").fillna(0).astype(int)
    except Exception:
        # fallback mapping from common strings
        df["Default"] = df["Default"].astype(str).str.strip().str.lower().map(
            lambda x: 1 if x in ("1","yes","y","true","t") else 0
        ).astype(int)

    return df

def aggregate_unique(df):
    # aggregate by unique FICO to reduce DP size
    agg = df.groupby("FICO", as_index=False).agg(count=("Default","size"), defaults=("Default","sum"))
    agg = agg.sort_values("FICO").reset_index(drop=True)
    return agg

def prefix_sums(scores, counts, defaults):
    prefix_n = np.concatenate([[0], np.cumsum(counts)])
    prefix_k = np.concatenate([[0], np.cumsum(defaults)])
    prefix_x = np.concatenate([[0], np.cumsum(counts * scores)])
    prefix_x2 = np.concatenate([[0], np.cumsum(counts * (scores**2))])
    return prefix_n, prefix_k, prefix_x, prefix_x2

def interval_stats(prefix_n, prefix_k, prefix_x, prefix_x2, i, j):
    # inclusive i..j (0-based indices on unique scores)
    n = prefix_n[j+1] - prefix_n[i]
    k = prefix_k[j+1] - prefix_k[i]
    s = prefix_x[j+1] - prefix_x[i]
    s2 = prefix_x2[j+1] - prefix_x2[i]
    return int(n), int(k), float(s), float(s2)

def mse_cost_matrix(scores, prefix_n, prefix_k, prefix_x, prefix_x2):
    m = len(scores)
    cost = np.full((m, m), np.inf)
    for i in range(m):
        for j in range(i, m):
            n, k, s, s2 = interval_stats(prefix_n, prefix_k, prefix_x, prefix_x2, i, j)
            if n == 0:
                cost[i, j] = 0.0
            else:
                # sum of squared deviations = s2 - s^2/n
                cost[i, j] = s2 - (s*s)/n
    return cost

def dp_optimal_partition(cost_matrix, K):
    m = cost_matrix.shape[0]
    # dp[k][t] = min cost partitioning first t items (0..t-1) into k buckets
    dp = np.full((K+1, m+1), np.inf)
    split = np.full((K+1, m+1), -1, dtype=int)
    dp[0,0] = 0.0
    for k in range(1, K+1):
        for t in range(1, m+1):
            best = np.inf
            best_s = -1
            # try last bucket starting at s (0..t-1)
            for s_idx in range(0, t):
                c = cost_matrix[s_idx, t-1] + dp[k-1, s_idx]
                if c < best:
                    best = c
                    best_s = s_idx
            dp[k, t] = best
            split[k, t] = best_s

    # reconstruct intervals
    boundaries = []
    t = m
    for k in range(K, 0, -1):
        s = split[k, t]
        boundaries.append(s)
        t = s
    boundaries = boundaries[::-1]  # ascending start indices
    bins = []
    for idx in range(len(boundaries)):
        start_idx = boundaries[idx]
        end_idx = boundaries[idx+1]-1 if idx+1 < len(boundaries) else m-1
        bins.append((start_idx, end_idx))
    return bins, dp[K, m]

def bins_to_dataframe(bins, scores, prefix_n, prefix_k, prefix_x, prefix_x2):
    rows = []
    for b_idx, (i, j) in enumerate(bins):
        n,k,s,s2 = interval_stats(prefix_n, prefix_k, prefix_x, prefix_x2, i, j)
        lower = scores[i]
        upper = scores[j]
        pd_hat = (k / n) if n>0 else float("nan")
        rows.append({
            "bin_index": b_idx,
            "lower_score": lower,
            "upper_score": upper,
            "count": n,
            "defaults": k,
            "PD": pd_hat
        })
    return pd.DataFrame(rows)

def assign_ratings_from_bins(bins_df):
    # sort bins ascending by lower_score (lowest to highest)
    df = bins_df.sort_values("lower_score").reset_index(drop=True)
    B = len(df)
    # rating 1 = best credit (highest FICO)
    df["rating"] = df.index.map(lambda idx: B - idx)
    df["interval"] = df.apply(lambda r: f"[{int(r['lower_score'])}, {int(r['upper_score'])}]", axis=1)
    return df[["rating","interval","lower_score","upper_score","count","defaults","PD"]].sort_values("rating")

def build_mapper_from_bins(bins, scores):
    intervals = [(scores[i], scores[j]) for i,j in bins]
    def map_fico(x):
        for idx, (low, up) in enumerate(intervals):
            if x >= low and x <= up:
                rating = len(intervals) - idx
                return rating
        # outside range: nearest
        if x < intervals[0][0]:
            return len(intervals)
        else:
            return 1
    return map_fico

def main():
    print("Loading CSV:", CSV_PATH)
    df = load_and_prepare(CSV_PATH)
    print(f"Loaded {len(df)} rows. Detecting unique FICO values and aggregating...")
    agg = aggregate_unique(df)
    scores = agg["FICO"].values
    counts = agg["count"].values.astype(int)
    defaults = agg["defaults"].values.astype(int)
    m = len(scores)
    print(f"Unique FICO values used in DP: {m}")

    prefix_n, prefix_k, prefix_x, prefix_x2 = prefix_sums(scores, counts, defaults)

    print("Computing MSE cost matrix (O(m^2))...")
    cost_mat = mse_cost_matrix(scores, prefix_n, prefix_k, prefix_x, prefix_x2)

    print(f"Running DP to find optimal partition into K = {NUM_BUCKETS} buckets...")
    bins, total_cost = dp_optimal_partition(cost_mat, NUM_BUCKETS)
    print("DP complete. Constructing rating map...")

    bins_df = bins_to_dataframe(bins, scores, prefix_n, prefix_k, prefix_x, prefix_x2)
    rating_df = assign_ratings_from_bins(bins_df)

    # Save to CSV
    rating_df.to_csv(OUTPUT_CSV, index=False)
    print(f"Saved rating map to: {OUTPUT_CSV}\n")

    print("Rating map (rating 1 = best credit):")
    print(rating_df.to_string(index=False))

    # Build mapper function and show PD by rating on original rows
    mapper = build_mapper_from_bins(bins, scores)
    df["mse_rating"] = df["FICO"].apply(mapper)
    summary = df.groupby("mse_rating")["Default"].agg(count="count", defaults="sum", PD="mean").reset_index().sort_values("mse_rating")
    print("\nEmpirical PD by rating (using MSE mapper):")
    print(summary.to_string(index=False))

    # Example mapping function accessible for future use
    print("\nExample mappings:")
    for x in [300, 550, 620, 700, 780, 820]:
        print(f"FICO {x} -> rating {mapper(x)}")

if __name__ == "__main__":
    main()
