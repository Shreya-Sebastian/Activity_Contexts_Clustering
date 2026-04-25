"""
Roll up the 100ms spatial-feature checkpoint into per-child epoch summaries.

Why floor + groupby instead of resample?
-----------------------------------------
The source recordings are sporadic (e.g. 13 days scattered across 2 years).
`DataFrame.resample('1Min')` builds a *continuous* time index from the first
to the last timestamp, which would emit roughly a million empty rows
spanning the gaps between recording days, only to be dropped afterward.

Flooring each timestamp to the epoch boundary and `groupby`-ing on it only
materialises epochs that actually contain measurements. Faster, leaner, and
the output table is exactly the rows you want.
"""

import pandas as pd

# ==========================================
# CONFIGURATION
# ==========================================
MACRO_EPOCH = '1Min'

CHECKPOINT_RAW = 'raw_spatial_100ms_checkpoint.csv'
SPATIAL_OUTPUT = 'spatial_features_1min.csv'

AGG_DICT = {
    'velocity': 'sum',
    'teacher_dist': 'mean',
    'is_in_group': 'mean',
    'is_hl_only_group': 'mean',
    'is_th_only_group': 'mean',
    'is_mixed_group': 'mean',
}

# Columns whose group-mean we want to express as a percentage of time
PCT_COLS = ['is_in_group', 'is_hl_only_group', 'is_th_only_group', 'is_mixed_group']

RENAME_MAP = {
    'velocity': 'Velocity_1min_TotalDist',
    'teacher_dist': 'Teacher_Dist_1min_Avg',
    'is_in_group': 'Time_In_Group_1min_Pct',
    'is_hl_only_group': 'Time_In_HL_Group_1min_Pct',
    'is_th_only_group': 'Time_In_TH_Group_1min_Pct',
    'is_mixed_group': 'Time_In_Mixed_Group_1min_Pct',
}


# ==========================================
# MAIN
# ==========================================
if __name__ == "__main__":
    print(f"\n{'=' * 50}\n AGGREGATION TO {MACRO_EPOCH} EPOCHS\n{'=' * 50}")
    print(f"Reading {CHECKPOINT_RAW}...")
    raw = pd.read_csv(CHECKPOINT_RAW)

    # ISO8601 + utc handles timestamps with/without fractional seconds and DST shifts.
    raw['TIME_LOCAL'] = pd.to_datetime(
        raw['TIME_LOCAL'], format='ISO8601', utc=True
    ).dt.tz_convert('America/New_York')
    raw = raw.dropna(subset=['TIME_LOCAL'])

    print(f"Flooring to {MACRO_EPOCH} and aggregating "
          f"({len(raw):,} input rows, {raw['SUBJECTID'].nunique()} subjects)...")

    # The whole rollup: floor timestamp -> groupby (subject, epoch) -> agg.
    # Only epochs that actually contain data appear in the output.
    raw['epoch'] = raw['TIME_LOCAL'].dt.floor(MACRO_EPOCH)
    out = (
        raw
        .groupby(['SUBJECTID', 'epoch'], sort=True)
        .agg(AGG_DICT)
        .reset_index()
    )

    out[PCT_COLS] *= 100
    out = out.rename(columns={'epoch': 'TIME_LOCAL', **RENAME_MAP})
    out = out[['TIME_LOCAL', 'SUBJECTID'] + list(RENAME_MAP.values())]

    out.to_csv(SPATIAL_OUTPUT, index=False)
    print(f"\nDone. Saved {len(out):,} rows to: {SPATIAL_OUTPUT}")