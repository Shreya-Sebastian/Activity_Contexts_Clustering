"""
Optional utility: roll up 1-minute feature CSVs to a coarser epoch
(e.g. 2Min, 5Min, 1H) for exploring different time resolutions.

Does not feed the main pipeline — outputs are named with the epoch
suffix so the baseline 1-minute files are preserved.
"""

import pandas as pd
import warnings

warnings.filterwarnings("ignore")

# ==========================================
# 1. CONFIGURATION
# ==========================================
TARGET_EPOCH = '5Min'

SPATIAL_IN = 'spatial_features_1min.csv'
ACOUSTIC_IN = 'acoustic_features_1min.csv'

SPATIAL_OUT = f"spatial_features_{TARGET_EPOCH}.csv"
ACOUSTIC_OUT = f"acoustic_features_{TARGET_EPOCH}.csv"

# ==========================================
# 2. AGGREGATION
# ==========================================
def aggregate_features(file_path, output_path, epoch_str):
    try:
        df = pd.read_csv(file_path)
    except FileNotFoundError:
        print(f" ERROR: Could not find {file_path}")
        return

    time_col = 'TIME_LOCAL' if 'TIME_LOCAL' in df.columns else 'TIME_UTC'
    if time_col not in df.columns:
        print(f" ERROR: No time column found in {file_path}")
        return

    # Parse as UTC first (handles mixed EST/EDT offsets in tz-aware strings
    # as well as naive UTC columns) then convert to America/New_York.
    df[time_col] = (pd.to_datetime(df[time_col], utc=True)
                    .dt.tz_convert('America/New_York'))

    # Sums for cumulative metrics, means for rates/distances/percentages
    agg_dict = {}
    for col in df.columns:
        if col in ['SUBJECTID', time_col]:
            continue
        elif 'Sum' in col or 'TotalDist' in col:
            agg_dict[col] = 'sum'
        elif 'Avg' in col or 'Pct' in col:
            agg_dict[col] = 'mean'

    print(f"Aggregating {file_path} to {epoch_str} windows...")
    macro_frames = []

    for subj, group in df.groupby('SUBJECTID'):
        g_resampled = group.set_index(time_col).resample(epoch_str).agg(agg_dict)
        g_resampled = g_resampled.dropna(how='all')
        g_resampled['SUBJECTID'] = subj
        macro_frames.append(g_resampled.reset_index())

    df_macro = pd.concat(macro_frames, ignore_index=True)

    # Rename '1min' -> new epoch label in column names
    rename_dict = {col: col.replace('1min', epoch_str.lower()) for col in df_macro.columns}
    df_macro = df_macro.rename(columns=rename_dict)

    cols = [time_col, 'SUBJECTID'] + [c for c in df_macro.columns if c not in [time_col, 'SUBJECTID']]
    df_macro = df_macro[cols]

    df_macro.to_csv(output_path, index=False)
    print(f" Saved {len(df_macro)} rows to {output_path}")

# ==========================================
# 3. MAIN
# ==========================================
if __name__ == "__main__":
    print("\n" + "=" * 50)
    print(f"   SCALING EPOCHS TO {TARGET_EPOCH}")
    print("=" * 50)

    aggregate_features(SPATIAL_IN, SPATIAL_OUT, TARGET_EPOCH)
    aggregate_features(ACOUSTIC_IN, ACOUSTIC_OUT, TARGET_EPOCH)

    print("=" * 50 + "\n")
    print(f"Done. Update cluster_room_states.py to point at the '{TARGET_EPOCH}' files if clustering at that scale.")