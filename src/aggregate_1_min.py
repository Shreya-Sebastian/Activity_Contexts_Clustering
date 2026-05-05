"""Roll up the 100ms spatial-feature checkpoint into 1-minute per-child summaries.

Floor + groupby (not resample) avoids materialising empty epochs across the
gaps between sporadic recording days.
"""

import pandas as pd

CHECKPOINT_RAW = 'raw_spatial_100ms.csv'
SPATIAL_OUTPUT = 'spatial_features_1min.csv'

AGG = {'velocity': 'sum', 'teacher_dist': 'mean',
       'is_in_group': 'mean', 'is_hl_only_group': 'mean',
       'is_th_only_group': 'mean', 'is_mixed_group': 'mean'}
PCT_COLS = ['is_in_group', 'is_hl_only_group', 'is_th_only_group', 'is_mixed_group']
RENAME = {
    'velocity': 'Velocity_1min_TotalDist',
    'teacher_dist': 'Teacher_Dist_1min_Avg',
    'is_in_group': 'Time_In_Group_1min_Pct',
    'is_hl_only_group': 'Time_In_HL_Group_1min_Pct',
    'is_th_only_group': 'Time_In_TH_Group_1min_Pct',
    'is_mixed_group': 'Time_In_Mixed_Group_1min_Pct',
}


if __name__ == "__main__":
    raw = pd.read_csv(CHECKPOINT_RAW)
    raw['TIME_LOCAL'] = (pd.to_datetime(raw['TIME_LOCAL'], format='ISO8601', utc=True)
                          .dt.tz_convert('America/New_York'))
    raw = raw.dropna(subset=['TIME_LOCAL'])
    print(f"Aggregating {len(raw):,} rows / {raw['SUBJECTID'].nunique()} subjects to 1-min epochs...")

    raw['epoch'] = raw['TIME_LOCAL'].dt.floor('1Min')
    out = raw.groupby(['SUBJECTID', 'epoch']).agg(AGG).reset_index()
    out[PCT_COLS] *= 100
    out = (out.rename(columns={'epoch': 'TIME_LOCAL', **RENAME})
              [['TIME_LOCAL', 'SUBJECTID', *RENAME.values()]])
    out.to_csv(SPATIAL_OUTPUT, index=False)
    print(f"Saved {len(out):,} rows to {SPATIAL_OUTPUT}")
