"""Optional utility: roll up 1-minute feature CSVs to a coarser epoch (5Min, 1H...).
Outputs use the epoch suffix; the baseline 1-min files are preserved."""

import warnings
import pandas as pd

warnings.filterwarnings("ignore")

TARGET_EPOCH = '5Min'
SPATIAL_IN   = 'spatial_features_1min.csv'
ACOUSTIC_IN  = 'acoustic_features_1min.csv'
SPATIAL_OUT  = f"spatial_features_{TARGET_EPOCH}.csv"
ACOUSTIC_OUT = f"acoustic_features_{TARGET_EPOCH}.csv"


def aggregate(in_path, out_path, epoch):
    try:
        df = pd.read_csv(in_path)
    except FileNotFoundError:
        print(f"Missing {in_path}"); return

    tcol = 'TIME_LOCAL' if 'TIME_LOCAL' in df.columns else 'TIME_UTC'
    if tcol not in df.columns:
        print(f"No time column in {in_path}"); return
    df[tcol] = pd.to_datetime(df[tcol], utc=True).dt.tz_convert('America/New_York')

    # Sums for cumulative metrics, means for rates/distances/percentages
    agg = {c: ('sum' if ('Sum' in c or 'TotalDist' in c) else 'mean')
           for c in df.columns if c not in ('SUBJECTID', tcol)
              and (any(t in c for t in ('Sum', 'TotalDist', 'Avg', 'Pct')))}

    print(f"Aggregating {in_path} to {epoch} windows...")
    out = pd.concat([
        g.set_index(tcol).resample(epoch).agg(agg).dropna(how='all').assign(SUBJECTID=s).reset_index()
        for s, g in df.groupby('SUBJECTID')
    ], ignore_index=True)
    out = out.rename(columns={c: c.replace('1min', epoch.lower()) for c in out.columns})
    out = out[[tcol, 'SUBJECTID'] + [c for c in out.columns if c not in (tcol, 'SUBJECTID')]]
    out.to_csv(out_path, index=False)
    print(f"  Saved {len(out)} rows to {out_path}")


if __name__ == "__main__":
    print(f"SCALING TO {TARGET_EPOCH}")
    aggregate(SPATIAL_IN,  SPATIAL_OUT,  TARGET_EPOCH)
    aggregate(ACOUSTIC_IN, ACOUSTIC_OUT, TARGET_EPOCH)
    print(f"Done. Update cluster_room_states.py to point at the {TARGET_EPOCH} files if clustering at that scale.")
