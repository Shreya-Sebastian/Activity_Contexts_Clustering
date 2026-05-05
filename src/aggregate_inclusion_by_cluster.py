"""Aggregate per-(child, cluster) inclusion metrics with attendance filter and diagnosis."""

import pandas as pd

CLUSTERED = 'clustered_epochs_7.csv'
BASE      = 'data/mapping/MAPPING_StarFish_2223_BASE_NONAMES.csv'
CONSOL    = 'data/mapping/MAPPING_CONSOLIDATED.csv'
OUTPUT    = 'child_inclusion_by_cluster_7.csv'


if __name__ == "__main__":
    df = pd.read_csv(CLUSTERED)
    df['Date'] = (pd.to_datetime(df['TIME_LOCAL'], utc=True)
                    .dt.tz_convert('America/New_York').dt.date.astype(str))

    # Attendance filter (PRESENT only) -- aligns row count with statistical_analysis.py
    consol = pd.read_csv(CONSOL)
    consol['Date'] = pd.to_datetime(consol['Date']).dt.date.astype(str)
    att = (consol.dropna(subset=['Subject_ID'])[['Subject_ID', 'Date', 'STATUS']]
                  .rename(columns={'Subject_ID': 'SUBJECTID'}))
    n_before = len(df)
    df = df.merge(att, on=['SUBJECTID', 'Date'], how='inner')
    df = df[df['STATUS'] == 'PRESENT'].drop(columns=['Date', 'STATUS'])
    print(f"Attendance filter: {n_before} -> {len(df)} minutes")

    # Diagnosis (HL/TH only)
    diag = (pd.read_csv(BASE).dropna(subset=['Subject_ID'])[['Subject_ID', 'Diagnosis']]
              .rename(columns={'Subject_ID': 'SUBJECTID'}))
    diag = diag[diag['Diagnosis'].isin(['HL', 'TH'])]

    summary = df.groupby(['SUBJECTID', 'Cluster_ID']).agg(
        Avg_Utt_Count=('Child_Utt_Count_1min', 'mean'),
        Avg_Time_In_Group_Pct=('Time_In_Group_1min_Pct', 'mean'),
        Avg_Time_HL_Group_Pct=('Time_In_HL_Group_1min_Pct', 'mean'),
        Avg_Time_TH_Group_Pct=('Time_In_TH_Group_1min_Pct', 'mean'),
        Avg_Time_Mixed_Group_Pct=('Time_In_Mixed_Group_1min_Pct', 'mean'),
    ).round(2).reset_index()

    final = summary.merge(diag, on='SUBJECTID', how='inner')
    final.to_csv(OUTPUT, index=False)
    print(f"Saved {len(final)} rows to {OUTPUT}")