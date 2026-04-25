"""
Aggregate per-child, per-cluster inclusion metrics and merge with diagnosis.
Output feeds build_dashboard.py.
"""

import pandas as pd

# ==========================================
# 1. CONFIGURATION
# ==========================================
CLUSTERED_DATA_FILE = 'clustered_epochs_7.csv'
BASE_MAPPING         = 'data/mapping/MAPPING_StarFish_2223_BASE_NONAMES.csv'
CONSOLIDATED_MAPPING = 'data/mapping/MAPPING_CONSOLIDATED.csv'
OUTPUT_FILE          = 'child_inclusion_by_cluster_7.csv'

# ==========================================
# 2. MAIN
# ==========================================
if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("      SOCIAL INCLUSION ANALYSIS (F-FORMATIONS & DEMOGRAPHICS)")
    print("=" * 70)

    print("1. Loading clustered data and demographics...")
    try:
        df = pd.read_csv(CLUSTERED_DATA_FILE)
    except FileNotFoundError:
        print(f"CRITICAL ERROR: Could not find {CLUSTERED_DATA_FILE}. Run cluster_room_states.py first.")
        exit()

    # ---- Attendance filter ----------------------------------------
    # Restrict to child-days marked PRESENT in the per-day attendance
    # log. This keeps the dashboard's input set aligned with the LMM's
    # input set in statistical_analysis.py; without this filter the
    # dashboard averages over 22,405 minutes but the LMM is fit on
    # 21,650, and the two sets of numbers can disagree.
    try:
        consol = pd.read_csv(CONSOLIDATED_MAPPING)
        consol['Date'] = pd.to_datetime(consol['Date']).dt.date.astype(str)
        attendance = (consol.dropna(subset=['Subject_ID'])
                      [['Subject_ID', 'Date', 'STATUS']]
                      .rename(columns={'Subject_ID': 'SUBJECTID'}))
        df['Date'] = (pd.to_datetime(df['TIME_LOCAL'], utc=True)
                      .dt.tz_convert('America/New_York')
                      .dt.date.astype(str))
        n_before = len(df)
        df = df.merge(attendance, on=['SUBJECTID', 'Date'], how='inner')
        df = df[df['STATUS'] == 'PRESENT']
        df = df.drop(columns=['Date', 'STATUS'])
        print(f"   -> Attendance filter: {n_before} -> {len(df)} minutes "
              f"({n_before - len(df)} dropped).")
    except FileNotFoundError:
        print(f"Warning: {CONSOLIDATED_MAPPING} not found. "
              "Skipping attendance filter (dashboard will NOT match the LMM row count).")

    # ---- Diagnosis ------------------------------------------------
    try:
        mapping_df = pd.read_csv(BASE_MAPPING)
        demographics = (mapping_df.dropna(subset=['Subject_ID'])
                        [['Subject_ID', 'Diagnosis']]
                        .rename(columns={'Subject_ID': 'SUBJECTID'}))
        demographics = demographics[demographics['Diagnosis'].isin(['HL', 'TH'])]
    except FileNotFoundError:
        print(f"Warning: Mapping file {BASE_MAPPING} not found. Proceeding without demographics.")
        demographics = pd.DataFrame()

    print("2. Aggregating inclusion metrics per child, per cluster...")
    inclusion_summary = df.groupby(['SUBJECTID', 'Cluster_ID']).agg(
        Avg_Utt_Count=('Child_Utt_Count_1min', 'mean'),
        Avg_Time_In_Group_Pct=('Time_In_Group_1min_Pct', 'mean'),
        Avg_Time_HL_Group_Pct=('Time_In_HL_Group_1min_Pct', 'mean'),
        Avg_Time_TH_Group_Pct=('Time_In_TH_Group_1min_Pct', 'mean'),
        Avg_Time_Mixed_Group_Pct=('Time_In_Mixed_Group_1min_Pct', 'mean'),
    ).reset_index()

    print("3. Merging diagnosis...")
    if not demographics.empty:
        final_df = pd.merge(inclusion_summary, demographics, on='SUBJECTID', how='inner')
    else:
        final_df = inclusion_summary

    cols_to_round = ['Avg_Utt_Count', 'Avg_Time_In_Group_Pct', 'Avg_Time_HL_Group_Pct',
                     'Avg_Time_TH_Group_Pct', 'Avg_Time_Mixed_Group_Pct']
    final_df[cols_to_round] = final_df[cols_to_round].round(2)

    print(f"4. Saving to {OUTPUT_FILE}...")
    final_df.to_csv(OUTPUT_FILE, index=False)

    print("\nDone. Run build_dashboard.py next.")