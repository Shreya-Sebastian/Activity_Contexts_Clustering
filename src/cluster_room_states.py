"""
Room-centric GMM clustering of classroom latent states.

Merges spatial + acoustic features, averages per minute across children
to capture whole-room context, then fits a Gaussian Mixture Model with
a fixed number of components (K) chosen by manual coding of the
recordings. See the thesis methodology chapter for the justification.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import PowerTransformer
from sklearn.mixture import GaussianMixture
import warnings

warnings.filterwarnings("ignore")

# ==========================================
# 1. CONFIGURATION
# ==========================================
SPATIAL_FILE = 'spatial_features_1min.csv'
ACOUSTIC_FILE = 'acoustic_features_1min.csv'
OUTPUT_FILE = 'clustered_epochs_7.csv'

# Number of latent classroom states, fixed by manual coding.
# See thesis methodology chapter for rationale.
N_COMPONENTS = 7

CLUSTER_FEATURES = [
    'AWC_1min_Sum',
    'Overlap_1min_Sum',
    'Velocity_1min_TotalDist',
    'Teacher_Dist_1min_Avg',
]

# ==========================================
# 2. MAIN
# ==========================================
if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("   ROOM-CENTRIC LATENT STATE CLUSTERING (GMM)")
    print("=" * 60)

    print("1. Loading and merging feature data...")
    try:
        df_spatial = pd.read_csv(SPATIAL_FILE)
        df_acoustic = pd.read_csv(ACOUSTIC_FILE)

        # Normalize spatial timestamps to Eastern (legacy files may use TIME_UTC)
        if 'TIME_UTC' in df_spatial.columns and 'TIME_LOCAL' not in df_spatial.columns:
            df_spatial = df_spatial.rename(columns={'TIME_UTC': 'TIME_LOCAL'})
        df_spatial['TIME_LOCAL'] = pd.to_datetime(
            df_spatial['TIME_LOCAL'], utc=True
        ).dt.tz_convert('America/New_York')

        df_acoustic['TIME_LOCAL'] = pd.to_datetime(
            df_acoustic['TIME_LOCAL'], utc=True
        ).dt.tz_convert('America/New_York')

    except Exception as e:
        print(f"\nCRITICAL ERROR: {e}")
        exit()

    df_clean = pd.merge(df_spatial, df_acoustic, on=['SUBJECTID', 'TIME_LOCAL'], how='inner')

    if df_clean.empty:
        print("ERROR: No overlapping data between spatial and acoustic files.")
        exit()

    print(f"   -> Merged {len(df_clean)} rows across {df_clean['SUBJECTID'].nunique()} subjects.")

    # ==========================================
    # 3. ROOM-CENTRIC AGGREGATION
    # ==========================================
    print("2. Averaging features per minute across the room...")

    # Require at least 3 children active in the minute
    children_per_minute = df_clean.groupby('TIME_LOCAL').size()
    valid_minutes = children_per_minute[children_per_minute >= 3].index
    df_clean = df_clean[df_clean['TIME_LOCAL'].isin(valid_minutes)]

    room_df = df_clean.groupby('TIME_LOCAL')[CLUSTER_FEATURES].mean().reset_index()
    room_df = room_df.dropna(subset=CLUSTER_FEATURES).copy()

    print("3. Power-transforming skewed features...")
    scaler = PowerTransformer(method='yeo-johnson')
    scaled_room_features = scaler.fit_transform(room_df[CLUSTER_FEATURES])

    # ==========================================
    # 4. FIT GMM WITH FIXED K
    # ==========================================
    print(f"4. Fitting GMM with K = {N_COMPONENTS} latent states "
          f"(K fixed by manual coding)...")
    gmm = GaussianMixture(n_components=N_COMPONENTS, covariance_type="full",
                          random_state=42, n_init=10)
    room_df['Cluster_ID'] = gmm.fit_predict(scaled_room_features)

    # Propagate cluster labels back to per-child rows
    df_final = pd.merge(df_clean, room_df[['TIME_LOCAL', 'Cluster_ID']],
                        on='TIME_LOCAL', how='inner')

    print("\n" + "=" * 60)
    print("        LATENT CLASSROOM PROFILES")
    print("=" * 60)
    profiles = room_df.groupby('Cluster_ID')[CLUSTER_FEATURES].mean().round(2)
    profiles['% of Day'] = (room_df['Cluster_ID'].value_counts(normalize=True) * 100).round(1)
    print(profiles.sort_values(by='% of Day', ascending=False).to_string())
    print("=" * 60 + "\n")

    df_final.to_csv(OUTPUT_FILE, index=False)
    print(f"Saved results to {OUTPUT_FILE}")