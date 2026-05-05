import warnings

import numpy as np
import pandas as pd
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import PowerTransformer

warnings.filterwarnings("ignore")

SPATIAL_FILE  = 'spatial_features_1min.csv'
ACOUSTIC_FILE = 'acoustic_features_1min.csv'

K_RANGE      = range(2, 15)
N_INIT       = 10
RANDOM_STATE = 42

CLUSTER_FEATURES = [
    'AWC_1min_Sum',
    'Overlap_1min_Sum',
    'Velocity_1min_TotalDist',
    'Teacher_Dist_1min_Avg',
]


def to_eastern(df):
    if 'TIME_UTC' in df.columns and 'TIME_LOCAL' not in df.columns:
        df = df.rename(columns={'TIME_UTC': 'TIME_LOCAL'})
    df['TIME_LOCAL'] = pd.to_datetime(df['TIME_LOCAL'], utc=True).dt.tz_convert('America/New_York')
    return df


if __name__ == "__main__":
    print(f"\n{'=' * 60}\n   ROOM-CENTRIC GMM CLUSTERING\n{'=' * 60}")

    # Load + merge on (subject, minute)
    df = pd.merge(
        to_eastern(pd.read_csv(SPATIAL_FILE)),
        to_eastern(pd.read_csv(ACOUSTIC_FILE)),
        on=['SUBJECTID', 'TIME_LOCAL'], how='inner',
    )

    # >= 3 children per minute, then room-level mean
    counts = df.groupby('TIME_LOCAL').size()
    df = df[df['TIME_LOCAL'].isin(counts[counts >= 3].index)]
    room = (df.groupby('TIME_LOCAL')[CLUSTER_FEATURES].mean()
              .reset_index().dropna(subset=CLUSTER_FEATURES))

    # Yeo-Johnson + BIC sweep
    X = PowerTransformer(method='yeo-johnson').fit_transform(room[CLUSTER_FEATURES])
    print(f"Room-level minutes: {len(X)}\n\n{'K':>3}  {'BIC':>11}  conv")
    fits = []
    for k in K_RANGE:
        gmm = GaussianMixture(n_components=k, covariance_type='full',
                              random_state=RANDOM_STATE, n_init=N_INIT).fit(X)
        bic = gmm.bic(X)
        fits.append((k, bic, gmm))
        print(f"{k:>3}  {bic:>11.1f}  {gmm.converged_}")
    best_k, best_bic, gmm = min(fits, key=lambda r: r[1])
    print(f"\nBest K by BIC: {best_k}  (BIC = {best_bic:.1f})")

    # Relabel by ascending AWC centroid -> deterministic C0..C(K-1)
    awc_idx = CLUSTER_FEATURES.index('AWC_1min_Sum')
    relabel = {int(old): new for new, old in enumerate(np.argsort(gmm.means_[:, awc_idx]))}
    room['Cluster_ID'] = pd.Series(gmm.predict(X)).map(relabel).values

    # Profiles
    profiles = room.groupby('Cluster_ID')[CLUSTER_FEATURES].mean().round(2)
    profiles['% of Day'] = (room['Cluster_ID'].value_counts(normalize=True)
                                              .sort_index() * 100).round(1)
    print(f"\nLATENT CLASSROOM PROFILES (sorted by AWC ascending):\n{profiles.to_string()}")

    # Propagate to per-child rows and save
    out = f'clustered_epochs_{best_k}.csv'
    (pd.merge(df, room[['TIME_LOCAL', 'Cluster_ID']], on='TIME_LOCAL', how='inner')
       .to_csv(out, index=False))
    print(f"\nSaved to {out}")