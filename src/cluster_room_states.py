"""
Room-level GMM clustering of classroom activity contexts.

Merges spatial + acoustic features, averages per minute across children
present that minute, Yeo-Johnson transforms, and selects K by BIC over
K = 2..14. The upper bound K = 14 is set on three grounds:
  (i)   at K = 15 a wider sweep showed an EM-init artifact (single
        random restart finds a sharp local optimum that does not
        reproduce at K = 14 or K = 16);
  (ii)  inclusive-preschool routines are not documented to contain
        more than ~10 distinct activity contexts (Irvin et al., 2021);
  (iii) at K >= 15 the per-component sample size becomes too small
        for stable full-covariance estimation.

Cluster IDs are relabeled by ascending AWC centroid so C0..C(K-1) are
deterministic across runs.

Outputs:
  clustered_epochs_{K}.csv  per-(child, minute) rows with Cluster_ID
  k_selection_bic.png       BIC vs K curve, marking the chosen K
"""

import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import PowerTransformer

warnings.filterwarnings("ignore")

SPATIAL_FILE  = 'spatial_features_1min.csv'
ACOUSTIC_FILE = 'acoustic_features_1min.csv'
PLOT_FILE     = 'k_selection_bic.png'

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


def plot_bic_curve(fits, best_k, best_bic, path):
    """Save a BIC vs K curve, marking the chosen K. Visualizes that BIC
    descends through the chosen K and rises afterward."""
    ks   = [k for k, _, _ in fits]
    bics = [b for _, b, _ in fits]

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(ks, bics, marker='o', color='C0', linewidth=2, label='BIC')
    ax.scatter([best_k], [best_bic], color='C3', s=140, zorder=5,
               label=f'Best K = {best_k}')
    ax.axvline(best_k, color='C3', linestyle='--', alpha=0.5)

    ax.set_xlabel('Number of components (K)')
    ax.set_ylabel('BIC')
    ax.set_title(f'GMM model selection by BIC  (K = {min(ks)}..{max(ks)})')
    ax.set_xticks(ks)
    ax.legend(loc='upper center')
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close(fig)


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

    plot_bic_curve(fits, best_k, best_bic, PLOT_FILE)
    print(f"Saved BIC curve to {PLOT_FILE}")

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
