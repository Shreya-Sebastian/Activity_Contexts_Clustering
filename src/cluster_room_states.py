"""
Room-level GMM clustering of classroom activity contexts.

Merges spatial + acoustic features, averages per minute across children
present that minute, Yeo-Johnson transforms, and selects K over K = 2..20
by a stability-aware BIC rule (thesis Sec 5.5.4). From K = 14 onward the
across-seed BIC standard deviation grows sharply (over an order of
magnitude larger than for K <= 13), the empirical signature of fragile EM
local optima at high K: a single random restart may find a sharp local
optimum that another seed won't reproduce. Mean pairwise ARI falls below
the 0.80 admissibility threshold at every K >= 7, so this instability
region never competes with the selected K regardless of where the sweep's
upper bound is set; K = 2..20 is swept (rather than stopping earlier) so
that is an observed result, not an assumption.

K selection: for each K, 20 GMM fits at 20 different seeds (full
covariance, n_init=10) give a mean BIC and a mean pairwise Adjusted
Rand Index across the 20 label sets. K is "admissible" if that mean ARI
is >= ARI_THRESHOLD (i.e. restarts agree on the partition, not just on
BIC). Among admissible K, the one with lowest mean BIC is selected --
this rejects K values that fit marginally better but are not
reproducible from run to run (see k_selection_bic.png). The reported
model for the selected K is the lowest-BIC fit *within that same sweep*
(not a separate single-seed refit): a lone
GaussianMixture(random_state=RANDOM_STATE) fit is not reliably
reproducible across sklearn/BLAS versions even with a fixed seed, since
EM can converge to a different local optimum from a tiny floating-point
difference in the k-means++ initialization. The sweep's seeds are all
deterministically derived from RANDOM_STATE + k, so this stays
reproducible.

Cluster IDs are relabeled by ascending AWC centroid so C0..C(K-1) are
deterministic across runs.

Outputs:
  clustered_epochs_{K}.csv  per-(child, minute) rows with Cluster_ID
  selected_k.txt            the chosen K, read by downstream stages
  k_selection_bic.png       BIC + stability curves, marking the chosen K
"""

import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import adjusted_rand_score
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import PowerTransformer

warnings.filterwarnings("ignore")

SPATIAL_FILE  = 'spatial_features_1min.csv'
ACOUSTIC_FILE = 'acoustic_features_1min.csv'
PLOT_FILE     = 'k_selection_bic.png'
SELECTED_K_FILE = 'selected_k.txt'

K_RANGE      = range(2, 21)
N_INIT       = 10
RANDOM_STATE = 42

N_STABILITY_SEEDS = 20
ARI_THRESHOLD     = 0.80

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


def stability_sweep(X, k_range, base_seed, n_seeds, n_init):
    """For each K, fit n_seeds GMMs at different seeds and return
    (K, mean_bic, mean_pairwise_ari, best_model) for each, where
    best_model is the lowest-BIC fit among the n_seeds (used as the
    canonical reported model for the selected K -- see select_k).
    Admissibility (mean ARI >= ARI_THRESHOLD) measures whether restarts
    agree on the partition, not just on BIC."""
    rows = []
    for k in k_range:
        rng = np.random.default_rng(base_seed + k)
        seeds = rng.integers(0, 10**6, size=n_seeds)
        fits = [GaussianMixture(n_components=k, covariance_type='full',
                                random_state=int(s), n_init=n_init).fit(X)
                for s in seeds]
        bics = np.array([m.bic(X) for m in fits])
        labels = [m.predict(X) for m in fits]
        aris = [adjusted_rand_score(labels[i], labels[j])
                for i in range(n_seeds) for j in range(i + 1, n_seeds)]
        best_model = fits[int(np.argmin(bics))]
        rows.append((k, bics.mean(), bics.std(), np.mean(aris), best_model))
        print(f"{k:>3}  {bics.mean():>11.1f}  {bics.std():>7.2f}  {np.mean(aris):>7.3f}")
    return rows


def select_k(stability_rows, ari_threshold):
    """Lowest mean-BIC K among those with mean ARI >= ari_threshold."""
    admissible = [r for r in stability_rows if r[3] >= ari_threshold]
    pool = admissible if admissible else stability_rows
    best = min(pool, key=lambda r: r[1])
    return best[0], bool(admissible)


def plot_bic_curve(stability_rows, best_k, ari_threshold, path):
    """Save BIC + stability curves vs K, marking the chosen K and the
    admissible (stable) set. Visualizes *why* K was chosen: not just
    the lowest BIC, but the lowest BIC among partitions that restarts
    agree on."""
    ks   = [r[0] for r in stability_rows]
    bics = [r[1] for r in stability_rows]
    aris = [r[3] for r in stability_rows]
    best_bic = dict(zip(ks, bics))[best_k]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9, 8), sharex=True)

    ax1.plot(ks, bics, marker='o', color='C0', linewidth=2, label='Mean BIC (20 seeds)')
    ax1.scatter([best_k], [best_bic], color='C3', s=140, zorder=5, label=f'Selected K = {best_k}')
    ax1.axvline(best_k, color='C3', linestyle='--', alpha=0.5)
    ax1.set_ylabel('BIC')
    ax1.set_title(f'GMM model selection  (K = {min(ks)}..{max(ks)})')
    ax1.legend(loc='upper center')
    ax1.grid(alpha=0.3)

    ax2.plot(ks, aris, marker='o', color='C2', linewidth=2, label='Mean pairwise ARI (stability)')
    ax2.axhline(ari_threshold, color='gray', linestyle=':', label=f'Admissibility threshold ({ari_threshold})')
    ax2.axvline(best_k, color='C3', linestyle='--', alpha=0.5)
    ax2.set_xlabel('Number of components (K)')
    ax2.set_ylabel('Stability (ARI)')
    ax2.set_xticks(ks)
    ax2.set_ylim(0, 1.05)
    ax2.legend(loc='lower left')
    ax2.grid(alpha=0.3)

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

    # Yeo-Johnson + stability-aware K selection
    X = PowerTransformer(method='yeo-johnson').fit_transform(room[CLUSTER_FEATURES])
    print(f"Room-level minutes: {len(X)}\n")
    print(f"STABILITY-AWARE K SELECTION  ({N_STABILITY_SEEDS} seeds/K, n_init={N_INIT})")
    print(f"{'K':>3}  {'meanBIC':>11}  {'sdBIC':>7}  {'meanARI':>7}")
    stability_rows = stability_sweep(X, K_RANGE, RANDOM_STATE, N_STABILITY_SEEDS, N_INIT)
    best_k, has_admissible = select_k(stability_rows, ARI_THRESHOLD)
    if not has_admissible:
        print(f"\nWARNING: no K in {K_RANGE} met the ARI >= {ARI_THRESHOLD} "
              f"stability threshold. Falling back to lowest mean BIC overall.")
    print(f"\nSelected K = {best_k}  (admissible set: "
          f"{[r[0] for r in stability_rows if r[3] >= ARI_THRESHOLD]})")

    plot_bic_curve(stability_rows, best_k, ARI_THRESHOLD, PLOT_FILE)
    print(f"Saved BIC + stability curves to {PLOT_FILE}")

    with open(SELECTED_K_FILE, 'w') as f:
        f.write(str(best_k))
    print(f"Saved selected K to {SELECTED_K_FILE}")

    # Reuse the lowest-BIC model already fit for the selected K in the
    # stability sweep, rather than a separate single-seed refit: a lone
    # GaussianMixture(random_state=42) fit is not reliably reproducible
    # across sklearn/BLAS versions even with a fixed seed (EM can converge
    # to a different local optimum from a tiny floating-point difference
    # in the k-means++ init), whereas the sweep's seeds are all derived
    # deterministically from RANDOM_STATE + k.
    gmm = next(r[4] for r in stability_rows if r[0] == best_k)

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
