"""
Data-driven K selection for the room-level GMM via BIC.

The primary criterion is BIC, evaluated on K in K_RANGE = range(2, 15)
(i.e. K = 2..14). The upper bound is set on three grounds:

  1. EM stability. A wider sweep (K = 2..29) shows that at K = 15 the
     GMM EM optimization becomes unstable on this data: log-likelihood
     at K = 15 jumps by ~240 nats below the smooth K = 14 trajectory
     and reverts to that trajectory at K = 16. This is the signature
     of a single random restart finding a sharp local optimum that
     neighbouring K values do not reproduce, i.e. a fitting artifact
     rather than structural recovery. Constraining K <= 14 excludes
     this region.

  2. Interpretability ceiling. Activity contexts in inclusive preschool
     classrooms are not documented to exceed roughly ten distinct types
     within a single classroom day (e.g. Irvin et al., 2021), so K much
     beyond ten is unlikely to admit interpretable activity-type labels
     even if it minimises a goodness-of-fit criterion.

  3. Sample size per component. With ~2,269 room-level minutes and a
     full-covariance GMM (14 parameters per component), fits at
     K >= 15 enter the regime where individual components are estimated
     from very few minutes and the covariance matrices become
     ill-conditioned.

Within K = 2..14, BIC has a unique interior minimum: it decreases
through K = 7 and increases monotonically from K = 8 onward, so K = 7
is the largest K at which adding a component improves BIC.

ICL (Biernacki, Celeux & Govaert, 2000) is reported alongside BIC as a
secondary diagnostic. ICL adds the entropy of the posterior allocation
to BIC, so a low ICL at the BIC-chosen K confirms that the BIC minimum
sits in a region of low component overlap (well-separated components)
rather than at a region where extra components carve overlapping
pieces. Lower is better for both criteria.

Manual coding is reported for comparison only and plays no role in the
selection.

Run from the project root:
    python src/select_k.py

Outputs (at project root):
    k_selection_results.csv     one row per K
    k_selection_plot.png        BIC and ICL curves with the BIC
                                 selection and manual reference
"""

import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.special import xlogy
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import PowerTransformer

warnings.filterwarnings("ignore")

# ==========================================
# 1. CONFIGURATION   (kept in sync with cluster_room_states.py)
# ==========================================
SPATIAL_FILE  = 'spatial_features_1min.csv'
ACOUSTIC_FILE = 'acoustic_features_1min.csv'
OUTPUT_CSV    = 'k_selection_results.csv'
OUTPUT_PLOT   = 'k_selection_plot.png'

CLUSTER_FEATURES = [
    'AWC_1min_Sum',
    'Overlap_1min_Sum',
    'Velocity_1min_TotalDist',
    'Teacher_Dist_1min_Avg',
]

# K = 2..14. Upper bound documented in the module docstring.
K_RANGE      = range(2, 15)
N_INIT       = 10
RANDOM_STATE = 42

# Comparison only. Plays no role in selection.
MANUAL_K_REFERENCE = 7


# ==========================================
# 2. ROOM-LEVEL FEATURES
# ==========================================
def load_room_features():
    """Replicates the preprocessing in cluster_room_states.py."""
    df_spatial  = pd.read_csv(SPATIAL_FILE)
    df_acoustic = pd.read_csv(ACOUSTIC_FILE)

    if 'TIME_UTC' in df_spatial.columns and 'TIME_LOCAL' not in df_spatial.columns:
        df_spatial = df_spatial.rename(columns={'TIME_UTC': 'TIME_LOCAL'})
    df_spatial['TIME_LOCAL']  = (pd.to_datetime(df_spatial['TIME_LOCAL'], utc=True)
                                   .dt.tz_convert('America/New_York'))
    df_acoustic['TIME_LOCAL'] = (pd.to_datetime(df_acoustic['TIME_LOCAL'], utc=True)
                                   .dt.tz_convert('America/New_York'))

    df = pd.merge(df_spatial, df_acoustic, on=['SUBJECTID', 'TIME_LOCAL'], how='inner')
    per_min = df.groupby('TIME_LOCAL').size()
    df = df[df['TIME_LOCAL'].isin(per_min[per_min >= 3].index)]

    room_df = (df.groupby('TIME_LOCAL')[CLUSTER_FEATURES].mean()
                 .reset_index()
                 .dropna(subset=CLUSTER_FEATURES))
    return PowerTransformer(method='yeo-johnson').fit_transform(room_df[CLUSTER_FEATURES])


# ==========================================
# 3. CRITERIA
# ==========================================
def compute_icl_and_entropy(gmm, X):
    """ICL = BIC + 2 * H. Returns (ICL, H)."""
    proba = gmm.predict_proba(X)
    # xlogy(0, 0) = 0 by convention; safe for components with zero posterior
    entropy = -np.sum(xlogy(proba, proba))
    return gmm.bic(X) + 2 * entropy, entropy


# ==========================================
# 4. MAIN
# ==========================================
def main():
    k_lo, k_hi = min(K_RANGE), max(K_RANGE)
    print("=" * 60)
    print(f"  K SELECTION VIA BIC  (K = {k_lo}..{k_hi})")
    print("=" * 60)
    X = load_room_features()
    print(f"Room-level minutes: {X.shape[0]} (features: {X.shape[1]})\n")

    print(f"{'K':>3}  {'log_lik':>11}  {'BIC':>11}  {'ICL':>11}  {'entropy':>9}  conv")
    rows = []
    for k in K_RANGE:
        gmm = GaussianMixture(
            n_components=k, covariance_type='full',
            random_state=RANDOM_STATE, n_init=N_INIT,
        ).fit(X)
        bic = gmm.bic(X)
        icl, entropy = compute_icl_and_entropy(gmm, X)
        ll = gmm.score(X) * len(X)
        rows.append({
            'K': k, 'log_lik': ll, 'BIC': bic, 'ICL': icl,
            'entropy': entropy, 'converged': gmm.converged_,
        })
        print(f"{k:>3}  {ll:>11.1f}  {bic:>11.1f}  {icl:>11.1f}  "
              f"{entropy:>9.1f}  {gmm.converged_}")

    df = pd.DataFrame(rows)
    df.to_csv(OUTPUT_CSV, index=False)

    best_bic = int(df.loc[df['BIC'].idxmin(), 'K'])
    best_icl = int(df.loc[df['ICL'].idxmin(), 'K'])
    bic_min  = df['BIC'].min()
    df_other = df[df['K'] != best_bic]
    runner_up_K   = int(df_other.loc[df_other['BIC'].idxmin(), 'K'])
    runner_up_gap = float(df_other['BIC'].min() - bic_min)

    # Kass-Raftery (1995) ΔBIC interpretive thresholds:
    #   < 2     not worth more than a bare mention
    #   2..6    positive
    #   6..10   strong
    #   > 10    very strong
    if   runner_up_gap >= 10: kr_label = 'very strong'
    elif runner_up_gap >=  6: kr_label = 'strong'
    elif runner_up_gap >=  2: kr_label = 'positive'
    else:                     kr_label = 'weak'

    print()
    print("-" * 60)
    print(f"  Best K by BIC (primary):       {best_bic}")
    print(f"  BIC runner-up:                 K={runner_up_K}   "
          f"(ΔBIC = +{runner_up_gap:.1f}, {kr_label} evidence)")
    print(f"  Best K by ICL (secondary):     {best_icl}")
    print(f"  Manual coding (comparison):    {MANUAL_K_REFERENCE}")
    print("-" * 60)

    # ==========================================
    # PLOT
    # ==========================================
    fig, ax = plt.subplots(figsize=(10, 5.5))
    ax.plot(df['K'], df['BIC'], marker='o', label='BIC (primary)')
    ax.plot(df['K'], df['ICL'], marker='s', label='ICL (secondary)')
    ax.axvline(best_bic, color='C0', linestyle=':', alpha=0.85,
               label=f'BIC selection (K={best_bic})')
    ax.axvline(MANUAL_K_REFERENCE, color='gray', linestyle='--', alpha=0.5,
               label=f'Manual coding reference (K={MANUAL_K_REFERENCE})')
    ax.set_xlabel('Number of components (K)')
    ax.set_ylabel('Criterion (lower = better)')
    ax.set_title(f'GMM K selection by BIC  (K = {k_lo}..{k_hi})')
    ax.legend(loc='upper right')
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUTPUT_PLOT, dpi=200)
    print(f"\nWrote {OUTPUT_CSV} and {OUTPUT_PLOT}")


if __name__ == '__main__':
    main()
