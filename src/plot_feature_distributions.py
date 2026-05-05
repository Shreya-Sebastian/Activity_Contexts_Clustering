"""
Visualize the four room-level cluster features before and after the
Yeo-Johnson transform.

Three figures, each addressing a different point in the YJ justification:

    feature_distributions.png   Raw vs YJ marginal histograms with a
                                Gaussian fit overlay and skew/excess
                                kurtosis annotations. Argues that YJ
                                makes the per-feature marginals
                                approximately Gaussian.

    feature_pairplot.png        Scatter matrix of YJ-transformed
                                features. Shows that joint structure
                                (and any non-linear coupling) survives
                                the per-feature transform.

    feature_per_cluster.png     Optional. YJ-transformed feature
                                histograms within each recovered cluster
                                (requires clustered_epochs_7.csv).
                                Shows that within-component
                                distributions remain non-Gaussian even
                                after YJ.

Run from the project root:
    python src/plot_feature_distributions.py
"""

import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats
from sklearn.preprocessing import PowerTransformer

warnings.filterwarnings("ignore")

# ==========================================
# 1. CONFIGURATION  (matches cluster_room_states.py)
# ==========================================
SPATIAL_FILE   = 'spatial_features_1min.csv'
ACOUSTIC_FILE  = 'acoustic_features_1min.csv'
CLUSTERED_FILE = 'clustered_epochs_7.csv'

OUTPUT_MARGINAL  = 'feature_distributions.png'
OUTPUT_PAIRPLOT  = 'feature_pairplot.png'
OUTPUT_PER_CLUST = 'feature_per_cluster.png'

CLUSTER_FEATURES = [
    'AWC_1min_Sum',
    'Overlap_1min_Sum',
    'Velocity_1min_TotalDist',
    'Teacher_Dist_1min_Avg',
]

LONG_LABELS = {
    'AWC_1min_Sum':            'Adult word count (per minute)',
    'Overlap_1min_Sum':        'Auditory overlap (s per minute)',
    'Velocity_1min_TotalDist': 'Cumulative displacement (m per minute)',
    'Teacher_Dist_1min_Avg':   'Mean teacher distance (m)',
}

SHORT_LABELS = {
    'AWC_1min_Sum':            'AWC',
    'Overlap_1min_Sum':        'Overlap',
    'Velocity_1min_TotalDist': 'Displacement',
    'Teacher_Dist_1min_Avg':   'Teacher dist.',
}


# ==========================================
# 2. ROOM-LEVEL FEATURES
# ==========================================
def load_room_features():
    """Returns room-level features (one row per minute, with TIME_LOCAL).
    Replicates the preprocessing in cluster_room_states.py."""
    df_spatial  = pd.read_csv(SPATIAL_FILE)
    df_acoustic = pd.read_csv(ACOUSTIC_FILE)

    if 'TIME_UTC' in df_spatial.columns and 'TIME_LOCAL' not in df_spatial.columns:
        df_spatial = df_spatial.rename(columns={'TIME_UTC': 'TIME_LOCAL'})
    df_spatial['TIME_LOCAL'] = (pd.to_datetime(df_spatial['TIME_LOCAL'], utc=True)
                                  .dt.tz_convert('America/New_York'))
    df_acoustic['TIME_LOCAL'] = (pd.to_datetime(df_acoustic['TIME_LOCAL'], utc=True)
                                   .dt.tz_convert('America/New_York'))

    df = pd.merge(df_spatial, df_acoustic, on=['SUBJECTID', 'TIME_LOCAL'], how='inner')
    per_min = df.groupby('TIME_LOCAL').size()
    df = df[df['TIME_LOCAL'].isin(per_min[per_min >= 3].index)]

    return (df.groupby('TIME_LOCAL')[CLUSTER_FEATURES].mean()
              .reset_index()
              .dropna(subset=CLUSTER_FEATURES))


# ==========================================
# 3. PLOTS
# ==========================================
def annotate_skew_kurtosis(ax, data):
    sk = stats.skew(data)
    kt = stats.kurtosis(data)  # Fisher (excess) kurtosis
    ax.text(
        0.97, 0.95, f'skew={sk:+.2f}\nex.kurt={kt:+.2f}',
        transform=ax.transAxes, ha='right', va='top',
        fontsize=9, family='monospace',
        bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.85),
    )


def plot_marginals(room_df, yj):
    """4 features x (raw, YJ) histograms with Gaussian fit overlay."""
    raw = room_df[CLUSTER_FEATURES].values
    transformed = yj.transform(raw)

    fig, axes = plt.subplots(len(CLUSTER_FEATURES), 2, figsize=(11, 12))

    for i, feat in enumerate(CLUSTER_FEATURES):
        for j, (data, suffix) in enumerate(
            [(raw[:, i], 'Raw'), (transformed[:, i], 'After Yeo-Johnson')]
        ):
            ax = axes[i, j]
            ax.hist(data, bins=50, density=True, alpha=0.65, color='C0',
                    edgecolor='white', linewidth=0.3)

            mu, sigma = data.mean(), data.std()
            xs = np.linspace(data.min(), data.max(), 200)
            ax.plot(xs, stats.norm.pdf(xs, mu, sigma), 'r-', linewidth=1.5,
                    label=f'N({mu:.2f}, {sigma:.2f}²)')
            annotate_skew_kurtosis(ax, data)

            ax.set_title(f'{LONG_LABELS[feat]} -- {suffix}', fontsize=10)
            ax.legend(fontsize=8, loc='upper right')
            if i == len(CLUSTER_FEATURES) - 1:
                ax.set_xlabel('Value')
            if j == 0:
                ax.set_ylabel('Density')

    plt.tight_layout()
    plt.savefig(OUTPUT_MARGINAL, dpi=200)
    plt.close(fig)
    print(f"Wrote {OUTPUT_MARGINAL}")


def plot_pairplot(room_df, yj):
    """Scatter matrix of YJ-transformed features. Joint structure that
    survives the per-feature normalization is the visual evidence that
    YJ does not produce a 4-D multivariate Gaussian."""
    transformed = yj.transform(room_df[CLUSTER_FEATURES])
    df_yj = pd.DataFrame(transformed, columns=[SHORT_LABELS[c] for c in CLUSTER_FEATURES])

    g = sns.pairplot(df_yj, height=2.4,
                     plot_kws=dict(alpha=0.25, s=10),
                     diag_kind='hist', diag_kws=dict(bins=40))
    g.fig.suptitle('YJ-transformed features: joint structure', y=1.01, fontsize=12)
    g.savefig(OUTPUT_PAIRPLOT, dpi=180, bbox_inches='tight')
    plt.close(g.fig)
    print(f"Wrote {OUTPUT_PAIRPLOT}")


def plot_per_cluster(room_df, yj):
    """If clustered_epochs_7.csv exists, show YJ-transformed feature
    distributions within each recovered cluster."""
    if not Path(CLUSTERED_FILE).exists():
        print(f"Skipping per-cluster plot ({CLUSTERED_FILE} not found).")
        return

    clust = pd.read_csv(CLUSTERED_FILE)
    clust['TIME_LOCAL'] = (pd.to_datetime(clust['TIME_LOCAL'], utc=True)
                             .dt.tz_convert('America/New_York'))

    # Cluster_ID is constant within each minute (room-level label
    # propagated to per-child rows), so first() suffices.
    minute_to_cluster = (clust.groupby('TIME_LOCAL')['Cluster_ID']
                              .first().reset_index())
    merged = room_df.merge(minute_to_cluster, on='TIME_LOCAL', how='inner')
    if merged.empty:
        print(f"Skipping per-cluster plot (no overlap with {CLUSTERED_FILE}).")
        return

    transformed = yj.transform(merged[CLUSTER_FEATURES])
    yj_df = pd.DataFrame(transformed, columns=CLUSTER_FEATURES)
    yj_df['Cluster_ID'] = merged['Cluster_ID'].values

    clusters = sorted(yj_df['Cluster_ID'].unique())
    n_feats, n_clusts = len(CLUSTER_FEATURES), len(clusters)
    fig, axes = plt.subplots(n_feats, n_clusts,
                             figsize=(2.0 * n_clusts, 2.0 * n_feats),
                             sharey='row')

    for i, feat in enumerate(CLUSTER_FEATURES):
        for j, c in enumerate(clusters):
            ax = axes[i, j]
            data = yj_df.loc[yj_df['Cluster_ID'] == c, feat].values
            if len(data) == 0:
                ax.set_axis_off()
                continue
            ax.hist(data, bins=20, density=True, alpha=0.75,
                    color=plt.cm.viridis(j / max(n_clusts - 1, 1)))

            mu, sigma = data.mean(), data.std()
            if sigma > 0:
                xs = np.linspace(data.min(), data.max(), 100)
                ax.plot(xs, stats.norm.pdf(xs, mu, sigma), 'r-', linewidth=1)

            sk = stats.skew(data) if len(data) > 2 else 0.0
            ax.text(0.97, 0.95, f'sk={sk:+.1f}\nn={len(data)}',
                    transform=ax.transAxes, ha='right', va='top',
                    fontsize=7, family='monospace',
                    bbox=dict(boxstyle='round,pad=0.2',
                              facecolor='white', alpha=0.85))
            if i == 0:
                ax.set_title(f'C{c}', fontsize=9)
            if j == 0:
                ax.set_ylabel(SHORT_LABELS[feat], fontsize=8)
            ax.set_xticks([])
            ax.set_yticks([])

    fig.suptitle('YJ-transformed features within each recovered cluster '
                 '(red = per-cluster Gaussian fit)', fontsize=11, y=1.01)
    plt.tight_layout()
    plt.savefig(OUTPUT_PER_CLUST, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f"Wrote {OUTPUT_PER_CLUST}")


# ==========================================
# 4. MAIN
# ==========================================
def main():
    print("=" * 60)
    print("  FEATURE DISTRIBUTIONS")
    print("=" * 60)
    room_df = load_room_features()
    print(f"Room-level minutes: {len(room_df)}\n")

    # Fit YJ once on the full set of minutes (matches cluster_room_states.py)
    # and reuse the same transform across all three plots.
    yj = PowerTransformer(method='yeo-johnson').fit(room_df[CLUSTER_FEATURES])

    plot_marginals(room_df, yj)
    plot_pairplot(room_df, yj)
    plot_per_cluster(room_df, yj)


if __name__ == '__main__':
    main()
