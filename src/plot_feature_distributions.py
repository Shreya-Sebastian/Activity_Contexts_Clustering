"""Visualize the four cluster features:
    feature_distributions.png  marginals (raw vs Yeo-Johnson) with Gaussian fit
    feature_pairplot.png       joint structure of YJ-transformed features
    feature_per_cluster.png    YJ-transformed marginals per cluster (optional)
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

SPATIAL   = 'spatial_features_1min.csv'
ACOUSTIC  = 'acoustic_features_1min.csv'
CLUSTERED = 'clustered_epochs_7.csv'

OUT_MARG  = 'feature_distributions.png'
OUT_PAIR  = 'feature_pairplot.png'
OUT_CLUST = 'feature_per_cluster.png'

CLUSTER_FEATURES = [
    'AWC_1min_Sum', 'Overlap_1min_Sum',
    'Velocity_1min_TotalDist', 'Teacher_Dist_1min_Avg',
]
LABELS_LONG = {
    'AWC_1min_Sum':            'Adult word count (per minute)',
    'Overlap_1min_Sum':        'Auditory overlap (s per minute)',
    'Velocity_1min_TotalDist': 'Cumulative displacement (m per minute)',
    'Teacher_Dist_1min_Avg':   'Mean teacher distance (m)',
}
LABELS_SHORT = {
    'AWC_1min_Sum': 'AWC', 'Overlap_1min_Sum': 'Overlap',
    'Velocity_1min_TotalDist': 'Displacement', 'Teacher_Dist_1min_Avg': 'Teacher dist.',
}


def load_room():
    sp, ac = pd.read_csv(SPATIAL), pd.read_csv(ACOUSTIC)
    if 'TIME_UTC' in sp.columns and 'TIME_LOCAL' not in sp.columns:
        sp = sp.rename(columns={'TIME_UTC': 'TIME_LOCAL'})
    for d in (sp, ac):
        d['TIME_LOCAL'] = pd.to_datetime(d['TIME_LOCAL'], utc=True).dt.tz_convert('America/New_York')
    df = sp.merge(ac, on=['SUBJECTID', 'TIME_LOCAL'], how='inner')
    cnt = df.groupby('TIME_LOCAL').size()
    df = df[df['TIME_LOCAL'].isin(cnt[cnt >= 3].index)]
    return df.groupby('TIME_LOCAL')[CLUSTER_FEATURES].mean().reset_index().dropna(subset=CLUSTER_FEATURES)


def _annotate(ax, data):
    ax.text(0.97, 0.95, f'skew={stats.skew(data):+.2f}\nex.kurt={stats.kurtosis(data):+.2f}',
            transform=ax.transAxes, ha='right', va='top', fontsize=9, family='monospace',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.85))


def plot_marginals(room, yj):
    raw = room[CLUSTER_FEATURES].values
    tx  = yj.transform(raw)
    fig, axes = plt.subplots(len(CLUSTER_FEATURES), 2, figsize=(11, 12))
    for i, feat in enumerate(CLUSTER_FEATURES):
        for j, (data, suffix) in enumerate([(raw[:, i], 'Raw'), (tx[:, i], 'After Yeo-Johnson')]):
            ax = axes[i, j]
            ax.hist(data, bins=50, density=True, alpha=0.65, color='C0',
                    edgecolor='white', linewidth=0.3)
            mu, sigma = data.mean(), data.std()
            xs = np.linspace(data.min(), data.max(), 200)
            ax.plot(xs, stats.norm.pdf(xs, mu, sigma), 'r-', linewidth=1.5,
                    label=f'N({mu:.2f}, {sigma:.2f}²)')
            _annotate(ax, data)
            ax.set_title(f'{LABELS_LONG[feat]} -- {suffix}', fontsize=10)
            ax.legend(fontsize=8, loc='upper right')
            if i == len(CLUSTER_FEATURES) - 1: ax.set_xlabel('Value')
            if j == 0:                         ax.set_ylabel('Density')
    plt.tight_layout(); plt.savefig(OUT_MARG, dpi=200); plt.close(fig)
    print(f"Wrote {OUT_MARG}")


def plot_pairplot(room, yj):
    tx = pd.DataFrame(yj.transform(room[CLUSTER_FEATURES]),
                      columns=[LABELS_SHORT[c] for c in CLUSTER_FEATURES])
    g = sns.pairplot(tx, height=2.4, plot_kws=dict(alpha=0.25, s=10),
                     diag_kind='hist', diag_kws=dict(bins=40))
    g.fig.suptitle('YJ-transformed features: joint structure', y=1.01, fontsize=12)
    g.savefig(OUT_PAIR, dpi=180, bbox_inches='tight'); plt.close(g.fig)
    print(f"Wrote {OUT_PAIR}")


def plot_per_cluster(room, yj):
    if not Path(CLUSTERED).exists():
        print(f"Skipping per-cluster plot ({CLUSTERED} not found).")
        return
    clust = pd.read_csv(CLUSTERED)
    clust['TIME_LOCAL'] = pd.to_datetime(clust['TIME_LOCAL'], utc=True).dt.tz_convert('America/New_York')
    minute_to_c = clust.groupby('TIME_LOCAL')['Cluster_ID'].first().reset_index()
    merged = room.merge(minute_to_c, on='TIME_LOCAL', how='inner')
    if merged.empty:
        print(f"No overlap with {CLUSTERED}."); return

    tx = pd.DataFrame(yj.transform(merged[CLUSTER_FEATURES]), columns=CLUSTER_FEATURES)
    tx['Cluster_ID'] = merged['Cluster_ID'].values

    clusters = sorted(tx['Cluster_ID'].unique())
    n_f, n_c = len(CLUSTER_FEATURES), len(clusters)
    fig, axes = plt.subplots(n_f, n_c, figsize=(2.0 * n_c, 2.0 * n_f), sharey='row')
    for i, feat in enumerate(CLUSTER_FEATURES):
        for j, c in enumerate(clusters):
            ax = axes[i, j]
            data = tx.loc[tx['Cluster_ID'] == c, feat].values
            if not len(data):
                ax.set_axis_off(); continue
            ax.hist(data, bins=20, density=True, alpha=0.75,
                    color=plt.cm.viridis(j / max(n_c - 1, 1)))
            mu, sigma = data.mean(), data.std()
            if sigma > 0:
                xs = np.linspace(data.min(), data.max(), 100)
                ax.plot(xs, stats.norm.pdf(xs, mu, sigma), 'r-', linewidth=1)
            sk = stats.skew(data) if len(data) > 2 else 0.0
            ax.text(0.97, 0.95, f'sk={sk:+.1f}\nn={len(data)}', transform=ax.transAxes,
                    ha='right', va='top', fontsize=7, family='monospace',
                    bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.85))
            if i == 0: ax.set_title(f'C{c}', fontsize=9)
            if j == 0: ax.set_ylabel(LABELS_SHORT[feat], fontsize=8)
            ax.set_xticks([]); ax.set_yticks([])
    fig.suptitle('YJ-transformed features within each recovered cluster '
                 '(red = per-cluster Gaussian fit)', fontsize=11, y=1.01)
    plt.tight_layout(); plt.savefig(OUT_CLUST, dpi=200, bbox_inches='tight'); plt.close(fig)
    print(f"Wrote {OUT_CLUST}")


if __name__ == '__main__':
    room = load_room()
    print(f"Room-level minutes: {len(room)}")
    yj = PowerTransformer(method='yeo-johnson').fit(room[CLUSTER_FEATURES])
    plot_marginals(room, yj)
    plot_pairplot(room, yj)
    plot_per_cluster(room, yj)
