"""Visualize activity contexts: proportions, time-of-day, per-day timeline,
and (if Ubisense data is present) 2D position density per cluster."""

import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

warnings.filterwarnings("ignore")

CLUSTERED       = 'clustered_epochs_7.csv'
UBISENSE_DIR    = 'data/ubisense/'
TEACHER_PATTERN = '_T|_Lab'

# Labels under the AWC-ascending cluster ordering applied in
# cluster_room_states.py. Identified by centroid profile
CLUSTER_LABELS = {
    0: 'Dispersed transition',           # lowest AWC, highest velocity + teacher dist
    1: 'Active peer interaction',        # mod-high teacher distance (~4.5 m)
    2: 'Quiet sedentary',                # lowest auditory overlap
    3: 'Free play',                      # largest cluster, high overlap, low teacher dist
    4: 'Structured seated conversation', # lowest velocity (~9 m/min)
    5: 'Teacher-proximal group activity',# lowest teacher distance, second-largest cluster
    6: 'Teacher-led direct instruction', # highest AWC
}


def load_clustered():
    df = pd.read_csv(CLUSTERED)
    df['TIME_LOCAL'] = pd.to_datetime(df['TIME_LOCAL'], utc=True).dt.tz_convert('America/New_York')
    df['minute']    = df['TIME_LOCAL'].dt.floor('1min')
    df['date']      = df['minute'].dt.date
    df['hour']      = df['minute'].dt.hour
    df['hour_dec']  = df['hour'] + df['minute'].dt.minute / 60.0
    return df


def room_minutes(df):
    return (df.groupby('minute').agg(Cluster_ID=('Cluster_ID', 'first'),
                                      date=('date', 'first'),
                                      hour=('hour', 'first'),
                                      hour_dec=('hour_dec', 'first')).reset_index())


def label_for(c):
    return CLUSTER_LABELS.get(int(c), f'Cluster {c}')


def plot_proportions(df, out='activity_proportions.png'):
    counts = room_minutes(df)['Cluster_ID'].value_counts().sort_index()
    pct = counts / counts.sum() * 100
    fig, ax = plt.subplots(figsize=(11, 5.5))
    cmap = plt.cm.viridis
    bars = ax.bar(counts.index.astype(int), pct.values,
                  color=[cmap(i / max(len(counts) - 1, 1)) for i in range(len(counts))])
    for bar, p in zip(bars, pct.values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
                f'{p:.1f}%', ha='center', fontsize=10)
    ax.set_xticks(counts.index.astype(int))
    ax.set_xticklabels([f'C{int(c)}\n{label_for(c)}' for c in counts.index], fontsize=8.5)
    ax.set_ylabel('% of observed minutes')
    ax.set_title('Activity-context proportions')
    ax.grid(alpha=0.3, axis='y')
    plt.tight_layout(); plt.savefig(out, dpi=200); plt.close(fig)
    print(f"Wrote {out}")


def plot_temporal(df, out='activity_temporal.png'):
    room = room_minutes(df)
    n_clusters = int(room['Cluster_ID'].max()) + 1
    cmap = plt.cm.tab10

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 9), gridspec_kw={'height_ratios': [3, 4]})

    pivot = room.groupby(['hour', 'Cluster_ID']).size().unstack(fill_value=0)
    sns.heatmap(pivot.div(pivot.sum(axis=1), axis=0).T * 100, ax=ax1, cmap='YlGnBu',
                cbar_kws={'label': '% of hour'})
    ax1.set(xlabel='Hour of day', ylabel='Cluster ID',
            title="Cluster occupancy by time of day  (% of each hour's minutes)")

    days = sorted(room['date'].unique())
    for i, day in enumerate(days):
        d = room[room['date'] == day]
        ax2.scatter(d['hour_dec'], np.full(len(d), i),
                    c=[cmap(int(c)) for c in d['Cluster_ID']],
                    marker='s', s=14, edgecolors='none')
    ax2.set_yticks(range(len(days)))
    ax2.set_yticklabels([str(d) for d in days], fontsize=8)
    ax2.set(xlabel='Time of day (hours)', ylabel='Recording day',
            title='Activity-context timeline by recording day')
    ax2.grid(alpha=0.3, axis='x')

    handles = [plt.Rectangle((0, 0), 1, 1, color=cmap(c)) for c in range(n_clusters)]
    ax2.legend(handles, [f"C{c}: {label_for(c)}" for c in range(n_clusters)],
               loc='center left', bbox_to_anchor=(1.02, 0.5), fontsize=8, frameon=False)
    plt.tight_layout(); plt.savefig(out, dpi=200, bbox_inches='tight'); plt.close(fig)
    print(f"Wrote {out}")


def load_ubisense_xy():
    """Mean (KC_X, KC_Y) per (child, minute) read from data/ubisense/."""
    if not Path(UBISENSE_DIR).exists():
        return None
    files = list(Path(UBISENSE_DIR).rglob('*.csv'))
    if not files:
        return None
    print(f"Loading {len(files)} Ubisense file(s)...")
    chunks = []
    for f in files:
        try:
            c = pd.read_csv(f)
            c['TIME'] = pd.to_datetime(c['TIME'])
            c['TIME_LOCAL'] = c['TIME'].dt.tz_localize(
                'America/New_York', ambiguous='NaT', nonexistent='shift_forward')
            chunks.append(c.dropna(subset=['KC_X', 'KC_Y', 'TIME_LOCAL'])
                           [['SUBJECTID', 'TIME_LOCAL', 'KC_X', 'KC_Y']])
        except Exception as e:
            print(f"  Failed {f.name}: {e}")
    if not chunks:
        return None
    all_ubi = pd.concat(chunks, ignore_index=True)
    all_ubi['minute'] = all_ubi['TIME_LOCAL'].dt.floor('1min')
    return all_ubi.groupby(['SUBJECTID', 'minute'])[['KC_X', 'KC_Y']].mean().reset_index()


def plot_spatial(df, mean_xy, out='activity_spatial.png'):
    mean_xy = mean_xy[~mean_xy['SUBJECTID'].str.contains(TEACHER_PATTERN, case=False, na=False)]
    df_min = df[['SUBJECTID', 'minute', 'Cluster_ID']].drop_duplicates(['SUBJECTID', 'minute'])
    merged = mean_xy.merge(df_min, on=['SUBJECTID', 'minute'], how='inner')
    if merged.empty:
        print("No overlap between Ubisense and clusters."); return
    print(f"Spatial: {len(merged):,} child-minutes across {merged['Cluster_ID'].nunique()} clusters")

    clusters = sorted(merged['Cluster_ID'].unique())
    n, cols = len(clusters), 4
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 3.6 * rows), sharex=True, sharey=True)
    axes = np.atleast_1d(axes).flatten()
    xmin, xmax = merged['KC_X'].quantile([0.005, 0.995])
    ymin, ymax = merged['KC_Y'].quantile([0.005, 0.995])

    for i, c in enumerate(clusters):
        ax, sub = axes[i], merged[merged['Cluster_ID'] == c]
        if len(sub):
            ax.hexbin(sub['KC_X'], sub['KC_Y'], gridsize=30, cmap='YlGnBu',
                      extent=(xmin, xmax, ymin, ymax), mincnt=1)
        ax.set_title(f"C{int(c)}: {label_for(c)} (n={len(sub):,})", fontsize=9)
        ax.set_aspect('equal', adjustable='box')
        if i % cols == 0:        ax.set_ylabel('Y (m)')
        if i // cols == rows - 1: ax.set_xlabel('X (m)')
    for i in range(n, len(axes)):
        axes[i].axis('off')

    fig.suptitle('Spatial distribution of children by activity context  '
                 '(mean position per child-minute)', y=1.0, fontsize=13)
    plt.tight_layout(); plt.savefig(out, dpi=200, bbox_inches='tight'); plt.close(fig)
    print(f"Wrote {out}")


if __name__ == '__main__':
    df = load_clustered()
    print(f"Loaded {len(df):,} rows, {df['Cluster_ID'].nunique()} clusters, "
          f"{df['date'].nunique()} days")
    plot_proportions(df)
    plot_temporal(df)
    xy = load_ubisense_xy()
    if xy is None or xy.empty:
        print(f"Skipping spatial plot (no Ubisense data in {UBISENSE_DIR}).")
    else:
        plot_spatial(df, xy)
