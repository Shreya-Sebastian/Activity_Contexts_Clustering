"""
Visualize the discovered activity contexts in temporal and spatial
dimensions.

Three figures:

    activity_proportions.png   Bar chart of % of observed minutes per
                               cluster — how often each activity type
                               occurs across the corpus.

    activity_temporal.png      Two panels:
                                 (top) cluster occupancy by hour of day
                                       (heatmap, % of each hour's
                                       minutes assigned to each cluster);
                                 (bottom) per-day timeline showing the
                                       sequence of cluster labels across
                                       each recording day.

    activity_spatial.png       2D position density per cluster — hexbin
                               of (KC_X, KC_Y) child positions during
                               minutes assigned to each cluster. Only
                               produced if data/ubisense/*.csv is
                               accessible (raw spatial features needed
                               for x,y are not stored in the per-minute
                               feature CSVs and are read fresh from
                               Ubisense).

Run from the project root:
    python src/plot_activity_contexts.py
"""

import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

warnings.filterwarnings("ignore")

# ==========================================
# 1. CONFIGURATION
# ==========================================
CLUSTERED_FILE = 'clustered_epochs_7.csv'
UBISENSE_DIR   = 'data/ubisense/'

OUTPUT_PROPORTIONS = 'activity_proportions.png'
OUTPUT_TEMPORAL    = 'activity_temporal.png'
OUTPUT_SPATIAL     = 'activity_spatial.png'

# Activity-type labels from the thesis §6.1 reading of the cluster
# centroids. If the GMM run on your machine produced different cluster
# IDs (the IDs are assigned by sklearn from random initialization), the
# label-to-ID mapping below may need to be updated to match your cluster
# profiles.
CLUSTER_LABELS = {
    0: 'Teacher-proximal group activity',
    1: 'Teacher-led direct instruction',
    2: 'Quiet sedentary',
    3: 'Structured seated conversation',
    4: 'Active peer interaction',
    5: 'Free play',
    6: 'Dispersed transition',
}

TEACHER_PATTERN = '_T|_Lab'   # SUBJECTID convention for non-children


# ==========================================
# 2. DATA LOADING
# ==========================================
def load_clustered():
    """(child, minute) rows with TIME_LOCAL parsed plus convenience
    fields (date, minute floor, hour, fractional hour-of-day)."""
    df = pd.read_csv(CLUSTERED_FILE)
    df['TIME_LOCAL'] = (pd.to_datetime(df['TIME_LOCAL'], utc=True)
                          .dt.tz_convert('America/New_York'))
    df['minute']   = df['TIME_LOCAL'].dt.floor('1min')
    df['date']     = df['minute'].dt.date
    df['hour']     = df['minute'].dt.hour
    df['hour_dec'] = df['hour'] + df['minute'].dt.minute / 60.0
    return df


def get_room_minutes(df):
    """Collapse per-(child, minute) rows to one row per minute. The
    room-level cluster label is constant within a minute, so first()
    is the right aggregator."""
    return (df.groupby('minute')
              .agg(Cluster_ID=('Cluster_ID', 'first'),
                   date=('date', 'first'),
                   hour=('hour', 'first'),
                   hour_dec=('hour_dec', 'first'))
              .reset_index())


def label_for(c):
    return CLUSTER_LABELS.get(int(c), f'Cluster {c}')


# ==========================================
# 3. PROPORTIONS
# ==========================================
def plot_proportions(df):
    room = get_room_minutes(df)
    counts = room['Cluster_ID'].value_counts().sort_index()
    pct = counts / counts.sum() * 100

    fig, ax = plt.subplots(figsize=(11, 5.5))
    cmap = plt.cm.viridis
    colors = [cmap(i / max(len(counts) - 1, 1)) for i in range(len(counts))]
    bars = ax.bar(counts.index.astype(int), pct.values, color=colors)
    for bar, p in zip(bars, pct.values):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.3, f'{p:.1f}%',
                ha='center', fontsize=10)
    ax.set_xticks(counts.index.astype(int))
    ax.set_xticklabels(
        [f'C{int(c)}\n{label_for(c)}' for c in counts.index],
        fontsize=8.5,
    )
    ax.set_ylabel('% of observed minutes')
    ax.set_title('Activity-context proportions across the observed corpus')
    ax.grid(alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig(OUTPUT_PROPORTIONS, dpi=200)
    plt.close(fig)
    print(f"Wrote {OUTPUT_PROPORTIONS}")


# ==========================================
# 4. TEMPORAL  (hour-of-day heatmap + per-day timeline)
# ==========================================
def plot_temporal(df):
    room = get_room_minutes(df)
    n_clusters = int(room['Cluster_ID'].max()) + 1
    cmap = plt.cm.tab10

    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(12, 9),
        gridspec_kw={'height_ratios': [3, 4]},
    )

    # ---- Top: hour-of-day heatmap ----
    pivot = (room.groupby(['hour', 'Cluster_ID']).size()
                  .unstack(fill_value=0))
    pivot_pct = pivot.div(pivot.sum(axis=1), axis=0) * 100
    sns.heatmap(
        pivot_pct.T, ax=ax1, cmap='YlGnBu', annot=False,
        cbar_kws={'label': '% of hour'},
    )
    ax1.set_xlabel('Hour of day')
    ax1.set_ylabel('Cluster ID')
    ax1.set_title('Cluster occupancy by time of day  '
                  '(% of each hour\'s minutes per cluster)')

    # ---- Bottom: per-day timeline ----
    days = sorted(room['date'].unique())
    hour_min = room['hour_dec'].min()
    hour_max = room['hour_dec'].max()

    for i, day in enumerate(days):
        d = room[room['date'] == day]
        ax2.scatter(
            d['hour_dec'], np.full(len(d), i),
            c=[cmap(int(c)) for c in d['Cluster_ID']],
            marker='s', s=14, edgecolors='none',
        )

    ax2.set_yticks(range(len(days)))
    ax2.set_yticklabels([str(d) for d in days], fontsize=8)
    ax2.set_xlabel('Time of day (hours, decimal)')
    ax2.set_ylabel('Recording day')
    ax2.set_title('Activity-context timeline by recording day')
    ax2.set_xlim(hour_min - 0.2, hour_max + 0.2)
    ax2.grid(alpha=0.3, axis='x')

    handles = [plt.Rectangle((0, 0), 1, 1, color=cmap(c)) for c in range(n_clusters)]
    labels  = [f"C{c}: {label_for(c)}" for c in range(n_clusters)]
    ax2.legend(handles, labels, loc='center left',
               bbox_to_anchor=(1.02, 0.5), fontsize=8, frameon=False)

    plt.tight_layout()
    plt.savefig(OUTPUT_TEMPORAL, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f"Wrote {OUTPUT_TEMPORAL}")


# ==========================================
# 5. SPATIAL  (Ubisense x,y read fresh)
# ==========================================
def load_ubisense_xy():
    """Mean (KC_X, KC_Y) per (child, minute) read from data/ubisense/.
    Returns None if the directory is missing or contains no usable CSVs."""
    if not Path(UBISENSE_DIR).exists():
        return None
    files = list(Path(UBISENSE_DIR).rglob('*.csv'))
    if not files:
        return None

    print(f"Loading {len(files)} Ubisense CSV file(s)...")
    chunks = []
    for f in files:
        try:
            chunk = pd.read_csv(f)
            chunk['TIME'] = pd.to_datetime(chunk['TIME'])
            chunk['TIME_LOCAL'] = chunk['TIME'].dt.tz_localize(
                'America/New_York', ambiguous='NaT', nonexistent='shift_forward'
            )
            chunk = chunk.dropna(subset=['KC_X', 'KC_Y', 'TIME_LOCAL'])
            chunks.append(chunk[['SUBJECTID', 'TIME_LOCAL', 'KC_X', 'KC_Y']])
        except Exception as e:
            print(f"  Failed to load {f.name}: {e}")
    if not chunks:
        return None

    all_ubi = pd.concat(chunks, ignore_index=True)
    all_ubi['minute'] = all_ubi['TIME_LOCAL'].dt.floor('1min')
    return (all_ubi.groupby(['SUBJECTID', 'minute'])[['KC_X', 'KC_Y']]
                    .mean().reset_index())


def plot_spatial(df, mean_xy):
    """One hexbin panel per cluster, on a shared (X, Y) extent."""
    is_child = ~mean_xy['SUBJECTID'].str.contains(TEACHER_PATTERN,
                                                   case=False, na=False)
    mean_xy = mean_xy[is_child].copy()

    df_min = (df[['SUBJECTID', 'minute', 'Cluster_ID']]
                .drop_duplicates(['SUBJECTID', 'minute']))
    merged = mean_xy.merge(df_min, on=['SUBJECTID', 'minute'], how='inner')
    if merged.empty:
        print("No overlap between Ubisense (child, minute) data and cluster labels.")
        return

    print(f"Spatial: {len(merged):,} (child, minute) points "
          f"across {merged['Cluster_ID'].nunique()} clusters.")

    clusters = sorted(merged['Cluster_ID'].unique())
    n_clusters = len(clusters)
    n_cols = 4
    n_rows = (n_clusters + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols,
                             figsize=(4.0 * n_cols, 3.6 * n_rows),
                             sharex=True, sharey=True)
    axes = np.atleast_1d(axes).flatten()

    # Common extent (1st-99th percentile to clip stray Ubisense errors)
    xmin, xmax = merged['KC_X'].quantile([0.005, 0.995])
    ymin, ymax = merged['KC_Y'].quantile([0.005, 0.995])

    for i, c in enumerate(clusters):
        ax = axes[i]
        sub = merged[merged['Cluster_ID'] == c]
        if len(sub) > 0:
            ax.hexbin(sub['KC_X'], sub['KC_Y'],
                      gridsize=30, cmap='YlGnBu',
                      extent=(xmin, xmax, ymin, ymax), mincnt=1)
        ax.set_title(f"C{int(c)}: {label_for(c)}\n(n = {len(sub):,})",
                     fontsize=9)
        ax.set_aspect('equal', adjustable='box')
        if i % n_cols == 0:
            ax.set_ylabel('Y (m)')
        if i // n_cols == n_rows - 1:
            ax.set_xlabel('X (m)')

    for i in range(n_clusters, len(axes)):
        axes[i].axis('off')

    fig.suptitle('Spatial distribution of children by activity context  '
                 '(mean position per child-minute)',
                 y=1.0, fontsize=13)
    plt.tight_layout()
    plt.savefig(OUTPUT_SPATIAL, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f"Wrote {OUTPUT_SPATIAL}")


# ==========================================
# 6. MAIN
# ==========================================
def main():
    print("=" * 60)
    print("  ACTIVITY-CONTEXT VISUALIZATION")
    print("=" * 60)
    df = load_clustered()
    print(f"Loaded {len(df):,} (child, minute) rows, "
          f"{df['Cluster_ID'].nunique()} clusters, "
          f"{df['date'].nunique()} days.\n")

    plot_proportions(df)
    plot_temporal(df)

    mean_xy = load_ubisense_xy()
    if mean_xy is None or mean_xy.empty:
        print(f"Skipping spatial plot (no usable CSVs in {UBISENSE_DIR}).")
    else:
        plot_spatial(df, mean_xy)


if __name__ == '__main__':
    main()
