"""
Build the inclusion dashboard:
  - Environmental context profiles per cluster
  - Time-in-group by Diagnosis × Cluster
  - Utterance count by Diagnosis × Cluster
  - Affiliation patterns (HL/TH/Mixed) split by Diagnosis
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
import matplotlib.gridspec as gridspec

# ==========================================
# 1. CONFIGURATION
# ==========================================
CLUSTERED_DATA_FILE = 'clustered_epochs_4.csv'
INCLUSION_DATA_FILE = 'child_inclusion_by_cluster_4.csv'
OUTPUT_IMAGE = 'Inclusion_Dashboard_4.png'

CONTEXT_COLS = ['AWC_1min_Sum', 'Overlap_1min_Sum',
                'Velocity_1min_TotalDist', 'Teacher_Dist_1min_Avg']

# ==========================================
# 2. MAIN
# ==========================================
if __name__ == "__main__":
    df_context = pd.read_csv(CLUSTERED_DATA_FILE)
    df_inclusion = pd.read_csv(INCLUSION_DATA_FILE).dropna(subset=['Diagnosis'])

    # Scale context features to [0, 1] for side-by-side comparison
    centroids_raw = df_context.groupby('Cluster_ID')[CONTEXT_COLS].mean()
    scaler = MinMaxScaler()
    centroids_scaled = pd.DataFrame(
        scaler.fit_transform(centroids_raw),
        columns=centroids_raw.columns,
        index=centroids_raw.index,
    ).reset_index()
    context_melted = centroids_scaled.melt(
        id_vars='Cluster_ID', var_name='Feature', value_name='Intensity'
    )

    sns.set_theme(style="whitegrid")
    fig = plt.figure(figsize=(24, 16))
    gs = gridspec.GridSpec(2, 2, figure=fig, wspace=0.15, hspace=0.3)
    cohort_colors = {'TH': '#4C72B0', 'HL': '#DD8452'}

    # Plot 1: context profiles
    ax1 = fig.add_subplot(gs[0, 0])
    sns.barplot(data=context_melted, x='Cluster_ID', y='Intensity',
                hue='Feature', ax=ax1, palette="viridis")
    ax1.set_title('Environmental Context Profiles', fontweight='bold')

    # Plot 2: physical inclusion
    ax2 = fig.add_subplot(gs[0, 1])
    sns.barplot(data=df_inclusion, x='Cluster_ID', y='Avg_Time_In_Group_Pct',
                hue='Diagnosis', ax=ax2, palette=cohort_colors, capsize=0.05)
    ax2.set_title('Peer Co-presence (% of minute in F-formation)', fontweight='bold')

    # Plot 3: expressive inclusion
    ax3 = fig.add_subplot(gs[1, 0])
    sns.barplot(data=df_inclusion, x='Cluster_ID', y='Avg_Utt_Count',
                hue='Diagnosis', ax=ax3, palette=cohort_colors, capsize=0.05)
    ax3.set_title('Vocal Participation Rate (utterances per minute)', fontweight='bold')
    ax3.legend(title='Diagnosis', loc='upper right')

    # Plot 4/5: association patterns per diagnosis
    demo_cols = ['Avg_Time_HL_Group_Pct', 'Avg_Time_TH_Group_Pct', 'Avg_Time_Mixed_Group_Pct']
    df_demo_melt = df_inclusion.melt(
        id_vars=['Cluster_ID', 'Diagnosis'],
        value_vars=demo_cols, var_name='Group', value_name='Pct'
    )
    gs_bottom_right = gridspec.GridSpecFromSubplotSpec(1, 2, subplot_spec=gs[1, 1], wspace=0.15)

    for i, diag in enumerate(['TH', 'HL']):
        ax = fig.add_subplot(gs_bottom_right[0, i])
        sns.barplot(data=df_demo_melt[df_demo_melt['Diagnosis'] == diag],
                    x='Cluster_ID', y='Pct', hue='Group',
                    ax=ax, palette="Set2", capsize=0.05)
        ax.set_title(f'{diag} Affiliation Patterns', fontweight='bold')
        if i == 1:
            ax.get_legend().remove()

    plt.savefig(OUTPUT_IMAGE, dpi=300, bbox_inches='tight')
    print(f"Dashboard saved as {OUTPUT_IMAGE}")