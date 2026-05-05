"""Build the inclusion dashboard PNG: context profiles, peer co-presence,
vocal participation, and affiliation patterns split by Diagnosis."""

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler

CLUSTERED = 'clustered_epochs_7.csv'
INCLUSION = 'child_inclusion_by_cluster_7.csv'
OUTPUT    = 'Inclusion_Dashboard_7.png'

CONTEXT_COLS = ['AWC_1min_Sum', 'Overlap_1min_Sum',
                'Velocity_1min_TotalDist', 'Teacher_Dist_1min_Avg']
DEMO_COLS    = ['Avg_Time_HL_Group_Pct', 'Avg_Time_TH_Group_Pct', 'Avg_Time_Mixed_Group_Pct']
PALETTE      = {'TH': '#4C72B0', 'HL': '#DD8452'}


if __name__ == "__main__":
    df_ctx = pd.read_csv(CLUSTERED)
    df_inc = pd.read_csv(INCLUSION).dropna(subset=['Diagnosis'])

    centroids = df_ctx.groupby('Cluster_ID')[CONTEXT_COLS].mean()
    scaled = pd.DataFrame(MinMaxScaler().fit_transform(centroids),
                          columns=CONTEXT_COLS, index=centroids.index).reset_index()
    ctx_melt = scaled.melt(id_vars='Cluster_ID', var_name='Feature', value_name='Intensity')
    demo_melt = df_inc.melt(id_vars=['Cluster_ID', 'Diagnosis'], value_vars=DEMO_COLS,
                            var_name='Group', value_name='Pct')

    sns.set_theme(style="whitegrid")
    fig = plt.figure(figsize=(24, 16))
    gs = gridspec.GridSpec(2, 2, figure=fig, wspace=0.15, hspace=0.3)

    sns.barplot(data=ctx_melt, x='Cluster_ID', y='Intensity', hue='Feature',
                ax=fig.add_subplot(gs[0, 0]), palette="viridis"
                ).set_title('Environmental Context Profiles', fontweight='bold')
    sns.barplot(data=df_inc, x='Cluster_ID', y='Avg_Time_In_Group_Pct', hue='Diagnosis',
                ax=fig.add_subplot(gs[0, 1]), palette=PALETTE, capsize=0.05
                ).set_title('Peer Co-presence (% in F-formation)', fontweight='bold')
    ax3 = fig.add_subplot(gs[1, 0])
    sns.barplot(data=df_inc, x='Cluster_ID', y='Avg_Utt_Count', hue='Diagnosis',
                ax=ax3, palette=PALETTE, capsize=0.05
                ).set_title('Vocal Participation Rate', fontweight='bold')
    ax3.legend(title='Diagnosis', loc='upper right')

    gs_br = gridspec.GridSpecFromSubplotSpec(1, 2, subplot_spec=gs[1, 1], wspace=0.15)
    for i, diag in enumerate(['TH', 'HL']):
        ax = fig.add_subplot(gs_br[0, i])
        sns.barplot(data=demo_melt[demo_melt['Diagnosis'] == diag],
                    x='Cluster_ID', y='Pct', hue='Group',
                    ax=ax, palette="Set2", capsize=0.05
                    ).set_title(f'{diag} Affiliation Patterns', fontweight='bold')
        if i == 1:
            ax.get_legend().remove()

    plt.savefig(OUTPUT, dpi=300, bbox_inches='tight')
    print(f"Saved {OUTPUT}")
