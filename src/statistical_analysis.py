"""
Social inclusion by cluster: exposure + physical + expressive + homophily.

For each cluster we report HL vs TH:
  - Exposure:   Mann-Whitney on per-child % of day spent in the cluster.
  - Physical:   LMM on Time_In_Group_1min_Pct; per-cluster HL-TH from Wald contrast.
  - Expressive: LMM on Child_Utt_Count_1min;   per-cluster HL-TH from Wald contrast.
  - Homophily:  LMM on same-diagnosis group time (Time_In_HL_Pct for HL kids,
                Time_In_TH_Pct for TH kids) and on mixed-group time.
  - Homophily index: child-level same / (same + mixed) group-minute totals,
                     Mann-Whitney per cluster (conditional on being grouped).

LMM: DV ~ C(Cluster_ID) * C(Diagnosis) + (1 | SUBJECTID) on minute-level data.
Cohen's d computed on child-level means per cluster.
"""

import warnings
import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.formula.api as smf

warnings.filterwarnings("ignore")

# ==========================================
# CONFIGURATION
# ==========================================
CLUSTERED_FILE = 'clustered_epochs_7.csv'
CONSOLIDATED   = 'data/mapping/MAPPING_CONSOLIDATED.csv'                 # per-day STATUS
BASE           = 'data/mapping/MAPPING_StarFish_2223_BASE_NONAMES.csv'   # static Diagnosis
OUTPUT         = 'statistical_analysis_results_7.txt'

PHYSICAL_DV   = 'Time_In_Group_1min_Pct'
EXPRESSIVE_DV = 'Child_Utt_Count_1min'
SAMEDX_DV     = 'Time_In_SameDx_Pct'
MIXED_DV      = 'Time_In_Mixed_Group_1min_Pct'


# ==========================================
# DATA
# ==========================================
def load_data():
    """Minute-level data + per-day STATUS + static Diagnosis."""
    df = pd.read_csv(CLUSTERED_FILE)
    df['Date'] = (pd.to_datetime(df['TIME_LOCAL'], utc=True)
                  .dt.tz_convert('America/New_York')
                  .dt.date.astype(str))

    consol = pd.read_csv(CONSOLIDATED)
    consol['Date'] = pd.to_datetime(consol['Date']).dt.date.astype(str)
    attendance = (consol.dropna(subset=['Subject_ID'])
                  [['Subject_ID', 'Date', 'STATUS']]
                  .rename(columns={'Subject_ID': 'SUBJECTID'}))

    base = pd.read_csv(BASE)
    diag = (base.dropna(subset=['Subject_ID'])
            [['Subject_ID', 'Diagnosis']]
            .rename(columns={'Subject_ID': 'SUBJECTID'}))
    diag = diag[diag['Diagnosis'].isin(['HL', 'TH'])]

    df = df.merge(attendance, on=['SUBJECTID', 'Date'], how='inner')
    df = df.merge(diag,       on='SUBJECTID',          how='inner')
    df = df[df['STATUS'] == 'PRESENT']
    df = df[~df['SUBJECTID'].str.contains('_T|_Lab', case=False, na=False)]
    df['Cluster_ID'] = df['Cluster_ID'].astype(str)

    # Derived: same-diagnosis group time. An HL child is never in a TH-only
    # group by construction (their presence makes it mixed) and vice versa,
    # so the "other" column is structurally ~0 and uninformative. The
    # same-dx column lets us compare HL-in-HL-group vs TH-in-TH-group.
    df[SAMEDX_DV] = np.where(
        df['Diagnosis'] == 'HL',
        df['Time_In_HL_Group_1min_Pct'],
        df['Time_In_TH_Group_1min_Pct'],
    )
    return df


def cohens_d(a, b):
    if len(a) < 2 or len(b) < 2:
        return np.nan
    pool = np.sqrt(((len(a)-1)*a.var(ddof=1) + (len(b)-1)*b.var(ddof=1)) / (len(a)+len(b)-2))
    return (a.mean() - b.mean()) / pool if pool > 0 else np.nan


# ==========================================
# ANALYSES
# ==========================================
def exposure(df, out):
    """% of each child's minutes in each cluster, HL vs TH."""
    out.append("\nEXPOSURE — % of each child's day per cluster (Mann-Whitney)")
    out.append("-" * 70)
    out.append(f"{'Cluster':<10} {'HL %':>8} {'TH %':>8} {'diff':>8} {'d':>8} {'p':>8}")

    per = df.groupby(['SUBJECTID', 'Diagnosis', 'Cluster_ID']).size().reset_index(name='n')
    tot = df.groupby('SUBJECTID').size().rename('tot')
    per = per.merge(tot, on='SUBJECTID')
    per['pct'] = per['n'] / per['tot'] * 100

    for c in sorted(df['Cluster_ID'].unique()):
        hl = per[(per['Cluster_ID'] == c) & (per['Diagnosis'] == 'HL')]['pct']
        th = per[(per['Cluster_ID'] == c) & (per['Diagnosis'] == 'TH')]['pct']
        if len(hl) >= 2 and len(th) >= 2:
            _, p = stats.mannwhitneyu(hl, th, alternative='two-sided')
        else:
            p = np.nan
        d = cohens_d(hl, th)
        out.append(f"{c:<10} {hl.mean():>8.1f} {th.mean():>8.1f} "
                   f"{hl.mean()-th.mean():>+8.1f} {d:>+8.2f} {p:>8.4f}")


def inclusion(df, dv, label, out):
    """LMM with per-cluster HL-TH Wald contrast."""
    out.append(f"\n{label.upper()} — {dv}")
    out.append(f"LMM: {dv} ~ C(Cluster_ID) * C(Diagnosis) + (1 | SUBJECTID)")
    out.append("-" * 70)
    out.append(f"{'Cluster':<10} {'HL':>8} {'TH':>8} {'diff':>8} {'d':>8} {'p':>8}")

    d = df.dropna(subset=[dv]).copy()
    d['Diagnosis'] = pd.Categorical(d['Diagnosis'], categories=['TH', 'HL'])

    try:
        model = smf.mixedlm(f"{dv} ~ C(Cluster_ID) * C(Diagnosis)",
                            data=d, groups=d['SUBJECTID']).fit(reml=True)
    except Exception as e:
        out.append(f"  Model failed: {e}")
        return

    # Child-level means for Cohen's d
    cm = d.groupby(['SUBJECTID', 'Diagnosis', 'Cluster_ID'])[dv].mean().reset_index()
    params = list(model.fe_params.index)
    clusters = sorted(d['Cluster_ID'].unique())
    ref = clusters[0]
    diag_term = 'C(Diagnosis)[T.HL]'

    for c in clusters:
        # Contrast = Diagnosis[T.HL] + (Cluster[T.c] : Diagnosis[T.HL]) when c != ref
        contrast = np.zeros(len(params))
        contrast[params.index(diag_term)] = 1
        if c != ref:
            inter = f'C(Cluster_ID)[T.{c}]:C(Diagnosis)[T.HL]'
            if inter in params:
                contrast[params.index(inter)] = 1
        t = model.t_test(contrast.reshape(1, -1))
        diff, p = float(t.effect[0]), float(t.pvalue)

        hl = cm[(cm['Cluster_ID'] == c) & (cm['Diagnosis'] == 'HL')][dv]
        th = cm[(cm['Cluster_ID'] == c) & (cm['Diagnosis'] == 'TH')][dv]
        d_eff = cohens_d(hl, th)
        out.append(f"{c:<10} {hl.mean():>8.2f} {th.mean():>8.2f} "
                   f"{diff:>+8.2f} {d_eff:>+8.2f} {p:>8.4f}")


def homophily_index(df, out):
    """
    Child-level homophily index: same / (same + mixed) group-minute totals
    per (child, cluster), conditional on the child being in some F-formation.

    Index = 1.0  -> always grouped with same-diagnosis peers when grouped.
    Index = 0.0  -> always in mixed groups when grouped.
    Note: baseline availability differs by cohort size, so absolute values
    reflect both preference and opportunity. The HL-vs-TH contrast is what
    matters, not the raw level.
    """
    out.append("\nHOMOPHILY INDEX — same / (same + mixed) group-minute totals")
    out.append("(per child, per cluster; conditional on being in an F-formation)")
    out.append("-" * 70)
    out.append(f"{'Cluster':<10} {'HL idx':>10} {'TH idx':>10} "
               f"{'diff':>8} {'d':>8} {'p':>8}")

    d = df.copy()
    # 'same' already computed as SAMEDX_DV in load_data(); recompute for clarity
    d['same']  = d[SAMEDX_DV]
    d['mixed'] = d[MIXED_DV]

    per = (d.groupby(['SUBJECTID', 'Diagnosis', 'Cluster_ID'])
             [['same', 'mixed']].sum().reset_index())
    per['grouped'] = per['same'] + per['mixed']
    per = per[per['grouped'] > 0].copy()
    per['idx'] = per['same'] / per['grouped']

    for c in sorted(df['Cluster_ID'].unique()):
        hl = per[(per['Cluster_ID'] == c) & (per['Diagnosis'] == 'HL')]['idx']
        th = per[(per['Cluster_ID'] == c) & (per['Diagnosis'] == 'TH')]['idx']
        if len(hl) >= 2 and len(th) >= 2:
            _, p = stats.mannwhitneyu(hl, th, alternative='two-sided')
        else:
            p = np.nan
        d_eff = cohens_d(hl, th)
        hl_mean = hl.mean() if len(hl) else np.nan
        th_mean = th.mean() if len(th) else np.nan
        diff = hl_mean - th_mean if (len(hl) and len(th)) else np.nan
        out.append(f"{c:<10} {hl_mean:>10.2f} {th_mean:>10.2f} "
                   f"{diff:>+8.2f} {d_eff:>+8.2f} {p:>8.4f}")


# ==========================================
# MAIN
# ==========================================
def main():
    df = load_data()
    n_hl = df[df['Diagnosis'] == 'HL']['SUBJECTID'].nunique()
    n_th = df[df['Diagnosis'] == 'TH']['SUBJECTID'].nunique()
    n_cl = df['Cluster_ID'].nunique()

    out = []
    out.append("SOCIAL INCLUSION BY CLUSTER")
    out.append("=" * 70)
    out.append(f"Sample: {n_hl} HL + {n_th} TH children, {len(df):,} minutes, {n_cl} clusters")
    out.append("All p-values compare HL vs TH within each cluster.")

    exposure(df, out)
    inclusion(df, PHYSICAL_DV,   "Peer Co-presence",                           out)
    inclusion(df, EXPRESSIVE_DV, "Vocal Participation Rate",                   out)
    inclusion(df, SAMEDX_DV,     "Peer Affiliation: Same-Diagnosis Group Time", out)
    inclusion(df, MIXED_DV,      "Peer Affiliation: Mixed-Diagnosis Group Time", out)
    homophily_index(df, out)

    text = '\n'.join(out)
    with open(OUTPUT, 'w') as f:
        f.write(text)
    print(text)
    print(f"\nSaved to {OUTPUT}")


if __name__ == '__main__':
    main()