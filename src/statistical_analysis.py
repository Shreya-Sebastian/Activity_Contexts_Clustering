"""Per-cluster HL vs TH inclusion analysis: exposure (Mann-Whitney),
peer co-presence and vocal participation (LMM with Wald contrasts),
peer affiliation (same-dx, mixed-dx) and homophily index."""

import warnings

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy import stats

warnings.filterwarnings("ignore")

CLUSTERED = 'clustered_epochs_7.csv'
CONSOL    = 'data/mapping/MAPPING_CONSOLIDATED.csv'
BASE      = 'data/mapping/MAPPING_StarFish_2223_BASE_NONAMES.csv'
OUTPUT    = 'statistical_analysis_results_7.txt'

PHYSICAL_DV   = 'Time_In_Group_1min_Pct'
EXPRESSIVE_DV = 'Child_Utt_Count_1min'
SAMEDX_DV     = 'Time_In_SameDx_Pct'
MIXED_DV      = 'Time_In_Mixed_Group_1min_Pct'


def load_data():
    df = pd.read_csv(CLUSTERED)
    df['Date'] = (pd.to_datetime(df['TIME_LOCAL'], utc=True)
                    .dt.tz_convert('America/New_York').dt.date.astype(str))

    consol = pd.read_csv(CONSOL)
    consol['Date'] = pd.to_datetime(consol['Date']).dt.date.astype(str)
    att = (consol.dropna(subset=['Subject_ID'])[['Subject_ID', 'Date', 'STATUS']]
                  .rename(columns={'Subject_ID': 'SUBJECTID'}))

    diag = (pd.read_csv(BASE).dropna(subset=['Subject_ID'])[['Subject_ID', 'Diagnosis']]
              .rename(columns={'Subject_ID': 'SUBJECTID'}))
    diag = diag[diag['Diagnosis'].isin(['HL', 'TH'])]

    df = df.merge(att, on=['SUBJECTID', 'Date'], how='inner').merge(diag, on='SUBJECTID', how='inner')
    df = df[(df['STATUS'] == 'PRESENT')
            & ~df['SUBJECTID'].str.contains('_T|_Lab', case=False, na=False)]
    df['Cluster_ID'] = df['Cluster_ID'].astype(str)
    # An HL child is never in a TH-only group by construction (and vice versa),
    # so the same-dx column lets us compare HL-in-HL-group vs TH-in-TH-group.
    df[SAMEDX_DV] = np.where(df['Diagnosis'] == 'HL',
                              df['Time_In_HL_Group_1min_Pct'], df['Time_In_TH_Group_1min_Pct'])
    return df


def cohens_d(a, b):
    if len(a) < 2 or len(b) < 2:
        return np.nan
    pool = np.sqrt(((len(a)-1)*a.var(ddof=1) + (len(b)-1)*b.var(ddof=1)) / (len(a)+len(b)-2))
    return (a.mean() - b.mean()) / pool if pool > 0 else np.nan


def _split(per, c, col='Cluster_ID'):
    return (per[(per[col] == c) & (per['Diagnosis'] == 'HL')],
            per[(per[col] == c) & (per['Diagnosis'] == 'TH')])


def exposure(df, out):
    out += ["", "EXPOSURE — % of each child's day per cluster (Mann-Whitney)",
            "-" * 70, f"{'Cluster':<10} {'HL %':>8} {'TH %':>8} {'diff':>8} {'d':>8} {'p':>8}"]
    per = df.groupby(['SUBJECTID', 'Diagnosis', 'Cluster_ID']).size().reset_index(name='n')
    per = per.merge(df.groupby('SUBJECTID').size().rename('tot'), on='SUBJECTID')
    per['pct'] = per['n'] / per['tot'] * 100
    for c in sorted(df['Cluster_ID'].unique()):
        hl, th = _split(per, c)
        hp, tp = hl['pct'], th['pct']
        p = stats.mannwhitneyu(hp, tp, alternative='two-sided').pvalue if len(hp) >= 2 and len(tp) >= 2 else np.nan
        out.append(f"{c:<10} {hp.mean():>8.1f} {tp.mean():>8.1f} "
                   f"{hp.mean() - tp.mean():>+8.1f} {cohens_d(hp, tp):>+8.2f} {p:>8.4f}")


def inclusion(df, dv, label, out):
    out += ["", f"{label.upper()} — {dv}",
            f"LMM: {dv} ~ C(Cluster_ID) * C(Diagnosis) + (1 | SUBJECTID)",
            "-" * 70, f"{'Cluster':<10} {'HL':>8} {'TH':>8} {'diff':>8} {'d':>8} {'p':>8}"]
    d = df.dropna(subset=[dv]).copy()
    d['Diagnosis'] = pd.Categorical(d['Diagnosis'], categories=['TH', 'HL'])
    try:
        model = smf.mixedlm(f"{dv} ~ C(Cluster_ID) * C(Diagnosis)",
                            data=d, groups=d['SUBJECTID']).fit(reml=True)
    except Exception as e:
        out.append(f"  Model failed: {e}"); return

    cm = d.groupby(['SUBJECTID', 'Diagnosis', 'Cluster_ID'])[dv].mean().reset_index()
    params = list(model.fe_params.index)
    clusters = sorted(d['Cluster_ID'].unique())
    ref = clusters[0]
    for c in clusters:
        contrast = np.zeros(len(params))
        contrast[params.index('C(Diagnosis)[T.HL]')] = 1
        if c != ref:
            inter = f'C(Cluster_ID)[T.{c}]:C(Diagnosis)[T.HL]'
            if inter in params:
                contrast[params.index(inter)] = 1
        t = model.t_test(contrast.reshape(1, -1))
        hl, th = _split(cm, c)
        out.append(f"{c:<10} {hl[dv].mean():>8.2f} {th[dv].mean():>8.2f} "
                   f"{float(t.effect[0]):>+8.2f} {cohens_d(hl[dv], th[dv]):>+8.2f} {float(t.pvalue):>8.4f}")


def homophily_index(df, out):
    """same / (same + mixed) per (child, cluster), conditional on being grouped.
    1.0 = always with same-diagnosis peers; 0.0 = always mixed."""
    out += ["", "HOMOPHILY INDEX — same / (same + mixed) group-minute totals",
            "(per child, per cluster; conditional on being in an F-formation)",
            "-" * 70, f"{'Cluster':<10} {'HL idx':>10} {'TH idx':>10} {'diff':>8} {'d':>8} {'p':>8}"]
    per = (df.assign(same=df[SAMEDX_DV], mixed=df[MIXED_DV])
             .groupby(['SUBJECTID', 'Diagnosis', 'Cluster_ID'])[['same', 'mixed']].sum().reset_index())
    per['grouped'] = per['same'] + per['mixed']
    per = per[per['grouped'] > 0].copy()
    per['idx'] = per['same'] / per['grouped']
    for c in sorted(df['Cluster_ID'].unique()):
        hl, th = _split(per, c)
        hi, ti = hl['idx'], th['idx']
        p = stats.mannwhitneyu(hi, ti, alternative='two-sided').pvalue if len(hi) >= 2 and len(ti) >= 2 else np.nan
        out.append(f"{c:<10} {hi.mean():>10.2f} {ti.mean():>10.2f} "
                   f"{hi.mean() - ti.mean():>+8.2f} {cohens_d(hi, ti):>+8.2f} {p:>8.4f}")


def main():
    df = load_data()
    n_hl = df[df['Diagnosis'] == 'HL']['SUBJECTID'].nunique()
    n_th = df[df['Diagnosis'] == 'TH']['SUBJECTID'].nunique()
    out = ["SOCIAL INCLUSION BY CLUSTER", "=" * 70,
           f"Sample: {n_hl} HL + {n_th} TH children, {len(df):,} minutes, "
           f"{df['Cluster_ID'].nunique()} clusters",
           "All p-values compare HL vs TH within each cluster."]
    exposure(df, out)
    inclusion(df, PHYSICAL_DV,   "Peer Co-presence",                            out)
    inclusion(df, EXPRESSIVE_DV, "Vocal Participation Rate",                    out)
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