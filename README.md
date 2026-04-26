<!--
DRAFT README for: https://github.com/Shreya-Sebastian/Activity_Contexts_Clustering
Currently has NO README on GitHub.

Drafted from the written thesis:
  Discovering Latent Interaction Contexts from Multimodal Behavioral Data
  for Social Inclusion Analysis (Shreya Sebastian, 2026-04-25, 38 pp + bib + appendix)

Items in [BRACKETS] need Shreya to confirm or fill in based on the actual repo
contents (entry-point file names, license, whether the thesis PDF will be
committed alongside the code, dataset access policy).

ALSO: set the repo description on GitHub (gear icon → "About"). Suggested:

    MSc thesis pipeline (TU Delft, 2026): GMM-based discovery of activity
    contexts from UWB + LENA data, with per-context HL/TH inclusion analysis.

This description is the highest-leverage change because it shows up next to
the repo name on the profile listing, where most reviewers stop scrolling.
-->

# Activity-Context Clustering for Social-Inclusion Analysis

> MSc thesis pipeline, TU Delft (2026). Discovers latent activity contexts in an inclusive preschool from fused ultra-wideband (UWB) spatial tracking and LENA acoustic data, then compares children with hearing loss (HL) and typically hearing (TH) peers on three sensor-derivable behavioral markers of inclusion *within each context*.

Companion thesis: *Discovering Latent Interaction Contexts from Multimodal Behavioral Data for Social Inclusion Analysis* — Shreya Sebastian, 2026.

## TL;DR

A Gaussian Mixture Model on four room-level features (adult word count, auditory overlap, cumulative displacement, teacher distance) recovers seven interpretable activity contexts in a single inclusive preschool (13 children, 6 HL + 7 TH). A linear mixed model with a per-child random intercept then tests HL vs TH on **peer co-presence**, **vocal participation rate**, and **peer affiliation patterns** within each context. The HL/TH signal is **not uniform**: it lives in specific cluster-by-diagnosis interactions and disappears when averaged across contexts.

## Why this problem

In inclusive classrooms, the physical presence of HL children does not guarantee social inclusion with TH peers. Self-report is biased and manual observation is too labor-intensive for fine-grained, full-day coverage. Wearable sensing (UWB position, child-worn LENA audio) can record continuous co-presence and vocal activity, but the meaning of those signals depends on what the *room* is doing — a quiet sedentary minute and a free-play minute are not comparable. The pipeline turns continuous sensor streams into (child, minute) rows labeled with an activity context, then tests inclusion *within* context.

## Research questions

> **RQ1.** Can a set of interpretable latent activity contexts in an inclusive classroom be recovered from fused UWB spatial data and LENA acoustic data using unsupervised clustering?
>
> **RQ2.** Within each recovered activity context, do HL and TH children differ on three sensor-derivable behavioral markers of inclusion: (a) peer co-presence, (b) vocal participation rate, (c) peer affiliation patterns?

The hypotheses follow modality-mismatch literature: HL children are expected to compensate for reduced access to rapid verbal exchange by remaining physically co-present while under-producing vocally, with the gap concentrating in contexts that combine high peer communicative demand with reduced adult scaffolding.

## Pipeline

```
UWB (10 Hz x, y, θ)        LENA (timestamped speaker-type segments)
        │                              │
        ▼                              ▼
 per-frame displacement,         proportional binning of
 F-formation membership          AWC / overlap / utterance counts
 via Dominant Sets [Hung &       to 1-min epochs
 Kröse 2011] on socio-spatial
 affinity (proximity × mutual                  │
 orientation, σ=0.5 m)                         ▼
        │                          per-child, per-minute
        ▼                          acoustic features
 per-child, per-minute                         │
 spatial features                              │
        │                                      │
        └──────────────► inner-join on (child, minute) ──────────────┐
                                                                     │
                                                                     ▼
                                          room-level averaging across children
                                                  (4 features:
                                                   AWC, auditory overlap,
                                                   displacement, teacher dist.)
                                                                     │
                                                                     ▼
                                              Yeo–Johnson power transform
                                                                     │
                                                                     ▼
                                              GMM, K=7, full covariance
                                                                     │
                                                                     ▼
                                       cluster label per (child, minute)
                                                                     │
                                                                     ▼
                                  LMM(outcome ~ cluster × HL/TH + (1|child))
                                  + Mann–Whitney U on per-child homophily
                                                                     │
                                                                     ▼
                                          per-context HL/TH contrasts
```

`K = 7` is fixed *a priori* from manual coding of this classroom's daily routine; cluster *assignment* and *profile* are recovered from sensor data. This makes the pipeline **semi-unsupervised** rather than fully unsupervised.

## Recovered activity contexts

| Cluster | Label                              | Approx. share | Sensor signature                                  |
|---------|------------------------------------|---------------|---------------------------------------------------|
| 0       | Teacher-proximal group activity    | ~25%          | High AWC, lowest teacher distance, low movement   |
| 1       | Teacher-led direct instruction     | ~14%          | Highest AWC, low overlap                          |
| 2       | Quiet sedentary activity           | ~6%           | Lowest auditory overlap, low utterances           |
| 3       | Structured seated conversation     | ~10%          | Near-zero displacement, moderate overlap          |
| 4       | Active peer interaction            | ~16%          | High displacement, moderate teacher distance      |
| 5       | Free play                          | ~22%          | High overlap, moderate-to-high displacement       |
| 6       | Dispersed high-movement transition | ~6%           | Min AWC, max overlap/displacement/teacher dist.   |

All seven components are interpretable as recognizable phases of an inclusive preschool day. The exposure check (Table 6.1 in the thesis) finds no significant HL/TH imbalance across contexts.

## Headline findings

| Marker                  | Pattern across the 7 contexts                                                                                       |
|-------------------------|---------------------------------------------------------------------------------------------------------------------|
| Peer co-presence        | No HL/TH difference in 6 of 7 contexts. HL > TH **only** in quiet sedentary (Cluster 2): +8.6 pp, *d* = 1.74, *p* < 0.0001. |
| Vocal participation     | HL deficit concentrated in active peer interaction, free play, and dispersed transition (all *p* < 0.05, *d* ≈ 0.7–0.9). Same direction but non-significant in the four teacher-anchored contexts. |
| Peer affiliation        | HL homophily index < TH homophily in **all 7** contexts (significantly in 4). HL children spend more grouped time in mixed-diagnosis F-formations than TH children across all 7 contexts (significantly in 6). The most consistent of the three markers. |

The pooled (across-context) HL/TH effect for peer co-presence is near-zero (*d* = +0.04). This is not because the signal is absent — it is because context-conditioning surfaces structure that unconditional comparison averages away.

## What I'd do differently

- **`K` was fixed from manual coding, not selected from the data.** A BIC sweep on held-out days, or replication on an independently coded cohort, would separate the manual-coding prior from the empirical structure.
- **The Dominant Sets F-formation extractor is not currently seeded** (random initialization on the simplex), so the spatial-feature table is stable up to convergence but not bit-for-bit reproducible. A one-line seed fix is flagged in the thesis (§7.3) and not yet applied.
- **F-formation affinity uses a single shoulder-orientation vector**, not a frustum/cone of attention as the F-formation literature recommends. In young children, shoulder orientation can dissociate from gaze, so the peer-co-presence measure carries an unknown amount of false-positive and false-negative group membership.
- **No multiple-comparisons correction** is applied to the 42 tests reported in Chapter 6. The findings most likely to survive correction are those at *p* < 0.001 (Cluster 2 peer co-presence; Clusters 3, 4, 5 same-diagnosis time; Clusters 0, 2, 3, 5 mixed-diagnosis time). Findings near *p* = 0.05 (Clusters 4, 5, 6 in vocal participation; Cluster 2 in same-diagnosis time) should be read with caution.
- **Cluster labels are weakly circular.** Each child contributes ~1/13 (~8%) of the room average that determines the cluster label for that minute, so a child's own behavior weakly influences the context within which their HL/TH comparison is made. The effect is diluted across 13 children but not eliminated.

## Limitations

- **Single setting, N = 13.** Both the seven-context structure and the directions of HL/TH differences may be specific to this classroom, age group, and curriculum. Multi-site replication is required before treating these findings as general.
- **The Cluster 2 HL-above-TH peer-co-presence effect** is large (*d* = 1.74) but rests on ~6% of the observed day. It is the most fragile of the reported findings.
- **The three markers are *behavioral correlates* of inclusion, not inclusion itself.** Belonging, acceptance, and response contingency (the developmental-literature constructs) require behaviors the pipeline does not capture. Peer affiliation tracks cross-group contact (a necessary but not sufficient precondition for sociometric inclusion); peer co-presence and vocal participation are more distant proxies.
- **The peer-affiliation gap is partly baseline-driven.** With 6 HL and 7 TH children, a TH child has 6/12 same-diagnosis classmates while an HL child has 5/12, giving the TH homophily index a built-in advantage of ~0.08 under purely random dyadic affiliation. The observed gap of 0.11–0.12 in the four significant clusters is only ~0.03–0.04 above this baseline.

## Reference pipeline

This work builds directly on two earlier artifacts from the same research group, referred to in the thesis as the *reference pipeline*:

- [TUDelft-SPC-Lab/group-detection](https://github.com/TUDelft-SPC-Lab/group-detection) — Dominant Sets F-formation extractor, adopted unchanged.
- [TUDelft-SPC-Lab/ICDL2025](https://github.com/TUDelft-SPC-Lab/ICDL2025) — socio-spatial affinity parameterization (proximity Gaussian × clipped mutual-orientation product, σ = 0.5 m), adopted unchanged.

The contributions of *this* repository relative to the reference pipeline are: (i) GMM-based unsupervised discovery of latent activity contexts from room-level features, (ii) a per-cluster linear mixed model with a per-child random intercept, and (iii) the three-marker decomposition of inclusion into peer co-presence, vocal participation rate, and peer affiliation patterns.

## Software

- **Python** 3.10+
- **Scientific stack:** *pandas*, *NumPy*, *SciPy* (non-parametric tests), *scikit-learn* (Yeo–Johnson, GMM), *statsmodels* (LMM, REML)
- **Plots:** *matplotlib*, *seaborn*

[CONFIRM: pin exact versions in `requirements.txt` or `environment.yml` if not already.]

## Reproducibility

- The GMM is fit with a fixed random seed and 10 random restarts → bit-for-bit reproducible on the same input.
- The Dominant Sets iteration is **not currently seeded** (flagged in thesis §7.3). The minute-level spatial-feature table is stable up to convergence but not bit-for-bit reproducible across independent runs.
- All numerical results in Chapter 6 are reported from a single run.
- Statistical analysis is restricted to (child, day) pairs marked present in the daily attendance record (21,650 of 22,405 child-minutes).

## Data

The dataset is **not redistributable**: child-worn audio (LENA) and indoor-tracking data of minors collected under a research-ethics protocol that does not permit public release. The repository contains the analysis pipeline, not the data.

[CONFIRM: link to the data-access procedure if applicable — e.g. via the SPC Lab at TU Delft, or note "data available on reasonable request to the authors".]

## Repository layout

[CONFIRM: list the actual top-level files/dirs once the repo is finalized. Likely shape:

- `pipeline/` or top-level `.py` files — feature extraction (UWB, LENA), fusion, GMM clustering, LMM analysis
- `notebooks/` — exploration, figure generation
- `requirements.txt` or `environment.yml` — pinned dependencies
- `figures/` — generated figures referenced in the thesis
- `thesis.pdf` — full written thesis [CONFIRM whether this will be committed alongside the code]
]

## AI-tool disclosure

LLM assistance was used during preparation of the written thesis for editing and structuring prose. No LLM was used to generate, interpret, or substitute for the statistical analyses themselves.

## License

[CONFIRM — likely MIT or Apache 2.0 for the analysis code. The data is governed by a separate ethics protocol and is not under the same license.]

## Acknowledgements

[CONFIRM — thesis advisors, the SPC Lab at TU Delft, the participating preschool. Cite the reference-pipeline authors explicitly.]
