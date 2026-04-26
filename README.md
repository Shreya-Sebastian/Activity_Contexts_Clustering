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

## Software

- **Python** 3.10+
- **Scientific stack:** *pandas*, *NumPy*, *SciPy*, *scikit-learn*, *statsmodels*
- **Plots:** *matplotlib*, *seaborn*

## Reproducibility

- The GMM is fit with a fixed random seed and 10 random restarts → bit-for-bit reproducible on the same input.
- The Dominant Sets iteration is **not currently seeded** (flagged in thesis §7.3). The minute-level spatial-feature table is stable up to convergence but not bit-for-bit reproducible across independent runs.
- All numerical results in Chapter 6 are reported from a single run.
- Statistical analysis is restricted to (child, day) pairs marked present in the daily attendance record (21,650 of 22,405 child-minutes).

## Data

The dataset is **not redistributable**: child-worn audio (LENA) and indoor-tracking data of minors collected under a research-ethics protocol that does not permit public release. The repository contains the analysis pipeline, not the data.


## Acknowledgements
Thesis supervised at **TU Delft** by **Hayley Hung** and **Stephanie Tan** (Socially Perceptive Computing Lab), with external supervision from **Daniel Messinger** and **Lynn Perry** (University of Miami).

The reference pipeline this work builds on is the product of two prior contributions from the same group:

- [TUDelft-SPC-Lab/group-detection](https://github.com/TUDelft-SPC-Lab/group-detection) — Dominant Sets F-formation extractor, by **Stephanie Tan**.
- [TUDelft-SPC-Lab/ICDL2025](https://github.com/TUDelft-SPC-Lab/ICDL2025) — socio-spatial affinity parameterization, by **Yuan Tian**
