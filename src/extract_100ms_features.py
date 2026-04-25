"""
Extract 100ms spatial features (velocity, teacher distance, F-formation
membership) from Ubisense tracker data. Output is a per-frame, per-child
checkpoint CSV that downstream scripts can roll up into any epoch size.

Running this script overwrites CHECKPOINT_RAW.
"""

import os
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
from tqdm import tqdm

import dominant_sets

warnings.filterwarnings("ignore")

# ==========================================
# 1. CONFIGURATION
# ==========================================
MAPPING_FILE = 'data/mapping/MAPPING_StarFish_2223_BASE_NONAMES.csv'
UBISENSE_DIR = 'data/ubisense/'

CHECKPOINT_RAW = 'raw_spatial_100ms.csv'

SPATIAL_SIGMA = 0.5      # proximity decay for F-formation affinity
VELOCITY_GAP_S = 5.0     # zero out velocity across gaps longer than this
FLUSH_EVERY_N_FRAMES = 1000
DEBUG_MODE = False       # if True, only process the first ubisense file


# ==========================================
# 2. F-FORMATION AFFINITY
# ==========================================
def build_affinity(positions, orientations_deg, spatial_sigma=0.5):
    """Socio-spatial affinity matrix from proximity x mutual facing."""
    n = len(positions)
    A = np.zeros((n, n))
    orientations_rad = np.deg2rad(orientations_deg)

    for i in range(n):
        xi, yi = positions[i]
        oi = orientations_rad[i]
        for j in range(n):
            if i == j:
                continue
            xj, yj = positions[j]
            oj = orientations_rad[j]

            dist = np.sqrt((xi - xj) ** 2 + (yi - yj) ** 2)
            spatial = np.exp(-(dist ** 2) / (2 * spatial_sigma ** 2))

            vec_ij = np.array([xj - xi, yj - yi])
            norm = np.linalg.norm(vec_ij)
            if norm == 0:
                continue
            vec_ij /= norm

            dir_i = np.array([np.cos(oi), np.sin(oi)])
            dir_j = np.array([np.cos(oj), np.sin(oj)])

            facing_i = np.dot(dir_i, vec_ij)
            facing_j = np.dot(dir_j, -vec_ij)
            mutual = max(0, facing_i) * max(0, facing_j)

            A[i, j] = spatial * mutual
    return A


# ==========================================
# 3. UBISENSE PIPELINE
# ==========================================
def process_ubisense(ubisense_file_path):
    """Read raw Ubisense CSV and snap timestamps to a 100ms grid."""
    df = pd.read_csv(ubisense_file_path, header=0)
    df['TIME'] = pd.to_datetime(df['TIME'])
    df['TIME_LOCAL'] = df['TIME'].dt.tz_localize(
        'America/New_York', ambiguous='NaT', nonexistent='shift_forward'
    )
    df['TIME_LOCAL'] = df['TIME_LOCAL'].dt.round('100ms')
    df = df.drop_duplicates(subset=['SUBJECTID', 'TIME_LOCAL'])
    return df.dropna(subset=['KC_X'])


def _flush(buffer, path):
    """Append a buffer of result dicts to the checkpoint CSV."""
    if not buffer:
        return
    pd.DataFrame(buffer).to_csv(
        path, mode='a',
        header=not os.path.exists(path),
        index=False,
    )


def extract_spatial_features(df, subject_to_diagnosis):
    """Run F-formation detection per 100ms frame and stream to checkpoint."""
    df = df.sort_values(['SUBJECTID', 'TIME_LOCAL'])
    df['is_teacher'] = df['SUBJECTID'].str.contains('_T|_Lab', case=False, na=False)

    # Velocity (zero out long gaps)
    print("Pre-calculating velocities...")
    df['prev_x'] = df.groupby('SUBJECTID')['KC_X'].shift(1)
    df['prev_y'] = df.groupby('SUBJECTID')['KC_Y'].shift(1)
    df['prev_time'] = df.groupby('SUBJECTID')['TIME_LOCAL'].shift(1)

    dist = np.sqrt((df['KC_X'] - df['prev_x']) ** 2 + (df['KC_Y'] - df['prev_y']) ** 2)
    time_gaps = (df['TIME_LOCAL'] - df['prev_time']).dt.total_seconds()
    df['velocity'] = pd.Series(
        np.where(time_gaps > VELOCITY_GAP_S, 0.0, dist), index=df.index
    ).fillna(0)

    # Fresh checkpoint
    if os.path.exists(CHECKPOINT_RAW):
        os.remove(CHECKPOINT_RAW)

    # Per-frame F-formation detection
    results_buffer = []
    df_sorted = df.sort_values('TIME_LOCAL')

    for i, (time, group) in enumerate(tqdm(
        df_sorted.groupby('TIME_LOCAL'),
        desc="Calculating 100ms F-Formations",
    )):
        teachers = group[group['is_teacher']][['KC_X', 'KC_Y']].values
        children = group[~group['is_teacher']]
        child_coords = children[['KC_X', 'KC_Y']].values
        child_oris = children['KC_O'].values
        child_ids = children['SUBJECTID'].values
        child_vels = children['velocity'].values

        if len(child_coords) == 0:
            continue

        # Teacher proximity
        if len(teachers) > 0:
            min_teacher_dist = cdist(child_coords, teachers, metric='euclidean').min(axis=1)
        else:
            min_teacher_dist = np.full(len(child_coords), np.nan)

        in_group, in_hl, in_th, in_mixed = (
            np.zeros(len(child_coords), dtype=int) for _ in range(4)
        )

        if len(child_coords) > 1:
            A = build_affinity(child_coords, child_oris, SPATIAL_SIGMA)
            if np.max(A) > 0.01:
                groups = dominant_sets.dominant_set_extraction(A, len(child_coords))
                for mask in groups:
                    if np.sum(mask) >= 2:
                        in_group[mask] = 1
                        members = child_ids[mask]

                        # Classify group by demographic composition
                        diags = set(
                            subject_to_diagnosis.get(m, 'Unknown') for m in members
                        ) - {'Unknown'}
                        if len(diags) == 0:
                            in_mixed[mask] = 1
                        elif diags == {'HL'}:
                            in_hl[mask] = 1
                        elif diags == {'TH'}:
                            in_th[mask] = 1
                        else:
                            in_mixed[mask] = 1

        for idx, subj in enumerate(child_ids):
            results_buffer.append({
                'TIME_LOCAL': time,
                'SUBJECTID': subj,
                'velocity': child_vels[idx],
                'teacher_dist': min_teacher_dist[idx],
                'is_in_group': in_group[idx],
                'is_hl_only_group': in_hl[idx],
                'is_th_only_group': in_th[idx],
                'is_mixed_group': in_mixed[idx],
            })

        # Live flush so the run is crash-recoverable and memory stays bounded
        if i % FLUSH_EVERY_N_FRAMES == 0 and i > 0:
            _flush(results_buffer, CHECKPOINT_RAW)
            results_buffer = []

    _flush(results_buffer, CHECKPOINT_RAW)


# ==========================================
# 4. MAIN
# ==========================================
if __name__ == "__main__":
    print(f"\n{'=' * 50}\n 100ms SPATIAL FEATURE EXTRACTION\n{'=' * 50}")

    try:
        mapping_df = pd.read_csv(MAPPING_FILE)
        subject_to_diagnosis = {
            row['Subject_ID']: row['Diagnosis']
            for _, row in mapping_df.dropna(subset=['Subject_ID']).iterrows()
        }
    except Exception as e:
        print(f"Error loading mapping: {e}")
        raise SystemExit(1)

    ubi_files = list(Path(UBISENSE_DIR).rglob('*.csv'))
    if DEBUG_MODE:
        ubi_files = ubi_files[:1]

    all_ubi = []
    for f in tqdm(ubi_files, desc="Loading Raw CSVs"):
        chunk = process_ubisense(f)
        if not chunk.empty:
            all_ubi.append(chunk)

    if not all_ubi:
        print("No data found.")
        raise SystemExit(1)

    extract_spatial_features(pd.concat(all_ubi), subject_to_diagnosis)
    print(f"\nDone. Saved: {CHECKPOINT_RAW}")