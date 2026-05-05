"""End-to-end pipeline runner. Skips stages whose output already exists.

Usage:
    python src/run_pipeline.py            # all stages, skip cached
    python src/run_pipeline.py --force    # force re-run
    python src/run_pipeline.py --from cluster
    python src/run_pipeline.py --only dashboard
"""

import argparse
import subprocess
import sys
import time
from pathlib import Path

SCRIPT_DIR   = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent

PIPELINE = [
    ('acoustic',       'extract_acoustic_features.py',      'acoustic_features_1min.csv'),
    ('spatial_100ms',  'extract_100ms_features.py',         'raw_spatial_100ms.csv'),
    ('spatial_1min',   'aggregate_1_min.py',                'spatial_features_1min.csv'),
    ('cluster',        'cluster_room_states.py',            'clustered_epochs_7.csv'),
    ('aggregate',      'aggregate_inclusion_by_cluster.py', 'child_inclusion_by_cluster_7.csv'),
    ('dashboard',      'build_dashboard.py',                'Inclusion_Dashboard_7.png'),
    ('stats',          'statistical_analysis.py',           'statistical_analysis_results_7.txt'),
]
NAMES = [s[0] for s in PIPELINE]


def run_step(script, output, force):
    sp, op = SCRIPT_DIR / script, PROJECT_ROOT / output
    if not sp.exists():
        print(f"   Script not found: {sp}"); return False
    if op.exists() and not force:
        print(f"Cached: {op}"); return True
    print(f"{script}")
    t0 = time.time()
    rc = subprocess.run([sys.executable, str(sp)], cwd=PROJECT_ROOT).returncode
    if rc:
        print(f"   Failed in {time.time()-t0:.1f}s (exit {rc})"); return False
    print(f"   {time.time()-t0:.1f}s"); return True


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--force', action='store_true')
    p.add_argument('--from', dest='from_step', choices=NAMES)
    p.add_argument('--only', choices=NAMES)
    args = p.parse_args()
    if args.from_step and args.only:
        p.error('--from and --only are mutually exclusive')

    if args.only:
        steps = [s for s in PIPELINE if s[0] == args.only]
    elif args.from_step:
        steps = PIPELINE[NAMES.index(args.from_step):]
    else:
        steps = PIPELINE

    print(f"INCLUSION PIPELINE  ({PROJECT_ROOT})")
    t0 = time.time()
    for i, (name, script, output) in enumerate(steps, 1):
        print(f"\n[{i}/{len(steps)}] {name.upper()}")
        if not run_step(script, output, args.force):
            print(f"\n Halted at {name}"); sys.exit(1)
    print(f"\n Pipeline complete ({time.time()-t0:.1f}s)")


if __name__ == '__main__':
    main()
