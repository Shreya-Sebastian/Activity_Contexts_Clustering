"""
End-to-end pipeline runner.

Assumed layout:
    project_root/
        data/               <-- input data (lena, ubisense, mapping)
        acoustic_features_1min.csv, ...   <-- outputs produced here
        src/
            run_pipeline.py               <-- this file
            extract_acoustic_features.py
            extract_spatial_features.py
            cluster_room_states.py
            aggregate_inclusion_by_cluster.py
            build_dashboard.py
            statistical_analysis.py
            dominant_sets.py
            scale_epochs.py

Each stage script is launched with cwd = project_root, so its relative
paths (like 'data/mapping/...' and 'acoustic_features_1min.csv') resolve
against the project root, not against src/.

By default, skips any stage whose output already exists. Use --force to
re-run everything, or --from STEP to resume from a specific stage.

Usage (from anywhere):
    python src/run_pipeline.py                 # run everything, skip cached
    python src/run_pipeline.py --force         # re-run everything
    python src/run_pipeline.py --from cluster  # resume from clustering
    python src/run_pipeline.py --only dashboard
"""

import argparse
import subprocess
import sys
import time
from pathlib import Path

# Scripts live next to this runner (src/), data and outputs live one level up.
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent

# (short_name, script_filename, expected_output)
PIPELINE = [
    ('acoustic',  'extract_acoustic_features.py',      'acoustic_features_1min.csv'),
    ('spatial',   'extract_spatial_features.py',       'spatial_features_1min.csv'),
    ('cluster',   'cluster_room_states.py',            'clustered_epochs.csv'),
    ('aggregate', 'aggregate_inclusion_by_cluster.py', 'child_inclusion_by_cluster.csv'),
    ('dashboard', 'build_dashboard.py',                'Inclusion_Dashboard.png'),
    ('stats',     'statistical_analysis.py',           'statistical_analysis_results.txt'),
]

STEP_NAMES = [s[0] for s in PIPELINE]


def banner(text, char='='):
    line = char * 70
    print(f"\n{line}\n  {text}\n{line}")


def run_step(script, output, force):
    """Run one pipeline script. Returns True on success."""
    script_path = SCRIPT_DIR / script
    output_path = PROJECT_ROOT / output

    if not script_path.exists():
        print(f"  ❌ Script not found: {script_path}")
        return False

    if output_path.exists() and not force:
        print(f"  ⏭  Skipping — output already exists: {output_path}")
        print(f"     (use --force to re-run)")
        return True

    print(f"  ▶ Running: {script}")
    print(f"     script : {script_path}")
    print(f"     cwd    : {PROJECT_ROOT}")
    t0 = time.time()
    result = subprocess.run(
        [sys.executable, str(script_path)],
        cwd=PROJECT_ROOT,
        check=False,
    )
    elapsed = time.time() - t0

    if result.returncode != 0:
        print(f"  ❌ Failed after {elapsed:.1f}s (exit code {result.returncode})")
        return False

    print(f"  ✅ Finished in {elapsed:.1f}s")
    return True


def resolve_steps(args):
    """Figure out which steps to run based on CLI args."""
    if args.only:
        if args.only not in STEP_NAMES:
            print(f"Unknown step: {args.only}. Valid: {STEP_NAMES}")
            sys.exit(2)
        return [s for s in PIPELINE if s[0] == args.only]

    if args.from_step:
        if args.from_step not in STEP_NAMES:
            print(f"Unknown step: {args.from_step}. Valid: {STEP_NAMES}")
            sys.exit(2)
        start = STEP_NAMES.index(args.from_step)
        return PIPELINE[start:]

    return PIPELINE


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--force', action='store_true',
                        help='Re-run all steps even if outputs exist')
    parser.add_argument('--from', dest='from_step', metavar='STEP', choices=STEP_NAMES,
                        help=f"Start from this step. Choices: {STEP_NAMES}")
    parser.add_argument('--only', metavar='STEP', choices=STEP_NAMES,
                        help=f"Run only this step. Choices: {STEP_NAMES}")
    args = parser.parse_args()

    if args.from_step and args.only:
        parser.error('--from and --only are mutually exclusive')

    steps = resolve_steps(args)

    banner('INCLUSION PIPELINE')
    print(f"  Script dir   : {SCRIPT_DIR}")
    print(f"  Project root : {PROJECT_ROOT}")
    print(f"  Steps to run : {[s[0] for s in steps]}")
    print(f"  Force re-run : {args.force}")

    pipeline_t0 = time.time()

    for i, (name, script, output) in enumerate(steps, 1):
        banner(f"[{i}/{len(steps)}]  {name.upper()}  ({script})", char='-')
        ok = run_step(script, output, args.force)
        if not ok:
            banner('❌ PIPELINE HALTED', char='=')
            print(f"  Failed at step: {name}")
            sys.exit(1)

    total = time.time() - pipeline_t0
    banner(f'✅ PIPELINE COMPLETE  ({total:.1f}s total)', char='=')


if __name__ == '__main__':
    main()