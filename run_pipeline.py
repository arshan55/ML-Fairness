"""
run_pipeline.py
===============
Orchestrates both fairness pipeline domains.

Usage:
  python run_pipeline.py                  # runs both domains
  python run_pipeline.py --domain salary      # salary / gender bias only
  python run_pipeline.py --domain healthcare  # healthcare / race bias only
"""
import os
import sys
import argparse
import subprocess


def run_step(script_name):
    print(f"\n{'='*60}")
    print(f" Executing: {os.path.relpath(script_name)}")
    print(f"{'='*60}")
    # subprocess.call handles paths with spaces correctly (no shell quoting issues)
    retval = subprocess.call([sys.executable, script_name])
    if retval != 0:
        print(f"[ERROR] Failed: {script_name}")
        sys.exit(1)


def get_salary_scripts(base_dir):
    src = os.path.join(base_dir, 'src')
    return [
        os.path.join(base_dir, 'datasets', 'fetch_data.py'),
        os.path.join(src, '01_preprocessing_and_symptoms.py'),
        os.path.join(src, '02_baseline_training.py'),
        os.path.join(src, '03_fairness_evaluation.py'),
        os.path.join(src, '04_mitigation.py'),
        os.path.join(src, '05_paper_metrics_exporter.py'),
        os.path.join(src, '06_journal_visualizations.py'),
    ]


def get_healthcare_scripts(base_dir):
    hc = os.path.join(base_dir, 'src', 'healthcare')
    return [
        os.path.join(hc, '01_healthcare_preprocessing.py'),
        os.path.join(hc, '02_healthcare_baseline.py'),
        os.path.join(hc, '03_healthcare_fairness_eval.py'),
        os.path.join(hc, '04_healthcare_mitigation.py'),
        os.path.join(hc, '05_healthcare_visualizations.py'),
    ]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ML Fairness Pipeline Runner")
    parser.add_argument(
        '--domain',
        choices=['salary', 'healthcare', 'all'],
        default='all',
        help="Which bias domain to run (default: all)"
    )
    args = parser.parse_args()

    base_dir = os.path.dirname(os.path.abspath(__file__))
    scripts  = []

    if args.domain in ('salary', 'all'):
        print("\n" + "#"*60)
        print("  DOMAIN 1: SALARY BIAS  (Gender | UCI Adult Income)")
        print("#"*60)
        scripts += get_salary_scripts(base_dir)

    if args.domain in ('healthcare', 'all'):
        print("\n" + "#"*60)
        print("  DOMAIN 2: HEALTHCARE BIAS  (Race | Insurance Charges)")
        print("#"*60)
        scripts += get_healthcare_scripts(base_dir)

    for script in scripts:
        run_step(script)

    print("\n" + "="*60)
    print(" Pipeline complete!")
    print(f" Results -> {os.path.join(base_dir, 'results')}")
    print("="*60)
