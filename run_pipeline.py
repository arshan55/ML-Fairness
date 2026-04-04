"""
run_pipeline.py
Orchestrates the entire fairness pipeline from data fetching to mitigation.
"""
import os
import sys

def run_step(script_name):
    print(f"\n{'='*50}")
    print(f"Executing: {script_name}")
    print(f"{'='*50}")
    retval = os.system(f'"{sys.executable}" {script_name}')
    if retval != 0:
        print(f"Error executing {script_name}.")
        sys.exit(1)

if __name__ == "__main__":
    base_dir = os.path.dirname(__file__)
    
    scripts = [
        os.path.join(base_dir, 'datasets', 'fetch_data.py'),
        os.path.join(base_dir, 'src', '01_preprocessing_and_symptoms.py'),
        os.path.join(base_dir, 'src', '02_baseline_training.py'),
        os.path.join(base_dir, 'src', '03_fairness_evaluation.py'),
        os.path.join(base_dir, 'src', '04_mitigation.py'),
        os.path.join(base_dir, 'src', '05_paper_metrics_exporter.py'),
        os.path.join(base_dir, 'src', '06_journal_visualizations.py') # Journal format outputs
    ]
    
    for script in scripts:
        run_step(script)
        
    print("\nPipeline execution complete!")
    print(f"Check the '{os.path.join(base_dir, 'results')}' directory for your research paper CSVs and tables.")
