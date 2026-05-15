"""
05_healthcare_visualizations.py - Healthcare Domain
====================================================
Generates 4 publication-quality figures for the healthcare bias case study.
Mirrors the visual format of the salary-domain journal visualizations for
side-by-side comparison across both domains in the research paper.

Figures produced:
  HC_Fig1 — Accuracy vs Fairness Trade-off Bar Chart (Baseline RF vs Fair NN)
  HC_Fig2 — ROC Curves (Baseline vs Fair model)
  HC_Fig3 — Racial Disparity Confusion Matrices (side-by-side: White / Non-White)
  HC_Fig4 — Prediction Probability Violin Plot by Race (Baseline vs Fair NN)
"""

import os
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from sklearn.metrics import roc_curve, auc, confusion_matrix
from sklearn.model_selection import train_test_split

RESULTS_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
    'results', 'healthcare'
)
os.makedirs(RESULTS_DIR, exist_ok=True)

# ── Shared style ─────────────────────────────────────────────────────────────
PALETTE = {
    'unfair_acc':  '#4A90D9',
    'unfair_bias': '#E74C3C',
    'fair_acc':    '#27AE60',
    'fair_bias':   '#F39C12',
    'white':       '#5B8FF9',
    'nonwhite':    '#FF6B6B',
}

plt.rcParams.update({
    'font.family':      'DejaVu Sans',
    'axes.spines.top':  False,
    'axes.spines.right': False,
    'figure.dpi':       150,
})


# ─────────────────────────────────────────────────────────────────────────────
# Figure 1 — Accuracy vs Fairness Trade-off
# ─────────────────────────────────────────────────────────────────────────────
def plot_hc_tradeoff(baseline_acc, baseline_dpd,
                      fair_acc,     fair_dpd):
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    fig.suptitle('Healthcare Insurance Bias: Accuracy vs Fairness Trade-off\n'
                 'Protected Attribute: Race (White vs Non-White)',
                 fontsize=13, fontweight='bold', y=1.02)

    models = ['Baseline RF', 'Fair NN']
    accs   = [baseline_acc, fair_acc]
    dpds   = [abs(baseline_dpd), abs(fair_dpd)]

    # Accuracy
    bars = axes[0].bar(models, accs,
                        color=[PALETTE['unfair_acc'], PALETTE['fair_acc']],
                        width=0.4, edgecolor='white', linewidth=1.5)
    axes[0].set_ylim(0, 1.0)
    axes[0].set_ylabel('Accuracy', fontsize=11)
    axes[0].set_title('Classification Accuracy', fontsize=11)
    for bar, v in zip(bars, accs):
        axes[0].text(bar.get_x() + bar.get_width()/2, v + 0.01,
                     f'{v:.3f}', ha='center', va='bottom', fontsize=10, fontweight='bold')

    # DPD (Bias)
    bars2 = axes[1].bar(models, dpds,
                         color=[PALETTE['unfair_bias'], PALETTE['fair_bias']],
                         width=0.4, edgecolor='white', linewidth=1.5)
    axes[1].set_ylim(0, max(dpds) * 1.3)
    axes[1].set_ylabel('|Demographic Parity Difference|', fontsize=11)
    axes[1].set_title('Racial Bias (lower = fairer)', fontsize=11)
    reduction = (1 - dpds[1] / dpds[0]) * 100 if dpds[0] > 0 else 0
    for bar, v in zip(bars2, dpds):
        axes[1].text(bar.get_x() + bar.get_width()/2, v + 0.003,
                     f'{v:.3f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
    axes[1].annotate(f'↓ {reduction:.0f}% bias\nreduction',
                     xy=(1, dpds[1]), xytext=(0.6, dpds[0]*0.8),
                     arrowprops=dict(arrowstyle='->', color='black'),
                     fontsize=9, color='darkgreen', fontweight='bold')

    plt.tight_layout()
    path = os.path.join(RESULTS_DIR, 'HC_Fig1_Tradeoff_BarChart.png')
    plt.savefig(path, bbox_inches='tight')
    plt.close()
    print(f"Saved: {path}")


# ─────────────────────────────────────────────────────────────────────────────
# Figure 2 — ROC Curves
# ─────────────────────────────────────────────────────────────────────────────
def plot_hc_roc(y_test, baseline_proba, fair_proba):
    fig, ax = plt.subplots(figsize=(7, 6))

    for label, proba, color, ls in [
        ('Baseline RF (Unfair)', baseline_proba, PALETTE['unfair_bias'], '-'),
        ('Fair NN',              fair_proba,      PALETTE['fair_acc'],   '--'),
    ]:
        fpr, tpr, _ = roc_curve(y_test, proba)
        roc_auc     = auc(fpr, tpr)
        ax.plot(fpr, tpr, color=color, lw=2.5, linestyle=ls,
                label=f'{label}  (AUC = {roc_auc:.3f})')

    ax.plot([0, 1], [0, 1], 'k--', lw=1, alpha=0.5)
    ax.set_xlabel('False Positive Rate', fontsize=11)
    ax.set_ylabel('True Positive Rate', fontsize=11)
    ax.set_title('ROC Curves — Healthcare Insurance Model\n'
                 'Fairness mitigation preserves predictive power',
                 fontsize=12, fontweight='bold')
    ax.legend(fontsize=10)
    ax.set_xlim([0, 1]); ax.set_ylim([0, 1.02])

    path = os.path.join(RESULTS_DIR, 'HC_Fig2_ROCCurve.png')
    plt.savefig(path, bbox_inches='tight', dpi=150)
    plt.close()
    print(f"Saved: {path}")


# ─────────────────────────────────────────────────────────────────────────────
# Figure 3 — Racial Disparity Confusion Matrices
# ─────────────────────────────────────────────────────────────────────────────
def plot_hc_confusion(y_test, baseline_preds, fair_preds, race_attr):
    fig, axes = plt.subplots(2, 2, figsize=(11, 9))
    fig.suptitle('Confusion Matrices by Race Group\n'
                 'Showing how bias affects each demographic differently',
                 fontsize=13, fontweight='bold')

    groups     = [(1, 'White (Privileged)'), (0, 'Non-White (Unprivileged)')]
    model_info = [
        ('Baseline RF (Biased)', baseline_preds, '#E74C3C'),
        ('Fair NN (Mitigated)',  fair_preds,      '#27AE60'),
    ]

    for col, (model_name, preds, color) in enumerate(model_info):
        for row, (group_val, group_name) in enumerate(groups):
            mask = race_attr == group_val
            cm   = confusion_matrix(y_test[mask], preds[mask])
            ax   = axes[row][col]
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                        ax=ax, cbar=False,
                        xticklabels=['Low Cost', 'High Cost'],
                        yticklabels=['Low Cost', 'High Cost'],
                        linewidths=0.5)
            ax.set_title(f'{model_name}\n{group_name}',
                         fontsize=10, fontweight='bold', color=color)
            ax.set_xlabel('Predicted', fontsize=9)
            ax.set_ylabel('Actual', fontsize=9)

    plt.tight_layout()
    path = os.path.join(RESULTS_DIR, 'HC_Fig3_ConfusionMatrix.png')
    plt.savefig(path, bbox_inches='tight', dpi=150)
    plt.close()
    print(f"Saved: {path}")


# ─────────────────────────────────────────────────────────────────────────────
# Figure 4 — Prediction Probability Violin Plot
# ─────────────────────────────────────────────────────────────────────────────
def plot_hc_violin(baseline_proba, fair_proba, race_attr):
    fig, axes = plt.subplots(1, 2, figsize=(12, 6), sharey=True)
    fig.suptitle('Prediction Probability Distribution by Race\n'
                 'Fair NN equalizes distributions across racial groups',
                 fontsize=13, fontweight='bold')

    race_labels = np.where(race_attr == 1, 'White', 'Non-White')

    for ax, (title, proba) in zip(axes, [
        ('Baseline RF (Biased)',  baseline_proba),
        ('Fair NN (Mitigated)',   fair_proba),
    ]):
        import pandas as pd
        plot_df = pd.DataFrame({'P(High Cost)': proba, 'Race': race_labels})
        sns.violinplot(
            data=plot_df, x='Race', y='P(High Cost)', ax=ax,
            palette={'White': PALETTE['white'], 'Non-White': PALETTE['nonwhite']},
            inner='quartile', linewidth=1.5
        )
        ax.set_title(title, fontsize=11, fontweight='bold')
        ax.set_ylabel('Predicted P(High Cost Patient)', fontsize=10)
        ax.set_xlabel('')
        ax.axhline(0.5, color='grey', linestyle='--', lw=1, alpha=0.7,
                   label='Decision threshold (0.5)')
        ax.legend(fontsize=8)

    path = os.path.join(RESULTS_DIR, 'HC_Fig4_ViolinPlot.png')
    plt.savefig(path, bbox_inches='tight', dpi=150)
    plt.close()
    print(f"Saved: {path}")


# ─────────────────────────────────────────────────────────────────────────────
# Main — run full visualization suite
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import torch
    from importlib.machinery import SourceFileLoader

    _hc_dir  = os.path.dirname(__file__)
    _src_dir = os.path.dirname(_hc_dir)
    sys.path.insert(0, _src_dir)
    from fairness_metrics import demographic_parity_difference

    preproc  = SourceFileLoader("preproc",  os.path.join(_hc_dir, "01_healthcare_preprocessing.py")).load_module()
    baseline = SourceFileLoader("baseline", os.path.join(_hc_dir, "02_healthcare_baseline.py")).load_module()
    mitigate = SourceFileLoader("mitigate", os.path.join(_hc_dir, "04_healthcare_mitigation.py")).load_module()

    df           = preproc.load_healthcare_data()
    df_processed = preproc.preprocess_healthcare(df)

    # Baseline
    bl_results, X_test, y_test = baseline.train_healthcare_baseline(df_processed)
    rf_model = bl_results['RandomForest']['model']
    bl_preds = rf_model.predict(X_test)
    bl_proba = rf_model.predict_proba(X_test)[:, 1]
    bl_dpd   = demographic_parity_difference(
                   y_test, bl_preds,
                   sensitive_features=X_test['race_binary'])

    # Fair NN
    fair_model, _, _, fair_preds, fair_proba = mitigate.train_healthcare_fair_model(df_processed)
    fair_dpd = demographic_parity_difference(
                   y_test, fair_preds,
                   sensitive_features=X_test['race_binary'])

    race_test = X_test['race_binary'].values

    # Generate all figures
    print("\n--- Generating Healthcare Visualizations ---")
    plot_hc_tradeoff(bl_results['RandomForest']['accuracy'], bl_dpd,
                     (fair_preds == y_test.values).mean(), fair_dpd)
    plot_hc_roc(y_test, bl_proba, fair_proba)
    plot_hc_confusion(y_test.values, bl_preds, fair_preds, race_test)
    plot_hc_violin(bl_proba, fair_proba, race_test)

    print(f"\nAll healthcare figures saved to: {RESULTS_DIR}")

