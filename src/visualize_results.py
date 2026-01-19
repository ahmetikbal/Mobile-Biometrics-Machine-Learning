"""
Confusion Matrix Visualizer
Generates heatmap visualizations for model confusion matrices.
"""

import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path


def load_results(json_path: str) -> dict:
    """Load evaluation results from JSON file."""
    with open(json_path, 'r') as f:
        return json.load(f)


def plot_confusion_matrix(cm: np.ndarray, model_name: str, classes: list, 
                          output_path: str, figsize: tuple = (12, 10)):
    """
    Plot a confusion matrix heatmap.
    
    Args:
        cm: Confusion matrix array
        model_name: Name of the model
        classes: List of class labels
        output_path: Path to save the figure
        figsize: Figure size
    """
    plt.figure(figsize=figsize)
    
    # Create heatmap
    sns.heatmap(cm, annot=False, fmt='d', cmap='Blues',
                xticklabels=classes, yticklabels=classes,
                square=True, cbar_kws={'shrink': 0.8})
    
    plt.title(f'{model_name} - Confusion Matrix\n(30 Users, Accuracy: {np.trace(cm)/np.sum(cm)*100:.2f}%)', 
              fontsize=14, fontweight='bold')
    plt.xlabel('Predicted Label', fontsize=12)
    plt.ylabel('True Label', fontsize=12)
    
    # Rotate tick labels
    plt.xticks(rotation=45, ha='right', fontsize=8)
    plt.yticks(rotation=0, fontsize=8)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"  Saved: {output_path}")


def plot_confusion_matrix_detailed(cm: np.ndarray, model_name: str, classes: list,
                                    output_path: str):
    """
    Plot confusion matrix with detailed annotations (for smaller matrices).
    """
    plt.figure(figsize=(14, 12))
    
    # Normalize for color intensity
    cm_normalized = cm.astype('float') / cm.sum(axis=1, keepdims=True)
    cm_normalized = np.nan_to_num(cm_normalized)
    
    # Create heatmap with annotations
    ax = sns.heatmap(cm_normalized, annot=cm, fmt='d', cmap='Blues',
                     xticklabels=classes, yticklabels=classes,
                     square=True, linewidths=0.5,
                     cbar_kws={'shrink': 0.8, 'label': 'Normalized Frequency'},
                     annot_kws={'size': 6})
    
    # Calculate metrics
    accuracy = np.trace(cm) / np.sum(cm)
    
    plt.title(f'{model_name} - Confusion Matrix\nAccuracy: {accuracy*100:.2f}%', 
              fontsize=16, fontweight='bold', pad=20)
    plt.xlabel('Predicted User', fontsize=12)
    plt.ylabel('Actual User', fontsize=12)
    
    plt.xticks(rotation=45, ha='right', fontsize=7)
    plt.yticks(rotation=0, fontsize=7)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"  Saved: {output_path}")


def plot_all_models_comparison(results: dict, output_path: str):
    """
    Plot all confusion matrices in a single figure for comparison.
    """
    n_models = len(results)
    fig, axes = plt.subplots(1, n_models, figsize=(6*n_models, 5))
    
    if n_models == 1:
        axes = [axes]
    
    for idx, (model_name, data) in enumerate(results.items()):
        cm = np.array(data['confusion_matrix'])
        cm_normalized = cm.astype('float') / cm.sum(axis=1, keepdims=True)
        cm_normalized = np.nan_to_num(cm_normalized)
        
        accuracy = np.trace(cm) / np.sum(cm)
        
        sns.heatmap(cm_normalized, ax=axes[idx], cmap='Blues',
                    cbar=True, square=True,
                    xticklabels=False, yticklabels=False)
        
        axes[idx].set_title(f'{model_name}\nAcc: {accuracy*100:.2f}%', 
                           fontsize=12, fontweight='bold')
        axes[idx].set_xlabel('Predicted', fontsize=10)
        if idx == 0:
            axes[idx].set_ylabel('Actual', fontsize=10)
    
    plt.suptitle('Model Comparison - Confusion Matrices (30 Users)', 
                 fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"  Saved: {output_path}")


def plot_error_analysis(results: dict, output_path: str):
    """
    Plot error analysis - shows which users are commonly misclassified.
    """
    fig, axes = plt.subplots(1, len(results), figsize=(5*len(results), 4))
    
    if len(results) == 1:
        axes = [axes]
    
    for idx, (model_name, data) in enumerate(results.items()):
        cm = np.array(data['confusion_matrix'])
        
        # Get errors per user
        errors_per_user = cm.sum(axis=1) - np.diag(cm)
        
        colors = ['red' if e > 0 else 'green' for e in errors_per_user]
        
        axes[idx].bar(range(len(errors_per_user)), errors_per_user, color=colors, alpha=0.7)
        axes[idx].set_title(f'{model_name}\nMisclassifications per User', fontsize=11)
        axes[idx].set_xlabel('User Index', fontsize=9)
        axes[idx].set_ylabel('Error Count', fontsize=9)
        axes[idx].axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"  Saved: {output_path}")


def plot_metrics_comparison(results: dict, output_path: str):
    """
    Plot bar chart comparing model metrics.
    """
    models = list(results.keys())
    metrics = ['accuracy', 'precision', 'recall', 'f1_score', 'eer', 'far', 'frr']
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Plot 1: Main metrics
    x = np.arange(len(models))
    width = 0.2
    
    for i, metric in enumerate(['accuracy', 'precision', 'recall', 'f1_score']):
        values = [results[m][metric] for m in models]
        axes[0].bar(x + i*width, values, width, label=metric.replace('_', ' ').title())
    
    axes[0].set_ylabel('Score', fontsize=11)
    axes[0].set_title('Classification Metrics', fontsize=12, fontweight='bold')
    axes[0].set_xticks(x + width * 1.5)
    axes[0].set_xticklabels(models, fontsize=10)
    axes[0].legend(loc='lower right')
    axes[0].set_ylim(0.9, 1.01)
    axes[0].grid(axis='y', alpha=0.3)
    
    # Plot 2: Biometric metrics (EER, FAR, FRR)
    width = 0.25
    for i, metric in enumerate(['eer', 'far', 'frr']):
        values = [results[m][metric] * 100 for m in models]  # Convert to percentage
        axes[1].bar(x + i*width, values, width, label=metric.upper())
    
    axes[1].set_ylabel('Rate (%)', fontsize=11)
    axes[1].set_title('Biometric Metrics (Lower is Better)', fontsize=12, fontweight='bold')
    axes[1].set_xticks(x + width)
    axes[1].set_xticklabels(models, fontsize=10)
    axes[1].legend(loc='upper right')
    axes[1].grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"  Saved: {output_path}")


def main():
    """Generate all confusion matrix visualizations."""
    project_root = Path(__file__).parent.parent
    results_path = project_root / "output" / "reports" / "evaluation_results.json"
    figures_path = project_root / "output" / "figures"
    
    figures_path.mkdir(parents=True, exist_ok=True)
    
    print("=" * 60)
    print("Confusion Matrix Visualization")
    print("=" * 60)
    
    # Load results
    print("\nLoading evaluation results...")
    results = load_results(str(results_path))
    
    # User labels
    users = [f'U{i+1}' for i in range(30)]
    
    # Generate individual confusion matrices
    print("\nGenerating individual confusion matrices...")
    for model_name, data in results.items():
        cm = np.array(data['confusion_matrix'])
        output_file = figures_path / f'confusion_matrix_{model_name.lower()}.png'
        plot_confusion_matrix_detailed(cm, model_name, users, str(output_file))
    
    # Generate comparison plot
    print("\nGenerating comparison plot...")
    plot_all_models_comparison(results, str(figures_path / 'confusion_matrices_comparison.png'))
    
    # Generate error analysis
    print("\nGenerating error analysis...")
    plot_error_analysis(results, str(figures_path / 'error_analysis.png'))
    
    # Generate metrics comparison
    print("\nGenerating metrics comparison...")
    plot_metrics_comparison(results, str(figures_path / 'metrics_comparison.png'))
    
    print("\n" + "=" * 60)
    print(f"All visualizations saved to: {figures_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()
