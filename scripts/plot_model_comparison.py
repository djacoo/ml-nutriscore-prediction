import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from datetime import datetime

sys.path.append(str(Path(__file__).parent.parent))


COLORS = {
    'primary': '#2E86DE',
    'success': '#10AC84',
    'warning': '#F79F1F',
    'danger': '#EE5A6F',
    'secondary': '#54A0FF',
    'muted': '#95A5A6'
}

BLUE = '\033[0;34m'
GREEN = '\033[0;32m'
YELLOW = '\033[1;33m'
CYAN = '\033[0;36m'
NC = '\033[0m'
BOLD = '\033[1m'


def print_header(text):
    width = 70
    print(f"\n{CYAN}{'='*width}{NC}")
    print(f"{BOLD}{text:^{width}}{NC}")
    print(f"{CYAN}{'='*width}{NC}\n")


def load_model_metadata(model_dir: Path) -> Dict:
    metadata_files = list(model_dir.rglob("*_metadata.json"))
    models_data = []

    for metadata_file in metadata_files:
        with open(metadata_file, 'r') as f:
            data = json.load(f)
            data['file_path'] = str(metadata_file)
            data['model_type'] = metadata_file.parent.name
            models_data.append(data)

    return models_data


def plot_performance_summary(models_data: List[Dict], output_dir: Path):
    fig = plt.figure(figsize=(14, 8))
    gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.3)

    model_names = [m['model_name'] for m in models_data]

    ax1 = fig.add_subplot(gs[0, 0])
    val_acc = [m['training_history']['val_metrics'].get('accuracy', 0) * 100 for m in models_data]
    bars = ax1.bar(model_names, val_acc, color=COLORS['primary'], alpha=0.8)
    ax1.set_title('Validation Accuracy', fontweight='bold')
    ax1.set_ylabel('Accuracy (%)', fontweight='bold')
    ax1.grid(axis='y', alpha=0.3)
    for bar, val in zip(bars, val_acc):
        ax1.text(bar.get_x() + bar.get_width()/2., bar.get_height(),
                f'{val:.1f}%', ha='center', va='bottom', fontsize=9)

    ax2 = fig.add_subplot(gs[0, 1])
    f1_scores = [m['training_history']['val_metrics'].get('f1_weighted', 0) * 100 for m in models_data]
    bars = ax2.bar(model_names, f1_scores, color=COLORS['success'], alpha=0.8)
    ax2.set_title('F1-Score (Weighted)', fontweight='bold')
    ax2.set_ylabel('F1-Score (%)', fontweight='bold')
    ax2.grid(axis='y', alpha=0.3)
    for bar, val in zip(bars, f1_scores):
        ax2.text(bar.get_x() + bar.get_width()/2., bar.get_height(),
                f'{val:.1f}%', ha='center', va='bottom', fontsize=9)

    ax3 = fig.add_subplot(gs[1, 0])
    train_acc = [m['training_history']['train_metrics']['accuracy'] * 100 for m in models_data]
    val_acc = [m['training_history']['val_metrics'].get('accuracy', 0) * 100 for m in models_data]
    gaps = [t - v for t, v in zip(train_acc, val_acc)]
    bars = ax3.bar(model_names, gaps, color=COLORS['warning'], alpha=0.8)
    ax3.set_title('Overfitting Gap', fontweight='bold')
    ax3.set_ylabel('Gap (%)', fontweight='bold')
    ax3.grid(axis='y', alpha=0.3)
    for bar, val in zip(bars, gaps):
        ax3.text(bar.get_x() + bar.get_width()/2., bar.get_height(),
                f'{val:.1f}%', ha='center', va='bottom', fontsize=9)

    ax4 = fig.add_subplot(gs[1, 1])
    training_times = [m['training_history']['training_time'] for m in models_data]
    bars = ax4.barh(model_names, training_times, color=COLORS['secondary'], alpha=0.8)
    ax4.set_title('Training Time', fontweight='bold')
    ax4.set_xlabel('Time (seconds)', fontweight='bold')
    ax4.grid(axis='x', alpha=0.3)
    for bar, val in zip(bars, training_times):
        ax4.text(bar.get_width(), bar.get_y() + bar.get_height()/2.,
                f'{val:.1f}s', ha='left', va='center', fontsize=9)

    fig.suptitle('Model Performance Summary', fontsize=16, fontweight='bold', y=0.98)

    plt.savefig(output_dir / 'performance_summary.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"{GREEN}✓{NC} Saved: performance_summary.png")


def plot_model_ranking(models_data: List[Dict], output_dir: Path):
    fig, ax = plt.subplots(figsize=(12, 6))

    model_names = [m['model_name'] for m in models_data]

    val_acc = np.array([m['training_history']['val_metrics'].get('accuracy', 0) * 100 for m in models_data])
    f1_scores = np.array([m['training_history']['val_metrics'].get('f1_weighted', 0) * 100 for m in models_data])
    training_times = np.array([m['training_history']['training_time'] for m in models_data])

    normalized_time = 100 - (training_times / training_times.max() * 100)

    overall_score = (val_acc * 0.5) + (f1_scores * 0.3) + (normalized_time * 0.2)

    sorted_indices = np.argsort(overall_score)[::-1]
    sorted_names = [model_names[i] for i in sorted_indices]
    sorted_scores = [overall_score[i] for i in sorted_indices]

    colors_gradient = plt.cm.RdYlGn(np.linspace(0.3, 0.9, len(sorted_names)))

    bars = ax.barh(sorted_names, sorted_scores, color=colors_gradient, alpha=0.8)

    ax.set_xlabel('Overall Score', fontsize=12, fontweight='bold')
    ax.set_ylabel('Model', fontsize=12, fontweight='bold')
    ax.set_title('Model Ranking (50% Accuracy + 30% F1 + 20% Speed)', fontsize=14, fontweight='bold')
    ax.grid(axis='x', alpha=0.3, linestyle='--')

    for i, (bar, score, idx) in enumerate(zip(bars, sorted_scores, sorted_indices)):
        width = bar.get_width()
        rank = i + 1
        medal = '🥇' if rank == 1 else '🥈' if rank == 2 else '🥉' if rank == 3 else f'#{rank}'
        ax.text(width, bar.get_y() + bar.get_height()/2.,
               f'  {medal} {score:.1f}',
               ha='left', va='center', fontsize=10, fontweight='bold')

    plt.tight_layout()
    plt.savefig(output_dir / 'model_ranking.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"{GREEN}✓{NC} Saved: model_ranking.png")


def generate_comparison_report(models_data: List[Dict], output_dir: Path):
    report_path = output_dir / 'comparison_report.txt'

    with open(report_path, 'w') as f:
        f.write("="*70 + "\n")
        f.write("MODEL COMPARISON REPORT\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("="*70 + "\n\n")

        for model in models_data:
            f.write(f"Model: {model['model_name']}\n")
            f.write("-" * 70 + "\n")
            f.write(f"Type: {model['model_type']}\n")

            f.write("\nHyperparameters:\n")
            for key, value in model['hyperparameters'].items():
                f.write(f"  {key}: {value}\n")

            f.write("\nPerformance Metrics:\n")
            train_acc = model['training_history']['train_metrics']['accuracy'] * 100
            val_metrics = model['training_history']['val_metrics']
            val_acc = val_metrics.get('accuracy', 0) * 100
            gap = train_acc - val_acc

            f.write(f"  Training Accuracy: {train_acc:.2f}%\n")
            f.write(f"  Validation Accuracy: {val_acc:.2f}%\n")
            f.write(f"  Train-Val Gap: {gap:.2f}% ")
            if gap < 5:
                f.write("(Excellent - No overfitting)\n")
            elif gap < 10:
                f.write("(Good - Minor overfitting)\n")
            else:
                f.write("(Warning - Significant overfitting)\n")

            f.write(f"  Precision (weighted): {val_metrics.get('precision_weighted', 0)*100:.2f}%\n")
            f.write(f"  Recall (weighted): {val_metrics.get('recall_weighted', 0)*100:.2f}%\n")
            f.write(f"  F1-Score (weighted): {val_metrics.get('f1_weighted', 0)*100:.2f}%\n")

            f.write(f"\nTraining Time: {model['training_history']['training_time']:.2f} seconds\n")
            f.write(f"Last Trained: {model['training_history']['last_trained']}\n")
            f.write("\n" + "="*70 + "\n\n")

        f.write("\nBEST MODEL ANALYSIS\n")
        f.write("="*70 + "\n")

        best_val_acc = max(models_data, key=lambda x: x['training_history']['val_metrics'].get('accuracy', 0))
        f.write(f"\nBest Validation Accuracy: {best_val_acc['model_name']}\n")
        f.write(f"  Accuracy: {best_val_acc['training_history']['val_metrics'].get('accuracy', 0)*100:.2f}%\n")

        best_f1 = max(models_data, key=lambda x: x['training_history']['val_metrics'].get('f1_weighted', 0))
        f.write(f"\nBest F1-Score: {best_f1['model_name']}\n")
        f.write(f"  F1-Score: {best_f1['training_history']['val_metrics'].get('f1_weighted', 0)*100:.2f}%\n")

        fastest = min(models_data, key=lambda x: x['training_history']['training_time'])
        f.write(f"\nFastest Training: {fastest['model_name']}\n")
        f.write(f"  Time: {fastest['training_history']['training_time']:.2f} seconds\n")

        train_acc_list = [m['training_history']['train_metrics']['accuracy'] * 100 for m in models_data]
        val_acc_list = [m['training_history']['val_metrics'].get('accuracy', 0) * 100 for m in models_data]
        gaps = [t - v for t, v in zip(train_acc_list, val_acc_list)]
        best_generalization_idx = gaps.index(min(gaps))
        best_gen_model = models_data[best_generalization_idx]

        f.write(f"\nBest Generalization: {best_gen_model['model_name']}\n")
        f.write(f"  Train-Val Gap: {gaps[best_generalization_idx]:.2f}%\n")

        f.write("\n" + "="*70 + "\n")

    print(f"{GREEN}✓{NC} Saved: comparison_report.txt")


def main():
    parser = argparse.ArgumentParser(description='Plot model comparison visualizations')
    parser.add_argument(
        '--models-dir',
        type=str,
        default='models/trained',
        help='Directory containing trained models'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='plots',
        help='Directory to save plots'
    )

    args = parser.parse_args()

    print_header("Model Performance Visualization")

    models_dir = Path(args.models_dir)
    output_dir = Path(args.output_dir)

    if not models_dir.exists():
        print(f"{YELLOW}Error: Models directory '{models_dir}' not found.{NC}")
        sys.exit(1)

    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"{BLUE}Loading model metadata...{NC}")
    models_data = load_model_metadata(models_dir)

    if not models_data:
        print(f"{YELLOW}No trained models found in '{models_dir}'.{NC}")
        sys.exit(1)

    print(f"{GREEN}✓{NC} Found {len(models_data)} trained model(s)\n")

    for model in models_data:
        print(f"  • {model['model_name']} ({model['model_type']})")
    print()

    print(f"{BLUE}Generating visualizations...{NC}\n")

    plot_performance_summary(models_data, output_dir)
    plot_model_ranking(models_data, output_dir)

    print(f"\n{BLUE}Generating comparison report...{NC}\n")
    generate_comparison_report(models_data, output_dir)

    print(f"\n{GREEN}{BOLD}✓ All visualizations saved to: {output_dir}/{NC}")
    print(f"\n{CYAN}Generated files:{NC}")
    print(f"  • performance_summary.png - Comprehensive 4-panel summary")
    print(f"  • model_ranking.png - Overall model ranking")
    print(f"  • comparison_report.txt - Detailed text analysis")


if __name__ == '__main__':
    sns.set_style("whitegrid")
    plt.rcParams['figure.facecolor'] = 'white'
    plt.rcParams['axes.facecolor'] = 'white'
    main()
