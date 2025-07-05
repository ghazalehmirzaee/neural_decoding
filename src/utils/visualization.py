# src/utils/visualization.py

import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_curve, auc

def plot_confusion_matrix(true_labels, predicted_labels, class_names, include_percentages=True,
                          title='Confusion Matrix', save_path=None):
    """
    Generate and save a confusion matrix visualization.
    """
    # Calculate confusion matrix - this gives us the raw counts
    cm = confusion_matrix(true_labels, predicted_labels)

    # Create figure with specific size for optimal appearance
    plt.figure(figsize=(10, 8))

    # Define annotation format and data for heatmap
    if include_percentages:
        # Calculate percentages by row (true class) - normalizes each row to sum to 100%
        # This shows what percentage of each true class was predicted as each class
        cm_perc = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100

        # Handle potential division by zero (creates NaN values) by replacing with 0
        cm_perc = np.nan_to_num(cm_perc, nan=0.0)

        annot = np.empty_like(cm, dtype=object)
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                annot[i, j] = f"{cm_perc[i, j]:.1f}%\n({cm[i, j]})"

        # Use percentage matrix for color scaling - this makes the color intensity
        # represent the proportion of predictions rather than absolute counts
        heatmap_data = cm_perc

        # Set color scale limits for consistent appearance
        vmin, vmax = 0, 100
        colorbar_label = 'Percentage (%)'

    else:
        annot = True 
        heatmap_data = cm
        vmin, vmax = None, None
        colorbar_label = 'Count'

    ax = sns.heatmap(
        heatmap_data,  
        annot=annot if include_percentages else True, 
        fmt='' if include_percentages else 'd', 
        cmap='Blues',  
        xticklabels=class_names,  
        yticklabels=class_names,  
        vmin=vmin,  
        vmax=vmax,  
        cbar_kws={'label': colorbar_label, 'shrink': 0.8},  
        annot_kws={'size': 14, 'weight': 'bold'},  
        linewidths=0.5,  
        linecolor='white', 
        square=True  
    )

    # Customize plot appearance to match your professional style
    ax.set_xlabel('Predicted Label', fontsize=14, fontweight='bold', labelpad=10)
    ax.set_ylabel('True Label', fontsize=14, fontweight='bold', labelpad=10)
    ax.set_title(title, fontsize=16, fontweight='bold', pad=20)

    # Style the tick labels for better readability
    ax.tick_params(axis='x', labelsize=12, rotation=0)  
    ax.tick_params(axis='y', labelsize=12, rotation=0)  

    # Ensure tick labels are properly positioned
    plt.xticks(rotation=0, ha='center')
    plt.yticks(rotation=0, va='center')

    # Adjust layout to prevent any label cutoff and ensure clean appearance
    plt.tight_layout()

    # Save figure if path provided, using high DPI for crisp output
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"Saved confusion matrix to {save_path}")

    plt.close()


def plot_roc_curves(true_labels, predicted_probs, class_names, title='ROC Curves', save_path=None):
    """
    Generate and save ROC curves.
    """
    plt.figure(figsize=(10, 8))

    # Number of classes
    n_classes = len(class_names)

    # Plot ROC curve for each class
    if n_classes == 2:
        # Binary classification
        fpr, tpr, _ = roc_curve(true_labels, predicted_probs[:, 1])
        roc_auc = auc(fpr, tpr)

        plt.plot(fpr, tpr, lw=2, label=f'{class_names[1]} (AUC = {roc_auc:.2f})')
    else:
        # Multi-class classification
        # One-vs-rest approach
        true_labels_one_hot = np.eye(n_classes)[true_labels]

        for i, class_name in enumerate(class_names):
            fpr, tpr, _ = roc_curve(true_labels_one_hot[:, i], predicted_probs[:, i])
            roc_auc = auc(fpr, tpr)

            plt.plot(fpr, tpr, lw=2, label=f'{class_name} (AUC = {roc_auc:.2f})')

    # Plot diagonal reference line (random classifier)
    plt.plot([0, 1], [0, 1], 'k--', lw=2)

    # Customize plot
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(title)
    plt.legend(loc='lower right')
    plt.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved ROC curves to {save_path}")

    plt.close()


def generate_all_visualizations(predictions, targets, model_type, output_dir):
    """
    Generate and save all required visualizations for model evaluation.
    """
    # Create output directory if it doesn't exist
    visualization_dir = os.path.join(output_dir, model_type, 'visualizations')
    os.makedirs(visualization_dir, exist_ok=True)

    # Define class names for visualization
    class_names = {
        'multiclass': ['No Footstep', 'Contralateral', 'Ipsilateral'],
        'contralateral': ['No Footstep', 'Contralateral'],
        'ipsilateral': ['No Footstep', 'Ipsilateral']
    }

    # Generate confusion matrices for all classification tasks
    for task in ['multiclass', 'contralateral', 'ipsilateral']:
        if task in predictions and task in targets:
            plot_confusion_matrix(
                true_labels=targets[task],
                predicted_labels=predictions[task],
                class_names=class_names[task],
                include_percentages=True,
                title=f"{model_type.upper()} Model - {task.capitalize()} Confusion Matrix",
                save_path=os.path.join(visualization_dir, f'{task}_confusion_matrix.png')
            )

    # Generate ROC curves for all classification tasks
    for task in ['multiclass', 'contralateral', 'ipsilateral']:
        if f"{task}_probs" in predictions and task in targets:
            plot_roc_curves(
                true_labels=targets[task],
                predicted_probs=predictions[f"{task}_probs"],
                class_names=class_names[task],
                title=f"{model_type.upper()} Model - {task.capitalize()} ROC Curves",
                save_path=os.path.join(visualization_dir, f'{task}_roc_curves.png')
            )

    # Generate combined metrics visualization
    plot_combined_metrics(
        predictions=predictions,
        targets=targets,
        class_names=class_names,
        model_type=model_type,
        save_path=os.path.join(visualization_dir, 'combined_metrics.png')
    )

def plot_combined_metrics(predictions, targets, class_names, model_type, save_path=None):
    """
    Generate a combined metrics visualization showing performance across all tasks.
    """
    # Calculate metrics for each task
    metrics = {}
    for task in ['multiclass', 'contralateral', 'ipsilateral']:
        if task in predictions and task in targets:
            # Calculate accuracy
            accuracy = np.mean(predictions[task] == targets[task])

            # Calculate per-class precision and recall
            n_classes = len(class_names[task])
            precision = np.zeros(n_classes)
            recall = np.zeros(n_classes)

            for i in range(n_classes):
                true_positives = np.sum((predictions[task] == i) & (targets[task] == i))
                false_positives = np.sum((predictions[task] == i) & (targets[task] != i))
                false_negatives = np.sum((predictions[task] != i) & (targets[task] == i))

                precision[i] = true_positives / max(true_positives + false_positives, 1)
                recall[i] = true_positives / max(true_positives + false_negatives, 1)

            # Store metrics
            metrics[task] = {
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall
            }

    # Create figure with subplots
    n_tasks = len(metrics)
    fig, axes = plt.subplots(n_tasks, 2, figsize=(15, 5 * n_tasks))

    # Handle single-task case
    if n_tasks == 1:
        axes = np.array([axes])

    # Plot metrics for each task
    for i, (task, task_metrics) in enumerate(metrics.items()):
        # Plot precision
        axes[i, 0].bar(range(len(class_names[task])), task_metrics['precision'], color='blue', alpha=0.7)
        axes[i, 0].set_xticks(range(len(class_names[task])))
        axes[i, 0].set_xticklabels(class_names[task])
        axes[i, 0].set_ylabel('Precision')
        axes[i, 0].set_title(f'{task.capitalize()} Precision')
        axes[i, 0].grid(True, alpha=0.3)

        # Add accuracy as text
        axes[i, 0].text(0.5, 0.9, f"Accuracy: {task_metrics['accuracy']:.4f}",
                        transform=axes[i, 0].transAxes, ha='center',
                        bbox=dict(facecolor='white', alpha=0.8))

        # Plot recall
        axes[i, 1].bar(range(len(class_names[task])), task_metrics['recall'], color='green', alpha=0.7)
        axes[i, 1].set_xticks(range(len(class_names[task])))
        axes[i, 1].set_xticklabels(class_names[task])
        axes[i, 1].set_ylabel('Recall')
        axes[i, 1].set_title(f'{task.capitalize()} Recall')
        axes[i, 1].grid(True, alpha=0.3)

    # Add title to the entire figure
    fig.suptitle(f"{model_type.upper()} Model - Performance Metrics", fontsize=16)

    # Adjust layout
    plt.tight_layout()
    plt.subplots_adjust(top=0.95)

    # Save figure if path provided
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved combined metrics to {save_path}")

    plt.close()


def plot_neural_activity_comparison(time_points, true_neural, pred_neural, true_behaviors, pred_behaviors,
                                    behavior_names=['No-footstep', 'Contralateral', 'Ipsilateral'],
                                    title='Neuronal Activity and Movement Prediction Comparison',
                                    save_path=None):
    import matplotlib.pyplot as plt
    import numpy as np
    from matplotlib.lines import Line2D

    behavior_colors = {
        0: "#FFDEDE",  # No Footstep
        1: "#FFB3D9",  # Contralateral
        2: "#FF66B2",  # Ipsilateral
    }

    fig = plt.figure(figsize=(12, 20))
    window_size = 100
    num_windows = min(4, len(time_points) // window_size)

    for w in range(num_windows):
        start_idx = w * window_size
        end_idx = min(start_idx + window_size, len(time_points))
        time_segment = np.arange(start_idx, end_idx)

        # ---- Upper panel: Neuronal Signal ----
        ax1 = fig.add_subplot(2 * num_windows, 1, 2 * w + 1)
        ax1.plot(true_neural[start_idx:end_idx], 'k-', linewidth=1.2)
        ax1.plot(pred_neural[start_idx:end_idx], 'g--', linewidth=1.2, alpha=0.7)

        # Background shading
        for i in range(end_idx - start_idx):
            idx = start_idx + i
            if idx < len(true_behaviors):
                behavior = true_behaviors[idx]
                color = behavior_colors.get(behavior)
                if behavior > 0 and color:
                    ax1.axvspan(i - 0.5, i + 0.5, color=color, alpha=0.1)

        ax1.set_ylabel('Neuronal Signal')
        ax1.set_ylim(-4, 4)  # More zoomed out
        ax1.set_xticks(np.linspace(0, end_idx - start_idx - 1, 5, dtype=int))
        ax1.set_xticklabels(
            [str(start_idx + int(i)) for i in np.linspace(0, end_idx - start_idx - 1, 5, dtype=int)]
        )

        # ---- Lower panel: Behavior Prediction ----
        ax2 = fig.add_subplot(2 * num_windows, 1, 2 * w + 2)
        true_beh = true_behaviors[start_idx:end_idx]
        pred_beh = pred_behaviors[start_idx:end_idx]

        ax2.step(time_segment, true_beh, where='post', color='blue', linewidth=1.5)
        ax2.step(time_segment, pred_beh, where='post', color='red', linestyle='--', linewidth=1.5)

        ax2.set_yticks(range(len(behavior_names)))
        ax2.set_yticklabels(behavior_names)
        ax2.set_ylim(-0.5, len(behavior_names) - 0.5)
        ax2.set_ylabel("Footstep")

        ax2.set_xticks(np.linspace(start_idx, end_idx - 1, 5, dtype=int))
        ax2.set_xticklabels(
            [str(int(i)) for i in np.linspace(start_idx, end_idx - 1, 5, dtype=int)]
        )

        if w != num_windows - 1:
            ax2.set_xticklabels([])

    # ---- Global legend and labels ----
    legend_elements = [
        Line2D([0], [0], color='k', linestyle='-', lw=1.5, label='True Neural'),
        Line2D([0], [0], color='g', linestyle='--', lw=1.5, label='Pred Neural'),
        Line2D([0], [0], color='blue', linestyle='-', lw=1.5, label='True Behavior'),
        Line2D([0], [0], color='red', linestyle='--', lw=1.5, label='Pred Behavior')
    ]
    fig.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(0.98, 0.99))

    plt.suptitle(title, fontsize=14)
    fig.text(0.5, 0.04, 'Frame Number', ha='center', fontsize=12) 
    plt.tight_layout(rect=[0, 0.05, 1, 0.95])  
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved neural activity visualization to {save_path}")

    plt.close(fig)
