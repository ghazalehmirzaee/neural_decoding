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

    Args:
        true_labels: Array of true class labels
        predicted_labels: Array of predicted class labels
        class_names: List of class names for labeling
        include_percentages: Whether to include percentages in cells
        title: Title for the plot
        save_path: Path to save the figure
    """
    # Calculate confusion matrix
    cm = confusion_matrix(true_labels, predicted_labels)

    # Create figure
    plt.figure(figsize=(10, 8))

    # Define annotation format
    if include_percentages:
        # Calculate percentages by row (true class)
        cm_perc = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100

        # Create annotations with count and percentage
        annot = np.empty_like(cm, dtype=object)
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                if np.isnan(cm_perc[i, j]):
                    annot[i, j] = f"{cm[i, j]}\n(0.0%)"
                else:
                    annot[i, j] = f"{cm[i, j]}\n({cm_perc[i, j]:.1f}%)"
    else:
        annot = cm

    # Plot using seaborn
    ax = sns.heatmap(cm, annot=annot if include_percentages else True, fmt='' if include_percentages else 'd',
                     cmap='Blues', xticklabels=class_names, yticklabels=class_names)

    # Customize plot
    ax.set_xlabel('Predicted Label')
    ax.set_ylabel('True Label')
    ax.set_title(title)

    # Adjust layout
    plt.tight_layout()

    # Save figure if path provided
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved confusion matrix to {save_path}")

    plt.close()


def plot_roc_curves(true_labels, predicted_probs, class_names, title='ROC Curves', save_path=None):
    """
    Generate and save ROC curves.

    Args:
        true_labels: Array of true class labels
        predicted_probs: Array of predicted probabilities (shape: n_samples, n_classes)
        class_names: List of class names for labeling
        title: Title for the plot
        save_path: Path to save the figure
    """
    # Create figure
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

    # Adjust layout
    plt.tight_layout()

    # Save figure if path provided
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved ROC curves to {save_path}")

    plt.close()


def plot_neural_activity(time_points, true_neural, pred_neural, true_behaviors, pred_behaviors,
                         behavior_names, window_size=200, title='Neural Activity and Behavior', save_path=None):
    """
    Plot neural activity and behavior predictions.

    Args:
        time_points: Array of time points
        true_neural: Array of true neural activity
        pred_neural: Array of predicted neural activity
        true_behaviors: Array of true behavior labels
        pred_behaviors: Array of predicted behavior labels
        behavior_names: List of behavior names
        window_size: Size of the window to display
        title: Title for the plot
        save_path: Path to save the figure
    """
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10), sharex=True, gridspec_kw={'height_ratios': [1, 2]})

    # Plot neural activity in the first subplot
    ax1.plot(time_points, true_neural, 'b-', label='True Neural Activity')
    ax1.plot(time_points, pred_neural, 'r--', label='Predicted Neural Activity')
    ax1.set_ylabel('Neural Activity')
    ax1.set_title('Neural Activity Comparison')
    ax1.legend(loc='upper right')
    ax1.grid(True, alpha=0.3)

    # Plot behavior in the second subplot
    cmap = plt.cm.get_cmap('tab10', len(behavior_names))

    # Plot true behavior
    for i, behavior in enumerate(behavior_names):
        mask = (true_behaviors == i)
        if np.any(mask):
            ax2.fill_between(time_points, i - 0.4, i + 0.4, where=mask, color=cmap(i), alpha=0.5,
                             label=f'True {behavior}')

    # Plot predicted behavior
    for i, behavior in enumerate(behavior_names):
        mask = (pred_behaviors == i)
        if np.any(mask):
            ax2.step(time_points, np.where(mask, i + 0.1, np.nan), 'k-', linewidth=2, where='post')

    # Customize behavior plot
    ax2.set_yticks(range(len(behavior_names)))
    ax2.set_yticklabels(behavior_names)
    ax2.set_xlabel('Time')
    ax2.set_ylabel('Behavior')
    ax2.set_ylim(-0.5, len(behavior_names) - 0.5)
    ax2.grid(True, alpha=0.3)

    # Add title to the entire figure
    fig.suptitle(title, fontsize=16)

    # Adjust layout
    plt.tight_layout()
    plt.subplots_adjust(top=0.9)

    # Save figure if path provided
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved neural activity plot to {save_path}")

    plt.close()


def generate_all_visualizations(predictions, targets, model_type, output_dir):
    """
    Generate and save all required visualizations for model evaluation.

    Args:
        predictions: Dictionary of model predictions
        targets: Dictionary of ground truth labels
        model_type: Type of model ('lstm' or 'hybrid')
        output_dir: Directory to save visualizations
    """
    # Create output directory if it doesn't exist
    visualization_dir = os.path.join(output_dir, model_type, 'visualizations')
    os.makedirs(visualization_dir, exist_ok=True)

    # Define class names for visualization
    class_names = {
        'multiclass': ['No Footstep', 'Contralateral', 'Ipsilateral'],
        'contralateral': ['Negative', 'Positive'],
        'ipsilateral': ['Negative', 'Positive']
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

    # Generate neural activity plots for hybrid model
    if model_type == 'hybrid' and 'neural_activity' in predictions and 'neural_activity' in targets:
        n_samples = len(targets['multiclass'])

        # Choose window size based on number of samples
        window_size = min(200, n_samples // 2)

        # Generate plots for different time windows
        for start_idx in range(0, n_samples, window_size):
            end_idx = min(start_idx + window_size, n_samples)

            plot_neural_activity(
                time_points=np.arange(start_idx, end_idx),
                true_neural=targets['neural_activity'][start_idx:end_idx],
                pred_neural=predictions['neural_activity'][start_idx:end_idx],
                true_behaviors=targets['multiclass'][start_idx:end_idx],
                pred_behaviors=predictions['multiclass'][start_idx:end_idx],
                behavior_names=class_names['multiclass'],
                window_size=window_size,
                title=f"{model_type.upper()} Model - Neural Activity and Behavior (Window {start_idx}-{end_idx})",
                save_path=os.path.join(visualization_dir, f'neural_activity_{start_idx}_{end_idx}.png')
            )


def plot_combined_metrics(predictions, targets, class_names, model_type, save_path=None):
    """
    Generate a combined metrics visualization showing performance across all tasks.

    Args:
        predictions: Dictionary of model predictions
        targets: Dictionary of ground truth labels
        class_names: Dictionary of class names for each task
        model_type: Type of model ('lstm' or 'hybrid')
        save_path: Path to save the figure
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

