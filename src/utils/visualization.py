# src/utils/visualization.py

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix


def plot_confusion_matrix(
        true_labels,
        predicted_labels,
        class_names=None,
        normalize=None,
        include_percentages=True,
        title=None,
        cmap='Blues',
        figsize=(10, 8),
        save_path=None
):
    """
    Plot confusion matrix with enhanced visualization.

    Args:
        true_labels: Ground truth labels
        predicted_labels: Predicted labels
        class_names: Names of classes
        normalize: Normalization method ('true', 'pred', 'all', or None)
        include_percentages: Whether to include percentages in cells
        title: Plot title
        cmap: Colormap for heatmap
        figsize: Figure size
        save_path: Path to save the figure
    """
    # Calculate confusion matrix
    cm = confusion_matrix(true_labels, predicted_labels)

    # Set default class names if not provided
    if class_names is None:
        class_names = [str(i) for i in range(cm.shape[0])]

    # Create figure
    plt.figure(figsize=figsize)

    # Create percentages matrix if requested
    if include_percentages and normalize is None:
        # Calculate percentages
        cm_sum = np.sum(cm, axis=1, keepdims=True)
        cm_perc = cm.astype('float') / cm_sum.astype('float') * 100
        cm_perc = np.nan_to_num(cm_perc)

        # Format annotation strings
        annot = np.empty_like(cm, dtype=object)
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                annot[i, j] = f"{cm[i, j]}\n({cm_perc[i, j]:.1f}%)"

        # Plot heatmap
        ax = sns.heatmap(
            cm,
            annot=annot,
            fmt='',
            cmap=cmap,
            square=True,
            linewidths=.5,
            cbar=True,
            cbar_kws={"shrink": 0.75}
        )
    else:
        # Apply normalization if specified
        if normalize is not None:
            if normalize == 'true':
                cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
                fmt = '.2f'
            elif normalize == 'pred':
                cm = cm.astype('float') / cm.sum(axis=0)[np.newaxis, :]
                fmt = '.2f'
            elif normalize == 'all':
                cm = cm.astype('float') / cm.sum()
                fmt = '.2f'
            cm = np.nan_to_num(cm)
        else:
            fmt = 'd'

        # Plot heatmap
        ax = sns.heatmap(
            cm,
            annot=True,
            fmt=fmt,
            cmap=cmap,
            square=True,
            linewidths=.5,
            cbar=True,
            cbar_kws={"shrink": 0.75}
        )

    # Set title and labels
    if title:
        plt.title(title, fontsize=14)
    plt.ylabel('True Label', fontsize=12)
    plt.xlabel('Predicted Label', fontsize=12)

    # Set tick labels
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks + 0.5, class_names, rotation=45, ha='right')
    plt.yticks(tick_marks + 0.5, class_names, rotation=0)

    # Tight layout
    plt.tight_layout()

    # Save figure if path is provided
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
        plt.close()
    else:
        plt.show()


def plot_roc_curves(
        true_labels,
        predicted_probs,
        class_names=None,
        title="ROC Curves",
        figsize=(10, 8),
        save_path=None
):
    """
    Plot ROC curves for binary or multiclass problems.

    Args:
        true_labels: Ground truth labels
        predicted_probs: Predicted probabilities
        class_names: Names of classes
        title: Plot title
        figsize: Figure size
        save_path: Path to save the figure
    """
    from sklearn.metrics import roc_curve, auc
    from sklearn.preprocessing import label_binarize

    # Get number of classes
    if len(predicted_probs.shape) > 1:
        n_classes = predicted_probs.shape[1]
    else:
        n_classes = 2

    # Set default class names if not provided
    if class_names is None:
        class_names = [f"Class {i}" for i in range(n_classes)]

    # Create figure
    plt.figure(figsize=figsize)

    if n_classes == 2:
        # Binary classification
        fpr, tpr, _ = roc_curve(
            true_labels,
            predicted_probs[:, 1] if len(predicted_probs.shape) > 1 else predicted_probs
        )
        roc_auc = auc(fpr, tpr)

        # Plot ROC curve
        plt.plot(
            fpr,
            tpr,
            lw=2,
            label=f'{class_names[1]} (AUC = {roc_auc:.2f})'
        )
    else:
        # Multiclass classification (one-vs-rest)
        # Binarize the labels
        y_bin = label_binarize(true_labels, classes=np.arange(n_classes))

        # Compute ROC curve and AUC for each class
        for i in range(n_classes):
            fpr, tpr, _ = roc_curve(y_bin[:, i], predicted_probs[:, i])
            roc_auc = auc(fpr, tpr)

            # Plot ROC curve
            plt.plot(
                fpr,
                tpr,
                lw=2,
                label=f'{class_names[i]} (AUC = {roc_auc:.2f})'
            )

    # Plot random chance line
    plt.plot([0, 1], [0, 1], 'k--', lw=2)

    # Set labels, title, and legend
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title(title, fontsize=14)
    plt.legend(loc='lower right', fontsize=10)
    plt.grid(True, alpha=0.3)

    # Save figure if path is provided
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
        plt.close()
    else:
        plt.show()


def plot_neural_activity(
        time_points,
        true_neural,
        pred_neural=None,
        true_behaviors=None,
        pred_behaviors=None,
        behavior_names=None,
        window_size=100,
        title="Neural Activity and Behavior Prediction",
        figsize=(15, 10),
        save_path=None
):
    """
    Plot neural activity and behavior predictions as in Figure 4 of the paper.

    Args:
        time_points: Time points for x-axis
        true_neural: True neural activity
        pred_neural: Predicted neural activity (optional)
        true_behaviors: True behavior labels (optional)
        pred_behaviors: Predicted behavior labels (optional)
        behavior_names: Names of behavior classes
        window_size: Size of window for display
        title: Plot title
        figsize: Figure size
        save_path: Path to save the figure
    """
    # Set default behavior names if not provided
    if behavior_names is None:
        behavior_names = ['No Footstep', 'Contralateral', 'Ipsilateral']

    # Create color mapping for behaviors (as in Figure 4)
    behavior_colors = ['#FFDEDE', '#FFB3D9', '#FF66B2']

    # Determine number of panels to create
    n_panels = 1
    if true_behaviors is not None:
        n_panels += 1

    # Create figure
    fig, axes = plt.subplots(n_panels, 1, figsize=figsize, height_ratios=[1] * n_panels)

    # Make axes a list if it's not already
    if n_panels == 1:
        axes = [axes]

    # Plot neural activity
    ax = axes[0]
    ax.plot(time_points, true_neural, 'k-', linewidth=1.5, label='True Neural Activity')

    if pred_neural is not None:
        ax.plot(time_points, pred_neural, 'g--', linewidth=1.5, alpha=0.7,
                label='Predicted Neural Activity')

    # Add behavior background shading if provided
    if true_behaviors is not None:
        for i, color in enumerate(behavior_colors):
            mask = true_behaviors == i
            if np.any(mask):
                ax.fill_between(
                    time_points,
                    ax.get_ylim()[0],
                    ax.get_ylim()[1],
                    where=mask,
                    color=color,
                    alpha=0.1
                )

    # Set title, labels, and legend
    ax.set_ylabel('Neural Activity', fontsize=12)
    ax.grid(True, alpha=0.3)
    if n_panels == 1:
        ax.set_xlabel('Time Steps', fontsize=12)
    ax.legend(loc='upper right', fontsize=10)

    # Plot behaviors if provided
    if true_behaviors is not None:
        ax = axes[1]

        # Plot true behaviors
        ax.step(
            time_points,
            true_behaviors,
            where='post',
            color='blue',
            linewidth=1.5,
            label='True Behaviors'
        )

        # Plot predicted behaviors if provided
        if pred_behaviors is not None:
            ax.step(
                time_points,
                pred_behaviors,
                where='post',
                color='red',
                linestyle='--',
                linewidth=1.5,
                label='Predicted Behaviors'
            )

        # Set ticks and labels
        ax.set_yticks(np.arange(len(behavior_names)))
        ax.set_yticklabels(behavior_names)
        ax.set_ylim(-0.2, len(behavior_names) - 0.8)
        ax.set_xlabel('Time Steps', fontsize=12)
        ax.set_ylabel('Behavior', fontsize=12)
        ax.legend(loc='upper right', fontsize=10)
        ax.grid(True, alpha=0.3)

    # Set overall title
    fig.suptitle(title, fontsize=14)

    # Format time periods
    period_size = window_size
    n_periods = (len(time_points) + period_size - 1) // period_size

    # Create period ticks and labels
    period_ticks = []
    period_labels = []

    for i in range(n_periods):
        period_start = i * period_size
        period_end = min(period_start + period_size, len(time_points))

        # Add the tick at the start of each period
        period_ticks.append(period_start)
        period_labels.append(f"{period_start}-{period_end}")

    # Apply the custom x-axis ticks and labels to both subplots
    for ax in axes:
        ax.set_xticks(period_ticks)
        ax.set_xticklabels(period_labels)

    # Tight layout
    plt.tight_layout(rect=[0, 0.03, 1, 0.97])

    # Save figure if path is provided
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
        plt.close()
    else:
        plt.show()

