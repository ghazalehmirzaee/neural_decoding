# src/utils/metrics.py

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
    roc_curve,
    auc
)


def calculate_metrics(true_labels, predicted_labels, task='multiclass'):
    """
    Calculate comprehensive metrics for model evaluation.

    Args:
        true_labels: Ground truth labels
        predicted_labels: Predicted labels
        task: Task name for appropriate averaging

    Returns:
        Dictionary of metrics
    """
    # Determine averaging method based on task
    if task == 'multiclass':
        average = 'weighted'
    else:
        average = 'binary'

    # Calculate accuracy
    accuracy = accuracy_score(true_labels, predicted_labels)

    # Calculate precision, recall, F1
    precision, recall, f1, _ = precision_recall_fscore_support(
        true_labels,
        predicted_labels,
        average=average,
        zero_division=0
    )

    # Return metrics dictionary
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1
    }


def calculate_confusion_matrix(true_labels, predicted_labels, normalize=None):
    """
    Calculate confusion matrix with optional normalization.

    Args:
        true_labels: Ground truth labels
        predicted_labels: Predicted labels
        normalize: Normalization method ('true', 'pred', 'all', or None)

    Returns:
        Confusion matrix as numpy array
    """
    # Calculate confusion matrix
    cm = confusion_matrix(true_labels, predicted_labels)

    # Apply normalization if specified
    if normalize is not None:
        if normalize == 'true':
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        elif normalize == 'pred':
            cm = cm.astype('float') / cm.sum(axis=0)[np.newaxis, :]
        elif normalize == 'all':
            cm = cm.astype('float') / cm.sum()

        # Handle division by zero
        cm = np.nan_to_num(cm)

    return cm


def calculate_roc_curve_data(true_labels, predicted_probs, num_classes=None):
    """
    Calculate ROC curve data for binary or multiclass problems.

    Args:
        true_labels: Ground truth labels
        predicted_probs: Predicted probabilities
        num_classes: Number of classes for multiclass problems

    Returns:
        Dictionary with FPR, TPR, and AUC for each class
    """
    # Determine number of classes
    if num_classes is None:
        if len(predicted_probs.shape) > 1:
            num_classes = predicted_probs.shape[1]
        else:
            num_classes = 2

    # Initialize result dictionaries
    fpr = {}
    tpr = {}
    roc_auc = {}

    if num_classes == 2:
        # Binary classification
        if len(predicted_probs.shape) > 1:
            probs = predicted_probs[:, 1]
        else:
            probs = predicted_probs

        fpr[0], tpr[0], _ = roc_curve(true_labels, probs)
        roc_auc[0] = auc(fpr[0], tpr[0])
    else:
        # Multiclass classification (one-vs-rest)
        for i in range(num_classes):
            # Convert to binary problem
            true_binary = (true_labels == i).astype(int)
            if len(predicted_probs.shape) > 1:
                pred_probs = predicted_probs[:, i]
            else:
                pred_probs = (predicted_labels == i).astype(float)

            # Calculate ROC curve
            fpr[i], tpr[i], _ = roc_curve(true_binary, pred_probs)
            roc_auc[i] = auc(fpr[i], tpr[i])

    return {
        'fpr': fpr,
        'tpr': tpr,
        'auc': roc_auc
    }

