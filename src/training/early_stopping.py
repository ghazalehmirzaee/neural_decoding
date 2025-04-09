# src/training/early_stopping.py

import numpy as np


class EarlyStopping:
    """
    Early stopping to prevent overfitting.

    As mentioned in the paper, early stopping was used with patience 7.
    """

    def __init__(self, patience=7, min_delta=0, mode='min'):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_score = None
        self.early_stop = False

        # Choose comparison function based on mode
        if mode == 'min':
            self.improved = lambda current, best: current < best - min_delta
        else:
            self.improved = lambda current, best: current > best + min_delta

    def __call__(self, score):
        """
        Check if training should stop.

        Args:
            score: Current validation score

        Returns:
            bool: Whether to stop training
        """
        if self.best_score is None:
            # First epoch
            self.best_score = score
            return False

        if self.improved(score, self.best_score):
            # Score improved
            self.best_score = score
            self.counter = 0
        else:
            # Score did not improve
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True

        return self.early_stop

