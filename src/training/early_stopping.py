# src/training/early_stopping.py

import numpy as np
import torch


class EarlyStopping:
    """
    Early stopping to prevent overfitting.

    As mentioned in the paper, early stopping was used with patience 7.
    """

    def __init__(self, patience=7, min_delta=0.0001, mode='min', verbose=True):
        """
        Initialize early stopping.

        Args:
            patience: Number of epochs without improvement before stopping
            min_delta: Minimum change in monitored value to qualify as improvement
            mode: 'min' (monitor is decreasing) or 'max' (monitor is increasing)
            verbose: Whether to print early stopping messages
        """
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.best_state_dict = None

        # Choose comparison function based on mode
        if mode == 'min':
            self.improved = lambda current, best: current < best - min_delta
        else:
            self.improved = lambda current, best: current > best + min_delta

    def __call__(self, score, model=None):
        """
        Check if training should stop and save best model if provided.

        Args:
            score: Current validation score
            model: Model to save (optional)

        Returns:
            bool: Whether to stop training
        """
        if self.best_score is None:
            # First epoch
            self.best_score = score
            if model is not None:
                self.best_state_dict = model.state_dict().copy()
            return False

        if self.improved(score, self.best_score):
            # Score improved
            if self.verbose:
                print(f"Validation score improved from {self.best_score:.4f} to {score:.4f}")
            self.best_score = score
            self.counter = 0
            if model is not None:
                self.best_state_dict = model.state_dict().copy()
        else:
            # Score did not improve
            self.counter += 1
            if self.verbose:
                print(f"No improvement in validation score: {self.counter}/{self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True
                if self.verbose:
                    print(f"Early stopping triggered after {self.counter} epochs without improvement")

        return self.early_stop

    def load_best_model(self, model):
        """
        Load the best model state dict into the provided model.

        Args:
            model: Model to load the best state dict into
        """
        if self.best_state_dict is not None:
            model.load_state_dict(self.best_state_dict)
            if self.verbose:
                print(f"Loaded best model with validation score: {self.best_score:.4f}")
        else:
            print("No best model state dict available")

