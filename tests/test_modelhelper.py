"""
tests/test_modelhelper.py
────────────────────────────────────────────────────────────────────────────
Unit tests for src/modelhelper.py helper functions.

Run with:
    pytest tests/test_modelhelper.py -v
"""

import sys
import os
import numpy as np
import pandas as pd
import pytest

from sklearn.linear_model import LogisticRegression
from sklearn.datasets import make_classification

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from modelhelper import (
    evaluate_model,
    plot_confusion_matrix,
    plot_precision_recall_curve,
    run_cross_validation,
)


# ─────────────────────────────────────────────────────────────────────────────
# FIXTURES
# ─────────────────────────────────────────────────────────────────────────────

@pytest.fixture
def fitted_model_and_data():
    """
    Returns a simple fitted LogisticRegression and matching X_test / y_test.
    Uses make_classification for a reproducible imbalanced dataset.
    """
    X, y = make_classification(
        n_samples=500,
        n_features=10,
        weights=[0.9, 0.1],   # 90% class 0, 10% class 1 (imbalanced)
        random_state=42,
    )
    X_df = pd.DataFrame(X, columns=[f'f{i}' for i in range(10)])
    y_s  = pd.Series(y, name='target')

    model = LogisticRegression(max_iter=1000, random_state=42)
    model.fit(X_df[:400], y_s[:400])

    return model, X_df[400:].reset_index(drop=True), y_s[400:].reset_index(drop=True)


@pytest.fixture
def unfitted_model():
    return LogisticRegression(max_iter=1000, random_state=42)


@pytest.fixture
def train_data():
    X, y = make_classification(
        n_samples=300, n_features=10, weights=[0.85, 0.15], random_state=0
    )
    X_df = pd.DataFrame(X, columns=[f'f{i}' for i in range(10)])
    y_s  = pd.Series(y)
    return X_df, y_s


# ─────────────────────────────────────────────────────────────────────────────
# TESTS — evaluate_model
# ─────────────────────────────────────────────────────────────────────────────

class TestEvaluateModel:

    def test_returns_dict(self, fitted_model_and_data):
        """evaluate_model must return a dictionary."""
        model, X_test, y_test = fitted_model_and_data
        result = evaluate_model(model, X_test, y_test, 'LR', 'Test')
        assert isinstance(result, dict)

    def test_required_keys(self, fitted_model_and_data):
        """Result must contain all required keys."""
        model, X_test, y_test = fitted_model_and_data
        result = evaluate_model(model, X_test, y_test, 'LR', 'Test')
        required = {'model', 'dataset', 'auc_pr', 'f1', 'precision', 'recall', 'pr_curve'}
        assert required.issubset(set(result.keys()))

    def test_auc_pr_in_range(self, fitted_model_and_data):
        """AUC-PR must be between 0 and 1."""
        model, X_test, y_test = fitted_model_and_data
        result = evaluate_model(model, X_test, y_test, 'LR', 'Test')
        assert 0.0 <= result['auc_pr'] <= 1.0

    def test_f1_in_range(self, fitted_model_and_data):
        """F1-Score must be between 0 and 1."""
        model, X_test, y_test = fitted_model_and_data
        result = evaluate_model(model, X_test, y_test, 'LR', 'Test')
        assert 0.0 <= result['f1'] <= 1.0

    def test_model_name_stored(self, fitted_model_and_data):
        """The model_name argument must be stored in result['model']."""
        model, X_test, y_test = fitted_model_and_data
        result = evaluate_model(model, X_test, y_test, 'MyModel', 'MyDataset')
        assert result['model'] == 'MyModel'
        assert result['dataset'] == 'MyDataset'

    def test_pr_curve_is_tuple(self, fitted_model_and_data):
        """pr_curve must be a tuple of two arrays (precision, recall)."""
        model, X_test, y_test = fitted_model_and_data
        result = evaluate_model(model, X_test, y_test, 'LR', 'Test')
        assert isinstance(result['pr_curve'], tuple)
        assert len(result['pr_curve']) == 2

    def test_works_without_axes(self, fitted_model_and_data):
        """evaluate_model must work when ax_cm=None (default)."""
        model, X_test, y_test = fitted_model_and_data
        # Should not raise any exception
        result = evaluate_model(model, X_test, y_test, 'LR', 'Test', ax_cm=None)
        assert result is not None


# ─────────────────────────────────────────────────────────────────────────────
# TESTS — plot_precision_recall_curve
# ─────────────────────────────────────────────────────────────────────────────

class TestPlotPrecisionRecallCurve:

    def test_runs_without_error(self, fitted_model_and_data):
        """plot_precision_recall_curve must run without raising an exception."""
        import matplotlib
        matplotlib.use('Agg')   # non-interactive backend for CI
        import matplotlib.pyplot as plt

        model, X_test, y_test = fitted_model_and_data
        result = evaluate_model(model, X_test, y_test, 'LR', 'Test')

        fig, ax = plt.subplots()
        # Should not raise
        plot_precision_recall_curve([result], 'Test PR Curve', ax)
        plt.close()

    def test_handles_multiple_models(self, fitted_model_and_data):
        """Should accept a list with more than one result dict."""
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        model, X_test, y_test = fitted_model_and_data
        r1 = evaluate_model(model, X_test, y_test, 'Model1', 'Test')
        r2 = evaluate_model(model, X_test, y_test, 'Model2', 'Test')

        fig, ax = plt.subplots()
        plot_precision_recall_curve([r1, r2], 'Multi-model PR Curve', ax)
        plt.close()


# ─────────────────────────────────────────────────────────────────────────────
# TESTS — run_cross_validation
# ─────────────────────────────────────────────────────────────────────────────

class TestRunCrossValidation:

    def test_returns_dict(self, unfitted_model, train_data):
        """run_cross_validation must return a dict."""
        X_train, y_train = train_data
        result = run_cross_validation(
            unfitted_model, X_train, y_train, 'LR', 'Test', cv=3
        )
        assert isinstance(result, dict)

    def test_required_keys(self, unfitted_model, train_data):
        """Result must contain mean and std for both metrics."""
        X_train, y_train = train_data
        result = run_cross_validation(
            unfitted_model, X_train, y_train, 'LR', 'Test', cv=3
        )
        required = {'auc_pr_mean', 'auc_pr_std', 'f1_mean', 'f1_std'}
        assert required.issubset(set(result.keys()))

    def test_metrics_in_range(self, unfitted_model, train_data):
        """All mean metrics must be between 0 and 1."""
        X_train, y_train = train_data
        result = run_cross_validation(
            unfitted_model, X_train, y_train, 'LR', 'Test', cv=3
        )
        assert 0.0 <= result['auc_pr_mean'] <= 1.0
        assert 0.0 <= result['f1_mean'] <= 1.0

    def test_std_non_negative(self, unfitted_model, train_data):
        """Standard deviations must be non-negative."""
        X_train, y_train = train_data
        result = run_cross_validation(
            unfitted_model, X_train, y_train, 'LR', 'Test', cv=3
        )
        assert result['auc_pr_std'] >= 0.0
        assert result['f1_std'] >= 0.0


# ─────────────────────────────────────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    pytest.main([__file__, '-v'])
