

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

try:
    import shap
except ImportError as e:
    raise ImportError(
        "SHAP is not installed. Run: pip install shap"
    ) from e


# 1. CREATE SHAP EXPLAINER

def get_shap_explainer(model):
    """
    Create a SHAP TreeExplainer for a tree-based model (XGBoost, RF, LightGBM).

    WHY TreeExplainer?
    ───────────────────
    SHAP has different explainer types for different model families:
      - TreeExplainer  → for decision trees, Random Forest, XGBoost, LightGBM
      - LinearExplainer → for linear models (Logistic Regression)
      - KernelExplainer → model-agnostic but very slow

    TreeExplainer uses an algorithm called TreeSHAP which runs in polynomial
    time (much faster than the exponential complexity of brute-force Shapley).
    This is why we chose XGBoost — SHAP support is best for tree models.

    Parameters
    ----------
    model : a fitted tree-based sklearn-compatible model

    Returns
    -------
    shap.TreeExplainer
    """
    try:
        explainer = shap.TreeExplainer(model)
        return explainer
    except Exception as e:
        raise RuntimeError(
            f"Failed to create SHAP TreeExplainer: {e}\n"
            "Ensure the model is a fitted tree-based model (XGBoost, RF, LightGBM)."
        ) from e


# 2. COMPUTE SHAP VALUES

def get_shap_values(explainer, X, max_samples=500):
    """
    Compute SHAP values for a feature matrix X.

    For large test sets, computing SHAP for every row can be slow.
    max_samples limits computation to a random sample — sufficient for
    global importance analysis (summary plots).

    WHAT ARE SHAP VALUES?
    ──────────────────────
    For a model with N features and M test samples, shap_values is an
    M × N matrix where:
      - shap_values[i, j] = the contribution of feature j to prediction i
      - Positive value → pushed this prediction toward FRAUD
      - Negative value → pushed this prediction toward LEGITIMATE
      - sum(shap_values[i, :]) ≈ model output[i] − baseline_output

    Parameters
    ----------
    explainer   : shap.TreeExplainer from get_shap_explainer()
    X           : feature DataFrame or numpy array
    max_samples : max rows to compute SHAP for (default 500)

    Returns
    -------
    tuple: (shap_values_array, X_sample)
        shap_values_array : numpy array shape (n_samples, n_features)
        X_sample          : the (possibly subsampled) feature DataFrame
    """
    try:
        if isinstance(X, pd.DataFrame):
            X_sample = X.sample(
                n=min(max_samples, len(X)),
                random_state=42
            ).reset_index(drop=True)
        else:
            idx = np.random.RandomState(42).choice(
                len(X), size=min(max_samples, len(X)), replace=False
            )
            X_sample = pd.DataFrame(X).iloc[idx].reset_index(drop=True)

        shap_values = explainer.shap_values(X_sample)

        # XGBoost / binary classifiers return a single 2-D array
        # Some models return a list [shap_class0, shap_class1] — take class 1 (fraud)
        if isinstance(shap_values, list):
            shap_values = shap_values[1]

        return shap_values, X_sample

    except Exception as e:
        raise RuntimeError(f"Failed to compute SHAP values: {e}") from e


# 3. SHAP SUMMARY PLOT — BAR CHART (GLOBAL IMPORTANCE)

def plot_shap_summary(shap_values, X_sample, title, top_n=10, save_path=None):
    """
    Bar chart of mean absolute SHAP values — global feature importance.

    WHY MEAN ABSOLUTE SHAP?
    ────────────────────────
    For each feature we compute mean(|SHAP value|) across all samples.
    Taking the absolute value means positive and negative contributions
    are treated equally — a feature that sometimes pushes toward fraud
    and sometimes toward legitimate is still important.

    This is more reliable than XGBoost's built-in 'gain' importance because:
      - It is consistent (doesn't depend on tree structure)
      - It accounts for feature interactions
      - It is directly tied to individual predictions

    Parameters
    ----------
    shap_values : numpy array (n_samples, n_features) from get_shap_values()
    X_sample    : feature DataFrame (same rows as shap_values)
    title       : plot title string
    top_n       : how many top features to show (default 10)
    save_path   : optional file path to save the figure
    """
    try:
        # mean(|SHAP|) per feature
        mean_abs_shap = pd.Series(
            np.abs(shap_values).mean(axis=0),
            index=X_sample.columns
        ).sort_values(ascending=False).head(top_n)

        fig, ax = plt.subplots(figsize=(10, 6))
        colors = plt.cm.RdYlBu_r(
            np.linspace(0.2, 0.8, len(mean_abs_shap))
        )
        bars = ax.barh(
            mean_abs_shap.index[::-1],
            mean_abs_shap.values[::-1],
            color=colors[::-1],
            edgecolor='white'
        )
        ax.set_xlabel('Mean |SHAP Value|  (average impact on model output)')
        ax.set_title(title, fontsize=12, fontweight='bold')
        ax.axvline(
            mean_abs_shap.values.mean(),
            color='navy', linestyle='--', linewidth=1.5,
            label='Mean across features'
        )
        ax.legend(fontsize=9)

        for bar, val in zip(bars, mean_abs_shap.values[::-1]):
            ax.text(
                val + mean_abs_shap.values.max() * 0.01,
                bar.get_y() + bar.get_height() / 2,
                f'{val:.4f}', va='center', fontsize=9
            )

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.show()
        plt.close()

        print(f"Top {top_n} features by mean |SHAP value|:")
        print(mean_abs_shap.to_string())
        return mean_abs_shap

    except Exception as e:
        raise RuntimeError(f"Failed to plot SHAP summary: {e}") from e


# 4. SHAP BEESWARM / DOT SUMMARY PLOT

def plot_shap_beeswarm(shap_values, X_sample, title, top_n=10, save_path=None):
    """
    Beeswarm (dot) summary plot — shows SHAP value distribution per feature.

    WHY THE BEESWARM ADDS VALUE OVER THE BAR CHART?
    ─────────────────────────────────────────────────
    The bar chart shows average importance. The beeswarm shows the FULL
    distribution of SHAP values for each feature:
      - Each dot = one prediction from the sample
      - X position = SHAP value (how much that feature pushed the score)
      - Colour = feature value (red = high, blue = low)

    This lets you see, for example:
      - A high 'time_since_signup_hours' (red) pushes the score LEFT (toward legit)
        → long-standing users are lower risk
      - A low 'time_since_signup_hours' (blue) pushes the score RIGHT (toward fraud)
        → new accounts making immediate purchases are high risk

    Parameters
    ----------
    shap_values : numpy array (n_samples, n_features)
    X_sample    : feature DataFrame
    title       : plot title
    top_n       : number of features to show
    save_path   : optional file path
    """
    try:
        fig, ax = plt.subplots(figsize=(10, 7))
        shap.summary_plot(
            shap_values,
            X_sample,
            max_display=top_n,
            show=False,
            plot_type='dot'
        )
        plt.title(title, fontsize=12, fontweight='bold')
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.show()
        plt.close()

    except Exception as e:
        raise RuntimeError(f"Failed to plot SHAP beeswarm: {e}") from e


# 5. SHAP FORCE PLOT (INDIVIDUAL PREDICTION)

def plot_shap_force(explainer, shap_values_row, X_row,
                   case_label, prediction_label, save_path=None):
    """
    Force plot for a single prediction — shows which features pushed
    the score up (toward fraud) or down (toward legitimate).

    HOW TO READ A FORCE PLOT
    ──────────────────────────
    The plot shows a horizontal arrow starting from the baseline
    (average model output across the dataset).

    Features highlighted in RED pushed the score HIGHER (toward fraud).
    Features highlighted in BLUE pushed the score LOWER (toward legitimate).
    The final score (model output) is shown on the right.

    Example for a True Positive (correctly caught fraud):
        Baseline: 0.094  (average fraud rate)
        time_since_signup_hours = 0.2h → RED  → pushes score up
        device_transaction_count = 15  → RED  → pushes score up
        country_Unknown            → BLUE → pushes score down slightly
        Final score: 0.89 → predicted FRAUD ✓

    Parameters
    ----------
    explainer          : shap.TreeExplainer
    shap_values_row    : 1-D numpy array of SHAP values for this sample
    X_row              : single-row DataFrame for this sample
    case_label         : 'True Positive', 'False Positive', or 'False Negative'
    prediction_label   : string describing prediction e.g. 'Predicted: FRAUD'
    save_path          : optional file path to save the figure
    """
    try:
        print(f"\n{'='*60}")
        print(f"  FORCE PLOT — {case_label}")
        print(f"  {prediction_label}")
        print(f"{'='*60}")

        # Top 8 features driving this individual prediction
        feature_impact = pd.Series(
            shap_values_row,
            index=X_row.columns
        ).sort_values(key=abs, ascending=False).head(8)

        fig, ax = plt.subplots(figsize=(12, 5))

        colors = ['tomato' if v > 0 else 'steelblue'
                  for v in feature_impact.values]
        bars = ax.barh(
            feature_impact.index,
            feature_impact.values,
            color=colors,
            edgecolor='white'
        )
        ax.axvline(0, color='black', linewidth=1)
        ax.set_xlabel('SHAP Value  (positive = toward fraud, negative = toward legit)')
        ax.set_title(
            f"Force Plot — {case_label}\n{prediction_label}",
            fontsize=11, fontweight='bold'
        )

        # Annotate bars with values
        for bar, val in zip(bars, feature_impact.values):
            xpos = val + (0.002 if val >= 0 else -0.002)
            ha   = 'left' if val >= 0 else 'right'
            ax.text(xpos, bar.get_y() + bar.get_height() / 2,
                    f'{val:+.4f}', va='center', ha=ha, fontsize=9)

        # Legend
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor='tomato',    label='Pushes toward FRAUD'),
            Patch(facecolor='steelblue', label='Pushes toward LEGITIMATE'),
        ]
        ax.legend(handles=legend_elements, loc='lower right', fontsize=9)

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.show()
        plt.close()

        print("\nTop contributing features:")
        print(feature_impact.to_string())

    except Exception as e:
        raise RuntimeError(f"Failed to plot SHAP force plot: {e}") from e


# 6. FIND CASE INDICES (TP, FP, FN)

def find_case_indices(model, X_test, y_test):
    """
    Find the index of one True Positive, one False Positive, and one False Negative.

    WHY THESE THREE CASES?
    ──────────────────────
    True Positive  (TP): fraud correctly caught
        → Shows what a clear fraud looks like to the model.

    False Positive (FP): legitimate transaction incorrectly flagged as fraud
        → Shows what misleads the model — which legitimate features look like fraud.
        → Directly linked to customer friction costs.

    False Negative (FN): fraud the model missed
        → Shows blind spots — which fraud patterns the model can't detect.
        → Directly linked to financial loss.

    Understanding all three is required to make actionable business recommendations.

    Parameters
    ----------
    model  : fitted classifier
    X_test : test features
    y_test : true labels (pandas Series or numpy array)

    Returns
    -------
    dict with keys: 'tp', 'fp', 'fn'
        Each value is the integer index in X_test / y_test for that case.
        Returns None for a case type if no example is found.
    """
    try:
        y_pred = model.predict(X_test)
        y_true = np.array(y_test)

        # Reset index to ensure positional alignment
        if isinstance(X_test, pd.DataFrame):
            X_test = X_test.reset_index(drop=True)
        y_true = pd.Series(y_true).reset_index(drop=True)
        y_pred = pd.Series(y_pred).reset_index(drop=True)

        tp_idx = y_true[(y_true == 1) & (y_pred == 1)].index
        fp_idx = y_true[(y_true == 0) & (y_pred == 1)].index
        fn_idx = y_true[(y_true == 1) & (y_pred == 0)].index

        cases = {
            'tp': int(tp_idx[0]) if len(tp_idx) > 0 else None,
            'fp': int(fp_idx[0]) if len(fp_idx) > 0 else None,
            'fn': int(fn_idx[0]) if len(fn_idx) > 0 else None,
        }

        print("Case indices found:")
        for case, idx in cases.items():
            label = {'tp': 'True Positive', 'fp': 'False Positive', 'fn': 'False Negative'}[case]
            status = f"index {idx}" if idx is not None else "NOT FOUND in test set"
            print(f"  {label:16s}: {status}")

        return cases

    except Exception as e:
        raise RuntimeError(f"Failed to find case indices: {e}") from e
