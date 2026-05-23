"""
F1 Winner Predictor - Model Training Module
Trains XGBoost, LightGBM, and Logistic Regression ensemble
Handles class imbalance, cross-validation, feature importance
"""

import pandas as pd
import numpy as np
import os
import joblib
import warnings
warnings.filterwarnings("ignore")

from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (classification_report, roc_auc_score,
                              confusion_matrix, log_loss, accuracy_score)
from sklearn.calibration import CalibratedClassifierCV
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
import xgboost as xgb
import lightgbm as lgb
import matplotlib.pyplot as plt
import seaborn as sns

plt.rcParams.update({
    "figure.facecolor": "#0f0f1a", "axes.facecolor": "#1a1a2e",
    "axes.edgecolor": "#e10600", "axes.labelcolor": "white",
    "xtick.color": "white", "ytick.color": "white",
    "text.color": "white", "grid.color": "#333355",
})

MODELS_DIR = os.path.join(os.path.dirname(__file__), "models")
PLOTS_DIR = os.path.join(os.path.dirname(__file__), "plots")
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(PLOTS_DIR, exist_ok=True)


def prepare_training_data(df: pd.DataFrame, feature_cols: list):
    """Prepare X, y with imputation — returns clean arrays."""
    available = [c for c in feature_cols if c in df.columns]
    df_clean = df.dropna(subset=["Winner"]).copy()

    X = df_clean[available].copy()
    y = df_clean["Winner"].astype(int).values

    # Impute missing values
    imputer = SimpleImputer(strategy="median")
    X_imp = imputer.fit_transform(X)

    return X_imp, y, available, imputer


def train_xgboost(X_train, y_train) -> xgb.XGBClassifier:
    """Train tuned XGBoost classifier."""
    # Handle class imbalance
    neg, pos = np.bincount(y_train)
    scale = neg / pos

    model = xgb.XGBClassifier(
        n_estimators=500,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=3,
        gamma=0.1,
        reg_alpha=0.1,
        reg_lambda=1.0,
        scale_pos_weight=scale,
        use_label_encoder=False,
        eval_metric="logloss",
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)
    return model


def train_lightgbm(X_train, y_train) -> lgb.LGBMClassifier:
    """Train tuned LightGBM classifier."""
    neg, pos = np.bincount(y_train)
    scale = neg / pos

    model = lgb.LGBMClassifier(
        n_estimators=500,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_samples=10,
        scale_pos_weight=scale,
        random_state=42,
        n_jobs=-1,
        verbose=-1,
    )
    model.fit(X_train, y_train)
    return model


def train_logistic(X_train, y_train) -> Pipeline:
    """Train Logistic Regression with scaling."""
    neg, pos = np.bincount(y_train)
    cw = {0: 1.0, 1: neg / pos}
    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(C=0.5, class_weight=cw, max_iter=1000, random_state=42)),
    ])
    pipeline.fit(X_train, y_train)
    return pipeline


def race_aware_cv(df: pd.DataFrame, X: np.ndarray, y: np.ndarray, model_fn, n_splits: int = 5):
    """
    Race-aware cross-validation: split by race (Year+Round), not by row.
    Prevents data leakage from same race appearing in train and test.
    """
    races = df[["Year", "Round"]].drop_duplicates().sort_values(["Year", "Round"])
    race_ids = list(zip(races["Year"], races["Round"]))
    n = len(race_ids)
    fold_size = n // n_splits

    auc_scores, acc_scores = [], []
    df_reset = df.reset_index(drop=True)

    for fold in range(n_splits):
        test_races = race_ids[fold * fold_size: (fold + 1) * fold_size]
        test_mask = df_reset.apply(
            lambda row: (int(row["Year"]), int(row["Round"])) in test_races, axis=1
        ).values
        train_mask = ~test_mask

        X_tr, X_te = X[train_mask], X[test_mask]
        y_tr, y_te = y[train_mask], y[test_mask]

        if len(np.unique(y_te)) < 2:
            continue

        model = model_fn(X_tr, y_tr)
        if hasattr(model, "predict_proba"):
            proba = model.predict_proba(X_te)[:, 1]
        else:
            proba = model.decision_function(X_te)

        auc = roc_auc_score(y_te, proba)
        acc = accuracy_score(y_te, model.predict(X_te))
        auc_scores.append(auc)
        acc_scores.append(acc)

    return np.mean(auc_scores), np.std(auc_scores), np.mean(acc_scores)


class EnsemblePredictor:
    """Weighted ensemble of XGBoost + LightGBM + LogReg."""

    def __init__(self, weights=(0.45, 0.40, 0.15)):
        self.weights = weights
        self.xgb_model = None
        self.lgb_model = None
        self.lr_model = None
        self.imputer = None
        self.feature_cols = None

    def fit(self, X, y):
        self.xgb_model = train_xgboost(X, y)
        self.lgb_model = train_lightgbm(X, y)
        self.lr_model = train_logistic(X, y)

    def predict_proba(self, X):
        p_xgb = self.xgb_model.predict_proba(X)[:, 1]
        p_lgb = self.lgb_model.predict_proba(X)[:, 1]
        p_lr = self.lr_model.predict_proba(X)[:, 1]
        return (
            self.weights[0] * p_xgb +
            self.weights[1] * p_lgb +
            self.weights[2] * p_lr
        )

    def predict(self, X, threshold=0.5):
        return (self.predict_proba(X) >= threshold).astype(int)


def plot_feature_importance(model, feature_names: list, title: str, save_name: str):
    """Plot feature importance from tree model."""
    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
    else:
        return

    idx = np.argsort(importances)[-20:]
    fig, ax = plt.subplots(figsize=(10, 8))
    colors = ["#e10600" if importances[i] > np.percentile(importances, 80) else "#4a90d9" for i in idx]
    ax.barh([feature_names[i] for i in idx], importances[idx], color=colors)
    ax.set_title(title, color="#e10600", fontsize=14)
    ax.set_xlabel("Importance")
    plt.tight_layout()
    path = os.path.join(PLOTS_DIR, save_name)
    plt.savefig(path, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"Saved: {path}")


def plot_confusion_matrix(y_true, y_pred, title="Confusion Matrix"):
    """Plot confusion matrix."""
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Reds", ax=ax,
                xticklabels=["No Win", "Win"], yticklabels=["No Win", "Win"],
                linewidths=2, linecolor="#0f0f1a")
    ax.set_title(title, color="#e10600", fontsize=14)
    ax.set_ylabel("Actual")
    ax.set_xlabel("Predicted")
    plt.tight_layout()
    path = os.path.join(PLOTS_DIR, "confusion_matrix.png")
    plt.savefig(path, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"Saved: {path}")


def train_and_evaluate(df: pd.DataFrame, feature_cols: list) -> EnsemblePredictor:
    """Full training pipeline with evaluation."""
    print("\n🏎️  Training F1 Winner Prediction Models...\n")

    X, y, available_cols, imputer = prepare_training_data(df, feature_cols)
    print(f"Training samples : {len(X):,}")
    print(f"Features used    : {len(available_cols)}")
    print(f"Positive rate    : {y.mean():.3f} (winners per entry)")

    # Race-aware CV evaluation
    print("\n── Cross-Validation (Race-Aware) ────────────")
    for name, fn in [("XGBoost", train_xgboost), ("LightGBM", train_lightgbm)]:
        auc_mean, auc_std, acc = race_aware_cv(df.dropna(subset=["Winner"]).reset_index(drop=True),
                                               X, y, fn)
        print(f"  {name:<12} AUC: {auc_mean:.4f} ± {auc_std:.4f}  |  Acc: {acc:.4f}")

    # Train final ensemble on full data
    print("\n── Training Final Ensemble ──────────────────")
    ensemble = EnsemblePredictor(weights=(0.45, 0.40, 0.15))
    ensemble.fit(X, y)
    ensemble.imputer = imputer
    ensemble.feature_cols = available_cols

    # Final train-set metrics (informational)
    proba = ensemble.predict_proba(X)
    pred = ensemble.predict(X)
    print(f"  Train AUC      : {roc_auc_score(y, proba):.4f}")
    print(f"  Train Log-Loss : {log_loss(y, proba):.4f}")
    print(f"  Train Accuracy : {accuracy_score(y, pred):.4f}")
    print("\n" + classification_report(y, pred, target_names=["No Win", "Win"]))

    # Feature importance plots
    plot_feature_importance(ensemble.xgb_model, available_cols,
                            "XGBoost Feature Importance", "xgb_importance.png")
    plot_feature_importance(ensemble.lgb_model, available_cols,
                            "LightGBM Feature Importance", "lgb_importance.png")
    plot_confusion_matrix(y, pred)

    # Save models
    joblib.dump(ensemble, os.path.join(MODELS_DIR, "ensemble.pkl"))
    joblib.dump(imputer, os.path.join(MODELS_DIR, "imputer.pkl"))
    pd.Series(available_cols).to_csv(os.path.join(MODELS_DIR, "feature_cols.csv"), index=False)
    print(f"\n✅ Models saved to: {MODELS_DIR}")

    return ensemble


def load_ensemble() -> EnsemblePredictor:
    """Load saved ensemble from disk."""
    ensemble = joblib.load(os.path.join(MODELS_DIR, "ensemble.pkl"))
    return ensemble


if __name__ == "__main__":
    from feature_engineering import get_feature_columns
    path = os.path.join(os.path.dirname(__file__), "data/engineered_data.csv")
    if os.path.exists(path):
        df = pd.read_csv(path)
        ensemble = train_and_evaluate(df, get_feature_columns())
    else:
        print("Run feature_engineering.py first.")
