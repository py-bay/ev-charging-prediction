"""Visualization tools for model evaluation."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from loguru import logger
from sklearn.metrics import auc, confusion_matrix, precision_recall_curve, roc_curve


class EvaluationVisualizer:
    """
    Handles creation of evaluation plots and visualizations.
    Separated from metrics calculation for better modularity.
    """

    @staticmethod
    def plot_confusion_matrix(
        y_true: np.ndarray,
        y_pred: np.ndarray,
        output_path: Path,
        model_name: str | None = None,
        class_names: list[str] | None = None,
    ) -> None:
        """
        Plot and save confusion matrix as heatmap.

        Args:
            y_true: True labels
            y_pred: Predicted labels
            output_path: Path to save the plot
            model_name: Optional model name for title
            class_names: Optional class labels (default: ["Negative", "Positive"])
        """
        output_path.parent.mkdir(parents=True, exist_ok=True)

        if class_names is None:
            class_names = ["Negative", "Positive"]

        cm = confusion_matrix(y_true, y_pred)

        plt.figure(figsize=(8, 6))
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=class_names,
            yticklabels=class_names,
            cbar_kws={"label": "Count"},
        )

        title = "Confusion Matrix"
        if model_name:
            title += f" - {model_name}"
        plt.title(title, fontsize=14, fontweight="bold")
        plt.ylabel("True Label", fontsize=12)
        plt.xlabel("Predicted Label", fontsize=12)

        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        plt.close()

        logger.info(f"Saved confusion matrix to {output_path}")

    @staticmethod
    def plot_roc_curve(
        y_true: np.ndarray, y_proba: np.ndarray, output_path: Path, model_name: str | None = None
    ) -> None:
        """
        Plot and save ROC curve with AUC score.

        Args:
            y_true: True labels
            y_proba: Predicted probabilities for positive class
            output_path: Path to save the plot
            model_name: Optional model name for title
        """
        output_path.parent.mkdir(parents=True, exist_ok=True)

        try:
            fpr, tpr, thresholds = roc_curve(y_true, y_proba)
            roc_auc = auc(fpr, tpr)

            plt.figure(figsize=(8, 6))
            plt.plot(fpr, tpr, color="darkorange", lw=2, label=f"ROC curve (AUC = {roc_auc:.4f})")
            plt.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--", label="Random classifier")

            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.xlabel("False Positive Rate", fontsize=12)
            plt.ylabel("True Positive Rate", fontsize=12)

            title = "ROC Curve"
            if model_name:
                title += f" - {model_name}"
            plt.title(title, fontsize=14, fontweight="bold")

            plt.legend(loc="lower right", fontsize=10)
            plt.grid(alpha=0.3, linestyle="--")

            plt.tight_layout()
            plt.savefig(output_path, dpi=300, bbox_inches="tight")
            plt.close()

            logger.info(f"Saved ROC curve to {output_path}")

        except Exception as e:
            logger.error(f"Failed to plot ROC curve: {e}")

    @staticmethod
    def plot_precision_recall_curve(
        y_true: np.ndarray, y_proba: np.ndarray, output_path: Path, model_name: str | None = None
    ) -> None:
        """
        Plot and save precision-recall curve.

        Args:
            y_true: True labels
            y_proba: Predicted probabilities for positive class
            output_path: Path to save the plot
            model_name: Optional model name for title
        """
        output_path.parent.mkdir(parents=True, exist_ok=True)

        try:
            precision, recall, thresholds = precision_recall_curve(y_true, y_proba)

            # Calculate area under PR curve
            pr_auc = auc(recall, precision)

            plt.figure(figsize=(8, 6))
            plt.plot(recall, precision, color="blue", lw=2, label=f"PR curve (AUC = {pr_auc:.4f})")

            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.xlabel("Recall", fontsize=12)
            plt.ylabel("Precision", fontsize=12)

            title = "Precision-Recall Curve"
            if model_name:
                title += f" - {model_name}"
            plt.title(title, fontsize=14, fontweight="bold")

            plt.legend(loc="lower left", fontsize=10)
            plt.grid(alpha=0.3, linestyle="--")

            plt.tight_layout()
            plt.savefig(output_path, dpi=300, bbox_inches="tight")
            plt.close()

            logger.info(f"Saved precision-recall curve to {output_path}")

        except Exception as e:
            logger.error(f"Failed to plot precision-recall curve: {e}")

    @staticmethod
    def plot_feature_importance(
        feature_names: list[str],
        importances: np.ndarray,
        output_path: Path,
        model_name: str | None = None,
        top_n: int = 20,
    ) -> None:
        """
        Plot feature importance bar chart.

        Args:
            feature_names: List of feature names
            importances: Feature importance values
            output_path: Path to save the plot
            model_name: Optional model name for title
            top_n: Number of top features to display
        """
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Sort features by importance
        indices = np.argsort(importances)[::-1][:top_n]
        top_features = [feature_names[i] for i in indices]
        top_importances = importances[indices]

        plt.figure(figsize=(10, 8))
        plt.barh(range(len(top_features)), top_importances, color="steelblue")
        plt.yticks(range(len(top_features)), top_features)
        plt.xlabel("Importance", fontsize=12)
        plt.ylabel("Feature", fontsize=12)

        title = f"Top {top_n} Feature Importances"
        if model_name:
            title += f" - {model_name}"
        plt.title(title, fontsize=14, fontweight="bold")

        plt.gca().invert_yaxis()  # Highest importance at top
        plt.grid(axis="x", alpha=0.3, linestyle="--")

        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        plt.close()

        logger.info(f"Saved feature importance plot to {output_path}")

    @staticmethod
    def plot_all_evaluation_curves(
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_proba: np.ndarray,
        output_dir: Path,
        model_name: str,
    ) -> None:
        """
        Generate and save all evaluation plots.

        Args:
            y_true: True labels
            y_pred: Predicted labels
            y_proba: Predicted probabilities for positive class
            output_dir: Directory to save plots
            model_name: Model name for file naming
        """
        output_dir.mkdir(parents=True, exist_ok=True)

        # Confusion matrix
        cm_path = output_dir / f"{model_name}_confusion_matrix.png"
        EvaluationVisualizer.plot_confusion_matrix(y_true, y_pred, cm_path, model_name)

        # ROC curve (if probabilities available)
        if y_proba is not None:
            roc_path = output_dir / f"{model_name}_roc_curve.png"
            EvaluationVisualizer.plot_roc_curve(y_true, y_proba, roc_path, model_name)

            # Precision-recall curve
            pr_path = output_dir / f"{model_name}_precision_recall_curve.png"
            EvaluationVisualizer.plot_precision_recall_curve(y_true, y_proba, pr_path, model_name)

        logger.info(f"All evaluation plots saved to {output_dir}")
