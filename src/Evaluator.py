from IPython.display import display, Markdown
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    classification_report,
    confusion_matrix,
    roc_curve,
)


class ModelEvaluator:
    """
    A utility class for evaluating machine learning models.
    """
    def __init__(self, y_true, y_pred, y_proba=None):
        """
        Initialize with true labels, predicted labels, and predicted probabilities.

        :param y_true: Ground truth target values.
        :param y_pred: Predicted target values.
        :param y_proba: Predicted probabilities for the positive class.
        """
        self.y_true = y_true
        self.y_pred = y_pred
        self.y_proba = y_proba

    def display_scores(self):
        """
        Calculate and display performance metrics.
        """
        precision = precision_score(self.y_true, self.y_pred)
        recall = recall_score(self.y_true, self.y_pred)
        f1 = f1_score(self.y_true, self.y_pred)
        accuracy = accuracy_score(self.y_true, self.y_pred)
        roc_auc = roc_auc_score(self.y_true, self.y_proba) if self.y_proba is not None else None

        display(Markdown("### **Scores**"))
        print(f"\033[1mAccuracy Score\033[0m\t: {accuracy:0.4f}")
        print(f"\033[1mPrecision\033[0m\t: {precision:0.4f}")
        print(f"\033[1mRecall\033[0m\t\t: {recall:0.4f}")
        print(f"\033[1mF1 Score\033[0m\t: {f1:0.4f}")
        if roc_auc is not None:
            print(f"\033[1mROC AUC Score\033[0m\t: {roc_auc:0.4f}")

        print("\nClassification Report:")
        print(classification_report(self.y_true, self.y_pred))

    def plot_confusion_matrix(self):
        """
        Plot the confusion matrix.
        """
        cm = confusion_matrix(self.y_true, self.y_pred)
        plt.figure(figsize=(6, 4))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
        plt.xlabel("Predicted Labels")
        plt.ylabel("True Labels")
        plt.title("Confusion Matrix")
        plt.show()

    def plot_roc_curve(self):
        """
        Plot the ROC curve using Plotly.
        """
        if self.y_proba is None:
            raise ValueError("Predicted probabilities are required to plot the ROC curve.")

        # Compute ROC curve
        fpr, tpr, thresholds = roc_curve(self.y_true, self.y_proba)
        roc_auc = roc_auc_score(self.y_true, self.y_proba)

        # Create a Plotly figure
        fig = go.Figure()

        # Add the ROC curve
        fig.add_trace(go.Scatter(x=fpr, y=tpr, mode="lines", name=f"AUC = {roc_auc:.2f}"))

        # Add the diagonal line
        fig.add_trace(
            go.Scatter(
                x=[0, 1],
                y=[0, 1],
                mode="lines",
                line=dict(dash="dash", color="red"),
                showlegend=False,
            )
        )

        # Update layout
        fig.update_layout(
            title="ROC Curve",
            xaxis_title="False Positive Rate",
            yaxis_title="True Positive Rate",
            showlegend=True,
            width=700,
            height=600,
            autosize=False,
        )

        fig.show()