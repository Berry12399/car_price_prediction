import os
import matplotlib.pyplot as plt
import seaborn as sns

from train_models import main as train_and_evaluate


def plot_actual_vs_predicted(y_test, lr_preds, dt_preds, out_path):
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.scatter(y_test, lr_preds, alpha=0.6, color="royalblue")
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], "r--")
    plt.xlabel("Actual Price")
    plt.ylabel("Predicted Price")
    plt.title("Linear Regression: Actual vs Predicted")

    plt.subplot(1, 2, 2)
    plt.scatter(y_test, dt_preds, alpha=0.6, color="seagreen")
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], "r--")
    plt.xlabel("Actual Price")
    plt.ylabel("Predicted Price")
    plt.title("Decision Tree: Actual vs Predicted")

    plt.tight_layout()
    plt.savefig(out_path)
    print(f"Saved plot: {out_path}")
    plt.close()


def plot_metric_comparison(lr_results, dt_results, out_path):
    metrics = ["mae", "rmse", "r2"]
    labels = ["MAE", "RMSE", "R2 Score"]

    lr_vals = [lr_results[m] for m in metrics]
    dt_vals = [dt_results[m] for m in metrics]

    x = range(len(metrics))
    width = 0.35

    plt.figure(figsize=(7, 5))
    plt.bar([i - width / 2 for i in x], lr_vals, width=width, label="Linear Regression", color="royalblue")
    plt.bar([i + width / 2 for i in x], dt_vals, width=width, label="Decision Tree", color="seagreen")
    plt.xticks(list(x), labels)
    plt.title("Model Comparison")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path)
    print(f"Saved plot: {out_path}")
    plt.close()


def main():
    os.makedirs("outputs", exist_ok=True)

    lr_results, dt_results, X_test, y_test = train_and_evaluate()

    plot_actual_vs_predicted(
        y_test, lr_results["preds"], dt_results["preds"],
        out_path="outputs/actual_vs_predicted.png"
    )
    plot_metric_comparison(
        lr_results, dt_results,
        out_path="outputs/metric_comparison.png"
    )

    print("\nAll done! Check the 'outputs' folder for charts and 'models' folder for saved models.")


if __name__ == "__main__":
    main()
