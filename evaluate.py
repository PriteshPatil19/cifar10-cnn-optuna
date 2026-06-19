import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

def plot_loss(train_loss,
              test_loss,
              save_path="results/loss_curve.png",
              epoch=None):

    plt.figure(figsize=(10, 6))

    plt.plot(train_loss, label="Train Loss", linewidth=2)
    plt.plot(test_loss, label="Test Loss", linewidth=2)

    plt.title("Train and Test Loss", fontsize=12, fontweight="bold")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.xlim(0, epoch)
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend(fontsize=11)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()



def plot_accuracy(train_acc,
                  test_acc,
                  save_path="results/accuracy_curve.png",
                  epoch=None):

    plt.figure(figsize=(10, 6))

    plt.plot(train_acc, label="Train Accuracy", linewidth=2)
    plt.plot(test_acc, label="Test Accuracy", linewidth=2)


    plt.title("Train and Test Accuracy", fontsize=12, fontweight="bold")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy (%)")
    plt.xlim(0, epoch)
    plt.ylim(0, 100)
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend(fontsize=11)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()



def plot_confusion_matrix(y_true,
                          y_pred,
                          save_path="results/confusion_matrix.png"):


    classes = (
        "plane",
        "car",
        "bird",
        "cat",
        "deer",
        "dog",
        "frog",
        "horse",
        "ship",
        "truck"
    )

    cm = confusion_matrix(y_true, y_pred)
    cm_percent = (cm.astype(float) / cm.sum(axis=1)[:, np.newaxis] * 100)

    fig, ax = plt.subplots(figsize=(9,8))

    disp = ConfusionMatrixDisplay(confusion_matrix=cm_percent, display_labels=classes)
    disp.plot(ax=ax, cmap="Blues", colorbar=True, values_format=".1f")

    plt.title("Normalized Confusion Matrix", fontsize=16, fontweight="bold")
    plt.xlabel("Predicted Class")
    plt.ylabel("True Class")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()