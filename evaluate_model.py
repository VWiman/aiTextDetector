import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, roc_curve, auc, precision_recall_curve, average_precision_score, confusion_matrix
# ============================================================
# EVALUATE MODEL
# ============================================================
def evaluate_model(model, history, X_test, y_test, le):
    # Predictions
    y_pred_prob = model.predict(X_test).flatten()
    y_pred = (y_pred_prob >= 0.5).astype(int)

    # Classification Report
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=le.classes_))

    # Test loss and accuracy
    test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=0)
    print(f"Test Loss:     {test_loss:.4f}")
    print(f"Test Accuracy: {test_accuracy:.4f}")

    # --- Plot: Loss and Accuracy over Epochs ---
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].plot(history.history['loss'], label='Training Loss', color='#2196F3')
    axes[0].plot(history.history['val_loss'], label='Validation Loss', color='#FF5722')
    axes[0].set_title('Loss over Epochs', fontsize=14)
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(history.history['accuracy'], label='Training Accuracy', color='#2196F3')
    axes[1].plot(history.history['val_accuracy'], label='Validation Accuracy', color='#FF5722')
    axes[1].set_title('Accuracy over Epochs', fontsize=14)
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Accuracy')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('training_curves.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved: training_curves.png")

    # --- Plot: Confusion Matrix ---
    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots(figsize=(7, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=le.classes_, yticklabels=le.classes_, ax=ax)
    ax.set_title('Confusion Matrix', fontsize=14)
    ax.set_xlabel('Predicted')
    ax.set_ylabel('Actual')
    plt.tight_layout()
    plt.savefig('confusion_matrix.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved: confusion_matrix.png")

    # --- Plot: ROC Curve & Precision-Recall Curve ---
    fpr, tpr, _ = roc_curve(y_test, y_pred_prob)
    roc_auc = auc(fpr, tpr)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].plot(fpr, tpr, color='#2196F3', lw=2, label=f'ROC curve (AUC = {roc_auc:.4f})')
    axes[0].plot([0, 1], [0, 1], color='gray', linestyle='--', lw=1)
    axes[0].set_title('ROC Curve', fontsize=14)
    axes[0].set_xlabel('False Positive Rate')
    axes[0].set_ylabel('True Positive Rate')
    axes[0].legend(loc='lower right')
    axes[0].grid(True, alpha=0.3)

    precision, recall, _ = precision_recall_curve(y_test, y_pred_prob)
    avg_precision = average_precision_score(y_test, y_pred_prob)

    axes[1].plot(recall, precision, color='#FF5722', lw=2,
                label=f'PR curve (AP = {avg_precision:.4f})')
    axes[1].set_title('Precision-Recall Curve', fontsize=14)
    axes[1].set_xlabel('Recall')
    axes[1].set_ylabel('Precision')
    axes[1].legend(loc='lower left')
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('roc_pr_curves.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved: roc_pr_curves.png")