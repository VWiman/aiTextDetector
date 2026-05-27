import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
from config import *
# ============================================================
# FEATURE IMPORTANCE
# ============================================================
def feature_importance(model, X_test, y_test, feature_names):
    # Step 1: Pre-screen using model weights
    first_layer_weights = np.abs(model.layers[0].get_weights()[0])
    weight_importance = first_layer_weights.sum(axis=1)
    candidate_idx = weight_importance.argsort()[-FI_CANDIDATES:]
    print(f"Pre-screened {FI_CANDIDATES} candidate features from model weights")

    # Step 2: Permutation importance only on candidates
    X_test_small = X_test[:FI_TEST_SAMPLES]
    y_test_small = y_test[:FI_TEST_SAMPLES]
    feature_names = np.asarray(feature_names)

    y_base_prob = model.predict(X_test_small, verbose=0).flatten()
    baseline_acc = np.mean((y_base_prob >= 0.5).astype(int) == y_test_small)

    importances = np.zeros(FI_CANDIDATES)
    for idx, feat_i in enumerate(tqdm(candidate_idx, desc="Permutation importance")):
        drops = []
        for r in range(FI_REPEATS):
            X_permuted = X_test_small.copy()
            np.random.seed(r)
            col = np.asarray(X_permuted[:, feat_i].todense()).flatten()
            X_permuted[:, feat_i] = np.random.permutation(col).reshape(-1, 1)
            y_perm_prob = model.predict(X_permuted, verbose=0).flatten()
            perm_acc = np.mean((y_perm_prob >= 0.5).astype(int) == y_test_small)
            drops.append(baseline_acc - perm_acc)
        importances[idx] = np.mean(drops)

    top_local_idx = importances.argsort()[-FI_TOP_N:]
    top_feature_idx = candidate_idx[top_local_idx]
    top_importances = importances[top_local_idx]

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(range(FI_TOP_N), top_importances, color='#2196F3')
    ax.set_yticks(range(FI_TOP_N))
    ax.set_yticklabels(feature_names[top_feature_idx])
    ax.set_title(f'Top {FI_TOP_N} Feature Importances (Permutation)', fontsize=14)
    ax.set_xlabel('Mean Accuracy Decrease')
    ax.grid(True, alpha=0.3, axis='x')
    plt.tight_layout()
    plt.savefig('feature_importance.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved: feature_importance.png")