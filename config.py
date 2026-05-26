"""
Configuration for ANN AI Text Detector
=======================================
Edit values here — no need to touch ann_ai_detector.py
"""

# ── Data ──────────────────────────────────────────────────
DATASET_PATH = 'ai_detector_dataset.csv'
SAMPLES_PER_CLASS = 100          # rows to sample from each label
RANDOM_SEED = 42
TEST_SIZE = 0.2                    # fraction held out for testing

# ── TF-IDF ────────────────────────────────────────────────
TFIDF_MAX_FEATURES = 10000
TFIDF_NGRAM_RANGE = (1, 2)        # unigrams + bigrams
TFIDF_STOP_WORDS = 'english'

# ── Network architecture ─────────────────────────────────
# Each tuple = (neurons, activation, dropout_rate)
# Layers are stacked in order; dropout_rate=0 means no Dropout layer
HIDDEN_LAYERS = [
    (32, 'relu', 0.4),
    (16, 'relu', 0.3),
]
OUTPUT_ACTIVATION = 'sigmoid'      # sigmoid for binary classification

# ── Compile ───────────────────────────────────────────────
OPTIMIZER = 'adam'
LEARNING_RATE = 0.00003            # default Adam LR (set to None to use optimizer default)
LOSS = 'binary_crossentropy'

# ── Training ──────────────────────────────────────────────
EPOCHS = 10
BATCH_SIZE = 32
VALIDATION_SPLIT = 0.2             # fraction of training data used for validation

# ── Early stopping ────────────────────────────────────────
ES_MONITOR = 'val_accuracy'
ES_MODE = 'max'                    # 'max' for accuracy, 'min' for loss
ES_PATIENCE = 3
ES_RESTORE_BEST = True

# ── Feature importance ────────────────────────────────────
FI_TOP_N = 3                      # how many features to display
FI_CANDIDATES = 10                 # pre-screen candidates via model weights
FI_TEST_SAMPLES = 100              # test samples used for permutation
FI_REPEATS = 2                     # permutation repeats per feature
