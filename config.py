"""
Configuration for ANN AI Text Detector
=======================================
Edit values here — no need to touch train.py
"""

# ── Data ──────────────────────────────────────────────────
DATASET_PATH = 'ai_detector_dataset.csv'
SAMPLES_PER_CLASS = 100000          # rows to sample from each label
RANDOM_SEED = 42
TEST_SIZE = 0.3                    # fraction held out for testing

# ── TF-IDF (word channel) ─────────────────────────────────
TFIDF_MAX_FEATURES = 20000
TFIDF_NGRAM_RANGE = (1, 2)        # unigrams + bigrams
TFIDF_STOP_WORDS = 'english'

# ── TF-IDF (char channel) ─────────────────────────────────
TFIDF_CHAR_MAX_FEATURES = 8000
TFIDF_CHAR_NGRAM_RANGE = (3, 4)   # char_wb 3- to 4-grams

# ── Network architecture ─────────────────────────────────
# Each tuple = (neurons, activation, dropout_rate)
# Layers are stacked in order; dropout_rate=0 means no Dropout layer
HIDDEN_LAYERS = [
    (32, 'tanh', 0.4),
    (16, 'tanh', 0.3),
]
OUTPUT_ACTIVATION = 'sigmoid'      # sigmoid for binary classification

# ── Compile ───────────────────────────────────────────────
OPTIMIZER = 'adam'
LEARNING_RATE = 0.00003            # default Adam LR (set to None to use optimizer default)
LOSS = 'binary_crossentropy'

# ── Training ──────────────────────────────────────────────
EPOCHS = 20
BATCH_SIZE = 32
VALIDATION_SPLIT = 0.2             # fraction of training data used for validation

# ── Early stopping ────────────────────────────────────────
ES_MONITOR = 'val_loss'
ES_MODE = 'min'                    # 'max' for accuracy, 'min' for loss
ES_PATIENCE = 3
ES_RESTORE_BEST = True

# ── Feature importance ────────────────────────────────────
FI_TOP_N = 5                      # how many features to display
FI_CANDIDATES = 20                 # pre-screen candidates via model weights
FI_TEST_SAMPLES = 200              # test samples used for permutation
FI_REPEATS = 4                     # permutation repeats per feature

# ── Grid Search ────────────────────────────────────
PARAM_GRID = {
    'model__n_neurons_layer1': [32],
    'model__n_neurons_layer2': [16],
    'model__dropout_rate': [0.2, 0.3, 0.4],
    'model__activation': ['relu', 'tanh'],
    'model__optimizer': [OPTIMIZER],
    'model__learning_rate': [0.001, LEARNING_RATE],
    'epochs': [EPOCHS],
    'batch_size': [BATCH_SIZE]
}