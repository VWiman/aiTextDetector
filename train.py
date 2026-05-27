import warnings
import tensorflow as tf
import pandas as pd
import numpy as np
import joblib
from scipy.sparse import hstack, csr_matrix
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from scikeras.wrappers import KerasClassifier
from config import *
from eda import eda
from evaluate_model import evaluate_model
from feature_importance import feature_importance
from text_features import remove_source_leaks, compute_text_stats
warnings.filterwarnings('ignore', category=Warning)

print("Tensorflow version: ", tf.__version__)
print("Tensorflow device: ", tf.config.list_physical_devices(
    device_type="GPU"
))

grid_search = False

# ============================================================
# 1. LOAD DATA & SAMPLE
# ============================================================
print("=" * 60)
print("1. LOADING DATA")
print("=" * 60)

df = pd.read_csv(DATASET_PATH)
print(f"Full dataset shape: {df.shape}")

df_human = df[df['label'] == 'Human'].sample(
    n=SAMPLES_PER_CLASS, random_state=RANDOM_SEED)
df_ai = df[df['label'] == 'AI'].sample(
    n=SAMPLES_PER_CLASS, random_state=RANDOM_SEED)
print(f"Sampled dataset human: {df_human.head()}")
print(f"Sampled dataset AI: {df_ai.head()}")
df_sample = pd.concat([df_human, df_ai]).reset_index(drop=True)

# ============================================================
# 2. EDA (Exploratory Data Analysis)
# ============================================================
print("=" * 60)
print("2. EDA (Exploratory Data Analysis)")
print("=" * 60)
eda(df_sample)

train_or_grid_search = input(
    'Do you want to skip training and do a grid search? "y" or "n": ')

if train_or_grid_search == "y":
    df = df_sample
    grid_search = True

# ============================================================
# 3. PREPROCESSING
# ============================================================
print("=" * 60)
print("3. PREPROCESSING DATA")
print("=" * 60)

# ============================================================
# 3.1 Clean data
# ============================================================
print(f"Cleaning data...")
df = df.dropna(subset=['text', 'label'])
df['text'] = df['text'].str.strip()
df['text'] = df['text'].apply(remove_source_leaks)
df = df[df['text'].str.len() > 0]
df = df.reset_index(drop=True)
print(f"After cleaning: {df.shape}")

# ============================================================
# 3.2 Split train/test (raw text, fits applied after split)
# ============================================================
X_text_train, X_text_test, y_train_raw, y_test_raw = train_test_split(
    df['text'], df['label'], test_size=TEST_SIZE,
    random_state=RANDOM_SEED, stratify=df['label'])

print(f"Training set: {X_text_train.shape[0]} samples")
print(f"Test set:     {X_text_test.shape[0]} samples")

# ============================================================
# 3.3 Encode labels (fit on train only)
# ============================================================
le = LabelEncoder()
y_train = le.fit_transform(y_train_raw)
y_test = le.transform(y_test_raw)
print(f"Label encoding: {dict(zip(le.classes_, le.transform(le.classes_)))}")

# ============================================================
# 3.4 Word TF-IDF Vectorization (fit on train only)
# ============================================================
print(f"Fitting word TF-IDF vectorizer (max {TFIDF_MAX_FEATURES} features)...")
tfidf = TfidfVectorizer(
    max_features=TFIDF_MAX_FEATURES,
    stop_words=TFIDF_STOP_WORDS,
    ngram_range=TFIDF_NGRAM_RANGE,
    dtype=np.float32
)
X_train_word = tfidf.fit_transform(X_text_train)
X_test_word = tfidf.transform(X_text_test)

print(f"Word TF-IDF matrix shape: {X_train_word.shape}")

# ============================================================
# 3.5 Char TF-IDF Vectorization (fit on train only)
# ============================================================
print(f"Fitting char TF-IDF vectorizer (max {TFIDF_CHAR_MAX_FEATURES} features)...")
tfidf_char = TfidfVectorizer(
    max_features=TFIDF_CHAR_MAX_FEATURES,
    analyzer='char_wb',
    ngram_range=TFIDF_CHAR_NGRAM_RANGE,
    dtype=np.float32
)
X_train_char = tfidf_char.fit_transform(X_text_train)
X_test_char = tfidf_char.transform(X_text_test)

print(f"Char TF-IDF matrix shape: {X_train_char.shape}")

# ============================================================
# 3.6 Compute text statistics (word count, avg sentence length, lexical diversity)
# ============================================================
print(f"Computing text statistics...")
stats_train = compute_text_stats(X_text_train)
stats_test = compute_text_stats(X_text_test)
print(f"Text stats matrix shape: {stats_train.shape}")

# ============================================================
# 3.7 Concatenate feature channels (word TF-IDF + char TF-IDF + text stats)
# ============================================================
X_train = hstack([X_train_word, X_train_char, csr_matrix(stats_train)], format='csr')
X_test = hstack([X_test_word, X_test_char, csr_matrix(stats_test)], format='csr')

# hstack(format='csr') doesn't sort column indices within rows; TF's SparseToDense requires it
X_train.sort_indices()
X_test.sort_indices()

print(f"Combined feature matrix shape: {X_train.shape}")

# Free per-channel matrices now that they are merged
del X_train_word, X_train_char, stats_train
del X_test_word, X_test_char, stats_test

# ============================================================
# 3.8 Scale features (fit on train only)
# ============================================================
scaler = StandardScaler(with_mean=False, copy=False)

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

if grid_search == False:
    # ============================================================
    # 4 BUILD MODEL
    # ============================================================
    print("=" * 60)
    print("4. BUILDING MODEL")
    print("=" * 60)

    model = tf.keras.models.Sequential()

    model.add(tf.keras.Input(shape=(X_train.shape[1],)))

    # ============================================================
    # 4.1 Add hidden layers
    # ============================================================
    for units, activation, dropout_rate in HIDDEN_LAYERS:
        model.add(
            tf.keras.layers.Dense(
                units,
                activation=activation,
                kernel_regularizer=tf.keras.regularizers.L2(1e-4)
            )
        )

        if dropout_rate > 0:
            model.add(tf.keras.layers.Dropout(dropout_rate))

    # ============================================================
    # 4.2 Add one output layer with binary classification
    # ============================================================
    model.add(tf.keras.layers.Dense(1, activation=OUTPUT_ACTIVATION))

    if LEARNING_RATE is not None:
        optimizer = tf.keras.optimizers.get({
            'class_name': OPTIMIZER,
            'config': {'learning_rate': LEARNING_RATE}
        })
    else:
        optimizer = OPTIMIZER

    model.compile(
        optimizer=optimizer,
        loss=LOSS,
        metrics=["accuracy"]
    )

    model.summary()

    # ============================================================
    # 5. TRAIN MODEL
    # ============================================================
    print("=" * 60)
    print("5. TRAINING MODEL")
    print("=" * 60)

    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor=ES_MONITOR,
        mode=ES_MODE,
        patience=ES_PATIENCE,
        restore_best_weights=ES_RESTORE_BEST,
        verbose=1
    )

    X_fit, X_val, y_fit, y_val = train_test_split(
        X_train,
        y_train,
        test_size=VALIDATION_SPLIT,
        random_state=RANDOM_SEED,
        stratify=y_train
    )

    history = model.fit(
        X_fit, y_fit,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        validation_data=(X_val, y_val),
        callbacks=[early_stopping],
        verbose=1
    )

    # ============================================================
    # 5.1 Flatten predictions
    # ============================================================
    y_pred = model.predict(X_test)
    y_pred_classes = (y_pred.ravel() >= 0.5).astype(int)

    # ============================================================
    # 6. SAVE MODEL
    # ============================================================
    print("\n" + "=" * 60)
    print("6. SAVING MODEL")
    print("=" * 60)

    model.save('ann_ai_detector_model.keras')
    print("Model saved: ann_ai_detector_model.keras")

    joblib.dump(tfidf, 'tfidf_vectorizer.joblib')
    joblib.dump(tfidf_char, 'tfidf_char_vectorizer.joblib')
    joblib.dump(le, 'label_encoder.joblib')
    joblib.dump(scaler, 'scaler.joblib')

    print("Word TF-IDF vectorizer saved: tfidf_vectorizer.joblib")
    print("Char TF-IDF vectorizer saved: tfidf_char_vectorizer.joblib")
    print("Label encoder saved: label_encoder.joblib")
    print("Standard saved: scaler.joblib")

    # ============================================================
    # 7. EVALUATE MODEL
    # ============================================================
    print("\n" + "=" * 60)
    print("7. EVALUATE MODEL")
    print("=" * 60)
    evaluate_model(model, history, X_test, y_test, le)

    # Classification Report:
    #               precision    recall  f1-score   support

    #           AI       0.83      0.82      0.82     49835
    #        Human       0.82      0.83      0.83     49835

    #     accuracy                           0.83     99670
    #    macro avg       0.83      0.83      0.83     99670
    # weighted avg       0.83      0.83      0.83     99670
    
    # ============================================================
    # 8. FEATURE IMPORTANCE
    # ============================================================
    print("\n" + "=" * 60)
    print("8. FEATURE IMPORTANCE")
    print("=" * 60)
    combined_feature_names = (
        list(tfidf.get_feature_names_out()) +
        list(tfidf_char.get_feature_names_out()) +
        ['word_count', 'avg_sent_len', 'lexical_diversity']
    )
    feature_importance(model, X_test, y_test, combined_feature_names)
else:
    # ============================================================
    # 9. GRID SEARCH
    # ============================================================
    def build_model(n_neurons_layer1=32, n_neurons_layer2=16,
                    dropout_rate=0.3, activation='relu',
                    learning_rate=LEARNING_RATE, optimizer=OPTIMIZER):
        m = tf.keras.models.Sequential()
        m.add(tf.keras.Input(shape=(X_train.shape[1],)))
        m.add(tf.keras.layers.Dense(n_neurons_layer1, activation=activation))
        m.add(tf.keras.layers.Dropout(dropout_rate))
        m.add(tf.keras.layers.Dense(n_neurons_layer2, activation=activation))
        m.add(tf.keras.layers.Dropout(dropout_rate))
        m.add(tf.keras.layers.Dense(1, activation='sigmoid'))
        opt = tf.keras.optimizers.get({
            'class_name': optimizer,
            'config': {'learning_rate': learning_rate}
        })
        m.compile(optimizer=opt, loss='binary_crossentropy', metrics=['accuracy'])
        return m

    estimator = KerasClassifier(
        model=build_model,
        verbose=1,
        random_state=RANDOM_SEED
    )

    param_grid = PARAM_GRID

    grid = GridSearchCV(estimator=estimator, param_grid=param_grid,
                        n_jobs=2, cv=3, scoring='balanced_accuracy', verbose=1)
    grid_result = grid.fit(X_train, y_train)

    print("Best parameters:", grid_result.best_params_)
    print("Best result (accuracy):", grid_result.best_score_)

    # Best parameters: {'batch_size': 32, 'epochs': 12, 'model__activation': 'tanh', 'model__dropout_rate': 0.4, 'model__learning_rate': 0.001, 'model__n_neurons_layer1': 32, 'model__n_neurons_layer2': 16, 'model__optimizer': 'adam'}
    # Best result (accuracy): 0.6095008051529791
    # We changed from 'reul' to 'tanh' based on these results