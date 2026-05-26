import tensorflow as tf
import pandas as pd
import joblib
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report
from config import *
warnings.filterwarnings('ignore')

print("Tensorflow version: ", tf.__version__)
print("Tensorflow device: ", tf.config.list_physical_devices(
    device_type="GPU"
))

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
df = df[df['text'].str.len() > 0]
df = df.reset_index(drop=True)
print(f"After cleaning: {df.shape}")

# ============================================================
# 3.2 Encode labels
# ============================================================
le = LabelEncoder()
df['label_encoded'] = le.fit_transform(df['label'])
print(f"Label encoding: {dict(zip(le.classes_, le.transform(le.classes_)))}")


# ============================================================
# 3.3 TF-IDF Vectorization
# ============================================================
print(f"Fitting TF-IDF vectorizer (max {TFIDF_MAX_FEATURES} features)...")
tfidf = TfidfVectorizer(
    max_features=TFIDF_MAX_FEATURES,
    stop_words=TFIDF_STOP_WORDS,
    ngram_range=TFIDF_NGRAM_RANGE
)
X = tfidf.fit_transform(df['text'])
y = df["label_encoded"]

print(f"TF-IDF matrix shape: {X.shape}")

# ============================================================
# 3.4 Split train/test + scale
# ============================================================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=TEST_SIZE, random_state=RANDOM_SEED, stratify=y)

print(f"Training set: {X_train.shape[0]} samples")
print(f"Test set:     {X_test.shape[0]} samples")

scaler = StandardScaler(with_mean=False)

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

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
    model.add(tf.keras.layers.Dense(units, activation=activation))

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

history = model.fit(
    X_train, y_train,
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    validation_split=VALIDATION_SPLIT,
    callbacks=[early_stopping],
    verbose=1
)

# ============================================================
# 5.1 Flatten predictions
# ============================================================
y_pred = model.predict(X_test)
y_pred_classes = (y_pred.ravel() >= 0.5).astype(int)

print(
    "Classification Report:\n",
    classification_report(
        y_test,
        y_pred_classes,
        target_names=le.classes_
    )
)

test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=0)
print(f"Test Loss:     {test_loss:.4f}")
print(f"Test Accuracy: {test_accuracy:.4f}")

# ============================================================
# 6. SAVE MODEL
# ============================================================
print("\n" + "=" * 60)
print("6. SAVING MODEL")
print("=" * 60)

model.save('ann_ai_detector_model.keras')
print("Model saved: ann_ai_detector_model.keras")

joblib.dump(tfidf, 'tfidf_vectorizer.joblib')
joblib.dump(le, 'label_encoder.joblib')

print("TF-IDF vectorizer saved: tfidf_vectorizer.joblib")
print("Label encoder saved: label_encoder.joblib")

# ============================================================
# 7. EVALUATE MODEL
# ============================================================

# ============================================================
# 8. FEATURE IMPORTANCE
# ============================================================