# Python code to build the hybrid malware detection pipeline the user requested.
# This will run in the notebook environment and display key outputs and figures.
# It attempts to be flexible and handle common dataset column layouts:
# - If there's a text column named 'api_sequence' (or similar), it will tokenize+pad and use an Embedding+LSTM.
# - If there are TF-IDF numeric columns (prefixed 'tfidf_' or 'X_tfidf'), it will reshape them into a sequence-like
#   shape for LSTM input (timesteps = number of TF-IDF features, features=1).
# Assumptions/notes:
# - Label column is searched as 'label', 'target', 'y', or the last column if none found.
# - Static features are taken as numeric columns excluding dynamic/text columns and label.
# - The script trains RandomForest and LSTM, saves both models, computes metrics, and creates plots.
# - If xgboost is available, it will optionally train XGBoost for the advanced comparison.
# - Uses matplotlib for plots (no seaborn), and caas_jupyter_tools.display_dataframe_to_user to show the summary table.
#
# Files saved:
# - /mnt/data/rf_model.joblib
# - /mnt/data/lstm_model (Keras SavedModel directory)
#
# Required packages: scikit-learn, tensorflow, pandas, numpy, matplotlib, joblib. xgboost optional.
#
# Run time may vary depending on dataset size and environment resources.

import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score, roc_curve, confusion_matrix
import joblib
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

# TensorFlow/Keras
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout, Masking, InputLayer
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

# helper for display
try:
    from caas_jupyter_tools import display_dataframe_to_user
except Exception:
    display_dataframe_to_user = None

DATA_PATH = "/mnt/data/Malware_Analysis.csv"
assert os.path.exists(DATA_PATH), f"Dataset not found at {DATA_PATH}"

# Load dataset
df = pd.read_csv(DATA_PATH)
print("Loaded dataset with shape:", df.shape)

# Detect label column
label_candidates = [c for c in df.columns if c.lower() in ("label","target","y","class")]
if label_candidates:
    label_col = label_candidates[0]
else:
    # fallback: last column
    label_col = df.columns[-1]
print("Using label column:", label_col)

# Detect dynamic column (text sequences) - common names: api_sequence, apis, calls, sequence
dynamic_text_candidates = [c for c in df.columns if any(k in c.lower() for k in ("api","call","sequence","trace")) and df[c].dtype == object]
dynamic_text_col = dynamic_text_candidates[0] if dynamic_text_candidates else None
if dynamic_text_col:
    print("Detected dynamic text column:", dynamic_text_col)
else:
    print("No obvious dynamic text column detected. Will look for TF-IDF numeric columns.")

# Detect TF-IDF numeric columns (prefixes)
tfidf_cols = [c for c in df.columns if c.lower().startswith("tfidf") or c.lower().startswith("x_tfidf") or c.lower().startswith("tfidf_")]
if tfidf_cols:
    print(f"Detected TF-IDF numeric columns (count {len(tfidf_cols)}).")
else:
    # also check for many numeric columns that might represent TF-IDF features (large number)
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    # exclude label if numeric
    numeric_cols_no_label = [c for c in numeric_cols if c != label_col]
    if len(numeric_cols_no_label) > 50:  # heuristic threshold
        # assume many numeric columns are TF-IDF or static combined; we will attempt to split by known static names (entropy, dll, registry)
        tfidf_cols = numeric_cols_no_label  # fallback; we'll separate static later by heuristics
        print(f"No explicit tfidf_* columns but found many numeric columns (count {len(tfidf_cols)}). Attempting heuristic split.")

# Prepare labels
y = df[label_col].copy()
if y.dtype == object or y.dtype == bool:
    le = LabelEncoder()
    y = le.fit_transform(y)
    classes = le.classes_
else:
    classes = np.unique(y)
n_classes = len(np.unique(y))
print("Number of classes:", n_classes)

# Identify static feature columns (heuristic): look for columns with keywords or small count of columns
exclude_cols = {label_col}
if dynamic_text_col:
    exclude_cols.add(dynamic_text_col)
for c in tfidf_cols:
    exclude_cols.add(c)

all_numeric = df.select_dtypes(include=[np.number]).columns.tolist()
static_candidates = [c for c in all_numeric if c not in exclude_cols]

# Heuristic: features that mention 'entropy','dll','registry','file','size' are likely static
static_keywords = ("entropy","dll","registry","file","size","count","entropy","import","hash","md5","sha","pe_","pe.")
static_features = [c for c in df.columns if any(k in c.lower() for k in static_keywords) and c != label_col]
# Ensure static_features are numeric
static_features = [c for c in static_features if c in df.select_dtypes(include=[np.number]).columns]

# If we didn't find explicit static features, use numeric columns that are not tfidf candidates
if not static_features:
    static_features = [c for c in all_numeric if c != label_col and c not in tfidf_cols]

print(f"Selected {len(static_features)} static feature columns (examples):", static_features[:8])

# Build X_static
X_static = df[static_features].copy()
# Impute and scale static features
imputer = SimpleImputer(strategy="median")
X_static_imputed = imputer.fit_transform(X_static)
scaler = StandardScaler()
X_static_scaled = scaler.fit_transform(X_static_imputed)

# Build dynamic features for LSTM
use_text_sequence_for_lstm = False
if dynamic_text_col:
    # Use Tokenizer => sequences => pad_sequences => Embedding + LSTM
    use_text_sequence_for_lstm = True
    texts = df[dynamic_text_col].fillna("").astype(str).tolist()
    # basic cleaning: replace commas with spaces if present
    texts = [t.replace(",", " ") for t in texts]
    MAX_NUM_WORDS = 20000
    tokenizer = Tokenizer(num_words=MAX_NUM_WORDS, oov_token="<OOV>")
    tokenizer.fit_on_texts(texts)
    sequences = tokenizer.texts_to_sequences(texts)
    # Choose maxlen as 95th percentile of lengths to avoid extreme padding
    lengths = [len(s) for s in sequences]
    maxlen = int(np.percentile(lengths, 95)) if len(lengths) > 0 else 100
    maxlen = max(20, maxlen)  # minimum sequence length
    X_seq = pad_sequences(sequences, maxlen=maxlen, padding="post", truncating="post")
    print("Prepared token sequences for LSTM with shape:", X_seq.shape)
else:
    # Use TF-IDF numeric columns (tfidf_cols must exist)
    if not tfidf_cols:
        raise RuntimeError("Could not find dynamic text column or TF-IDF numeric columns in dataset. Please provide one of them.")
    X_tfidf = df[tfidf_cols].fillna(0).astype(float).values
    # Heuristic reshape: treat each TF-IDF dimension as a timestep with feature=1
    # So LSTM input shape => (samples, timesteps, features) = (n_samples, n_tfidf_features, 1)
    X_seq = X_tfidf.reshape((X_tfidf.shape[0], X_tfidf.shape[1], 1))
    print("Prepared TF-IDF numeric data reshaped for LSTM with shape:", X_seq.shape)

# Use consistent train/test split for both models
RANDOM_STATE = 42
Xs_train, Xs_test, y_train, y_test = train_test_split(X_static_scaled, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y if n_classes>1 else None)
if use_text_sequence_for_lstm:
    Xd_train, Xd_test = train_test_split(X_seq, test_size=0.2, random_state=RANDOM_STATE, stratify=y if n_classes>1 else None)
else:
    Xd_train, Xd_test = train_test_split(X_seq, test_size=0.2, random_state=RANDOM_STATE, stratify=y if n_classes>1 else None)

print("Static train/test shapes:", Xs_train.shape, Xs_test.shape)
print("Dynamic (LSTM) train/test shapes:", Xd_train.shape, Xd_test.shape)
print("y_train distribution:", np.bincount(y_train) if n_classes>1 else np.bincount(y_train.astype(int)))

# ------------------- RandomForest on static features -------------------
rf = RandomForestClassifier(n_estimators=200, random_state=RANDOM_STATE, n_jobs=-1)
print("Training RandomForest on static features...")
rf.fit(Xs_train, y_train)
print("RandomForest training complete.")

# Save RandomForest model
rf_path = "/mnt/data/rf_model.joblib"
joblib.dump({"model": rf, "imputer": imputer, "scaler": scaler, "static_features": static_features}, rf_path)
print("Saved RandomForest pipeline to:", rf_path)

# RF predictions (probabilities)
if n_classes == 2:
    rf_proba_test = rf.predict_proba(Xs_test)[:,1]
else:
    rf_proba_test = rf.predict_proba(Xs_test)  # shape (n_samples, n_classes)

# ------------------- LSTM on dynamic features -------------------
# Build LSTM model depending on whether we used token sequences (with vocab) or reshaped TF-IDF numeric timesteps
tf.keras.backend.clear_session()
if use_text_sequence_for_lstm:
    vocab_size = min(MAX_NUM_WORDS, len(tokenizer.word_index) + 1)
    embedding_dim = 128
    lstm_units = 128
    model = Sequential([
        InputLayer(input_shape=(Xd_train.shape[1],)),  # sequences of ints
        Embedding(input_dim=vocab_size, output_dim=embedding_dim, mask_zero=True),
        LSTM(lstm_units, return_sequences=False),
        Dropout(0.3),
        Dense(64, activation="relu"),
        Dense(n_classes, activation="softmax" if n_classes>2 else "sigmoid")
    ])
    # For binary, we'll keep a single output with sigmoid; for multi-class, softmax and categorical labels
    if n_classes == 2:
        model = Sequential([
            InputLayer(input_shape=(Xd_train.shape[1],)),
            Embedding(input_dim=vocab_size, output_dim=embedding_dim, mask_zero=True),
            LSTM(lstm_units),
            Dropout(0.3),
            Dense(32, activation="relu"),
            Dense(1, activation="sigmoid")
        ])
        loss = "binary_crossentropy"
        metrics = ["accuracy"]
    else:
        model = Sequential([
            InputLayer(input_shape=(Xd_train.shape[1],)),
            Embedding(input_dim=vocab_size, output_dim=embedding_dim, mask_zero=True),
            LSTM(lstm_units),
            Dropout(0.3),
            Dense(64, activation="relu"),
            Dense(n_classes, activation="softmax")
        ])
        loss = "sparse_categorical_crossentropy"
        metrics = ["accuracy"]
    model.compile(optimizer="adam", loss=loss, metrics=metrics)
else:
    # Numeric timesteps (e.g., TF-IDF reshaped)
    timesteps = Xd_train.shape[1]
    features_per_timestep = Xd_train.shape[2]
    lstm_units = 128
    model = Sequential([
        InputLayer(input_shape=(timesteps, features_per_timestep)),
        LSTM(lstm_units),
        Dropout(0.3),
        Dense(64, activation="relu"),
        Dense(n_classes, activation="softmax" if n_classes>2 else "sigmoid")
    ])
    if n_classes == 2:
        model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    else:
        model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])

model.summary()

# Prepare labels for LSTM training
if n_classes == 2:
    y_train_lstm = y_train
    y_test_lstm = y_test
else:
    y_train_lstm = y_train
    y_test_lstm = y_test

# Callbacks and training
epochs = 25
batch_size = 64 if Xd_train.shape[0] >= 64 else 16
es = EarlyStopping(monitor="val_loss", patience=4, restore_best_weights=True)
lstm_model_path = "/mnt/data/lstm_model_best.h5"
mc = ModelCheckpoint(lstm_model_path, monitor="val_loss", save_best_only=True, save_weights_only=False)

print("Training LSTM...")
history = model.fit(Xd_train, y_train_lstm, validation_data=(Xd_test, y_test_lstm), epochs=epochs, batch_size=batch_size, callbacks=[es, mc], verbose=2)
print("LSTM training complete. Best model saved to:", lstm_model_path)

# Save final model (SavedModel format)
saved_lstm_dir = "/mnt/data/lstm_model"
model.save(saved_lstm_dir)
print("Saved full LSTM model to:", saved_lstm_dir)

# Plot LSTM training curves (loss and accuracy)
fig1 = plt.figure(figsize=(8,4))
plt.plot(history.history.get("loss", []))
plt.plot(history.history.get("val_loss", []))
plt.title("LSTM Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend(["train_loss", "val_loss"])
plt.grid(True)
plt.show()

# Accuracy plot if available
if "accuracy" in history.history:
    fig2 = plt.figure(figsize=(8,4))
    plt.plot(history.history.get("accuracy", []))
    plt.plot(history.history.get("val_accuracy", []))
    plt.title("LSTM Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend(["train_acc", "val_acc"])
    plt.grid(True)
    plt.show()

# LSTM predicted probabilities
if n_classes == 2:
    lstm_proba_test = model.predict(Xd_test).ravel()
else:
    lstm_proba_test = model.predict(Xd_test)  # shape (n_samples, n_classes)

# ------------------- Ensemble via weighted averaging -------------------
weight_lstm = 0.6
weight_rf = 0.4

# For binary classification
if n_classes == 2:
    # Ensure rf_proba_test and lstm_proba_test are aligned probabilities for the positive class
    # rf_proba_test: shape (n_samples,), lstm_proba_test: shape (n_samples,)
    ensemble_proba = weight_lstm * lstm_proba_test + weight_rf * rf_proba_test
    ensemble_pred = (ensemble_proba >= 0.5).astype(int)
else:
    # multiclass: rf_proba_test and lstm_proba_test are arrays (n_samples, n_classes)
    ensemble_proba = weight_lstm * lstm_proba_test + weight_rf * rf_proba_test
    ensemble_pred = np.argmax(ensemble_proba, axis=1)

# ------------------- Metrics -------------------
def compute_metrics(y_true, y_pred, y_proba=None):
    acc = accuracy_score(y_true, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="weighted", zero_division=0)
    if y_proba is not None:
        try:
            if n_classes == 2:
                roc = roc_auc_score(y_true, y_proba)
            else:
                roc = roc_auc_score(y_true, y_proba, multi_class="ovr")
        except Exception:
            roc = np.nan
    else:
        roc = np.nan
    return {"accuracy": acc, "precision": precision, "recall": recall, "f1": f1, "roc_auc": roc}

# RF preds
rf_pred = rf.predict(Xs_test)
rf_metrics = compute_metrics(y_test, rf_pred, rf_proba_test if n_classes==2 else rf_proba_test)
# LSTM preds
if n_classes == 2:
    lstm_pred = (lstm_proba_test >= 0.5).astype(int)
else:
    lstm_pred = np.argmax(lstm_proba_test, axis=1)
lstm_metrics = compute_metrics(y_test, lstm_pred, lstm_proba_test)
# Ensemble metrics
ensemble_metrics = compute_metrics(y_test, ensemble_pred, ensemble_proba)

# Summary table
results_df = pd.DataFrame([
    {"model":"RandomForest", **rf_metrics},
    {"model":"LSTM", **lstm_metrics},
    {"model":"Ensemble (0.6 LSTM + 0.4 RF)", **ensemble_metrics},
])
print("\nEvaluation Results:")
print(results_df)

# Display dataframe to user if helper exists, else print
if display_dataframe_to_user is not None:
    display_dataframe_to_user("Model Comparison Results", results_df)
else:
    display(results_df)

# ------------------- ROC curves -------------------
plt.figure(figsize=(8,6))
if n_classes == 2:
    # RF
    fpr_rf, tpr_rf, _ = roc_curve(y_test, rf_proba_test)
    fpr_lstm, tpr_lstm, _ = roc_curve(y_test, lstm_proba_test)
    fpr_ens, tpr_ens, _ = roc_curve(y_test, ensemble_proba)
    plt.plot(fpr_rf, tpr_rf, label=f"RandomForest (AUC={rf_metrics['roc_auc']:.3f})")
    plt.plot(fpr_lstm, tpr_lstm, label=f"LSTM (AUC={lstm_metrics['roc_auc']:.3f})")
    plt.plot(fpr_ens, tpr_ens, label=f"Ensemble (AUC={ensemble_metrics['roc_auc']:.3f})")
    plt.plot([0,1],[0,1],"--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve Comparison for Malware Detection Models")
    plt.legend()
    plt.grid(True)
    plt.show()
else:
    print("ROC curves for multiclass classification require one-vs-rest plotting; omitted for brevity.")

# ------------------- Confusion matrices -------------------
cm_rf = confusion_matrix(y_test, rf_pred)
cm_lstm = confusion_matrix(y_test, lstm_pred)
cm_ens = confusion_matrix(y_test, ensemble_pred)

# Plot side-by-side confusion matrices (each as its own figure to respect python_user_visible rules)
def plot_confusion(cm, title):
    plt.figure(figsize=(5,4))
    plt.imshow(cm, interpolation="nearest")
    plt.title(title)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, int(cm[i,j]), ha="center", va="center")
    plt.colorbar()
    plt.show()

plot_confusion(cm_rf, "Confusion Matrix - RandomForest")
plot_confusion(cm_lstm, "Confusion Matrix - LSTM")
plot_confusion(cm_ens, "Confusion Matrix - Ensemble")

# ------------------- Optional: XGBoost and feature importances -------------------
try:
    import xgboost as xgb
    print("Training XGBoost for comparison...")
    xgb_model = xgb.XGBClassifier(use_label_encoder=False, eval_metric="logloss", n_estimators=200, random_state=RANDOM_STATE)
    # Train XGBoost on static features (like RF)
    xgb_model.fit(Xs_train, y_train)
    xgb_proba_test = xgb_model.predict_proba(Xs_test)[:,1] if n_classes==2 else xgb_model.predict_proba(Xs_test)
    xgb_pred = xgb_model.predict(Xs_test)
    xgb_metrics = compute_metrics(y_test, xgb_pred, xgb_proba_test)
    results_df = results_df.append({"model":"XGBoost", **xgb_metrics}, ignore_index=True)
    if display_dataframe_to_user is not None:
        display_dataframe_to_user("Extended Model Comparison Results", results_df)
    else:
        display(results_df)
    # Feature importances
    try:
        importances_rf = rf.feature_importances_
        top_idx = np.argsort(importances_rf)[::-1][:20]
        top_features = [static_features[i] for i in top_idx]
        top_values = importances_rf[top_idx]
        feat_df = pd.DataFrame({"feature": top_features, "importance": top_values})
        if display_dataframe_to_user is not None:
            display_dataframe_to_user("Top RandomForest Feature Importances", feat_df)
        else:
            display(feat_df)
    except Exception as e:
        print("Could not compute RF feature importances:", e)
except Exception as e:
    print("xgboost not available or failed to run - skipping XGBoost step. Error:", e)

# Final: write out ensemble predictions to CSV for review
out_df = pd.DataFrame({"y_true": y_test, "rf_pred": rf_pred, "lstm_pred": lstm_pred, "ensemble_pred": ensemble_pred, "ensemble_proba": ensemble_proba if n_classes==2 else None})
out_csv = "/mnt/data/predictions_ensemble.csv"
out_df.to_csv(out_csv, index=False)
print("Saved ensemble predictions to:", out_csv)

print("\nCompleted training & evaluation. Models saved at:")
print(" - RandomForest:", rf_path)
print(" - LSTM (SavedModel):", saved_lstm_dir)
print(" - LSTM (best h5 checkpoint):", lstm_model_path)

