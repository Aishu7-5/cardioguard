"""
Cardiovascular Stroke Prediction – Deep Learning Model
=======================================================
Architecture : Deep Neural Network (DNN) with residual connections
Framework    : TensorFlow / Keras
"""

import os, warnings
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import joblib, json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection   import train_test_split, StratifiedKFold
from sklearn.preprocessing     import StandardScaler, LabelEncoder
from sklearn.metrics           import (classification_report,
                                       confusion_matrix,
                                       roc_auc_score, roc_curve)
from sklearn.utils.class_weight import compute_class_weight

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, regularizers, callbacks  # type: ignore

# ── Reproducibility ───────────────────────────────────────────────────────
SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)

# ─────────────────────────────────────────────────────────────────────────
#  1. LOAD DATA
# ─────────────────────────────────────────────────────────────────────────
print("=" * 60)
print("  CARDIOVASCULAR STROKE PREDICTION – MODEL TRAINING")
print("=" * 60)

df = pd.read_csv("stroke_clinical_dataset.csv")
print(f"\n✅  Loaded dataset  →  {df.shape[0]:,} rows × {df.shape[1]} cols")
print(f"   Stroke prevalence : {df.stroke.mean()*100:.1f}%")

# ─────────────────────────────────────────────────────────────────────────
#  2. FEATURE ENGINEERING
# ─────────────────────────────────────────────────────────────────────────
def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()

    # Composite cardiovascular risk score (0-10 scale)
    d["risk_score"] = (
        (d.hypertension       * 2.0) +
        (d.heart_disease      * 2.0) +
        (d.family_history     * 1.5) +
        ((d.bmi > 30).astype(float)             * 1.0) +
        ((d.avg_glucose_level > 125).astype(float) * 1.5) +
        ((d.cholesterol > 240).astype(float)    * 1.0) +
        ((d.hdl < 40).astype(float)             * 1.0)
    ).clip(0, 10)

    # Lifestyle risk (higher = worse)
    d["lifestyle_risk"] = (
        (d.smoking_status    * 1.5) +
        (d.alcohol_intake    * 0.8) +
        ((3 - d.physical_activity) * 1.2) +
        (d.stress_level      * 0.3)
    )

    # Age group (ordinal)
    d["age_group"] = pd.cut(
        d.age,
        bins=[0, 30, 45, 60, 75, 120],
        labels=[0, 1, 2, 3, 4]
    ).astype(float)

    # Pulse pressure (systolic - diastolic)
    d["pulse_pressure"] = d.systolic_bp - d.diastolic_bp

    # Mean arterial pressure
    d["mean_arterial_pressure"] = d.diastolic_bp + d.pulse_pressure / 3

    # Cholesterol ratio
    d["chol_hdl_ratio"] = (d.cholesterol / d.hdl.clip(1)).clip(0, 20)

    # BMI category
    d["bmi_category"] = pd.cut(
        d.bmi,
        bins=[0, 18.5, 25, 30, 35, 100],
        labels=[0, 1, 2, 3, 4]
    ).astype(float)

    # Glucose-BMI interaction
    d["glucose_bmi_interaction"] = (d.avg_glucose_level * d.bmi) / 1000

    # Hypertension severity
    d["bp_severity"] = np.where(
        (d.systolic_bp >= 180) | (d.diastolic_bp >= 120), 3,
        np.where((d.systolic_bp >= 160) | (d.diastolic_bp >= 100), 2,
        np.where((d.systolic_bp >= 130) | (d.diastolic_bp >= 80),  1, 0))
    )

    return d

df = engineer_features(df)

FEATURE_COLS = [
    # Original 20
    "age", "gender", "bmi", "height", "weight",
    "hypertension", "heart_disease", "ever_married", "family_history",
    "avg_glucose_level", "fasting_blood_sugar", "cholesterol", "hdl", "ldl",
    "smoking_status", "alcohol_intake", "physical_activity", "stress_level",
    "systolic_bp", "diastolic_bp",
    # Engineered
    "risk_score", "lifestyle_risk", "age_group", "pulse_pressure",
    "mean_arterial_pressure", "chol_hdl_ratio", "bmi_category",
    "glucose_bmi_interaction", "bp_severity",
]

X = df[FEATURE_COLS].values.astype(np.float32)
y = df["stroke"].values.astype(np.float32)

print(f"\n📊  Feature matrix  : {X.shape}")

# ─────────────────────────────────────────────────────────────────────────
#  3. SPLIT & SCALE
# ─────────────────────────────────────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=SEED, stratify=y
)
X_train, X_val, y_train, y_val = train_test_split(
    X_train, y_train, test_size=0.15, random_state=SEED, stratify=y_train
)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_val   = scaler.transform(X_val)
X_test  = scaler.transform(X_test)

joblib.dump(scaler, "scaler.pkl")
print(f"   Train : {len(X_train):,}  |  Val : {len(X_val):,}  |  Test : {len(X_test):,}")

# ─────────────────────────────────────────────────────────────────────────
#  4. CLASS WEIGHTS
# ─────────────────────────────────────────────────────────────────────────
classes = np.unique(y_train)
cw = compute_class_weight("balanced", classes=classes, y=y_train)
class_weight = {int(c): float(w) for c, w in zip(classes, cw)}
print(f"\n⚖️   Class weights : {class_weight}")

# ─────────────────────────────────────────────────────────────────────────
#  5. MODEL ARCHITECTURE – Deep Residual Network
# ─────────────────────────────────────────────────────────────────────────
INPUT_DIM = X_train.shape[1]

def residual_block(x, units, dropout=0.3):
    """A dense residual block with skip connection."""
    shortcut = layers.Dense(units)(x)
    x = layers.Dense(units, kernel_regularizer=regularizers.l2(1e-4))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.Dropout(dropout)(x)
    x = layers.Dense(units, kernel_regularizer=regularizers.l2(1e-4))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Add()([x, shortcut])
    x = layers.Activation("relu")(x)
    return x

def build_model(input_dim: int) -> keras.Model:
    inputs = keras.Input(shape=(input_dim,), name="clinical_features")

    # Entry block
    x = layers.Dense(256, kernel_regularizer=regularizers.l2(1e-4))(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.Dropout(0.35)(x)

    # Residual blocks
    x = residual_block(x, 256, 0.35)
    x = residual_block(x, 128, 0.30)
    x = residual_block(x, 64,  0.25)

    # Bottleneck
    x = layers.Dense(32, activation="relu")(x)
    x = layers.Dropout(0.20)(x)

    # Output
    output = layers.Dense(1, activation="sigmoid", name="stroke_probability")(x)

    model = keras.Model(inputs, output, name="StrokePredictor_DNN")
    return model

model = build_model(INPUT_DIM)

model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=1e-3),
    loss="binary_crossentropy",
    metrics=[
        "accuracy",
        keras.metrics.AUC(name="auc"),
        keras.metrics.Precision(name="precision"),
        keras.metrics.Recall(name="recall"),
    ],
)
model.summary()

# ─────────────────────────────────────────────────────────────────────────
#  6. CALLBACKS
# ─────────────────────────────────────────────────────────────────────────
cb_list = [
    callbacks.EarlyStopping(
        monitor="val_auc", patience=15, restore_best_weights=True, mode="max"
    ),
    callbacks.ReduceLROnPlateau(
        monitor="val_auc", factor=0.5, patience=7, min_lr=1e-6, mode="max"
    ),
    callbacks.ModelCheckpoint(
        "best_stroke_model.keras", monitor="val_auc",
        save_best_only=True, mode="max", verbose=0,
    ),
]

# ─────────────────────────────────────────────────────────────────────────
#  7. TRAINING
# ─────────────────────────────────────────────────────────────────────────
print("\n🚀  Training …")
history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=120,
    batch_size=64,
    class_weight=class_weight,
    callbacks=cb_list,
    verbose=1,
)

# ─────────────────────────────────────────────────────────────────────────
#  8. EVALUATION
# ─────────────────────────────────────────────────────────────────────────
print("\n📈  Evaluating on test set …")
y_prob = model.predict(X_test, verbose=0).flatten()
y_pred = (y_prob >= 0.40).astype(int)        # lower threshold → higher recall

auc = roc_auc_score(y_test, y_prob)
cm  = confusion_matrix(y_test, y_pred)
report = classification_report(y_test, y_pred, target_names=["No Stroke","Stroke"])

print(f"\n{'='*50}")
print(f"  ROC-AUC Score  : {auc:.4f}")
print(f"{'='*50}")
print(report)
print("Confusion Matrix:")
print(cm)

# ─────────────────────────────────────────────────────────────────────────
#  9. PLOTS
# ─────────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
fig.suptitle("Stroke Prediction Model – Evaluation", fontsize=14, fontweight="bold")

# Loss curve
axes[0].plot(history.history["loss"],     label="Train Loss", color="#E74C3C")
axes[0].plot(history.history["val_loss"], label="Val Loss",   color="#3498DB", linestyle="--")
axes[0].set_title("Loss"); axes[0].legend(); axes[0].set_xlabel("Epoch")

# AUC curve
axes[1].plot(history.history["auc"],     label="Train AUC", color="#27AE60")
axes[1].plot(history.history["val_auc"], label="Val AUC",   color="#E67E22", linestyle="--")
axes[1].set_title(f"AUC  (Test={auc:.3f})"); axes[1].legend(); axes[1].set_xlabel("Epoch")

# Confusion matrix
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=axes[2],
            xticklabels=["No Stroke","Stroke"],
            yticklabels=["No Stroke","Stroke"])
axes[2].set_title("Confusion Matrix"); axes[2].set_ylabel("Actual"); axes[2].set_xlabel("Predicted")

plt.tight_layout()
plt.savefig("model_evaluation.png", dpi=150)
plt.close()
print("\n✅  Plots saved → model_evaluation.png")

# ─────────────────────────────────────────────────────────────────────────
#  10. SAVE MODEL & METADATA
# ─────────────────────────────────────────────────────────────────────────
model.save("stroke_model.keras")

metadata = {
    "feature_columns": FEATURE_COLS,
    "original_20_features": [
        "age","gender","bmi","height","weight",
        "hypertension","heart_disease","ever_married","family_history",
        "avg_glucose_level","fasting_blood_sugar","cholesterol","hdl","ldl",
        "smoking_status","alcohol_intake","physical_activity","stress_level",
        "systolic_bp","diastolic_bp",
    ],
    "engineered_features": [
        "risk_score","lifestyle_risk","age_group","pulse_pressure",
        "mean_arterial_pressure","chol_hdl_ratio","bmi_category",
        "glucose_bmi_interaction","bp_severity",
    ],
    "threshold": 0.40,
    "test_auc":  round(auc, 4),
    "input_dim": INPUT_DIM,
}
with open("model_metadata.json", "w") as f:
    json.dump(metadata, f, indent=2)

print("\n✅  Saved: stroke_model.keras | scaler.pkl | model_metadata.json")
print(f"   Test AUC = {auc:.4f}")

# ─────────────────────────────────────────────────────────────────────────
#  11. SAMPLE PREDICTIONS
# ─────────────────────────────────────────────────────────────────────────
print("\n─── Sample predictions (first 5 test samples) ───")
for i in range(5):
    prob = y_prob[i] * 100
    risk = "🔴 HIGH" if prob >= 60 else ("🟡 MODERATE" if prob >= 30 else "🟢 LOW")
    actual = "Stroke" if y_test[i] == 1 else "No Stroke"
    print(f"  [{i+1}] Prob={prob:5.1f}%  Risk={risk:<14}  Actual={actual}")

print("\n✅  Training complete!")