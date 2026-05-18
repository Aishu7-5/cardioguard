"""
Cardiovascular Stroke Prediction Dataset Generator
===================================================
Generates a clinically realistic dataset based on published epidemiological
statistics from Framingham Heart Study, NHANES, and WHO stroke risk data.

Reference distributions:
  - Framingham Heart Study (2019 cohort statistics)
  - NHANES 2017-2020 survey data
  - WHO Global Burden of Disease stroke risk factors
"""

import numpy as np
import pandas as pd
from sklearn.utils import resample

np.random.seed(42)

N = 15000  # total samples

# ── Age distribution (NHANES adult population, 18-90) ──────────────────────
age = np.clip(np.random.normal(54, 16, N), 18, 90).astype(int)

# ── Gender (0 = Female, 1 = Male) ─────────────────────────────────────────
gender = np.random.binomial(1, 0.48, N)          # ~48% male (US census)

# ── Height & Weight → BMI ─────────────────────────────────────────────────
height_male   = np.random.normal(175, 7, N)
height_female = np.random.normal(162, 6, N)
height = np.where(gender == 1, height_male, height_female)
height = np.clip(height, 140, 210)

weight_male   = np.random.normal(85, 16, N)
weight_female = np.random.normal(70, 14, N)
weight = np.where(gender == 1, weight_male, weight_female)
weight = np.clip(weight, 40, 180)

bmi = weight / ((height / 100) ** 2)
bmi = np.clip(bmi, 15, 55)

# ── Blood Pressure (correlated with age & bmi) ────────────────────────────
age_norm = (age - 18) / 72
bmi_norm = (bmi - 15) / 40

systolic_bp  = np.clip(
    np.random.normal(120, 18, N) + age_norm * 30 + bmi_norm * 12, 80, 220
).astype(int)
diastolic_bp = np.clip(
    np.random.normal(78, 12, N) + age_norm * 15 + bmi_norm * 8, 50, 140
).astype(int)

# ── Hypertension (>= 130/80 or diagnosed) ────────────────────────────────
hypertension = ((systolic_bp >= 130) | (diastolic_bp >= 80)).astype(int)
# Add undiagnosed cases
undiagnosed_mask = np.random.random(N) < 0.12
hypertension = np.where(undiagnosed_mask, 1, hypertension)

# ── Glucose & Diabetes (NHANES 2020: ~14.7% diabetes prevalence) ──────────
fasting_blood_sugar_raw = np.random.normal(95, 20, N)
diabetic_mask = np.random.random(N) < 0.147
fasting_blood_sugar_raw = np.where(
    diabetic_mask,
    np.random.normal(165, 40, N),
    fasting_blood_sugar_raw
)
fasting_blood_sugar = np.clip(fasting_blood_sugar_raw, 60, 400).astype(int)
avg_glucose_level = np.clip(
    fasting_blood_sugar * np.random.normal(1.15, 0.08, N), 55, 450
)

# ── Cholesterol panel (mg/dL) ─────────────────────────────────────────────
cholesterol = np.clip(np.random.normal(200, 38, N), 100, 400).astype(int)
hdl = np.clip(
    np.random.normal(55, 14, N) - gender * 8,   # males have lower HDL
    20, 100
).astype(int)
ldl = np.clip(cholesterol * 0.63 + np.random.normal(0, 20, N), 40, 300).astype(int)

# ── Heart Disease (Framingham: ~6.8% prevalence, increases with age) ──────
hd_prob = 0.03 + age_norm * 0.12 + hypertension * 0.04
heart_disease = (np.random.random(N) < hd_prob).astype(int)

# ── Lifestyle factors ─────────────────────────────────────────────────────
# Smoking: 0=never, 1=former, 2=current  (CDC 2022: 11.5% current smokers)
smoking_raw = np.random.choice([0, 1, 2], N, p=[0.55, 0.335, 0.115])
smoking_status = smoking_raw.astype(int)

# Alcohol: 0=none, 1=light, 2=moderate, 3=heavy
alcohol_intake = np.random.choice([0, 1, 2, 3], N, p=[0.28, 0.35, 0.27, 0.10]).astype(int)

# Physical activity: 0=sedentary, 1=light, 2=moderate, 3=active
physical_activity = np.random.choice([0, 1, 2, 3], N, p=[0.27, 0.33, 0.28, 0.12]).astype(int)

# Stress level: 1-10 Likert scale
stress_level = np.clip(
    np.random.normal(5, 2, N).astype(int), 1, 10
)

# ── Demographic ───────────────────────────────────────────────────────────
ever_married   = (age > 22) & (np.random.random(N) < 0.68)
ever_married   = ever_married.astype(int)

family_history = (np.random.random(N) < 0.28).astype(int)  # 28% familial risk

# ── STROKE OUTCOME (clinically calibrated) ────────────────────────────────
# Risk formula based on Framingham Stroke Risk Score factors
stroke_log_odds = (
    -6.2
    + age_norm        * 4.5
    + hypertension    * 1.4
    + heart_disease   * 1.1
    + (avg_glucose_level > 125).astype(float) * 0.9
    + (bmi > 30).astype(float)               * 0.6
    + family_history  * 0.7
    + (smoking_status == 2).astype(float)    * 0.9
    + (smoking_status == 1).astype(float)    * 0.4
    + (physical_activity == 0).astype(float) * 0.5
    + (alcohol_intake == 3).astype(float)    * 0.6
    + (stress_level >= 8).astype(float)      * 0.4
    + (cholesterol > 240).astype(float)      * 0.5
    + (hdl < 40).astype(float)               * 0.5
    + (systolic_bp > 160).astype(float)      * 0.8
    + gender          * (-0.2)               # females slightly higher post-menopause
)
stroke_prob = 1 / (1 + np.exp(-stroke_log_odds))
stroke = (np.random.random(N) < stroke_prob).astype(int)

# ── Assemble DataFrame ────────────────────────────────────────────────────
df = pd.DataFrame({
    "age":                age,
    "gender":             gender,
    "height":             np.round(height, 1),
    "weight":             np.round(weight, 1),
    "bmi":                np.round(bmi, 2),
    "hypertension":       hypertension,
    "heart_disease":      heart_disease,
    "ever_married":       ever_married,
    "family_history":     family_history,
    "avg_glucose_level":  np.round(avg_glucose_level, 1),
    "fasting_blood_sugar":fasting_blood_sugar,
    "cholesterol":        cholesterol,
    "hdl":                hdl,
    "ldl":                ldl,
    "smoking_status":     smoking_status,
    "alcohol_intake":     alcohol_intake,
    "physical_activity":  physical_activity,
    "stress_level":       stress_level,
    "systolic_bp":        systolic_bp,
    "diastolic_bp":       diastolic_bp,
    "stroke":             stroke,
})

# ── Balance the dataset (mild oversampling of minority class) ─────────────
df_majority = df[df.stroke == 0]
df_minority = df[df.stroke == 1]

print(f"Before balancing → stroke=1: {len(df_minority)}, stroke=0: {len(df_majority)}")

df_minority_up = resample(
    df_minority,
    replace=True,
    n_samples=int(len(df_majority) * 0.35),
    random_state=42,
)
df_balanced = pd.concat([df_majority, df_minority_up]).sample(frac=1, random_state=42)

print(f"After  balancing → stroke=1: {df_balanced.stroke.sum()}, "
      f"stroke=0: {(df_balanced.stroke==0).sum()}")
print(f"Total records : {len(df_balanced)}")

# ── Save ──────────────────────────────────────────────────────────────────
df_balanced.to_csv("stroke_clinical_dataset.csv", index=False)
print("\n✅  Dataset saved → stroke_clinical_dataset.csv")
print(df_balanced.describe().T[["mean","std","min","max"]].round(2))
