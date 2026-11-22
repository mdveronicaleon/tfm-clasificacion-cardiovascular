"""
03_preprocessing.py — Preparación de datos para el TFM de clasificación cardiovascular
- Target binario: 0 = sin enfermedad, 1 = con enfermedad (grados 1-4 agrupados)
- Codificación de categóricas nominales (cp, restecg, thal) con one-hot
- slope y ca se mantienen como numéricas ordinales
- División train/test estratificada 80/20
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
import joblib

OUT = "/home/claude/tfm_cardiovascular/outputs"
df = pd.read_pickle(f"{OUT}/heart_raw.pkl")

# 1. Target binario
df["target"] = (df["num"] > 0).astype(int)
df = df.drop(columns=["num"])

# 2. Tipado de variables
continuous = ["age", "trestbps", "chol", "thalach", "oldpeak"]
binary = ["sex", "fbs", "exang"]
ordinal = ["slope", "ca"]
nominal = ["cp", "restecg", "thal"]

X = df.drop(columns=["target"])
y = df["target"]

# 3. Split estratificado (antes de ajustar cualquier transformación, para evitar fuga de información)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)
print("Train:", X_train.shape, "Test:", X_test.shape)
print("Proporción clase positiva - train:", y_train.mean().round(3), "test:", y_test.mean().round(3))

# 4. ColumnTransformer: escalado de continuas + one-hot de nominales; binarias y ordinales pasan igual
preprocessor = ColumnTransformer(
    transformers=[
        ("cont", StandardScaler(), continuous),
        ("nom", OneHotEncoder(drop="if_binary", handle_unknown="ignore"), nominal),
    ],
    remainder="passthrough",  # deja pasar binary + ordinal sin transformar
)

X_train_proc = preprocessor.fit_transform(X_train)
X_test_proc = preprocessor.transform(X_test)

feature_names = (
    continuous
    + list(preprocessor.named_transformers_["nom"].get_feature_names_out(nominal))
    + binary
    + ordinal
)

print("\nNúmero de variables tras preprocesado:", X_train_proc.shape[1])
print("Variables:", feature_names)

joblib.dump(
    {
        "X_train": X_train_proc, "X_test": X_test_proc,
        "y_train": y_train.values, "y_test": y_test.values,
        "feature_names": feature_names, "preprocessor": preprocessor,
    },
    f"{OUT}/processed_data.joblib",
)
print("\nGuardado processed_data.joblib")
