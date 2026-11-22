"""
01_eda.py — Análisis exploratorio inicial del dataset heart.csv (UCI Heart Disease, Cleveland)
TFM: Clasificación de enfermedades cardiovasculares
"""
import pandas as pd
import numpy as np

pd.set_option("display.max_columns", None)
pd.set_option("display.width", 160)

df = pd.read_csv("/home/claude/tfm_cardiovascular/data/heart.csv")

print("=== Forma del dataset ===")
print(df.shape)

print("\n=== Tipos de datos ===")
print(df.dtypes)

print("\n=== Primeras filas ===")
print(df.head())

print("\n=== Estadísticos descriptivos ===")
print(df.describe().T)

print("\n=== Valores únicos por columna (para detectar categóricas y códigos especiales) ===")
for col in df.columns:
    uniq = sorted(df[col].unique())
    print(f"{col}: n_unique={len(uniq)} -> {uniq if len(uniq) <= 15 else uniq[:15]+['...']}")

print("\n=== Nulos explícitos (NaN) ===")
print(df.isna().sum())

print("\n=== Distribución de la variable objetivo 'num' (0=sin enfermedad, 1-4=grados de enfermedad) ===")
print(df["num"].value_counts().sort_index())
print("\nBinarizada (0 vs 1+):")
print((df["num"] > 0).value_counts())

print("\n=== Posibles valores faltantes disfrazados (0 en variables donde 0 no es fisiológicamente válido) ===")
for col in ["trestbps", "chol", "thalach"]:
    n_zero = (df[col] == 0).sum()
    print(f"{col}: ceros = {n_zero}")

print("\n=== Filas duplicadas ===")
print(df.duplicated().sum())

df.to_pickle("/home/claude/tfm_cardiovascular/outputs/heart_raw.pkl")
print("\nGuardado heart_raw.pkl")
