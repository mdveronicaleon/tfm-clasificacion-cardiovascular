"""
07_shap.py — Interpretabilidad con SHAP del modelo final (Regresión Logística optimizada)
- Summary plot (beeswarm): impacto global de cada variable
- Waterfall plot: explicación individual de un paciente concreto
"""
import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import shap

OUT = "/home/claude/tfm_cardiovascular/outputs"
data = joblib.load(f"{OUT}/processed_data.joblib")
X_train, X_test = data["X_train"], data["X_test"]
y_test = data["y_test"]
feature_names = data["feature_names"]
model = joblib.load(f"{OUT}/modelo_final_logreg.joblib")

X_test_df = pd.DataFrame(X_test, columns=feature_names)
X_train_df = pd.DataFrame(X_train, columns=feature_names)

# Para un modelo lineal, LinearExplainer es exacto y rápido
explainer = shap.LinearExplainer(model, X_train_df)
shap_values = explainer(X_test_df)

# 1. Summary plot (beeswarm) — impacto global de cada variable sobre la predicción
plt.figure(figsize=(8, 7))
shap.summary_plot(shap_values, X_test_df, show=False)
plt.title("SHAP — impacto de cada variable en la predicción (test set)")
plt.tight_layout()
plt.savefig(f"{OUT}/fig09_shap_summary.png", dpi=150, bbox_inches="tight")
plt.close()

# 2. Importancia media absoluta (barras) — complementaria al beeswarm
plt.figure(figsize=(7, 6))
shap.summary_plot(shap_values, X_test_df, plot_type="bar", show=False)
plt.title("SHAP — importancia media de cada variable (|valor SHAP|)")
plt.tight_layout()
plt.savefig(f"{OUT}/fig10_shap_bar.png", dpi=150, bbox_inches="tight")
plt.close()

# 3. Waterfall plot — explicación individual de un paciente con enfermedad y otro sin ella
idx_enfermo = int(np.where(y_test == 1)[0][0])
idx_sano = int(np.where(y_test == 0)[0][0])

for idx, etiqueta in [(idx_enfermo, "con_enfermedad"), (idx_sano, "sin_enfermedad")]:
    plt.figure(figsize=(8, 6))
    shap.plots.waterfall(shap_values[idx], show=False)
    plt.title(f"SHAP — explicación individual (paciente real, {etiqueta.replace('_', ' ')})")
    plt.tight_layout()
    plt.savefig(f"{OUT}/fig11_shap_waterfall_{etiqueta}.png", dpi=150, bbox_inches="tight")
    plt.close()

print("Figuras SHAP guardadas: fig09_shap_summary.png, fig10_shap_bar.png, "
      "fig11_shap_waterfall_con_enfermedad.png, fig11_shap_waterfall_sin_enfermedad.png")

# Tabla de importancia media |SHAP| por variable, para la memoria
mean_abs_shap = pd.Series(
    np.abs(shap_values.values).mean(axis=0), index=feature_names
).sort_values(ascending=False)
mean_abs_shap.to_csv(f"{OUT}/shap_importancia_media.csv", header=["importancia_media_abs_shap"])
print("\nImportancia media |SHAP| por variable:")
print(mean_abs_shap)
