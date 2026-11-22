"""
05_comparacion_modelos_plot.py — Curvas ROC comparadas + matriz de confusión del modelo elegido
"""
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import RocCurveDisplay, ConfusionMatrixDisplay
import pandas as pd

OUT = "/home/claude/tfm_cardiovascular/outputs"
data = joblib.load(f"{OUT}/processed_data.joblib")
X_test, y_test = data["X_test"], data["y_test"]
models = joblib.load(f"{OUT}/fitted_models.joblib")

# 1. Curvas ROC de todos los modelos
fig, ax = plt.subplots(figsize=(6.5, 6))
colors = ["#2c5f7c", "#c0392b", "#27ae60", "#8e44ad", "#d68910", "#17a2b8"]
for (name, model), color in zip(models.items(), colors):
    RocCurveDisplay.from_estimator(model, X_test, y_test, ax=ax, name=name, plot_chance_level=False)
    ax.lines[-1].set_color(color)
    ax.lines[-1].set_alpha(0.85)
ax.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Azar")
ax.set_title("Curvas ROC — comparación de modelos (test set)")
plt.tight_layout()
plt.savefig(f"{OUT}/fig06_curvas_roc.png", dpi=150)
plt.close()

# 2. Matriz de confusión de la Regresión Logística (candidata a modelo final por interpretabilidad + robustez)
fig, ax = plt.subplots(figsize=(5, 4.5))
ConfusionMatrixDisplay.from_estimator(
    models["Regresión Logística"], X_test, y_test, ax=ax,
    display_labels=["Sin enfermedad", "Con enfermedad"], cmap="Blues", colorbar=False
)
ax.set_title("Matriz de confusión — Regresión Logística (test)")
plt.tight_layout()
plt.savefig(f"{OUT}/fig07_matriz_confusion_logreg.png", dpi=150)
plt.close()

print("Figuras guardadas")
