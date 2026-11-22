"""
06_optimizacion_interpretacion.py
- Ajuste de hiperparámetros de la Regresión Logística (modelo final elegido) mediante GridSearchCV
- Interpretación: coeficientes / odds ratios de cada variable
- Evaluación final del modelo optimizado sobre el test hold-out
"""
import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    confusion_matrix, classification_report
)

OUT = "/home/claude/tfm_cardiovascular/outputs"
data = joblib.load(f"{OUT}/processed_data.joblib")
X_train, X_test = data["X_train"], data["X_test"]
y_train, y_test = data["y_train"], data["y_test"]
feature_names = data["feature_names"]

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# 1. Búsqueda de hiperparámetros (regularización)
param_grid = [
    {"penalty": ["l2"], "C": [0.01, 0.03, 0.1, 0.3, 1, 3, 10], "solver": ["lbfgs"]},
    {"penalty": ["l1"], "C": [0.01, 0.03, 0.1, 0.3, 1, 3, 10], "solver": ["liblinear"]},
]
grid = GridSearchCV(
    LogisticRegression(max_iter=5000, random_state=42),
    param_grid=param_grid, scoring="roc_auc", cv=cv, n_jobs=-1,
)
grid.fit(X_train, y_train)

print("=== Mejores hiperparámetros (GridSearchCV, optimizando AUC-ROC) ===")
print(grid.best_params_)
print(f"AUC-ROC (CV) con mejores hiperparámetros: {grid.best_score_:.3f}")

best_model = grid.best_estimator_

# 2. Evaluación en test
y_pred = best_model.predict(X_test)
y_proba = best_model.predict_proba(X_test)[:, 1]
tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
specificity = tn / (tn + fp)

print("\n=== Regresión Logística optimizada — resultados en TEST ===")
print(f"Accuracy:       {accuracy_score(y_test, y_pred):.3f}")
print(f"Precisión:      {precision_score(y_test, y_pred):.3f}")
print(f"Sensibilidad:   {recall_score(y_test, y_pred):.3f}")
print(f"Especificidad:  {specificity:.3f}")
print(f"F1:             {f1_score(y_test, y_pred):.3f}")
print(f"AUC-ROC:        {roc_auc_score(y_test, y_proba):.3f}")
print("\n", classification_report(y_test, y_pred, target_names=["Sin enfermedad", "Con enfermedad"]))

# 3. Interpretación: coeficientes -> odds ratios
coefs = best_model.coef_[0]
odds_ratios = np.exp(coefs)
interp_df = pd.DataFrame({
    "Variable": feature_names,
    "Coeficiente": coefs,
    "Odds Ratio": odds_ratios,
}).sort_values("Odds Ratio", ascending=False)

pd.set_option("display.float_format", lambda x: f"{x:.3f}")
print("\n=== Interpretación — Odds Ratios (Regresión Logística optimizada) ===")
print(interp_df.to_string(index=False))
interp_df.to_csv(f"{OUT}/logreg_odds_ratios.csv", index=False)

# 4. Gráfica de odds ratios
fig, ax = plt.subplots(figsize=(7, 6))
plot_df = interp_df.copy()
colors = ["#c0392b" if v > 1 else "#2c5f7c" for v in plot_df["Odds Ratio"]]
ax.barh(plot_df["Variable"], plot_df["Odds Ratio"] - 1, left=1, color=colors)
ax.axvline(1, color="black", linewidth=0.8)
ax.set_xlabel("Odds Ratio (referencia = 1)")
ax.set_title("Odds Ratios — Regresión Logística (modelo final optimizado)")
ax.invert_yaxis()
plt.tight_layout()
plt.savefig(f"{OUT}/fig08_odds_ratios.png", dpi=150)
plt.close()

joblib.dump(best_model, f"{OUT}/modelo_final_logreg.joblib")
print("\nGuardado: logreg_odds_ratios.csv, fig08_odds_ratios.png, modelo_final_logreg.joblib")
