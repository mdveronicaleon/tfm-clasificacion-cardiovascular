"""
04_modelado.py — Entrenamiento y comparación de modelos de clasificación
"""
import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    confusion_matrix, classification_report
)

OUT = "/home/claude/tfm_cardiovascular/outputs"
data = joblib.load(f"{OUT}/processed_data.joblib")
X_train, X_test = data["X_train"], data["X_test"]
y_train, y_test = data["y_train"], data["y_test"]
feature_names = data["feature_names"]

models = {
    "Regresión Logística": LogisticRegression(max_iter=2000, random_state=42),
    "Árbol de Decisión": DecisionTreeClassifier(random_state=42, max_depth=5),
    "Random Forest": RandomForestClassifier(n_estimators=300, random_state=42),
    "Gradient Boosting (XGBoost)": XGBClassifier(
        n_estimators=200, max_depth=3, learning_rate=0.05,
        eval_metric="logloss", random_state=42,
    ),
    "SVM (RBF)": SVC(kernel="rbf", probability=True, random_state=42),
    "K-Vecinos (KNN)": KNeighborsClassifier(n_neighbors=9),
}

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
scoring = ["accuracy", "precision", "recall", "f1", "roc_auc"]

cv_results = []
test_results = []
fitted_models = {}

for name, model in models.items():
    # 5-fold CV sobre el conjunto de entrenamiento
    cvres = cross_validate(model, X_train, y_train, cv=cv, scoring=scoring)
    cv_results.append({
        "Modelo": name,
        "CV Accuracy": cvres["test_accuracy"].mean(),
        "CV Precision": cvres["test_precision"].mean(),
        "CV Recall (Sensibilidad)": cvres["test_recall"].mean(),
        "CV F1": cvres["test_f1"].mean(),
        "CV AUC-ROC": cvres["test_roc_auc"].mean(),
    })

    # Ajuste final sobre todo el train, evaluación en test (held-out)
    model.fit(X_train, y_train)
    fitted_models[name] = model
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
    specificity = tn / (tn + fp)

    test_results.append({
        "Modelo": name,
        "Accuracy": accuracy_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred),
        "Sensibilidad (Recall)": recall_score(y_test, y_pred),
        "Especificidad": specificity,
        "F1": f1_score(y_test, y_pred),
        "AUC-ROC": roc_auc_score(y_test, y_proba),
    })

cv_df = pd.DataFrame(cv_results).sort_values("CV AUC-ROC", ascending=False)
test_df = pd.DataFrame(test_results).sort_values("AUC-ROC", ascending=False)

pd.set_option("display.float_format", lambda x: f"{x:.3f}")
print("=== Resultados validación cruzada (5-fold) sobre TRAIN ===")
print(cv_df.to_string(index=False))
print("\n=== Resultados sobre conjunto de TEST (hold-out) ===")
print(test_df.to_string(index=False))

cv_df.to_csv(f"{OUT}/cv_results.csv", index=False)
test_df.to_csv(f"{OUT}/test_results.csv", index=False)
joblib.dump(fitted_models, f"{OUT}/fitted_models.joblib")

# Reporte de clasificación detallado del mejor modelo por AUC-ROC en test
best_name = test_df.iloc[0]["Modelo"]
best_model = fitted_models[best_name]
print(f"\n=== Classification report del mejor modelo ({best_name}) ===")
print(classification_report(y_test, best_model.predict(X_test),
                             target_names=["Sin enfermedad", "Con enfermedad"]))

# Importancia de variables (si el modelo lo soporta) del mejor modelo basado en árboles, y de RF siempre
if hasattr(best_model, "feature_importances_"):
    imp = pd.Series(best_model.feature_importances_, index=feature_names).sort_values(ascending=False)
    print(f"\nImportancia de variables ({best_name}):")
    print(imp.head(10))
