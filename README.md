# TFM — Clasificación de enfermedades cardiovasculares

Trabajo Fin de Máster — Máster en Inteligencia Artificial en Salud.

## Objetivo

Desarrollar y comparar modelos de aprendizaje automático para la clasificación binaria de presencia/ausencia de enfermedad cardiovascular, a partir del dataset público **UCI Heart Disease (Cleveland)**.

- Fuente de los datos: https://archive.ics.uci.edu/dataset/45/heart+disease
- 303 pacientes, 13 variables predictoras + variable objetivo (`num`, grado de enfermedad 0–4, binarizada aquí en 0 = sin enfermedad / 1 = con enfermedad).

## Estructura del repositorio

```
tfm_cardiovascular/
├── data/
│   └── heart.csv                  # Dataset original (UCI Heart Disease)
├── scripts/
│   ├── 01_eda.py                   # Análisis exploratorio inicial (estadísticos, nulos, duplicados)
│   ├── 02_eda_plots.py             # Gráficas de EDA
│   ├── 03_preprocessing.py         # Limpieza, codificación, split train/test estratificado
│   ├── 04_modelado.py              # Entrenamiento y comparación de 6 modelos (CV 5-fold + test)
│   ├── 05_comparacion_modelos_plot.py  # Curvas ROC y matriz de confusión del modelo final
│   ├── 06_optimizacion_interpretacion.py  # Ajuste de hiperparámetros + interpretabilidad del modelo final
│   └── 07_shap.py                  # Interpretabilidad adicional con SHAP (impacto global e individual)
├── outputs/                        # Figuras, tablas de resultados y objetos entrenados (generados al ejecutar)
├── requirements.txt
└── README.md
```

## Cómo reproducir

```bash
pip install -r requirements.txt
cd scripts
python 01_eda.py
python 02_eda_plots.py
python 03_preprocessing.py
python 04_modelado.py
python 05_comparacion_modelos_plot.py
python 06_optimizacion_interpretacion.py
python 07_shap.py
```

Cada script guarda sus resultados (figuras, tablas `.csv`, objetos `.joblib`) en `outputs/`.

## Modelos comparados

Regresión Logística, Árbol de Decisión, Random Forest, Gradient Boosting (XGBoost), SVM (kernel RBF) y K-Vecinos (KNN), evaluados mediante validación cruzada estratificada (5-fold) sobre el conjunto de entrenamiento y sobre un conjunto de test independiente (20%, estratificado).

**Modelo final seleccionado:** Regresión Logística, por su rendimiento consistente entre validación cruzada y test, y por su interpretabilidad clínica (coeficientes interpretables como odds ratios).

## Autoría

Trabajo Fin de Máster realizado por Verónica León.
