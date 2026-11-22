"""
02_eda_plots.py — Gráficas de EDA para la memoria del TFM
"""
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

OUT = "/home/claude/tfm_cardiovascular/outputs"
df = pd.read_pickle(f"{OUT}/heart_raw.pkl")

# Target binarizado: 0 = sin enfermedad, 1 = enfermedad presente (grados 1-4 agrupados)
df["target"] = (df["num"] > 0).astype(int)

sns.set_theme(style="whitegrid", font_scale=0.9)

# 1. Distribución de la variable objetivo (multiclase y binaria)
fig, axes = plt.subplots(1, 2, figsize=(10, 4))
df["num"].value_counts().sort_index().plot(kind="bar", ax=axes[0], color="#2c5f7c")
axes[0].set_title("Distribución original 'num' (0-4)")
axes[0].set_xlabel("Grado de enfermedad")
axes[0].set_ylabel("Nº pacientes")
df["target"].value_counts().sort_index().plot(kind="bar", ax=axes[1], color=["#2c5f7c", "#c0392b"])
axes[1].set_title("Distribución binarizada")
axes[1].set_xticklabels(["Sin enfermedad (0)", "Con enfermedad (1)"], rotation=0)
axes[1].set_ylabel("Nº pacientes")
plt.tight_layout()
plt.savefig(f"{OUT}/fig01_distribucion_target.png", dpi=150)
plt.close()

# 2. Histogramas de variables numéricas continuas
num_vars = ["age", "trestbps", "chol", "thalach", "oldpeak"]
fig, axes = plt.subplots(2, 3, figsize=(13, 7))
axes = axes.flatten()
for i, col in enumerate(num_vars):
    sns.histplot(df, x=col, hue="target", kde=True, ax=axes[i], palette=["#2c5f7c", "#c0392b"], alpha=0.5)
    axes[i].set_title(col)
axes[-1].axis("off")
plt.tight_layout()
plt.savefig(f"{OUT}/fig02_histogramas_numericas.png", dpi=150)
plt.close()

# 3. Matriz de correlación
fig, ax = plt.subplots(figsize=(9, 7))
corr = df.drop(columns=["num"]).corr()
sns.heatmap(corr, annot=True, fmt=".2f", cmap="RdBu_r", center=0, ax=ax, annot_kws={"size": 7})
ax.set_title("Matriz de correlación")
plt.tight_layout()
plt.savefig(f"{OUT}/fig03_correlacion.png", dpi=150)
plt.close()

# 4. Variables categóricas vs target
cat_vars = ["sex", "cp", "fbs", "restecg", "exang", "slope", "ca", "thal"]
fig, axes = plt.subplots(2, 4, figsize=(15, 7))
axes = axes.flatten()
for i, col in enumerate(cat_vars):
    ct = pd.crosstab(df[col], df["target"], normalize="index")
    ct.plot(kind="bar", stacked=True, ax=axes[i], color=["#2c5f7c", "#c0392b"], legend=False)
    axes[i].set_title(col)
    axes[i].set_ylabel("Proporción")
handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, ["Sin enfermedad", "Con enfermedad"], loc="lower center", ncol=2)
plt.tight_layout(rect=[0, 0.04, 1, 1])
plt.savefig(f"{OUT}/fig04_categoricas_vs_target.png", dpi=150)
plt.close()

# 5. Edad y sexo vs enfermedad (boxplot)
fig, ax = plt.subplots(figsize=(6, 4.5))
sns.boxplot(data=df, x="target", y="age", hue="sex", ax=ax, palette=["#f2a0a0", "#5a8fb5"])
ax.set_xticklabels(["Sin enfermedad", "Con enfermedad"])
ax.set_title("Edad por presencia de enfermedad y sexo")
ax.legend(title="Sexo", labels=["Mujer (0)", "Hombre (1)"])
plt.tight_layout()
plt.savefig(f"{OUT}/fig05_edad_sexo_target.png", dpi=150)
plt.close()

print("Figuras guardadas en", OUT)
print([f for f in __import__("os").listdir(OUT) if f.endswith(".png")])
