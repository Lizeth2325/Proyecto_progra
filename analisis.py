# %% [markdown]
# # Proyecto Programación II - Análisis de Supervivencia Titanic
# Integrantes: Lizeth (Integrante A), Isabella (Integrante B)

# %%
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# Carga de los conjuntos de datos
train = pd.read_csv("datos/train.csv")
test = pd.read_csv("datos/test.csv")
gender = pd.read_csv("datos/gender_submission.csv")

# Integración para consolidar los más de 1000 registros
test_full = test.merge(gender, on="PassengerId", how="left")
df = pd.concat([train, test_full], ignore_index=True)

# %% [markdown]
# ## 1. Carga e inspección inicial de los datos (Integrante A)

# %% [markdown]
# ## 2. Manejo de datos faltantes (Integrante B)

# %% [markdown]
# ## 3. Identificación de outliers (Integrante A)

# %% [markdown]
# ## 4. Valores inconsistentes y duplicados (Integrante B)

# %% [markdown]
# ## 5. Creación de variables (Integrante A)

# %% [markdown]
# ## 6. Filtrado y condiciones en pandas (Integrante B)

# %% [markdown]
# ## 7. Agrupación con groupby (Integrante A)

# %% [markdown]
# ## 8. Visualizaciones (Integrante B)

# %% [markdown]
# ## 9. Conclusiones finales (Integrante A)
# 1. Tamaño del conjunto de datos
filas, columnas = df.shape
print(f"Dimensiones del DataFrame: {filas} registros y {columnas} variables.")

# 2. Tipos de variables e información de valores no nulos
print("\n--- Estructura y tipos de datos ---")
df.info()

# 3. Resumen estadístico de variables numéricas
print("\n--- Estadísticas descriptivas de variables numéricas ---")
display(df.describe().round(2))

# 4. Resumen descriptivo de variables categóricas
print("\n--- Estadísticas descriptivas de variables categóricas ---")
display(df.describe(include=['object']))
# %% [markdown]
# ## 1. Carga e inspección inicial de los datos  (Integrante A)
# %%
# 1. Tamaño del conjunto de datos
filas, columnas = df.shape
print(f"Dimensiones del DataFrame: {filas} registros y {columnas} variables.")

# 2. Tipos de variables e información de valores no nulos
print("\n--- Estructura y tipos de datos ---")
df.info()

# 3. Resumen estadístico de variables numéricas
print("\n--- Estadísticas descriptivas de variables numéricas ---")
display(df.describe().round(2))

# 4. Resumen descriptivo de variables categóricas
print("\n--- Estadísticas descriptivas de variables categóricas ---")
display(df.describe(include=['object']))

# %% [markdown]
# **Conclusión Ítem 1:**
# 1. El conjunto de datos unificado contiene 1309 registros y 12 columnas.
# 2. Contiene variables categóricas (`Name`, `Sex`, `Ticket`, `Cabin`, `Embarked`), numéricas discretas (`PassengerId`, `Pclass`, `SibSp`, `Parch`, `Survived`) y continuas (`Age`, `Fare`).
# 3. La edad promedio es de 29.88 años y la tarifa promedio de 33.30 con alta variabilidad.
# 4. Existen valores nulos considerables en `Cabin` (más del 70%) y en `Age` (263 registros), así como datos ausentes en `Fare` (1) y `Embarked` (2), que serán tratados en el Ítem 2.
# %% [markdown]
# ## 3. Identificación de outliers (Integrante A) 
# %%
import matplotlib.pyplot as plt
import seaborn as sns

# 1. Visualización de variables numéricas continuas mediante Boxplots
fig, axes = plt.subplots(1, 2, figsize=(13, 5))

sns.boxplot(y=df['Fare'], ax=axes[0], color='skyblue')
axes[0].set_title('Distribución y Outliers: Tarifa (Fare)')
axes[0].set_ylabel('Tarifa ($)')

sns.boxplot(y=df['Age'], ax=axes[1], color='lightcoral')
axes[1].set_title('Distribución y Outliers: Edad (Age)')
axes[1].set_ylabel('Años')

plt.tight_layout()
plt.show()

# 2. Cuantificación matemática de outliers mediante Rango Intercuartílico (IQR)
# Análisis para Fare
q1_f = df['Fare'].quantile(0.25)
q3_f = df['Fare'].quantile(0.75)
iqr_f = q3_f - q1_f
lim_sup_f = q3_f + 1.5 * iqr_f
outliers_f = (df['Fare'] > lim_sup_f).sum()

# Análisis para Age
q1_a = df['Age'].quantile(0.25)
q3_a = df['Age'].quantile(0.75)
iqr_a = q3_a - q1_a
lim_sup_a = q3_a + 1.5 * iqr_a
lim_inf_a = q1_a - 1.5 * iqr_a
outliers_a = ((df['Age'] > lim_sup_a) | (df['Age'] < lim_inf_a)).sum()

print(f"Fare: Límite superior = {lim_sup_f:.2f} | Outliers = {outliers_f} ({outliers_f/len(df)*100:.2f}%)")
print(f"Age: Límite superior = {lim_sup_a:.2f}, Límite inferior = {lim_inf_a:.2f} | Outliers = {outliers_a}")

# %% [markdown]
# **Conclusión Ítem 3:**
# 1. **Tarifa (`Fare`):** Exhibe una distribución marcadamente asimétrica a la derecha con 171 valores atípicos (13.06% del total) por encima de $66.34, alcanzando un máximo de $512.33. No se consideran errores de registro sino boletos de suites de primera clase y viajes familiares en grupo.
# 2. **Edad (`Age`):** Muestra una distribución cercana a la normalidad con únicamente 9 valores atípicos superiores (pasajeros entre 66 y 80 años) y 0 observaciones atípicas inferiores.
# 3. **Decisión analítica:** Se conservan estos datos atípicos para no descartar información socioeconómica y demográfica determinante en la supervivencia.git add analisis.py
