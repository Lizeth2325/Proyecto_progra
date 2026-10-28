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
