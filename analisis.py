# %% [markdown]
# # Proyecto Programación II - Análisis de Supervivencia Titanic
# Integrantes: Isabella (Integrante A), Lizeth (Integrante B)

# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Carga y consolidación de los conjuntos de datos
train = pd.read_csv("datos/train.csv")
test = pd.read_csv("datos/test.csv")
df = pd.concat([train, test], ignore_index=True)

# %% [markdown]
# ## 1. Carga e inspección inicial de los datos  (Integrante A)

# %%
print("Filas y columnas:", df.shape)
df.info()

# %%
df.describe()

# %%
df.select_dtypes(exclude="number").describe()

# %% [markdown]
# **Conclusión:** el conjunto une los archivos `train` y `test` del Titanic y
# tiene 1309 registros y 12 variables. Las categóricas son `Name`, `Sex`,
# `Ticket`, `Cabin`, `Embarked`, `Pclass` y `Survived`; las numéricas continuas
# son `Age` y `Fare`, y las discretas son `SibSp` y `Parch` (`PassengerId` solo
# identifica a cada pasajero). Hay datos faltantes en `Cabin` (1014, cerca del
# 77 %), `Age` (263, cerca del 20 %), `Embarked` (2) y `Fare` (1). `Survived`
# solo tiene valores para 891 pasajeros porque el archivo `test` no la incluye.
# En `Fare` la media (33.3) es mucho mayor que la mediana (14.5) y el máximo es
# 512.3, lo que sugiere valores atípicos para revisar en el ítem 3.
# %% [markdown]
# ## 2. Manejo de datos faltantes (Integrante B)

# %% [markdown]
# ## 3. Identificación de outliers (Integrante A)

# %% [markdown]
# ## 4. Inconsistencias y duplicados (Integrante B)

# %% [markdown]
# ## 5. Creación de variables (Integrante A)

# %% [markdown]
# ## 6. Filtrado y condiciones (Integrante B)

# %% [markdown]
# ## 7. Agrupación con groupby (Integrante A)

# %% [markdown]
# ## 8. Visualizaciones (Integrante B)

# %% [markdown]
# ## 9. Conclusiones finales (Integrante A)