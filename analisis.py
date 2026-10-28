# %% [markdown]
# # Proyecto Programación II - Análisis de Supervivencia Titanic
# Integrantes: Isabella (Integrante A), Lizeth (Integrante B)

# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Carga de los conjuntos de datos
train = pd.read_csv("datos/train.csv")
test = pd.read_csv("datos/test.csv")
gender = pd.read_csv("datos/gender_submission.csv")

# Consolidación del conjunto completo
test_completo = test.merge(gender, on="PassengerId", how="left")
df = pd.concat([train, test_completo], ignore_index=True)

# %% [markdown]
# ## 1. Carga e inspección inicial de los datos (Integrante A)

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