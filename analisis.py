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
