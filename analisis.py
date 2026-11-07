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

# Se encontraron faltantes en cuatro columnas: Cabin (1014, 77,5 %), Age (263, 20,1 %), Embarked (2, 0,2 %) y Fare (1, 0,1 %). Cabin se eliminó porque más de las tres cuartas partes estaban vacías. Age y Fare se imputaron con la mediana (28 años y 14,45), porque son numéricas y la mediana no se distorsiona con valores extremos. Embarked se imputó con la moda (S), por ser categórica. El segundo mapa de calor queda uniforme: el conjunto quedó con 1309 registros y 11 variables, sin faltantes.

# %% [markdown]
# ## 3. Identificación de outliers (Integrante A)

# %% [markdown]
# ## 4. Inconsistencias y duplicados (Integrante B)

# Solo Name tenía espacios sobrantes (4 registros), que se corrigieron con strip(). Sex y Embarked estaban uniformes. No hay filas duplicadas, porque PassengerId es único, así que no se eliminó ningún registro. Hay 2 nombres repetidos que corresponden a filas con otros datos distintos, y 380 tiquetes repetidos, que no son un error: familias o grupos viajaban con el mismo tiquete.

# %% [markdown]
# ## 5. Creación de variables (Integrante A)

# %% [markdown]
# ## 6. Filtrado y condiciones (Integrante B)

# 1. Las mujeres de primera clase (144) sobrevivieron en un 97,9 %, frente al 37,7 % general.
# 2. Los niños menores de 12 años (91) sobrevivieron en un 53,8 %, por encima del promedio.
# 3. Los hombres de tercera clase (493) tuvieron la menor supervivencia: 9,5 %.
# 4. Quienes viajaban con familia (519) sobrevivieron más (50,7 %) que quienes viajaban solos (790, 29,2 %). Esto no prueba que la familia sea la causa, porque influyen el sexo y la clase.
# 5. Las 5 tarifas más altas son de primera clase: cuatro pagaron exactamente 512,33 (probablemente un tiquete compartido) y los cuatro sobrevivieron; el quinto pagó 263 y no sobrevivió.

# %% [markdown]
# ## 7. Agrupación con groupby (Integrante A)

# %% [markdown]
# ## 8. Visualizaciones (Integrante B)

# 1. Sobrevivieron 494 de 1309 pasajeros (37,7 %) y murieron 815: el conjunto no está equilibrado.
# 2. Las mujeres sobrevivieron más que los hombres en las tres clases (en primera, 97,9 % frente a 25,1 %; en tercera, 66,7 % frente a 9,5 %).
# 3. La mediana de edad es 28 años tanto en sobrevivientes como en no sobrevivientes, así que la edad sola casi no diferencia. El pico en 28 se debe en parte a la imputación con la mediana.
# 4. La tarifa sube con la clase (mediana de 60,0 en primera, 15,0 en segunda y 8,0 en tercera), con valores altos y aislados en primera, de hasta 512,33.
# 5. Las variables más relacionadas con Survived son Pclass (−0,26) y Fare (0,23). Son relaciones débiles: ninguna explica por sí sola la supervivencia.

# %% [markdown]
# ## 9. Conclusiones finales (Integrante A)