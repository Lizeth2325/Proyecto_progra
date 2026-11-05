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
# ## 2. Manejo de datos faltantes  (Integrante B)

# %% [markdown]
# ## 3. Identificación de outliers  (Integrante A)

# %%
variables_numericas = ["Age", "Fare", "SibSp", "Parch"]
fig, axes = plt.subplots(1, 4, figsize=(16, 4))
for ax, col in zip(axes, variables_numericas):
    sns.boxplot(y=df[col], ax=ax)
    ax.set_title(col)
plt.tight_layout()
plt.show()

# %%
for col in variables_numericas:
    q1, q3 = df[col].quantile([0.25, 0.75])
    iqr = q3 - q1
    inferior, superior = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    n = ((df[col] < inferior) | (df[col] > superior)).sum()
    print(f"{col}: límites [{inferior:.2f}, {superior:.2f}] -> {n} outliers")

# %% [markdown]
# **Conclusión:** `Fare` es la variable con valores atípicos más relevantes
# (171 por el criterio del IQR, con un máximo de 512.3). Se conservan porque
# corresponden a tarifas altas de primera clase, no a errores. En `Age` hay
# solo 9 (pasajeros mayores de 66 años) y en `SibSp` 57 (familias numerosas).
# `Parch` marca 307 porque su rango intercuartílico es 0 (la mayoría viaja sin
# padres ni hijos), así que ese resultado no indica errores.

# %% [markdown]
# ## 4. Valores inconsistentes y duplicados  (Integrante B)

# %% [markdown]
# ## 5. Creación de variables  (Integrante A)

# %%
df["FamilySize"] = df["SibSp"] + df["Parch"] + 1
df["IsAlone"] = (df["FamilySize"] == 1).astype(int)
df["AgeGroup"] = pd.cut(
    df["Age"],
    bins=[0, 12, 18, 35, 60, 100],
    labels=["Niño", "Adolescente", "Adulto joven", "Adulto", "Mayor"],
)
df[["SibSp", "Parch", "FamilySize", "IsAlone", "Age", "AgeGroup"]].head(10)

# %% [markdown]
# **Conclusión:** se crearon tres variables. `FamilySize` suma `SibSp`, `Parch`
# y el propio pasajero; `IsAlone` vale 1 si viaja solo (790 pasajeros) y 0 si
# va acompañado (519); `AgeGroup` agrupa la edad en cinco categorías y queda
# vacía en los 263 pasajeros sin edad.

# %% [markdown]
# ## 6. Filtrado y condiciones en pandas  (Integrante B)

# %% [markdown]
# ## 7. Agrupación con groupby  (Integrante A)

# %%
train_df = df[df["Survived"].notna()]
train_df.groupby("Sex")["Survived"].mean()

# %% [markdown]
# **Conclusión 1:** sobrevivió el 74.2 % de las mujeres frente al 18.9 % de los
# hombres.

# %%
train_df.groupby("Pclass")["Survived"].mean()

# %% [markdown]
# **Conclusión 2:** la supervivencia baja con la clase: 63.0 % en primera,
# 47.3 % en segunda y 24.2 % en tercera.

# %%
df.groupby("Pclass")["Fare"].mean()

# %% [markdown]
# **Conclusión 3:** la tarifa promedio cae de 87.5 en primera clase a 21.2 en
# segunda y 13.3 en tercera.

# %%
train_df.groupby("IsAlone")["Survived"].mean()

# %% [markdown]
# **Conclusión 4:** quienes viajaban acompañados sobrevivieron más (50.6 %)
# que quienes viajaban solos (30.4 %).

# %%
train_df.groupby("AgeGroup", observed=True)["Survived"].agg(["mean", "count"])

# %% [markdown]
# **Conclusión 5:** los niños fueron el grupo con mayor supervivencia (58.0 %,
# 69 pasajeros) y los mayores de 60 el de menor (22.7 %, solo 22 pasajeros).

# %% [markdown]
# ## 8. Visualizaciones  (Integrante B)

# %% [markdown]
# ## 9. Conclusiones finales  (Integrante A)

# %% [markdown]
# 1. **Sexo y supervivencia:** el sexo fue el factor con mayor diferencia.
#    Sobrevivió el 74.2 % de las mujeres frente al 18.9 % de los hombres
#    (solo pasajeros del `train`, que son los que tienen `Survived`).
# 2. **Clase social y tarifa:** la supervivencia baja con la clase: 63.0 % en
#    primera, 47.3 % en segunda y 24.2 % en tercera. La tarifa promedio también
#    cae de 87.5 en primera a 21.2 en segunda y 13.3 en tercera.
# 3. **Valores atípicos:** `Fare` tiene 171 valores atípicos por el criterio
#    del IQR y llega hasta 512.3. Se conservan porque corresponden a tarifas
#    altas de primera clase, no a errores de captura.
# 4. **Variables nuevas:** viajar acompañado se asocia con más supervivencia
#    (50.6 % frente a 30.4 % de quienes viajaban solos, que son 790 de los
#    1309 pasajeros). Por edad, los niños sobrevivieron más (58.0 %) y los
#    mayores de 60 menos (22.7 %, aunque solo hay 22 pasajeros en ese grupo).
# 5. **Limitaciones de los datos:** `Cabin` no tiene dato en 1014 pasajeros
#    (cerca del 77 %) y `Age` en 263 (cerca del 20 %). Además, `Survived` solo
#    existe para 891 pasajeros, así que las conclusiones de supervivencia se
#    basan solo en el `train`.