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

# Se encontraron faltantes en cuatro columnas: Cabin (1014, 77,5 %), Age (263, 20,1 %), Embarked (2, 0,2 %) y Fare (1, 0,1 %). Cabin se eliminó porque más de las tres cuartas partes estaban vacías. Age y Fare se imputaron con la mediana (28 años y 14,45), porque son numéricas y la mediana no se distorsiona con valores extremos. Embarked se imputó con la moda (S), por ser categórica. El segundo mapa de calor queda uniforme: el conjunto quedó con 1309 registros y 11 variables, sin faltantes.

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

# Solo Name tenía espacios sobrantes (4 registros), que se corrigieron con strip(). Sex y Embarked estaban uniformes. No hay filas duplicadas, porque PassengerId es único, así que no se eliminó ningún registro. Hay 2 nombres repetidos que corresponden a filas con otros datos distintos, y 380 tiquetes repetidos, que no son un error: familias o grupos viajaban con el mismo tiquete.

# %% [markdown]
# **Conclusión:** `Fare` es la variable con valores atípicos más relevantes
# (171 por el criterio del IQR, con un máximo de 512.3). Se conservan porque
# corresponden a tarifas altas de primera clase, no a errores. En `Age` hay
# solo 9 (pasajeros mayores de 66 años) y en `SibSp` 57 (familias numerosas).
# `Parch` marca 307 porque su rango intercuartílico es 0 (la mayoría viaja sin
# padres ni hijos), así que ese resultado no indica errores.

# %% [markdown]
# ## 4. Valores inconsistentes y duplicados  (Integrante B)

# 1. Las mujeres de primera clase (144) sobrevivieron en un 97,9 %, frente al 37,7 % general.
# 2. Los niños menores de 12 años (91) sobrevivieron en un 53,8 %, por encima del promedio.
# 3. Los hombres de tercera clase (493) tuvieron la menor supervivencia: 9,5 %.
# 4. Quienes viajaban con familia (519) sobrevivieron más (50,7 %) que quienes viajaban solos (790, 29,2 %). Esto no prueba que la familia sea la causa, porque influyen el sexo y la clase.
# 5. Las 5 tarifas más altas son de primera clase: cuatro pagaron exactamente 512,33 (probablemente un tiquete compartido) y los cuatro sobrevivieron; el quinto pagó 263 y no sobrevivió.

# %% [markdown]

# ## 5. Creación de variables  (Integrante A)
# %%
df["FamilySize"] = df["SibSp"] + df["Parch"] + 1
df["IsAlone"] = (df["FamilySize"] == 1).astype(int)
df["FarePerPerson"] = df["Fare"] / df["FamilySize"]
df["Title"] = df["Name"].str.extract(r" ([A-Za-z]+)\.", expand=False)
df["AgeGroup"] = pd.cut(
    df["Age"],
    bins=[0, 12, 18, 35, 60, 100],
    labels=["Niño", "Adolescente", "Adulto joven", "Adulto", "Mayor"],
)
df[["SibSp", "Parch", "FamilySize", "IsAlone", "FarePerPerson", "Title", "Age", "AgeGroup"]].head(10)

# %% [markdown]
# **Conclusión:** se crearon cinco variables. `FamilySize` suma `SibSp`, `Parch`
# y el propio pasajero; `IsAlone` vale 1 si viaja solo (790 pasajeros) y 0 si
# va acompañado (519); `FarePerPerson` divide la tarifa entre el tamaño de la
# familia (variable numérica continua); `Title` extrae el tratamiento del
# nombre (Mr, Mrs, Miss, etc.); `AgeGroup` agrupa la edad en cinco categorías
# y queda vacía en los 263 pasajeros sin edad.

# 1. Sobrevivieron 494 de 1309 pasajeros (37,7 %) y murieron 815: el conjunto no está equilibrado.
# 2. Las mujeres sobrevivieron más que los hombres en las tres clases (en primera, 97,9 % frente a 25,1 %; en tercera, 66,7 % frente a 9,5 %).
# 3. La mediana de edad es 28 años tanto en sobrevivientes como en no sobrevivientes, así que la edad sola casi no diferencia. El pico en 28 se debe en parte a la imputación con la mediana.
# 4. La tarifa sube con la clase (mediana de 60,0 en primera, 15,0 en segunda y 8,0 en tercera), con valores altos y aislados en primera, de hasta 512,33.
# 5. Las variables más relacionadas con Survived son Pclass (−0,26) y Fare (0,23). Son relaciones débiles: ninguna explica por sí sola la supervivencia.

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
# **Conclusión 2:

#  **Clase social y tarifa:** la supervivencia baja con la clase: 63.0 % en primera, 47.3 % en segunda y 24.2 % en tercera. La tarifa promedio también cae de 87.5 en primera a 21.2 en segunda y 13.3 en tercera.

# %%
df.groupby("Pclass")["Fare"].mean()

# %% [markdown]
# **Conclusión 3:** la tarifa promedio cae de 87.5 en primera clase a 21.2 en
# segunda y 13.3 en tercera.

# %%
train_df.groupby("IsAlone")["Survived"].mean()

# %% [markdown]
# **Conclusión 4:
# 
# **Variables nuevas:** viajar acompañado se asocia con más supervivencia (50.6 % frente a 30.4 % de quienes viajaban solos, que son 790 de los 1309 pasajeros). Por edad, los niños sobrevivieron más (58.0 %) y los mayores de 60 menos (22.7 %, aunque solo hay 22 pasajeros en ese grupo).
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
#    (cerca del 77 %) y se eliminó; `Age` falta en 263 (cerca del 20 %) y se
#    imputó con la mediana, lo que limita el análisis por edad. No había
#    duplicados. `Survived` solo existe para 891 pasajeros, así que las
#    conclusiones de supervivencia se basan solo en el `train`. Además, las
#    relaciones con `Survived` son moderadas o débiles (`Pclass` −0.34,
#    `Fare` 0.26), por lo que ninguna variable la explica por sí sola.