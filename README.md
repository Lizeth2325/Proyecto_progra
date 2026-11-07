# Análisis exploratorio del Titanic

## Integrantes
- Isabella Renteria (Integrante A)
- Lizeth Bohórquez (Integrante B)

## Fuente de los datos
Dataset "Titanic - Machine Learning from Disaster" de Kaggle.
https://www.kaggle.com/code/parvathicg/titanic/input

## Estructura de los datos
1309 registros y 12 variables. Cada fila representa un pasajero. Tiene 5 variables categóricas, 2 numéricas continuas y 5 numéricas discretas.

## Diccionario de datos
| Variable | Tipo | Descripción | Valores / unidades |
|---|---|---|---|
| PassengerId | Numérica discreta | Identificador del pasajero | 1 a 891 |
| Survived | Numérica discreta | Si sobrevivió | 0 = no, 1 = sí |
| Pclass | Numérica discreta | Clase del tiquete | 1, 2, 3 |
| Name | Categórica | Nombre del pasajero | Texto |
| Sex | Categórica | Sexo | male, female |
| Age | Numérica continua | Edad | Años |
| SibSp | Numérica discreta | Hermanos o cónyuge a bordo | 0 a 8 |
| Parch | Numérica discreta | Padres o hijos a bordo | 0 a 6 |
| Ticket | Categórica | Número de tiquete | Texto |
| Fare | Numérica continua | Tarifa pagada | Libras |
| Cabin | Categórica | Cabina | Texto, con muchos vacíos |
| Embarked | Categórica | Puerto de embarque | C, Q, S |

## Organización del repositorio
- `analisis.py`: solución final del proyecto.
- `cuadernos/`: cuadernos de exploración de cada integrante.
- `datos/`: base de datos descargada de Kaggle.

## Conclusiones
1. (Integrante A)

*Sexo y supervivencia:* el sexo fue el factor con mayor diferencia. Sobrevivió el 74,2 % de las mujeres frente al 18,9 % de los hombres (solo pasajeros del train, que son los que tienen Survived).

2. (Integrante B)

El sexo fue el factor más fuerte: sobrevivió el 74,2 % de las mujeres frente al 18,9 % de los hombres. La clase refuerza la diferencia, porque el 96,8 % de las mujeres de primera clase sobrevivió, mientras que solo el 13,5 % de los hombres de tercera.


3. (Integrante A)

*Valores atípicos:* Fare tiene 171 valores atípicos por el criterio del IQR y llega hasta 512,3. Se conservan porque corresponden a tarifas altas de primera clase, no a errores de captura.



4. (Integrante B)

La calidad de los datos es buena, con una excepción: Cabin se eliminó por tener 77,1 % de valores vacíos. No había duplicados, y Age se imputó con la mediana (19,9 % de faltantes), lo que limita el análisis de la edad. Las relaciones con Survived son moderadas o débiles (Pclass −0,34 y Fare 0,26), así que ninguna variable explica la supervivencia por sí sola.


5. (Integrante A)

*Limitaciones de los datos:* Cabin no tiene dato en 1014 pasajeros (cerca del 77 %) y se eliminó; Age falta en 263 (cerca del 20 %) y se imputó con la mediana, lo que limita el análisis por edad. No había duplicados. Survived solo existe para 891 pasajeros, así que las conclusiones de supervivencia se basan solo en el train. Además, las relaciones con Survived son moderadas o débiles (Pclass −0,34, Fare 0,26), por lo que ninguna variable la explica por sí sola.