# Análisis de Mercado de Automóviles para Consultora de Renombre

## Descripción del Proyecto

Hemos sido contratados por una consultora de renombre para formar parte de su equipo de ciencias de datos. Nuestro cliente es una importante automotriz china que desea ingresar al mercado local de automóviles. El objetivo de este proyecto es analizar las características de los vehículos presentes en el mercado actual, con un enfoque especial en diferenciar entre vehículos de gama alta y gama baja.

### Objetivo

El propósito principal de este estudio es identificar y comparar las características de los vehículos de gama alta y gama baja disponibles en el mercado. Esto permitirá a nuestro cliente ajustar su catálogo de modelos según el gusto y las necesidades de los consumidores locales, abarcando todos los públicos objetivos y estableciendo precios competitivos.

## Contenido del Proyecto

1. **Data Collection (Recopilación de Datos)**
   - Fuentes de datos utilizadas
   - Métodos de recopilación de datos

2. **Data Cleaning (Limpieza de Datos)**
   - Procedimientos de limpieza de datos
   - Tratamiento de valores faltantes y anómalos

3. **Data Analysis (Análisis de Datos)**
   - Herramientas y técnicas utilizadas para el análisis
   - Comparación de características entre vehículos de gama alta y baja

4. **Resultados**
   - Insights clave obtenidos del análisis
   - Recomendaciones para el cliente basadas en los resultados

## Conclusión

El análisis realizado proporciona una visión clara de las características predominantes en los vehículos de gama alta y baja en el mercado local. Estas conclusiones permiten a nuestro cliente:
- Ajustar su oferta de vehículos para satisfacer las preferencias del mercado local.
- Establecer precios competitivos que reflejen las expectativas de los consumidores.
- Maximizar su cobertura del mercado al ofrecer una gama de vehículos que atraigan a todos los segmentos del público objetivo.

<ul><h2>segun el estudio las variables a considerar son las siguientes:</h2>
  <li>wheelbase</li>
  <li>carlength</li>
  <li>carwidth</li>
  <li>curbweight</li>
  <li>cylindernumber</li>
  <li>enginesize</li>
  <li>boreratio</li>
  <li>horsepower</li>
  <li>drivewheel</li>
  <li>fuelsystem</li>
  <li>citympg</li>
  <li>highwaympg</li>
</ul>
  

Este proyecto representa un paso crucial para la entrada exitosa de nuestro cliente en el mercado local de automóviles, asegurando que su catálogo esté alineado con las demandas y preferencias de los consumidores.

## Contacto

Para más información sobre este proyecto, por favor contactar a:

- **Rammer Gomez**
- **Rammer@rammerbot.com**
- **[LinkedIn](https://www.linkedin.com/in/rammer-gomez/)**

## Descripción de los Análisis y Procedimientos

## Importar librerias necesarias

```python
import pandas as pd
from collections import Counter

import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.metrics import confusion_matrix, roc_auc_score, roc_curve, classification_report, precision_recall_curve
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold, GridSearchCV, RandomizedSearchCV
from xgboost import XGBClassifier
from collections import Counter
# optimizar el modelo con optimizacion de hiperparametros optuna
import optuna
from optuna import Trial
from optuna.samplers import TPESampler
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import RobustScaler, StandardScaler
from sklearn.decomposition import PCA
```
## Cargar dataset.
```python
path = 'ML_cars.csv'

datos = pd.read_csv(path)
datos
```
![image](https://github.com/rammerbot/Proyecto_Machine_learning_car/assets/123994694/7fd1bf20-0a70-4506-84e9-2a5a09c156b3)


> Posterior a la carga de los datos se evaluan para comprobar la calidad y realizar un analisis exploratorio.

## inspeccionar dataset.
```python
datos['fuelsystem'].unique()
```
![image](https://github.com/rammerbot/Proyecto_Machine_learning_car/assets/123994694/bcc4adae-f340-4000-9743-83058cc135b6)

```python
datos.info()
```
![image](https://github.com/rammerbot/Proyecto_Machine_learning_car/assets/123994694/be1a73fb-ee62-4551-b63b-d9726f986048)
> De manera sencilla evaluamos cuantos datos tiene el dataset, que tipo de datos contiene cada columna. dejando en total 205 filas con 26 columnas, 10 de ellas categoricas y 16 numericas.

```python
datos.isnull().sum()
```
> en la verificacion de los datos nulos pueden observarse que el dataset no contiene datos nulos dentro de sus filas lo que facilitaria el trabajo de limpieza.

```python
datos.describe()
```
![image](https://github.com/rammerbot/Proyecto_Machine_learning_car/assets/123994694/fca1efa4-de9a-4e13-a965-9732a1c23709)

> Con Describe verificamos los datos numericos, en este caso no se obsevan datos atipicos o anormalidades en el dataset.

## Evaluar datos categoricos
```python
cat_cols =  ['CarName','fueltype', 'aspiration', 'doornumber','carbody',
             'drivewheel', 'enginelocation','enginetype', 'cylindernumber','fuelsystem']

fig, ax= plt.subplots(nrows=len(cat_cols), ncols=1, figsize=(10,35))

for i, col in enumerate(cat_cols):
    sns.countplot(x=col, data=datos, ax=ax[i])
    ax[i].set_title(col)
    ax[i].set_xticklabels(ax[i].get_xticklabels(), rotation=30)
![image](https://github.com/rammerbot/Proyecto_Machine_learning_car/assets/123994694/c0e40f81-dd8c-4477-86f6-243d5209344d)
```
> al graficar los datos categoricos se evalua el comportamiento de los mismos, denotando que hay vario datos sesgados y desbalance entre las columnas categoricas.

### Eliminar Columnas

```python
# variables que a utilizar en el modelo.
"""wheelbase
carlength
carwidth
curbweight
cylindernumber
enginesize
boreratio
horsepower
drivewheel
fuelsystem
citympg
highwaympg
"""
# Lista de variables a eliminar.
drop_colums = ['car_ID', 'symboling', 'CarName','fueltype', 'aspiration', 'doornumber',
               'carbody', 'enginelocation', 'carheight', 'enginetype', 'stroke','compressionratio', 'peakrpm' ]

datos = datos.drop(columns=drop_colums)
datos
```
![image](https://github.com/rammerbot/Proyecto_Machine_learning_car/assets/123994694/da3116a8-ef79-4ef3-a3a0-7b60ecb1a411)
> en este punto se eliminan las columnas con menor grado de correlacion con respecto a la columna precio.

### Proceso de Transformacion de datos

> Transformar variables categoricas a numericas binarias.
```python
# Criterios para transformar categoricas

col_cat = {
    'cylindernumber': {'two':2,'three':3,'four':4, 'five':5, 'six':6,'eight':8, 'twelve':12}
}

for col,v0 in zip(col_cat,col_cat.values()):
    datos[col] = datos[col].map(v0)
datos
```

![image](https://github.com/rammerbot/Proyecto_Machine_learning_car/assets/123994694/98f40218-e315-4e98-9419-921e952ebd26)

> Para normalizar los datos se han transformado los valores categoricos a numericos en la columna cylindernumber.
```python
### Obtener variables dummies
> Transformar variables categoricas a dummies

col_cat = ['drivewheel', 'fuelsystem']
datos = pd.get_dummies(datos, columns=col_cat, drop_first=True)
datos
```
![image](https://github.com/rammerbot/Proyecto_Machine_learning_car/assets/123994694/7e3185a4-30ab-40e4-94a2-c6cd4dc376ef)

> se Transforman columnas categoricas a dummies, esta accion se realiza para poder cargar los datos al modelo y no arroje error, debido a que los modelos trabajan solo con datos numericos.

## Analisis de Correlacion

```python
corr = datos.corr()
```
```python
corr
```
![image](https://github.com/rammerbot/Proyecto_Machine_learning_car/assets/123994694/dca3189c-d78f-4379-9730-847111d2a06c)
> Evaluacion de correlacion entre las columnas con el fin de seleccionar las colunas para el modelo y reducir su complejidad, se hara una seleccion de las columnas que tengan una correlacion mayo a 0.5 y menor a -0.5 con respecto de la columna 'price'.

```python
# Variable con correlacion de datos desde el Dataset con pandas.
corr = datos.corr()

# Diagaramado con matplotlib y Seaborn
plt.figure(figsize=(20,20))
sns.heatmap(corr, cbar = True,  square = True, annot=True, fmt= '.2f',annot_kws={'size': 15},
           xticklabels= datos.columns, 
           yticklabels= datos.columns,
           cmap= 'coolwarm')
plt.show()
```
![image](https://github.com/rammerbot/Proyecto_Machine_learning_car/assets/123994694/29499623-e51b-4862-a8ae-8b3832ae51d7)

## Crear Rango de gamas

```python
# Cálculo de los límites
precio_min = datos['price'].min()
precio_max = datos['price'].max()
mitad = (precio_max - precio_min) / 2

# Límites ajustados
limite_baja_alta = precio_min + mitad

# Discretización del precio
bins = [precio_min, limite_baja_alta, precio_max]
labels = ['Baja', 'Alta']

# Discretización del precio
datos['gama'] = pd.cut(datos['price'], bins=bins, labels=labels, include_lowest=True)

datos
```
![image](https://github.com/rammerbot/Proyecto_Machine_learning_car/assets/123994694/86fbdd01-ce8b-49e5-9930-1e4c397f38f0)

> Se ha creado una variable nueva llamada 'gama' donde los autos estarian categorizados en gama-alta y gama-baja tomando en cuenta el rango de precio de los vehiculos, esto con el fin de dar respuesta mediante nuestro modelo a la solicitud del cliente.
```python
sns.countplot(x='gama', data=datos)
```
> Mediante el siguiente grafico se puede observar un desvalance entre las clases Baja y Alta.

## Castear Gama

```python
datos['gama'] = datos['gama'].map(
    {'Baja':0, 'Alta':1 }
)
```
```python
datos['gama'].unique()
```
![image](https://github.com/rammerbot/Proyecto_Machine_learning_car/assets/123994694/3844209e-ee4e-4f0e-b306-a18fd5c17e84)
> Se realiza un casteo en donde la clase Alta recibira el valor de 1 y la baja de 0

## Seleccionar Variables X, Y.

```python
columnas_drop = ['price', 'gama']

X = datos.drop(columns=columnas_drop, axis=1)
Y = datos['gama']

```
> Mediante el siguiente comando se selecionan las variables X y Y, en donde Y sera 'Gama' y X todas las demas menos 'price' y 'gama'

### Division de datos

```python
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size= 0.35, random_state=1990, stratify=Y)
```
mediante el metodo train_tes_split se dividen los datos para el entrenamiento y las pruebas del modelo.

### Prueba de modelos de categorizacion

```python
# Variable con las listas de elementos
modelos = []
resultados = []
nombres_modelos = []

# Agregar Modelos a las listas
modelos.append(('Regresion Logistica', LogisticRegression()))
modelos.append(('Arbol de Decision', DecisionTreeClassifier()))
modelos.append(('Bosque de Clasificacion', RandomForestClassifier()))
```

```python
# For para instanciar los modelos con cross_val_score y obtener los parametros de manera individual
for nombre_modelo, model in modelos:
    kfold = StratifiedKFold(n_splits=3,shuffle=True,random_state=1990)
    resultados_cross_value = cross_val_score(model, X_train, Y_train, cv=kfold, scoring='roc_auc')
    nombres_modelos.append(nombre_modelo)
    resultados.append(resultados_cross_value)

# imprimir resultado de modelos.
for i in range(len(nombres_modelos)):
    print(nombres_modelos[i], resultados[i].mean())
```

![image](https://github.com/rammerbot/Proyecto_Machine_learning_car/assets/123994694/58f84956-78c1-4627-98a1-651e543391b6)

> se realiza una medicion con cross validation para escoger el mejor modelo a utilizar en el proyecto, en este caso se realizara mediante Regresion Logistoca.

### Crear un Pipeline

```python
# crear Pipeline
modelo = Pipeline((
    ('scale', StandardScaler()), ('log_reg', LogisticRegression(C=10, solver='lbfgs', n_jobs=-1,fit_intercept=True))
))

modelo.fit(X_train,Y_train)
```
![image](https://github.com/rammerbot/Proyecto_Machine_learning_car/assets/123994694/8c0338f3-20e6-4762-a4ae-b43234cf609d)


> Mediante pipline se crea un flojo de trabajo para automatizar la creacion del modelo y con el metodo fit se entrena con los datos X_train y Y_train respectivamente que representan el 20 % del total de los datos.

## Probar Modelo entrenado.
```python
Y_fit_train = modelo.predict(X_train)
Y_fit_test = modelo.predict(X_test)
```
> Se procede a realizar una prediccion con los datos que el modelo conoce en X_train y posteriormente con datos no desconocidos por el modelo con X_test.

### Crear un Pipeline
```python
print(classification_report(Y_train, Y_fit_train))
```
![image](https://github.com/rammerbot/Proyecto_Machine_learning_car/assets/123994694/520fe4bc-ea92-49a7-a9f5-388e374223f0)

```python
print(classification_report(Y_test, Y_fit_test))
```
![image](https://github.com/rammerbot/Proyecto_Machine_learning_car/assets/123994694/206d4bd6-3fdd-4694-b943-601899150b71)

```python
matiz_train = confusion_matrix(Y_train, Y_fit_train)
matiz_train
```
![image](https://github.com/rammerbot/Proyecto_Machine_learning_car/assets/123994694/6eecf516-4693-468d-9941-beb196d71c38)

```python
matiz_test = confusion_matrix(Y_test, Y_fit_test)
matiz_test
```
![image](https://github.com/rammerbot/Proyecto_Machine_learning_car/assets/123994694/74e436a3-9ab6-46db-a6e7-068966d54667)
