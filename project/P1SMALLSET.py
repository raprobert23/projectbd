import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt2
import numpy as np
import imblearn
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
plt.style.use('ggplot') 

# Load dataset (El de entrenamiento para el classificar.Tambien lo usaremos para ver de forma mas clara como se distribuyen los datos)
df = pd.read_table("../project/finalset.data", sep=",", header=None)
col_names = ['SUIT 1', 'CARD 1', 'SUIT 2', 'CARD 2', 'SUIT 3', 'CARD 3', 'SUIT 4', 'CARD 4', 'SUIT 5', 'CARD 5', 'HAND']
df.columns = col_names

# Get info
df.shape
df.head()
df.mode()
df = df.fillna(df.mode().iloc[0])
df.head()
df.HAND.value_counts() # este comando nos devuelve el número de jugadas que se han dado en nuestro dataset.
                       #Las manos menos probables, se corresponden a los valores más bajos y viceversa (para mas info consultar.names)



##############################################################################################################

# =============================================================================
# - Podemos darnos cuenta que esos datos estan totalmente descompensados en un aspecto fundamental
# y es que apenas tenemos jugadas altas (8,9) pero la gran mayoría de muestras ertencen a las jugadas (0,1).
# Esto no provocara un desvalanceo cuando entrenemos a los classificadores pues solo aprenderan a discernir
# entre un 0 y un 1 (jugadas más probables que representan el 95% del dataset anterior). Para ello hemos de
# utilizar una herramienta de sobre-muestreo.
# =============================================================================
  
X = df.iloc[:,:-1]
y = df.iloc[:,-1]

print('Class labels:', np.unique(y)) #Tenemos 10 clases (5 cartas y 5 palos)
print('Labels counts in y:', np.bincount(y))  #Número de jugadas asociadas a cada una de las manos


# Resampling Imbalanced Data
# conda install -c conda-forge imbalanced-learn
# https://imbalanced-learn.readthedocs.io/en/stable/generated/imblearn.over_sampling.SMOTE.html
# not majority': resample all classes but the majority class
# 'minority': resample only the minority class
# 'not majority': resample all classes but the majority class
# 'all': resample all classes

from imblearn import under_sampling, over_sampling
from imblearn.over_sampling import RandomOverSampler
sm = RandomOverSampler(sampling_strategy='auto', random_state = 13)
X_resampled, y_resampled = sm.fit_sample(X, y)
X_resampled = pd.DataFrame(X_resampled, columns=X.columns)

print('Class labels:', np.unique(y_resampled))
print('Labels counts in y_resampled:', np.bincount(y_resampled))




############################################################################################################


X = df[['SUIT 1', 'CARD 1', 'SUIT 2', 'CARD 2', 'SUIT 3', 'CARD 3', 'SUIT 4', 'CARD 4', 'SUIT 5', 'CARD 5']]
y = df['HAND']
X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.50, random_state=21, stratify=y_resampled)


#############################################################################################################

# KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=150)

# Fit the classifier to the data

knn.fit(X_train, y_train)
y_pred = knn.predict(X_test)

# Accuracy
accknn = knn.score(X_test, y_test)

print("Accuracy: {0:.2f}".format(accknn))

print("Test set good labels: {}".format(y_test))
print("Test set predictions: {}".format(y_pred))
print('Well classified samples: {}'.format((y_test == y_pred).sum()))
print('Misclassified samples: {}'.format((y_test != y_pred).sum()))



# DecisionTreeClassifier
# https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html
# model train criterion='entropy', max_depth=2
dtc = DecisionTreeClassifier(criterion='entropy', max_depth=150, random_state=4)
dtc.fit(X_train, y_train)


dtc.fit(X_train, y_train)
y_pred = dtc.predict(X_test)

# Accuracy
accdtc = dtc.score(X_test, y_test)

print("Accuracy: {0:.2f}".format(accdtc))

print("Test set good labels: {}".format(y_test))
print("Test set predictions: {}".format(y_pred))
print('Well classified samples: {}'.format((y_test == y_pred).sum()))
print('Misclassified samples: {}'.format((y_test != y_pred).sum()))



# GaussianNB
# https://scikit-learn.org/stable/modules/generated/sklearn.naive_bayes.GaussianNB.html
# 
gnb = GaussianNB()
gnb.fit(X_train, y_train)

gnb.fit(X_train, y_train)
y_pred = gnb.predict(X_test)

# Accuracy
accgnb = gnb.score(X_test, y_test)

print("Accuracy: {0:.2f}".format(accgnb))

print("Test set good labels: {}".format(y_test))
print("Test set predictions: {}".format(y_pred))
print('Well classified samples: {}'.format((y_test == y_pred).sum()))
print('Misclassified samples: {}'.format((y_test != y_pred).sum()))



"""

 Idea para el desarollo del proyecto en una web API:
    
    Marcariamos en 4 casillas el valor numerico de la carta (1 al 13) y su respectivo 
    palo ( del 1 al 4)
    
    Se enviaria esa informacion y se cotejarian dicha combinación con todas las manos del dataset
    que contengan esas dos cartas, se haria un recuento de el numero de manos existentes en el dataset.
    El objetivo sería que se intuyese con esas dos cartas donde calssificar la jugada una vez esta se
    finalize (cuando se entregan las 3 cartas restantes). Se daria el resultado en funcion de que jugada
    cotejada anteriormente, se haya con mayor probabilidad. Luego se generarian aleatoriamente las 3 cartas
    restantes para completar la mano y se daria la jugada que efectivamente ha resultado. 
    
    Con esto veriamos la capacidad del classificador de con tan solo el 40% de la información, poder predecir
    que jugada obtendremos con mayor probabilidad una vez se complete el reparto de la mano


"""
    

