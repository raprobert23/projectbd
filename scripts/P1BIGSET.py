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
from sklearn.neural_network import MLPClassifier 
from joblib import dump, load
plt.style.use('ggplot') 


"""

 Idea para el desarollo del proyecto en una web API:
    
    Marcariamos en 4 casillas el valor numerico de la carta (1 al 13) y su respectivo 
    palo ( del 1 al 4)
    
    Se enviaria esa informacion y se cotejarian dicha combinación con todas las manos del dataset
    que contengan esas dos cartas, se haria un recuento de el numero de manos existentes en el dataset.
    El objetivo sería que se intuyese con esas dos cartas donde clasificar la jugada una vez esta se
    finalize (cuando se entregan las 3 cartas restantes). Se daria el resultado en funcion de que jugada
    cotejada anteriormente, se haya con mayor probabilidad. Luego se generarian aleatoriamente las 3 cartas
    restantes para completar la mano y se daria la jugada que efectivamente ha resultado. 
    
    Con esto veriamos la capacidad del classificador de con tan solo el 40% de la información, poder predecir
    que jugada obtendremos con mayor probabilidad una vez se complete el reparto de la mano.
    
    Para ello necesitaremos generar varios modelos con un dataset particular (es decir un dataset
    con las 4 primeras columnas (2 cartas) otro  con las 6 primeras (3 cartas) y otro con 8).
    
    


"""

################# Modelo número 1 (2 cartas seleccionadas) ##################################

m1 = pd.read_table("../project/poker-hand-training-true.data", sep=",", header=None)
col_names = ['SUIT 1', 'CARD 1', 'SUIT 2', 'CARD 2', 'SUIT 3', 'CARD 3', 'SUIT 4', 'CARD 4', 'SUIT 5', 'CARD 5', 'HAND']
m1.columns = col_names

X1 = m1.iloc[:,:-7]
y1 = m1.iloc[:,-1]



print('Class labels:', np.unique(y1)) #Tenemos 4 clases (2 cartas y 2 palos)
print('Labels counts in y:', np.bincount(y1))  #Número de jugadas asociadas a cada una de las manos


# Resampling Imbalanced Data
# conda install -c conda-forge imbalanced-learn
# https://imbalanced-learn.readthedocs.io/en/stable/generated/imblearn.over_sampling.SMOTE.html
# not majority': resample all classes but the majority class
# 'minority': resample only the minority class
# 'not majority': resample all classes but the majority class
# 'all': resample all classes

from imblearn import under_sampling, over_sampling
from imblearn.over_sampling import RandomOverSampler
sm = RandomOverSampler(sampling_strategy='auto', random_state = 21)
X1_resampled, y1_resampled = sm.fit_sample(X1, y1)
X1_resampled = pd.DataFrame(X1_resampled, columns=X1.columns)

print('Class labels:', np.unique(y1_resampled))
print('Labels counts in y_resampled:', np.bincount(y1_resampled))



X1_train, X1_test, y1_train, y1_test = train_test_split(X1_resampled, y1_resampled, test_size=0.40, random_state=16, stratify=y1_resampled)




# DecisionTreeClassifier
# https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html
# model train criterion='entropy', max_depth=2
dtc = DecisionTreeClassifier(criterion='entropy', max_depth=150, random_state=10)
dtc.fit(X1_train, y1_train)

y1_pred = dtc.predict(X1_test)

# Accuracy
accdtc = dtc.score(X1_test, y1_test)

print("Accuracy: {0:.2f}".format(accdtc))

print("Test set good labels: {}".format(y1_test))
print("Test set predictions: {}".format(y1_pred))
print('Well classified samples: {}'.format((y1_test == y1_pred).sum()))
print('Misclassified samples: {}'.format((y1_test != y1_pred).sum()))

dump(dtc,'dtc2cards.joblib') #### Guardamos el modelo para poder usarlo en otro script




# KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=150)

# Fit the classifier to the data

knn.fit(X1_train, y1_train)
y1_pred = knn.predict(X1_test)

# Accuracy
accknn = knn.score(X1_test, y1_test)

print("Accuracy: {0:.2f}".format(accknn))

print("Test set good labels: {}".format(y1_test))
print("Test set predictions: {}".format(y1_pred))
print('Well classified samples: {}'.format((y1_test == y1_pred).sum()))
print('Misclassified samples: {}'.format((y1_test != y1_pred).sum()))

dump(knn,'knn2cards.joblib') #### Guardamos el modelo para poder usarlo en otro script




# GaussianNB
# https://scikit-learn.org/stable/modules/generated/sklearn.naive_bayes.GaussianNB.html
# 
gnb = GaussianNB()
gnb.fit(X1_train, y1_train)

gnb.fit(X1_train, y1_train)
y1_pred = gnb.predict(X1_test)

# Accuracy
accgnb = gnb.score(X1_test, y1_test)

print("Accuracy: {0:.2f}".format(accgnb))

print("Test set good labels: {}".format(y1_test))
print("Test set predictions: {}".format(y1_pred))
print('Well classified samples: {}'.format((y1_test == y1_pred).sum()))
print('Misclassified samples: {}'.format((y1_test != y1_pred).sum()))

dump(gnb,'gnb2cards.joblib') #### Guardamos el modelo para poder usarlo en otro script




# Multi-layer Perceptron classifier
# https://scikit-learn.org/stable/modules/generated/sklearn.neural_network.MLPClassifier.html
# https://scikit-learn.org/stable/modules/neural_networks_supervised.html

# model train 
mlpc = MLPClassifier(alpha=1, max_iter=1000, hidden_layer_sizes=(200,100,100, ), early_stopping = True, warm_start = True)
mlpc.fit(X1_train, y1_train)
y1_pred = mlpc.predict(X1_test)

# Accuracy
accmlpc = mlpc.score(X1_test, y1_test)

print("Accuracy: {0:.2f}".format(accmlpc))

print("Test set good labels: {}".format(y1_test))
print("Test set predictions: {}".format(y1_pred))
print('Well classified samples: {}'.format((y1_test == y1_pred).sum()))
print('Misclassified samples: {}'.format((y1_test != y1_pred).sum()))

dump(mlpc,'mlpc2cards.joblib') #### Guardamos el modelo para poder usarlo en otro script





############################################################################################################



################# Modelo número 2 (3 cartas seleccionadas) ##################################

m1 = pd.read_table("../project/poker-hand-training-true.data", sep=",", header=None)
col_names = ['SUIT 1', 'CARD 1', 'SUIT 2', 'CARD 2', 'SUIT 3', 'CARD 3', 'SUIT 4', 'CARD 4', 'SUIT 5', 'CARD 5', 'HAND']
m1.columns = col_names

X2 = m1.iloc[:,:-5]
y2 = m1.iloc[:,-1]



print('Class labels:', np.unique(y2)) #Tenemos 4 clases (2 cartas y 2 palos)
print('Labels counts in y:', np.bincount(y2))  #Número de jugadas asociadas a cada una de las manos


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
X2_resampled, y2_resampled = sm.fit_sample(X2, y2)
X2_resampled = pd.DataFrame(X2_resampled, columns=X2.columns)

print('Class labels:', np.unique(y2_resampled))
print('Labels counts in y_resampled:', np.bincount(y2_resampled))



X2_train, X2_test, y2_train, y2_test = train_test_split(X2_resampled, y2_resampled, test_size=0.40, random_state=4, stratify=y2_resampled)




# DecisionTreeClassifier
# https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html
# model train criterion='entropy', max_depth=2
dtc2 = DecisionTreeClassifier(criterion='entropy', max_depth=150, random_state=10)
dtc2.fit(X2_train, y2_train)


y2_pred = dtc2.predict(X2_test)

# Accuracy
accdtc2 = dtc2.score(X2_test, y2_test)

print("Accuracy: {0:.2f}".format(accdtc2))

print("Test set good labels: {}".format(y2_test))
print("Test set predictions: {}".format(y2_pred))
print('Well classified samples: {}'.format((y2_test == y2_pred).sum()))
print('Misclassified samples: {}'.format((y2_test != y2_pred).sum()))

dump(dtc2,'dtc3cards.joblib') #### Guardamos el modelo para poder usarlo en otro script




# KNeighborsClassifier
knn2 = KNeighborsClassifier(n_neighbors=150)

# Fit the classifier to the data

knn2.fit(X2_train, y2_train)
y2_pred = knn2.predict(X2_test)

# Accuracy
accknn2 = knn2.score(X2_test, y2_test)

print("Accuracy: {0:.2f}".format(accknn2))

print("Test set good labels: {}".format(y2_test))
print("Test set predictions: {}".format(y2_pred))
print('Well classified samples: {}'.format((y2_test == y2_pred).sum()))
print('Misclassified samples: {}'.format((y2_test != y2_pred).sum()))

dump(knn2,'knn3cards.joblib') #### Guardamos el modelo para poder usarlo en otro script




# GaussianNB
# https://scikit-learn.org/stable/modules/generated/sklearn.naive_bayes.GaussianNB.html
# 
gnb2 = GaussianNB()
gnb2.fit(X2_train, y2_train)

gnb2.fit(X2_train, y2_train)
y2_pred = gnb2.predict(X2_test)

# Accuracy
accgnb2 = gnb2.score(X2_test, y2_test)

print("Accuracy: {0:.2f}".format(accgnb2))

print("Test set good labels: {}".format(y2_test))
print("Test set predictions: {}".format(y2_pred))
print('Well classified samples: {}'.format((y2_test == y2_pred).sum()))
print('Misclassified samples: {}'.format((y2_test != y2_pred).sum()))

dump(gnb2,'gnb3cards.joblib') #### Guardamos el modelo para poder usarlo en otro script



    
# Multi-layer Perceptron classifier
# https://scikit-learn.org/stable/modules/generated/sklearn.neural_network.MLPClassifier.html
# https://scikit-learn.org/stable/modules/neural_networks_supervised.html

# model train 
mlpc2 = MLPClassifier(alpha=1, max_iter=1000, hidden_layer_sizes=(200,100,100, ), early_stopping = True, warm_start = True)
mlpc2.fit(X2_train, y2_train)
y2_pred = mlpc2.predict(X2_test)

# Accuracy
accmlpc2 = mlpc2.score(X2_test, y2_test)

print("Accuracy: {0:.2f}".format(accmlpc2))

print("Test set good labels: {}".format(y2_test))
print("Test set predictions: {}".format(y2_pred))
print('Well classified samples: {}'.format((y2_test == y2_pred).sum()))
print('Misclassified samples: {}'.format((y2_test != y2_pred).sum()))

dump(mlpc2,'mlpc3cards.joblib') #### Guardamos el modelo para poder usarlo en otro script




############################################################################################################




################# Modelo número 3 (4 cartas seleccionadas) ##################################

m1 = pd.read_table("../project/poker-hand-training-true.data", sep=",", header=None)
col_names = ['SUIT 1', 'CARD 1', 'SUIT 2', 'CARD 2', 'SUIT 3', 'CARD 3', 'SUIT 4', 'CARD 4', 'SUIT 5', 'CARD 5', 'HAND']
m1.columns = col_names

X3 = m1.iloc[:,:-3]
y3 = m1.iloc[:,-1]



print('Class labels:', np.unique(y3)) #Tenemos 4 clases (2 cartas y 2 palos)
print('Labels counts in y:', np.bincount(y3))  #Número de jugadas asociadas a cada una de las manos


# Resampling Imbalanced Data
# conda install -c conda-forge imbalanced-learn
# https://imbalanced-learn.readthedocs.io/en/stable/generated/imblearn.over_sampling.SMOTE.html
# not majority': resample all classes but the majority class
# 'minority': resample only the minority class
# 'not majority': resample all classes but the majority class
# 'all': resample all classes

from imblearn import under_sampling, over_sampling
from imblearn.over_sampling import RandomOverSampler
sm = RandomOverSampler(sampling_strategy='auto', random_state = 7)
X3_resampled, y3_resampled = sm.fit_sample(X3, y3)
X3_resampled = pd.DataFrame(X3_resampled, columns=X3.columns)

print('Class labels:', np.unique(y3_resampled))
print('Labels counts in y_resampled:', np.bincount(y3_resampled))



X3_train, X3_test, y3_train, y3_test = train_test_split(X3_resampled, y3_resampled, test_size=0.30, random_state=23, stratify=y3_resampled)




# DecisionTreeClassifier
# https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html
# model train criterion='entropy', max_depth=2
dtc3 = DecisionTreeClassifier(criterion='entropy', max_depth=150, random_state=10)
dtc3.fit(X3_train, y3_train)

y3_pred = dtc3.predict(X3_test)

# Accuracy
accdtc3 = dtc3.score(X3_test, y3_test)

print("Accuracy: {0:.2f}".format(accdtc3))

print("Test set good labels: {}".format(y3_test))
print("Test set predictions: {}".format(y3_pred))
print('Well classified samples: {}'.format((y3_test == y3_pred).sum()))
print('Misclassified samples: {}'.format((y3_test != y3_pred).sum()))

dump(dtc3,'dtc4cards.joblib') #### Guardamos el modelo para poder usarlo en otro script




# KNeighborsClassifier
knn3 = KNeighborsClassifier(n_neighbors=150)

# Fit the classifier to the data

knn3.fit(X3_train, y3_train)
y3_pred = knn3.predict(X3_test)

# Accuracy
accknn3 = knn3.score(X3_test, y3_test)

print("Accuracy: {0:.2f}".format(accknn3))

print("Test set good labels: {}".format(y3_test))
print("Test set predictions: {}".format(y3_pred))
print('Well classified samples: {}'.format((y3_test == y3_pred).sum()))
print('Misclassified samples: {}'.format((y3_test != y3_pred).sum()))

dump(knn3,'knn4cards.joblib') #### Guardamos el modelo para poder usarlo en otro script




# GaussianNB
# https://scikit-learn.org/stable/modules/generated/sklearn.naive_bayes.GaussianNB.html
# 
gnb3 = GaussianNB()
gnb3.fit(X3_train, y3_train)

gnb3.fit(X3_train, y3_train)
y3_pred = gnb3.predict(X3_test)

# Accuracy
accgnb3 = gnb3.score(X3_test, y3_test)

print("Accuracy: {0:.2f}".format(accgnb3))

print("Test set good labels: {}".format(y3_test))
print("Test set predictions: {}".format(y3_pred))
print('Well classified samples: {}'.format((y3_test == y3_pred).sum()))
print('Misclassified samples: {}'.format((y3_test != y3_pred).sum()))

dump(gnb3,'gnb4cards.joblib') #### Guardamos el modelo para poder usarlo en otro script



    
# Multi-layer Perceptron classifier
# https://scikit-learn.org/stable/modules/generated/sklearn.neural_network.MLPClassifier.html
# https://scikit-learn.org/stable/modules/neural_networks_supervised.html

# model train 
mlpc3 = MLPClassifier(alpha=1, max_iter=1000, hidden_layer_sizes=(200,100,100, ), early_stopping = True, warm_start = True)
mlpc3.fit(X3_train, y3_train)
y3_pred = mlpc3.predict(X3_test)

# Accuracy
accmlpc3 = mlpc3.score(X3_test, y3_test)

print("Accuracy: {0:.2f}".format(accmlpc3))

print("Test set good labels: {}".format(y3_test))
print("Test set predictions: {}".format(y3_pred))
print('Well classified samples: {}'.format((y3_test == y3_pred).sum()))
print('Misclassified samples: {}'.format((y3_test != y3_pred).sum()))

dump(mlpc3,'mlpc4cards.joblib') #### Guardamos el modelo para poder usarlo en otro script



