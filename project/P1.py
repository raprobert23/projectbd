import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from util import plot_decision_regions
import matplotlib.pyplot as plt2
import numpy as np
import imblearn
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
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



# Plot 1: Observamos la cantidad obtenida en el dataset de cada una de las manos
plt.figure(figsize =(10,8), facecolor="lightyellow", edgecolor="black")
sns.countplot(x='HAND', hue='SUIT 4', data=df, palette='RdBu')
plt.xticks([0,1,2,3,4,5,6,7,8,9], ['0', '1','2','3','4','5','6','7','8','9' ])
plt.yticks(np.arange(0,3400,step=150))
plt.show()

# Plot 2: Observamos que de cada carta que nos entregan (CARD 1), obtenemos una equiprobabilidad de entre las 13 existentes
#Esta equiprobabilidad se mantiene tambien para los palos de la baraja (SUIT 1)
plt2.figure(figsize =(10,8), facecolor="lightyellow", edgecolor="black")
sns.countplot(x='CARD 1', hue='SUIT 1', data=df, palette='RdBu')
plt2.xticks([0,1,2,3,4,5,6,7,8,9,10,11,12], ['1','2','3','4','5','6','7','8','9','10','11','12','13'])
plt2.yticks(np.arange(0,675,step=25))
plt2.show()

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
X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.70, random_state=12, stratify=y_resampled)


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
dtc = DecisionTreeClassifier(criterion='entropy', max_depth=150, random_state=1)
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



# Support vector classifier SVC
# https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html
# https://scikit-learn.org/stable/modules/svm.html
# 
svm = SVC(kernel='linear')
svm.fit(X_train, y_train)

svm.fit(X_train, y_train)
y_pred = svm.predict(X_test)

# Accuracy
accsvm = svm.score(X_test, y_test)

print("Accuracy: {0:.2f}".format(accsvm))

print("Test set good labels: {}".format(y_test))
print("Test set predictions: {}".format(y_pred))
print('Well classified samples: {}'.format((y_test == y_pred).sum()))
print('Misclassified samples: {}'.format((y_test != y_pred).sum()))

