import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt2
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
plt.style.use('ggplot') 

# Load dataset (El de entrenamiento para el classificar.Tambien lo usaremos para ver de forma mas clara como se distribuyen los datos)
df = pd.read_table("../project/poker-hand-training-true.data", sep=",", header=None)
col_names = ['SUIT 1', 'CARD 1', 'SUIT 2', 'CARD 2', 'SUIT 3', 'CARD 3', 'SUIT 4', 'CARD 4', 'SUIT 5', 'CARD 5', 'HAND']
df.columns = col_names

df.head()
df.info()
df.describe()

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