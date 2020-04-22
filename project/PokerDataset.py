# -*- coding: utf-8 -*-
"""
Created on Fri Mar 20 18:07:55 2020

Attribute Information:
   1) S1 �Suit of card #1�
      Ordinal (1-4) representing {Hearts, Spades, Diamonds, Clubs}

   2) C1 �Rank of card #1�
      Numerical (1-13) representing (Ace, 2, 3, ... , Queen, King)

   3) S2 �Suit of card #2�
      Ordinal (1-4) representing {Hearts, Spades, Diamonds, Clubs}

   4) C2 �Rank of card #2�
      Numerical (1-13) representing (Ace, 2, 3, ... , Queen, King)

   5) S3 �Suit of card #3�
      Ordinal (1-4) representing {Hearts, Spades, Diamonds, Clubs}

   6) C3 �Rank of card #3�
      Numerical (1-13) representing (Ace, 2, 3, ... , Queen, King)

   7) S4 �Suit of card #4�
      Ordinal (1-4) representing {Hearts, Spades, Diamonds, Clubs}

   8) C4 �Rank of card #4�
      Numerical (1-13) representing (Ace, 2, 3, ... , Queen, King)

   9) S5 �Suit of card #5�
      Ordinal (1-4) representing {Hearts, Spades, Diamonds, Clubs}

   10) C5 �Rank of card 5�
      Numerical (1-13) representing (Ace, 2, 3, ... , Queen, King)

   11) CLASS �Poker Hand�
      Ordinal (0-9)

      0: Nothing in hand; not a recognized poker hand 
      1: One pair; one pair of equal ranks within five cards
      2: Two pairs; two pairs of equal ranks within five cards
      3: Three of a kind; three equal ranks within five cards
      4: Straight; five cards, sequentially ranked with no gaps
      5: Flush; five cards with the same suit
      6: Full house; pair + different rank three of a kind
      7: Four of a kind; four equal ranks within five cards
      8: Straight flush; straight + flush
      9: Royal flush; {Ace, King, Queen, Jack, Ten} + flush


@author: Adri
"""
import pandas as pd
import time
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import datasets
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_moons, make_circles, make_classification
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report


data = pd.read_table("poker-hand-testing.data", sep=",", header=None, na_values="?")
col_names = ['S1', 'C1', 'S2', 'C2', 'S3', 'C3', 'S4', 'C4', 'S5', 'C5', 'CLASS']
data.columns = col_names

X =data.drop('CLASS',axis=1)
y= data['CLASS']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=21, stratify=y)#srandom state es el seed

# Create a k-NN classifier with 5 neighbors #####################################
knn = KNeighborsClassifier(n_neighbors=5)


# Fit the classifier to the data
t1 = time.time()
knn.fit(X_train, y_train)
t2 = time.time()

tf = round(t2-t1,3)
print("KNN(n=5) Train cost: {0:.3f}".format(tf))


t1 = time.time()
# Fit the classifier to the data
y_pred = knn.predict(X_test)
t2 = time.time()
tf = round(t2-t1,3)

print("KNN(n=5) Test cost: {0:.3f}".format(tf))



print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))