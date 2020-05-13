import pandas as pd
import seaborn as sns
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier 
from joblib import dump, load



'''
Remember to uncomment the commented lines and delete the test_data line in order to pass the values
as the paratmethers of the function

'''


#def model3cartas (suit1,suit2,card1,card2,suit3,card3):
    #test_data2 = [[suit1,card1,suit2,card2,suit3,card3]]
    test_data2 = [[1,3,4,8,4,8]]
    print (test_data2)
    dtc2 = load('dtc3cards.joblib')
    knn2 = load('knn3cards.joblib')
    gnb2 = load('gnb3cards.joblib')
    mlpc2 = load('mlpc3cards.joblib')
    
    dtcpred2 = dtc2.predict_proba(test_data2)
    print ("Prediction accuracy: {}".format(dtcpred2))
    
    knnpred2 = knn2.predict_proba(test_data2)
    print ("Prediction accuracy: {}".format(knnpred2))
    
    gnbpred2 = gnb2.predict_proba(test_data2)
    print ("Prediction accuracy: {}".format(gnbpred2))
    
    mlpcpred2 = mlpc2.predict_proba(test_data2)
    print ("Prediction accuracy: {}".format(mlpcpred2))
    
#model3cartas(1,3,4,8,4,8)