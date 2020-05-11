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


#def model4cartas (suit1,suit2,card1,card2,suit3,card3,suit4,card4):
    #test_data3 = [[suit1,card1,suit2,card2,suit3,card3,suit4,card4]]
    test_data3 = [[1,3,4,8,4,8,3,11]]
    print (test_data3)
    dtc3 = load('dtc4cards.joblib')
    knn3 = load('knn4cards.joblib')
    gnb3 = load('gnb4cards.joblib')
    mlpc3 = load('mlpc4cards.joblib')
    
    dtcpred3 = dtc3.predict_proba(test_data3)
    print ("Prediction accuracy: {}".format(dtcpred3))
    
    knnpred3 = knn3.predict_proba(test_data3)
    print ("Prediction accuracy: {}".format(knnpred3))
    
    gnbpred3 = gnb3.predict_proba(test_data3)
    print ("Prediction accuracy: {}".format(gnbpred3))
    
    mlpcpred3 = mlpc3.predict_proba(test_data3)
    print ("Prediction accuracy: {}".format(mlpcpred3))
    
#model3cartas(1,3,4,8,4,8)