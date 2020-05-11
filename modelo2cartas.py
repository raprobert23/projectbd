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


#def model2cartas (suit1,suit2,card1,card2):
    #test_data = [[suit1,card1,suit2,card2]]
    test_data = [[1,3,4,8]]
    print (test_data)
    dtc = load('dtc2cards.joblib')
    knn = load('knn2cards.joblib')
    gnb = load('gnb2cards.joblib')
    mlpc = load('mlpc2cards.joblib')
    
    dtcpred = dtc.predict_proba(test_data)
    print ("Prediction accuracy: {}".format(dtcpred))
    
    knnpred = knn.predict_proba(test_data)
    print ("Prediction accuracy: {}".format(knnpred))
    
    gnbpred = gnb.predict_proba(test_data)
    print ("Prediction accuracy: {}".format(gnbpred))
    
    mlpcpred = mlpc.predict_proba(test_data)
    print ("Prediction accuracy: {}".format(mlpcpred))
    
#model2cartas(1,3,4,8)



