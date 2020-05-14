from joblib import dump, load
import os

THIS_FOLDER = os.path.dirname(os.path.abspath(__file__ )) 
import sys

s1 = int(sys.argv[1])
c1 = int(sys.argv[2])
s2 = int(sys.argv[3])
c2 = int(sys.argv[4])
s3 = int(sys.argv[5])
c3 = int(sys.argv[6])
s4 = int(sys.argv[7])
c4 = int(sys.argv[8])

dtc4cards = os.path.join(THIS_FOLDER, 'dtc4cards.joblib')
knn4cards = os.path.join(THIS_FOLDER, 'knn4cards.joblib')
gnb4cards = os.path.join(THIS_FOLDER, 'gnb4cards.joblib')
mlpc4cards = os.path.join(THIS_FOLDER, 'mlpc4cards.joblib')

'''
Remember to uncomment the commented lines and delete the test_data line in order to pass the values
as the paratmethers of the function

'''


# def model4cartas (s1, c1, s2, c2, s3, c3, s4, c4):
test_data3 = [[s1, c1, s2, c2, s3, c3, s4, c4]]
print(test_data3)

dtc3 = load(dtc4cards)
knn3 = load(knn4cards)
gnb3 = load(gnb4cards)
mlpc3 = load(mlpc4cards)

    
dtcpred3 = dtc3.predict_proba(test_data3)
dtcpred3 = dtcpred3.round(3)
print ("Prediction accuracy: {}".format(dtcpred3))
    
knnpred3 = knn3.predict_proba(test_data3)
knnpred3 = knnpred3.round(3)
print ("Prediction accuracy: {}".format(knnpred3))
    
gnbpred3 = gnb3.predict_proba(test_data3)
gnbpred3 = gnbpred3.round(3)
print ("Prediction accuracy: {}".format(gnbpred3))
    
mlpcpred3 = mlpc3.predict_proba(test_data3)
mlpcpred3 = mlpcpred3.round(3)
print ("Prediction accuracy: {}".format(mlpcpred3))
    
