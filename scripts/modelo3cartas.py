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

dtc3cards = os.path.join(THIS_FOLDER, 'dtc3cards.joblib')
knn3cards = os.path.join(THIS_FOLDER, 'knn3cards.joblib')
gnb3cards = os.path.join(THIS_FOLDER, 'gnb3cards.joblib')
mlpc3cards = os.path.join(THIS_FOLDER, 'mlpc3cards.joblib')

'''
Remember to uncomment the commented lines and delete the test_data line in order to pass the values
as the paratmethers of the function

'''


# def model3cartas (s1, c1, s2, c2, s3, c3):
test_data2 = [[s1, c1, s2, c2, s3, c3]]
print(test_data2)

dtc2 = load(dtc3cards)
knn2 = load(knn3cards)
gnb2 = load(gnb3cards)
mlpc2 = load(mlpc3cards)

    
dtcpred2 = dtc2.predict_proba(test_data2)
dtcpred2 = dtcpred2.round(3)
print ("Prediction accuracy: {}".format(dtcpred2))
    
knnpred2 = knn2.predict_proba(test_data2)
knnpred2 = knnpred2.round(3)
print ("Prediction accuracy: {}".format(knnpred2))
    
gnbpred2 = gnb2.predict_proba(test_data2)
gnbpred2 = gnbpred2.round(3)
print ("Prediction accuracy: {}".format(gnbpred2))
    
mlpcpred2 = mlpc2.predict_proba(test_data2)
mlpcpred2 = mlpcpred2.round(3)
print ("Prediction accuracy: {}".format(mlpcpred2))
    
