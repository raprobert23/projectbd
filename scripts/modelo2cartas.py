
from joblib import dump, load
import os

THIS_FOLDER = os.path.dirname(os.path.abspath(__file__))
import sys

s1 = int(sys.argv[1])
c1 = int(sys.argv[2])
s2 = int(sys.argv[3])
c2 = int(sys.argv[4])


dtc2cards = os.path.join(THIS_FOLDER, 'dtc2cards.joblib')
knn2cards = os.path.join(THIS_FOLDER, 'dtc2cards.joblib')
gnb2cards = os.path.join(THIS_FOLDER, 'dtc2cards.joblib')
mlpc2cards = os.path.join(THIS_FOLDER, 'dtc2cards.joblib')

'''
Remember to uncomment the commented lines and delete the test_data line in order to pass the values
as the paratmethers of the function

'''


# def model2cartas (suit1,suit2,card1,card2):
#test_data = [[suit1,card1,suit2,card2]]
test_data = [[s1, c1, s2, c2]]
print(test_data)

dtc = load(dtc2cards)
knn = load(knn2cards)
gnb = load(gnb2cards)
mlpc = load(mlpc2cards)

dtcpred = dtc.predict_proba(test_data)
print("Prediction accuracy: {}".format(dtcpred))

print(dtcpred[0,0])
print(dtcpred[0,1])
print(dtcpred[0,2])
print(dtcpred[0,3])
print(dtcpred[0,4])
print(dtcpred[0,5])
print(dtcpred[0,6])
print(dtcpred[0,7])
print(dtcpred[0,8])
print(dtcpred[0,9])


knnpred = knn.predict_proba(test_data)
print("Prediction accuracy: {}".format(knnpred))

gnbpred = gnb.predict_proba(test_data)
print("Prediction accuracy: {}".format(gnbpred))

mlpcpred = mlpc.predict_proba(test_data)
print("Prediction accuracy: {}".format(mlpcpred))

# model2cartas(1,3,4,8)
