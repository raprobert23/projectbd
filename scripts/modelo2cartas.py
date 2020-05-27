
from joblib import dump, load
import os

THIS_FOLDER = os.path.dirname(os.path.abspath(__file__))
import sys

s1 = int(sys.argv[1])
c1 = int(sys.argv[2])
s2 = int(sys.argv[3])
c2 = int(sys.argv[4])


dtc2cards = os.path.join(THIS_FOLDER, 'dtc2cards.joblib')

'''
Remember to uncomment the commented lines and delete the test_data line in order to pass the values
as the paratmethers of the function

'''


# def model2cartas (suit1,suit2,card1,card2):
#test_data = [[suit1,card1,suit2,card2]]
test_data = [[s1, c1, s2, c2]]

dtc = load(dtc2cards)

dtcpred = dtc.predict_proba(test_data)


print(round((dtcpred[0,0]),3))
print(round((dtcpred[0,1]),3))
print(round((dtcpred[0,2]),3))
print(round((dtcpred[0,3]),3))
print(round((dtcpred[0,4]),3))
print(round((dtcpred[0,5]),3))
print(round((dtcpred[0,6]),3))
print(round((dtcpred[0,7]),3))
print(round((dtcpred[0,8]),3))
print(round((dtcpred[0,9]),3))
