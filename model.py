import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn
from sklearn import linear_model
import pickle

data = pd.read_csv('student-mat.csv', sep = ';', encoding = 'utf8')

data = data[['G1','G2','G3']]
predict = 'G3'

X = np.array(data.drop([predict],1))
Y  =np.array(data[predict])

X_train, X_test, Y_train, Y_test = sklearn.model_selection.train_test_split(X,Y, test_size = 0.33, random_state = 42)

regr = linear_model.LinearRegression() 
regr.fit(X_train,Y_train)

#accuracy of the model
accuracy = regr.score(X_test,Y_test)

#coefficients of G1 and G2 respectively
regr.coef_

#intercept made by the line
regr.intercept_

predictions = regr.predict(X_test)

#saving model 
pickle.dump(regr,open('model.pkl','wb'))

