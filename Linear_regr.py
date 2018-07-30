import matplotlib.pyplot as plt
import numpy as np
from sklearn import linear_model

filename = 'fl_fant_5.csv'
data = np.loadtxt(filename, delimiter=',')
X = data
T = data


lot_set_x_train = X[:100] / 100
lot_set_x_test = X[101:170] / 100

lot_set_t_train = T[:100] / 100
lot_set_t_test = T[101:170] / 100 
#[64:129]

#Create linear regression object

regr = linear_model.LinearRegression()

#Train the model using the training sets
regr.fit(lot_set_x_train, lot_set_t_train)

#The coefficients
print('Coefficients: \n', regr.coef_)
#The Mean squared error
print("Mean square error: %.2f"
	% np.mean((regr.predict(lot_set_x_test) - lot_set_t_test) ** 2))
# Explained variance score: 1 is perfect prediction
print('Variance score: %.2f' % regr.score(lot_set_x_test, lot_set_t_test))


#plot outputs
plt.scatter(lot_set_x_test, lot_set_t_test, color='red')
plt.plot(lot_set_x_test, regr.predict(lot_set_t_test), color='blue',
		linewidth=1)
		
#plt.xticks(())
#plt.yticks(())

plt.show()

#print("Mean square error: %.2f" (regr.predict(lot_set_x_test)))