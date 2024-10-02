from sklearn.svm import SVR
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
# nacitanie dat
excel_data = pd.read_excel('build.xlsx', sheet_name=0)
data = pd.DataFrame(excel_data, columns=['x_data', 'y_data'])
# prevedenie stlpcov na numpy array
x_data = data.iloc[:, 0].values
y_data = data.iloc[:, 1].values


# transformacia z 1d pola na 2d pole (tak si pyta model SVR), y-ove netreba - maju dobry rozmer
x_data = np.reshape(x_data, (-1, 1))
# x-ove data pre regresnu krivku

'''
x_reg = np.arange(0, max(x_data) + 1).reshape(-1, 1)

# instancia modelu
svr = SVR(kernel="linear")  # "poly" pre polynomicku regresiu
# naucenie modelu
svr.fit(x_data, y_data)
# predikovanie y-hodnoty, ktora je zavisla od x
y_reg = svr.predict(x_reg)

# vypis do grafu
plt.title("Data and regress line")
plt.plot(y_reg, x_reg, color='r', label="Regress line")
plt.scatter(x_data, y_data, s=1, color='k', label="Data")

plt.legend()
plt.xlim(0, max(x_data))

plt.show()
'''

#doing numeral prediction

scaler = StandardScaler()
scaler.fit(x_data[::])
x_data[::] = scaler.transform(x_data[::]) 

x_train = x_data
y_train = y_data
'''
x_train, x_test, y_train, y_test = train_test_split(
        x_data, y_data, test_size=0.25, random_state=0)
'''

#skusanie linearRegression:

svr = LinearRegression()

grid_search_params_svr = {
    'fit_intercept': [True, False],
    'normalize': [True, False],
    'copy_X': [True, False],
    'n_jobs': [1,2,3,4,5,-1],
    'positive': [True, False],
    #'gamma': ['scale','auto']
   }

gs = GridSearchCV(svr, grid_search_params_svr, cv=3, scoring='r2')
gs.fit(x_train, y_train)
print(gs.best_estimator_)
print(gs.best_score_)


gs = GridSearchCV(svr, grid_search_params_svr, cv=3, scoring='neg_root_mean_squared_error')
gs.fit(x_train, y_train)
print(gs.best_estimator_)
print(gs.best_score_)
svr = LinearRegression()


'''
#vysledky z linearregression:
LinearRegression(n_jobs=1)
0.26438885021898206
LinearRegression(n_jobs=1, normalize=True, positive=True)
-1647.5512984661002
'''

svr = SVR()

grid_search_params_svr = {
    #'C': [150000000,50000000],
    'C': [310000500],
    'kernel': ['linear', 'poly','rbf','sigmoid'],
    'degree': [1,2],
    #'gamma': ['scale','auto']
   }

gs = GridSearchCV(svr, grid_search_params_svr, cv=3, scoring='r2')
gs.fit(x_train, y_train)
print(gs.best_estimator_)
print(gs.best_score_)

grid_search_params_svr = {
    'C': [1048,1049,1050],
    'kernel': ['linear', 'poly','rbf','sigmoid'],
    'degree': [1,2,3,4,5],
    #'gamma': ['scale','auto']
   }
gs = GridSearchCV(svr, grid_search_params_svr, cv=3, scoring='neg_root_mean_squared_error')
gs.fit(x_train, y_train)
print(gs.best_estimator_)
print(gs.best_score_)


'''
vysledky z SVR:
SVR(C=310000500, degree=1, kernel='poly')
0.2811711612306741
SVR(C=1049, degree=1, kernel='poly')
-1686.6295440817728
'''

#plot graph

data = pd.DataFrame(excel_data, columns=['x_data', 'y_data'])
# prevedenie stlpcov na numpy array
x_data = data.iloc[:, 0].values
y_data = data.iloc[:, 1].values
# transformacia z 1d pola na 2d pole (tak si pyta model SVR), y-ove netreba - maju dobry rozmer
x_data = np.reshape(x_data, (-1, 1))

scaler = StandardScaler()
scaler.fit(x_data[::])
x_data[::] = scaler.transform(x_data[::])

x_train = x_data
y_train = y_data

#svr_poly = LinearRegression()
#svr_poly = SVR(C=4229, degree=1, kernel='poly')
svr_poly = SVR(C=310000500, degree=1, kernel='poly')
y_poly = svr_poly.fit(x_train, y_train).predict(x_train)

lw = 1
plt.scatter(x_train, y_train, color='red', label='váhy budov')
#plt.hold('on')
plt.plot(x_train, y_poly, color='cornflowerblue', lw=lw, label='lineárny model')
plt.xlabel('váhy budov odvodené od objemu')
plt.ylabel('počet obyvateľov')
plt.title('Graf regresie (SVR)')
plt.legend()
plt.show()


'''
y_predict = gs.predict(x_test)
mae = metrics.mean_absolute_error(y_test, y_predict)
mse = metrics.mean_squared_error(y_test, y_predict)
r2 = metrics.r2_score(y_test, y_predict)

print("The model performance for testing set")
print("--------------------------------------")
print('MAE is {}'.format(mae))
print('MSE is {}'.format(mse))
print('R2 score is {}'.format(r2))
'''