from sklearn.svm import SVR
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression

from sklearn import datasets
import matplotlib.pyplot as plt
import seaborn as sns
# nacitanie dat
excel_data = pd.read_excel('points_func.xlsx', sheet_name=0)
data = pd.DataFrame(excel_data, columns=['x_data', 'y_data'])


# prevedenie stlpcov na numpy array
x_data = data.iloc[:, 0].values
y_data = data.iloc[:, 1].values


# transformacia z 1d pola na 2d pole (tak si pyta model SVR), y-ove netreba - maju dobry rozmer
x_data = np.reshape(x_data, (-1, 1))


'''
# x-ove data pre regresnu krivku
scaler = StandardScaler()
scaler.fit(x_data[::])
x_data[::] = scaler.transform(x_data[::]) 
'''

'''
x_reg = np.arange(0, max(x_data) + 1).reshape(-1, 1)

scaler = StandardScaler()
scaler.fit(x_reg[::])
x_reg[::] = scaler.transform(x_reg[::]) 

# instancia modelu
svr = SVR(C=2622.5, degree=1, gamma='auto',kernel="linear")  # "poly" pre polynomicku regresiu
# naucenie modelu
svr.fit(x_data, y_data)
# predikovanie y-hodnoty, ktora je zavisla od x
y_reg = svr.predict(x_reg)

# vypis do grafu
plt.title("Data and regress line")
plt.plot(y_reg, x_reg, color='b', label="Regress line")
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
        x_data, y_data, test_size=0.20, random_state=0)
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


gs = GridSearchCV(svr, grid_search_params_svr, cv=10, scoring='neg_root_mean_squared_error')
gs.fit(x_train, y_train)
print(gs.best_estimator_)
print(gs.best_score_)



'''
#vysledky z linearregression:
LinearRegression(n_jobs=1)
0.37780157069995896
LinearRegression(n_jobs=1, normalize=True, positive=True)
-1245.636802933412

'''


svr = SVR()
grid_search_params_svr = {
    'C': [1121,1121.5,1122,1122.5,1123],
    'kernel': ['linear', 'poly','rbf','sigmoid'],
    'degree': [1,2,3,4,5],
    'gamma': ['scale','auto']
   }

gs = GridSearchCV(svr, grid_search_params_svr, cv=3, scoring='r2')
gs.fit(x_train, y_train)
print(gs.best_estimator_)
print(gs.best_score_)

grid_search_params_svr = {
    'C': [1081,1082,1083],
    'kernel': ['linear', 'poly','rbf','sigmoid'],
    'degree': [1,2,3,4,5],
    'gamma': ['scale','auto']
   }

gs = GridSearchCV(svr, grid_search_params_svr, cv=6, scoring='neg_root_mean_squared_error')
gs.fit(x_train, y_train)
print(gs.best_estimator_)
print(gs.best_score_)


'''
klesajuca f
SVR(C=1122, degree=1, gamma='auto')
0.44346506746291087
SVR(C=1082, degree=1, gamma='auto')
-1258.1571712221387
'''

#plot graph

data = pd.DataFrame(excel_data, columns=['x_data', 'y_data'])
# prevedenie stlpcov na numpy array
x_data = data.iloc[:, 0].values
y_data = data.iloc[:, 1].values
# transformacia z 1d pola na 2d pole (tak si pyta model SVR), y-ove netreba - maju dobry rozmer
x_data = np.reshape(x_data, (-1, 1))


x_train = x_data
y_train = y_data

svr_poly = SVR(C=1122, degree=1, gamma='auto')
y_poly = svr_poly.fit(x_train, y_train).predict(x_train)

lw = 1
plt.scatter(x_train, y_train, color='darkorange', label='hustota bodov')
#plt.hold('on')
plt.plot(x_train, y_poly, color='cornflowerblue', lw=lw, label='lineárny model')
plt.xlabel('hustota bodov')
plt.ylabel('počet obyvateľov')
plt.title('Graf regresie (SVR)')
plt.legend()
plt.show()


'''
model = LinearRegression()
#gs = GridSearchCV(svr, grid_search_params_svr, cv=4, scoring='neg_root_mean_squared_error')
model.fit(x_data, y_data)
#print(gs.best_estimator_)
#print(gs.best_score_)
y_predict = model.predict(x_data)

r_sq = model.score(x_data, y_data)
print("vysledok", r_sq)
import matplotlib.pyplot as plt
plt.scatter(x_data,y_data)
plt.plot(x_data,y_predict)

plt.show()
'''