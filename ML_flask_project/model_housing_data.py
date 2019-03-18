import pandas as pd
import numpy as np
from pandas.api.types import is_numeric_dtype
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error, accuracy_score
import warnings
warnings.filterwarnings("ignore")
import pickle
from sklearn.linear_model import LinearRegression


dataset = pd.read_csv("kc_house_data.csv")
dataset = dataset.drop(columns=['id', 'date',
       'sqft_lot', 'floors', 'waterfront', 'view', 'condition', 'grade',
       'sqft_above', 'sqft_basement', 'yr_built', 'yr_renovated', 'zipcode',
       'lat', 'long', 'sqft_living15', 'sqft_lot15'])


X = dataset.iloc[:,1:]
y = dataset.iloc[:,:1]


# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X)

# Import linear regression class
from sklearn.linear_model import LinearRegression
regression = LinearRegression()
regression.fit(X, y)

y_pred_train = regression.predict(X)

# Saving model to disk
pickle.dump(regression, open('model_housing_data.pkl','wb'))
# Loading model to compare the results
model = pickle.load(open('model_housing_data.pkl','rb'))
# print(model.predict([[2,2,2000]]))