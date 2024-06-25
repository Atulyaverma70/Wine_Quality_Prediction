import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn.metrics import mean_absolute_error
from sklearn.svm import SVR
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import SGDRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
import joblib
import seaborn as sns
import matplotlib.pyplot as plt

# Load the data
data = pd.read_csv('WineQuality.csv')
print(data.head())

#drop the unnamed column
data = data.drop('Unnamed: 0', axis=1)

# Data Preprocessing
# Check for missing values
print(data.isnull().sum())

# Check for duplicate data
print(data.duplicated().sum())

# Check for the distribution of the target variable
print(data['quality'].value_counts())

#labelencode the Type column
labelencoder = LabelEncoder()
data['Type'] = labelencoder.fit_transform(data['Type'])
print(data.head())


#build
# Split the data into features and target variable
X = data.drop('quality', axis=1)
y = data['quality']

# Split the data into training and testing data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize the data
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

#Logistic Regression(Multiclass 1-10)
# Instantiate the model
lr = LogisticRegression(max_iter=1000)

# Fit the model
lr.fit(X_train, y_train)

# Make predictions
y_pred = lr.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
print("Logistic Regression")
print('Mean Squared Error:', mse)
print('Root Mean Squared Error:', rmse)
print('R-squared:', r2)
print('Mean Absolute Error:', mae)
print('\n')



#Decision Tree
# Instantiate the model
dt = DecisionTreeRegressor()

# Fit the model
dt.fit(X_train, y_train)

# Make predictions
y_pred = dt.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
print("Decision Tree")
print('Mean Squared Error:', mse)
print('Root Mean Squared Error:', rmse)
print('R-squared:', r2)
print('Mean Absolute Error:', mae)
print('\n')

#neural network
# Instantiate the model
ann = MLPRegressor(max_iter=1000)

# Fit the model
ann.fit(X_train, y_train)

# Make predictions
y_pred = ann.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
print("Artificial Neural Network")
print('Mean Squared Error:', mse)
print('Root Mean Squared Error:', rmse)
print('R-squared:', r2)
print('Mean Absolute Error:', mae)
print('\n')




#Random Forest
# Instantiate the model
rf = RandomForestRegressor()

# Fit the model
rf.fit(X_train, y_train)


# Make predictions
y_pred = rf.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
print("Random Forest")
print('Mean Squared Error:', mse)
print('Root Mean Squared Error:', rmse)
print('R-squared:', r2)
print('Mean Absolute Error:', mae)
print('\n')


#K-Nearest Neighbors
# Instantiate the model
knn = KNeighborsRegressor()

# Fit the model
knn.fit(X_train, y_train)

# Make predictions
y_pred = knn.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
print("K-Nearest Neighbors")
print('Mean Squared Error:', mse)
print('Root Mean Squared Error:', rmse)
print('R-squared:', r2)
print('Mean Absolute Error:', mae)
print('\n')



# Support Vector Machine
# Instantiate the model
svr = SVR()

# Fit the model
svr.fit(X_train, y_train)

# Make predictions
y_pred = svr.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
print("Support Vector Machine")
print('Mean Squared Error:', mse)
print('Root Mean Squared Error:', rmse)
print('R-squared:', r2)
print('Mean Absolute Error:', mae)
print('\n')

# Gradient Boosting Regressor
# Instantiate the model
gbr = GradientBoostingRegressor()

# Fit the model
gbr.fit(X_train, y_train)

# Make predictions
y_pred = gbr.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
print("Gradient Boosting Regressor")
print('Mean Squared Error:', mse)
print('Root Mean Squared Error:', rmse)
print('R-squared:', r2)
print('Mean Absolute Error:', mae)
print('\n')


# Naive Bayes
# Instantiate the model
nb = GaussianNB()

# Fit the model
nb.fit(X_train, y_train)

# Make predictions
y_pred = nb.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
print("Naive Bayes")
print('Mean Squared Error:', mse)
print('Root Mean Squared Error:', rmse)
print('R-squared:', r2)
print('Mean Absolute Error:', mae)
print('\n')

# Stochastic Gradient Descent
# Instantiate the model
sgd = SGDRegressor()

# Fit the model
sgd.fit(X_train, y_train)

# Make predictions
y_pred = sgd.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
print("Stochastic Gradient Descent")
print('Mean Squared Error:', mse)
print('Root Mean Squared Error:', rmse)
print('R-squared:', r2)
print('Mean Absolute Error:', mae)
print('\n')


#save all the models, with the names

joblib.dump(lr, 'lr_model.pkl')
joblib.dump(dt, 'dt_model.pkl')
joblib.dump(ann, 'ann_model.pkl')
joblib.dump(rf, 'rf_model.pkl')
joblib.dump(knn, 'knn_model.pkl')
joblib.dump(svr, 'svr_model.pkl')
joblib.dump(gbr, 'gbr_model.pkl')
joblib.dump(nb, 'nb_model.pkl')
joblib.dump(sgd, 'sgd_model.pkl')

#correlation plot

plt.figure(figsize=(12, 8))
sns.heatmap(data.corr(), annot=True, cmap='viridis')
plt.title('Correlation Plot')
plt.show()