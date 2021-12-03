import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn import tree
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor 
from sklearn.model_selection import cross_val_score

# Importing the dataset

def load_data(file_name):
    return pd.read_csv(file_name)
sin_data = load_data('sin_data.csv')

def look_data():
    print(sin_data.head())
    print('-----------------------------------------------------')
    print(sin_data.tail())
    print('-----------------------------------------------------')
    print(sin_data.describe())
    print('-----------------------------------------------------')

    sin_data.plot(x='x_values', y='y_values', kind='scatter')
    plt.show()

# Now we will split the dataset into training and testing sets, we don't need to stratify the data!
train_test, test_set = train_test_split(sin_data, test_size=0.2, random_state=42)

'''
We will use the training set to train our model, and the test set to test our model.
'''

def copy_data():
    return train_test.drop(['y_values'], axis=1)

def separate_data():
    return train_test['y_values'].copy()

#Now we can train our data by using the training set and some model selection techniques.

def train_linear():
    lin_reg = LinearRegression()
    lin_reg.fit(copy_data(), separate_data())
    some_data = copy_data().iloc[:5]
    some_labels = separate_data().iloc[:5]
    print('Predictions:', lin_reg.predict(some_data))
    print('Labels:', list(some_labels))
    print('-----------------------------------------------------')
    sin_predictions = lin_reg.predict(copy_data())
    lmse = mean_squared_error(list(separate_data()), sin_predictions)
    print('Linear Model MSE:', lmse)
    print('-----------------------------------------------------')
    rmse = np.sqrt(lmse)
    print('RMSE:', rmse)
    plt.plot(train_test['x_values'], train_test['y_values'], 'r.', label='Real Data')
    plt.plot(copy_data(), list(sin_predictions), 'b-', label='Predictions')
    plt.legend()
    plt.show()
    #using the same data, we can see that the predictions are too far off, so we will use a decision tree to improve the predictions.

def train_decision():
    tree_reg = DecisionTreeRegressor()
    tree_reg.fit(copy_data(), separate_data())
    some_data = copy_data().iloc[:5]
    some_labels = separate_data().iloc[:5]
    print('Predictions:', tree_reg.predict(some_data))
    print('Labels:', list(some_labels))
    print('-----------------------------------------------------')
    sin_predictions = tree_reg.predict(copy_data())
    lmse = mean_squared_error(list(separate_data()), sin_predictions)
    print('Decision Tree Model MSE:', lmse)
    print('-----------------------------------------------------')
    rmse = np.sqrt(lmse)
    print('RMSE:', rmse)
    plt.plot(train_test['x_values'], train_test['y_values'], 'r.', label='Real Data')
    plt.plot(copy_data(), list(sin_predictions), 'b. ', label='Predictions')
    plt.legend()
    plt.show()

    scores = cross_val_score(tree_reg, copy_data(), separate_data(), scoring='neg_mean_squared_error', cv=10)
    tree_rmse_scores = np.sqrt(-scores)
    print('Cross Validation Scores:', tree_rmse_scores)
    print('-----------------------------------------------------')
    print('Mean Cross Validation Score:',  tree_rmse_scores.mean())
    print('-----------------------------------------------------')
    print('Standard Deviation:', tree_rmse_scores.std())
    print('-----------------------------------------------------')


    #using the same data, we can see that the predictions are too far off, so we will use a random forest to improve the predictions.

def train_forest():
    forest_reg = RandomForestRegressor()
    forest_reg.fit(copy_data(), separate_data())
    some_data = copy_data().iloc[:5]
    some_labels = separate_data().iloc[:5]
    print('Predictions:', forest_reg.predict(some_data))
    print('Labels:', list(some_labels))
    print('-----------------------------------------------------')
    sin_predictions = forest_reg.predict(copy_data())
    lmse = mean_squared_error(list(separate_data()), sin_predictions)
    rmse = np.sqrt(lmse)
    print('Random Forest Model RMSE:', rmse)
    print('-----------------------------------------------------')
    plt.plot(train_test['x_values'], train_test['y_values'], 'r.', label='Real Data')
    plt.plot(copy_data(), list(sin_predictions), 'b. ', label='Predictions')
    plt.legend()
    plt.grid(True)
    plt.show()

def evaluate_model():
    forest_reg = RandomForestRegressor()
    Xtest = test_set.drop(['y_values'], axis=1)
    ytest = test_set['y_values'].copy()
    forest_reg.fit(Xtest, ytest)
    some_data = Xtest.iloc[:5]
    some_labels = ytest.iloc[:5]
    print('Predictions:', forest_reg.predict(some_data))
    print('Labels:', list(some_labels))
    print('-----------------------------------------------------')
    print('Random Forest Regression')
    print('-----------------------------------------------------')
    y_pred = forest_reg.predict(Xtest)
    rmse = np.sqrt(mean_squared_error(ytest, y_pred))
    print('RMSE:', rmse)
    print('-----------------------------------------------------')
    plt.plot(test_set['x_values'], test_set['y_values'], 'r.', label='Real Data')
    plt.plot(Xtest, list(y_pred), 'b. ', label='Predictions')
    plt.grid(True)
    plt.legend()
    plt.show()

def predict(x):
    forest_reg = RandomForestRegressor()
    forest_reg.fit(copy_data(), separate_data())
    return forest_reg.predict(x)

  

if __name__ == "__main__":
    #look_data()
    '''print(train_test.head())
    print('-----------------------------------------------------')
    print(test_set.head())'''
    #train_linear()
    #train_decision()
    #train_forest()
    #evaluate_model()
    print(predict([[1]]))
