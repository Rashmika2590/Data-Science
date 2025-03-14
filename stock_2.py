import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
from sklearn.linear_model import LinearRegression
from sklearn import metrics
from sklearn.model_selection import train_test_split

# read dataset
df_Stock = pd.read_csv('./AAPL.csv')
df_Stock.info()

df_Stock = df_Stock.rename(columns={'Close(t)':'Close'})
print(df_Stock.head())
print(df_Stock.tail(5))
print(df_Stock.shape)
print(df_Stock.columns)
df_Stock = df_Stock.drop(columns='Date')
df_Stock = df_Stock.drop(columns='Date_col')

# plot a particular column / feature
colName = 'Close'
df_Stock[colName].plot(figsize=(10, 7))
plt.title("Stock Price", fontsize=17)
plt.ylabel('Price', fontsize=14)
plt.xlabel('Time', fontsize=14)
plt.grid(which="major", color='k', linestyle='-.', linewidth=0.5)
plt.show()

'''
def create_train_test_set(df_Stock, p = 0.8):
    features = df_Stock.drop(columns=['Close_forcast'], axis=1)
    target = df_Stock['Close_forcast']
    
    data_len = df_Stock.shape[0]
    train_split = int(data_len * p)
    X_train, X_test = features[:train_split], features[train_split:]
    Y_train, Y_test = target[:train_split], target[train_split:]
    
    print('Total - ', str(data_len))
    print('Training: ', X_train.shape, Y_train.shape)
    print('Testing: ', X_test.shape, Y_test.shape)
    
    return X_train, X_test, Y_train, Y_test
	
X_train, X_test, Y_train, Y_test = create_train_test_set(df_Stock)
'''

X = df_Stock.drop(columns=['Close_forcast'], axis=1)
y = df_Stock['Close_forcast']
X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
	
	
lr = LinearRegression()
lr.fit(X_train, Y_train)
print('LR Coefficients: \n', lr.coef_)
print('LR Intercept: \n', lr.intercept_)

Y_train_pred = lr.predict(X_train)
Y_test_pred = lr.predict(X_test)
print("Training R-squared: ", round(metrics.r2_score(Y_train, Y_train_pred),2))
print("Testing R-squared: ", round(metrics.r2_score(Y_test, Y_test_pred),2))

df_pred = pd.DataFrame(Y_test.values, columns=['Actual'])
df_pred['Predicted'] = Y_test_pred
print(df_pred)

df_pred[['Actual', 'Predicted']].plot()
plt.show()