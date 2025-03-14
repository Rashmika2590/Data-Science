import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split


data = pd.read_csv('cancer.csv')
data.info()

y = data['diagnosis']
print(y.value_counts())

sns.countplot(data = y)
plt.show()

#load X Variables into s Pandas Dataframe with columns
X = data.drop(['id', 'diagnosis', 'Unnamed: 32'], axis=1)
X.head()
print(X.shape)

print(X.isnull().sum)
#we do not have missing values

#split the data
X_train,X_test,Y_train,Y_test = train_test_split(X,y, test_size =0.2)

#fit the value
logModel = LogisticRegression(max_iter=5000)
logModel.fit(X_train,Y_train)
acc_Tr =logModel.score(X_train,Y_train)
acc_Ts =logModel.score(X_test,Y_test)

print('Tr Accuracy - %2.2f :',acc_Tr)
print('Test Accuracy - %2.2f :',acc_Ts)

predictions = logModel.predict(X_test)

df_pred = pd.DataFrame(Y_test.values, columns=['Actual'])
df_pred['Predicted'] = predictions
print(df_pred)
accuracy = accuracy_score(Y_test,predictions)
print('Test Accuracy  =  :',accuracy)

