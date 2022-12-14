import pandas as pd
import numpy as np
import random
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error


calories = pd.read_csv("calories.csv")
print(calories.head())

print()

exercise= pd.read_csv("exercise.csv")
print(exercise.head())
print()

data = pd.merge(exercise, calories,  on='User_ID')
print(data.head())
print()


print(data.info())
print()
print(data.describe())

corr_matrix= data.corr()
corr_matrix['Calories'].sort_values(ascending=True)
print(corr_matrix)


data.hist(bins=50,figsize=(12,8))
plt.title('Histogram ploting of features')



sns.pairplot(data)
plt.title('Pairplotting with Entire features') 


data= data[['Duration','Calories']]

print(f'Data features is {data.head()}')
print()
print(f'shape is {data.shape}')

data= data[data['Calories'] < 300]
print()
print(f'shape is {data.shape}')


train_set,test_set=train_test_split(data,test_size=0.3,random_state=42)

print(f'Rows of the traning set is {len(train_set)} and shape is {train_set.shape}\nRows of the test set is {len(test_set)} and shape is {test_set.shape}')



plt.plot(data['Duration'],data['Calories'])
plt.xlabel('Duration')
plt.ylabel('Calories')
plt.title('Calories versus Duration')




plt.plot(data['Duration'], data['Calories'], 'bo')
plt.xlabel('Duration (min)', size = 18)
plt.ylabel('Calories', size = 18)
plt.title('Calories burned vs Duration of Exercise', size = 20)
# plt.show()
    
x_train=np.array(train_set[['Duration',]])
y_train= np.array(train_set[['Calories']])

x_test=np.array(test_set[['Duration',]])
y_test= np.array(test_set[['Calories']])

print()
print(f'Rows of the x_traning set is {len(x_train)} and Rows of the y_traning set is {len(y_train)} shape is {x_train.shape}\nRows of the x_test set is {len(x_test)} and Rows of the y_test set is {len(y_test)} shape is {x_test.shape}')

model=LinearRegression()
model.fit(x_train,y_train)
print()

print(f'Intercept is : {model.intercept_}')
print()
print(f'Coefficent is : {model.coef_}')

print()

y_pred=model.predict(x_test)

R=r2_score(y_test,y_pred)

print(f'R2 score of this model is {R}')

print()

MSE= mean_squared_error(y_test,y_pred)
print(f'Mean Squared Error is {MSE}')\
    
MAE= mean_absolute_error(y_test,y_pred)
print(f'Mean Absolute Error is {MAE}')



## Text 
print(model.intercept_+model.coef_*29.0) 

print(model.intercept_+model.coef_*14.0)


