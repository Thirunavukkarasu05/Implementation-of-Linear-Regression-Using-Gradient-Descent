# Implementation-of-Linear-Regression-Using-Gradient-Descent

## AIM:
To write a program to predict the profit of a city using the linear regression model with gradient descent.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. 
2. 
3. 
4. 

## Program:
```
/*
Program to implement the linear regression using gradient descent.
Developed by: THIRUNAVUKKARASU P
RegisterNumber: 212222040173

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
def linear_regression(X1,y,learning_rate=0.1,num_iters=1000):
  X=np.c_[np.ones(len(X1)),X1]
  theta=np.zeros(X.shape[1]).reshape(-1,1)
  for _ in range(num_iters):
#calculate predictions
    predictions=(X).dot(theta).reshape(-1,1)
#calculate errors
    errors=(predictions-y).reshape(-1,1)
#Update theta using gradient descent
    theta-=learning_rate*(1/len(X1))*X.T.dot(errors)
    return theta
data=pd.read_csv("/content/50_Startups.csv")
data.head()
x=(data.iloc[1:,:-2].values)
x1=x.astype(float)
scaler=StandardScaler()
y=(data.iloc[1:,-1].values).reshape(-1,1)
x1_scaled=scaler.fit_transform(x1)
y1_scaled=scaler.fit_transform(y)
print(x)
print(x1_scaled)
theta=linear_regression(x1_scaled,y1_scaled)
new_data=np.array([165349.2,136897.8,471784.1]).reshape(-1,1)
new_scaled=scaler.fit_transform(new_data)
prediction=np.dot(np.append(1,new_scaled),theta)
prediction=prediction.reshape(-1,1)
pre=scaler.inverse_transform(prediction)
print(prediction)
print(f"Predicted valeue: {pre}")
*/
```

## Output:
![Screenshot 2024-03-04 151026](https://github.com/Thirunavukkarasu05/Implementation-of-Linear-Regression-Using-Gradient-Descent/assets/119291645/df7fb3a6-ed23-4ba3-9ea5-46fd345f7169)
![Screenshot 2024-03-04 151041](https://github.com/Thirunavukkarasu05/Implementation-of-Linear-Regression-Using-Gradient-Descent/assets/119291645/467aa730-2a2f-4dcb-a848-b557399eeb6e)
![Screenshot 2024-03-04 151106](https://github.com/Thirunavukkarasu05/Implementation-of-Linear-Regression-Using-Gradient-Descent/assets/119291645/383e8329-52c7-4ff2-869c-1bbafd65a8c5)



## Result:
Thus the program to implement the linear regression using gradient descent is written and verified using python programming.
