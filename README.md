# Implementation-of-SVM-For-Spam-Mail-Detection

## AIM:
To write a program to implement the SVM For Spam Mail Detection.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import the required packages.
2. Import the dataset to operate on
3. Split the dataset.
4. Predict the required output.

## Program:
```
/*
Program to implement the SVM For Spam Mail Detection..
Developed by: SASINTHAR P
RegisterNumber:  212223230199
*/


import chardet
file='/content/spam.csv'
with open(file,'rb') as rawdata:
  result = chardet.detect(rawdata.read(100000))
result


import pandas as pd
data=pd.read_csv('/content/spam.csv',encoding='Windows-1252')

data.head()

data.info()

data.isnull().sum()

x=data["v2"].values
y=data["v1"].values

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)

from sklearn.feature_extraction.text import CountVectorizer
cv=CountVectorizer()

x_train=cv.fit_transform(x_train)
x_test=cv.transform(x_test)

from sklearn.svm import SVC
svc=SVC()
svc.fit(x_train,y_train)
y_pred=svc.predict(x_test)
y_pred

from sklearn import metrics
accuracy=metrics.accuracy_score(y_test,y_pred)
accuracy
```
## Output:
# result output:

![image](https://github.com/user-attachments/assets/d4ed74ee-011d-44d2-b43e-798da6e767c2)

# data head():
![image](https://github.com/user-attachments/assets/c3e43887-bfeb-4ee7-8f76-d9d9db3880b9)
# data info:
![image](https://github.com/user-attachments/assets/7775936d-e549-4883-9a64-c61e9a64c04b)
# data.isnull().sum():
![image](https://github.com/user-attachments/assets/0db8df00-4c75-4237-8988-a45f0dc4ca45)
# Y_prediction:
![image](https://github.com/user-attachments/assets/e385495c-5b50-41b8-bb24-d1790b6a0ab1)
# accuracy:
![image](https://github.com/user-attachments/assets/a24c2c2e-dfc8-4588-a46c-f1fe2aa6df1f)

## Result:
Thus the program to implement the SVM For Spam Mail Detection is written and verified using python programming.
