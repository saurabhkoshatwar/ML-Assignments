import pandas as pd 
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt 

dataset = pd.read_csv("ass1data.csv")
X = dataset.iloc[:,0].values.reshape(-1,1)
Y = dataset.iloc[:,1].values.reshape(-1,1)
trainx,testx,trainy,testy = train_test_split(X,Y,test_size=0.25,random_state=0)

model = LinearRegression()
model.fit(trainx,trainy)
predicty = model.predict(testx)
print(predicty)
print("m",model.coef_,"c",model.intercept_)

plt.scatter(trainx,trainy,color='blue')
plt.plot(trainx,model.predict(trainx),color='red')
plt.xlabel('xlabel')
plt.ylabel('ylabel')
plt.title('title')
plt.show()