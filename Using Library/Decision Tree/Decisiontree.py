from sklearn import tree
import pandas as pd 
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score , confusion_matrix

dataset = pd.read_csv("ml_assi2_dataset.csv")
encoded_data = dataset.apply(preprocessing.LabelEncoder().fit_transform)

X = encoded_data.iloc[:,1:5]
Y = encoded_data.iloc[:,5]
trainx,testx,trainy,testy = train_test_split(X,Y,test_size=0.25,random_state=25)

model = tree.DecisionTreeClassifier()
model.fit(trainx,trainy)
predicty = model.predict(testx)
print(accuracy_score(testy,predicty))
print(confusion_matrix(testy,predicty))