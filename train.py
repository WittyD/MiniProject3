from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pandas as pd
data=pd.read_csv('E:\\train\\witty.csv')
false_data=pd.read_csv('E:\\train\\witty_false.csv')
data=data.append(false_data,ignore_index=True)
x = data.iloc[:,:14].values
y = data.iloc[:, -1].values
print(data)
# x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.25, random_state = 0)
# classifier = LogisticRegression(random_state = 0,max_iter=7600)
# classifier.fit(x_train, y_train)
# y_pred = classifier.predict(x_test)
# print ("Accuracy : ", accuracy_score(y_test, y_pred))
