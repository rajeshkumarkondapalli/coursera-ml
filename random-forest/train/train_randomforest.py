from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


iris = load_iris()
X = iris.data
y = iris.target

#split test and train data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#build model
clf = RandomForestClassifier(n_estimators = 10)

#train classifier
clf.fit(X_train, y_train)

#prediction
predicted = clf.predict(X_test)

#check accuracy
print(accuracy_score(predicted, y_test))

import pickle
with open('./rf.pkl', 'wb') as model_pkl:
    pickle.dump(clf, model_pkl)