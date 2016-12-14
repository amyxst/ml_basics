import numpy as np
from sklearn import preprocessing, cross_validation, neighbors
import pandas as pd

df = pd.read_csv('breast-cancer-wisconsin.data')
df.replace('?', -9999, inplace=True) # will allow those points to be treated as outliers (sometimes a large portion of data may be missing)
df.drop(['id'], 1, inplace=True) # not useful for our algorithm, messes with KNN greatly

X = np.array(df.drop(['class'], 1))
y = np.array(df['class'])

X_train, X_test, y_train, y_test = cross_validation.train_test_split(X,y,test_size=0.2) # 20%

clf = neighbors.KNeighborsClassifier()
clf.fit(X_train, y_train)

accuracy = clf.score(X_test, y_test)
print(accuracy)

example_measures = np.array([4,2,1,1,1,2,3,2,1])
example_measures = example_measures.reshape(1,-1)
prediction = clf.predict(example_measures)
print(prediction)