import matplotlib.pyplot as plt
from matplotlib import style
style.use('ggplot')
import numpy as np
from sklearn.cluster import KMeans
from sklearn import preprocessing, cross_validation
import pandas as pd

df = pd.read_excel('titanic.xls')
df.drop(['body', 'name'], 1, inplace=True)
df.apply(pd.to_numeric, errors='ignore')
# df = pd.to_numeric()
df.fillna(0, inplace=True)

print(df.head())

def handle_non_numerical_data(df):
	columns = df.columns.values # column headings

	for column in columns:
		text_digit_vals = {}
		def convert_to_int(val):
			return text_digit_vals[val]

		if df[column].dtype != np.int64 and df[column].dtype != np.float64:
			column_contents = df[column].values.tolist()
			unique_elements = set(column_contents)
			x = 0
			for unique in unique_elements:
				if unique not in text_digit_vals:
					text_digit_vals[unique] = x
					x += 1

			df[column] = list(map(convert_to_int, df[column]))

	return df

df = handle_non_numerical_data(df)
# print(df.head())

df.drop(['boat'], 1)

X = np.array(df.drop(['survived'], 1).astype(float))
X = preprocessing.scale(X)
print("X: {}".format(X))
y = np.array(df['survived'])

clf = KMeans(n_clusters=2)
clf.fit(X)
labels = clf.labels_
print("labels: {}".format(clf.labels_))

# label method
c1 = 0
for i in range(len(labels)):
	if labels[i] == y[i]:
		c1 += 1

print("c1: {}".format(c1))

# predicting each point method
correct = 0
for i in range(len(X)):
	predict_me = np.array(X[i].astype(float))
	# print("before reshape {}".format(predict_me))
	predict_me = predict_me.reshape(-1, len(predict_me))
	# print("after reshape {}".format(predict_me))
	prediction = clf.predict(predict_me)
	# print("prediction: {}".format(prediction))
	if prediction[0] == y[i]:
		correct += 1

print("correct: {}".format(correct))
print(correct/len(X)) # keeps printing the reciprocal as clusters are assigned randomly
