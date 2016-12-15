import matplotlib.pyplot as plt
from matplotlib import style
style.use('ggplot')
import numpy as np
from sklearn.cluster import MeanShift
from sklearn import preprocessing, cross_validation
import pandas as pd

pd.options.mode.chained_assignment = None

df = pd.read_excel('titanic.xls')
original_df = pd.DataFrame.copy(df)

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

X = np.array(df.drop(['survived'], 1).astype(float))
X = preprocessing.scale(X)
print("X: {}".format(X))
y = np.array(df['survived'])

clf = MeanShift()
clf.fit(X)

labels = clf.labels_
cluster_centers = clf.cluster_centers_

original_df['cluster_group'] = np.nan

# label each point with cluster from meanshift
for i in range(len(X)):
	original_df['cluster_group'].iloc[i] = labels[i]

# number of clusters identified by meanshift in the dataset
n_clusters_ = len(np.unique(labels))

# dict to keep track of survival rate of each cluster
survival_rates = {}

# for each cluster
# note: previously we assumed each cluster maps to survival rate, now we're seeing how true that is
for i in range(n_clusters_):
	temp_df = original_df[(original_df['cluster_group']==float(i))]
	print("temp {}".format(temp_df.head()))
	survival_cluster = temp_df[ (temp_df['survived']==1) ]
	# survival rate in that particular cluster
	survival_rate = len(survival_cluster)/len(temp_df)
	survival_rates[i] = survival_rate

print(survival_rates)

# will see that depending on the cluster group, actually clustered by mostly ticket class, and survival rate also correlates
print(original_df[(original_df['cluster_group']==1)].describe())

# see the survival rate of first class passengers in 
cluster_0 = original_df[ (original_df['cluster_group']==0) ]
cluster_0_fc = cluster_0[ (cluster_0['pclass']==1) ]
cluster_0_fc.describe()


'''
key insights:
- groups clustered largely on fare
- also clustered based on survival rate
'''