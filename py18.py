import numpy as np
from math import sqrt
import warnings
from collections import Counter
import pandas as pd # pandas used to load in dataset
import random # shuffling the dataset

dataset = {'k':[[1,2],[2,3],[3,1]], 'r':[[6,5],[7,7],[8,6]]}
new_features = [5,7]

# [[plt.scatter(ii[0],ii[1],s=100,color=i) for ii in dataset[i]] for i in dataset]
# plt.scatter(new_features[0],new_features[1],s=100)
# plt.show()

def k_nearest_neighbors(data, predict, k=3):
	if len(data) >= k:
		warnings.warn('K is set to a value less than total voting groups!')

	# not using the previously written euclidean distance formula as it only handles 2 dimensions
	# an alt would be to use numpy sqrt and sum
	# faster alternative below:

	distances = []
	for group in data:
		for features in data[group]:
			euclidean_distance = np.linalg.norm(np.array(features)-np.array(predict))
			distances.append([euclidean_distance, group])

	votes = [i[1] for i in sorted(distances)[:k]] # getting top k closest point groups
	# print(Counter(votes).most_common(1))
	vote_result = Counter(votes).most_common(1)[0][0]
	confidence = Counter(votes).most_common(1)[0][1] / k

	# print('Confidence: {}'.format(confidence))
	return vote_result, confidence

accuracies = []

for i in range(5): # finding avg accuracy over range
	df = pd.read_csv('breast-cancer-wisconsin.data')
	df.replace('?', -99999, inplace=True)
	df.drop(['id'], 1, inplace=True)
	full_data = df.astype(float).values.tolist() # converts each row to list
	random.shuffle(full_data) # shuffling changes the test/train sets

	test_size = 0.4
	train_set = {2:[], 4:[]}
	test_set = {2:[], 4:[]}
	train_data = full_data[:-int(test_size*len(full_data))]
	test_data = full_data[-int(test_size*len(full_data)):]

	for i in train_data:
		train_set[i[-1]].append(i[:-1])

	for i in test_data:
		test_set[i[-1]].append(i[:-1])

	correct = 0
	total = 0

	for group in test_set:
		for data in test_set[group]:
			vote, confidence = k_nearest_neighbors(train_set, data, k=5)
			if vote == group: # scikit learn is using default 5
				correct += 1
			# else:
				# print(confidence)
			total += 1

	# print('Accuracy: {}'.format(correct/total))
	accuracies.append(correct/total)

print(sum(accuracies)/len(accuracies))
