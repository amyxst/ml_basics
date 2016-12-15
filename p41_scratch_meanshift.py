import matplotlib.pyplot as plt
from matplotlib import style
style.use('ggplot')
import numpy as np

X = np.array([[1,2],
			[1.5,1.8],
			[5,8],
			[8,8],
			[1,0.6],
			[9,11],
			[8,2],
			[10,2],
			[9,3]])

# plt.scatter(X[:,0],X[:,1], s=150)
# plt.show()

colors = 10*["g", "r", "c", "b", "k"]

class Mean_Shift:
	def __init__(self, radius=4):
		self.radius = radius

	# can have max iter and tol, but scikit learn doesn't include it
	def fit(self, data):
		centroids = {}

		# start by creating centroid at each data point
		for i in range(len(data)):
			centroids[i] = data[i]

		while True:
			new_centroids = []
			for i in centroids:
				in_bandwidth = []
				centroid = centroids[i]
				# see the format of a centroid
				print("working with centroid {}".format(centroid))
				for d in data:
					if np.linalg.norm(d-centroid) <= self.radius:
						in_bandwidth.append(d)
				# mean vector out of all the vectors
				new_centroid = np.average(in_bandwidth, axis=0)
				print("new_cent {}".format(new_centroid))
				new_centroids.append(tuple(new_centroid))

			# as we get convergence, will start getting same centroids
			uniques = sorted(list(set(new_centroids))) # can use set on tuples

			prev_centroids = dict(centroids) # copying centroids dict
			centroids = {}

			for i in range(len(uniques)):
				centroids[i] = np.array(uniques[i])

			optimized = True

			for i in centroids:
				if not np.array_equal(centroids[i], prev_centroids[i]):
					optimized = False

				if not optimized:
					break

			if optimized:
				break

		self.centroids = centroids

	def predict(self, data):
		pass

clf = Mean_Shift()
clf.fit(X)

centroids = clf.centroids

plt.scatter(X[:,0],X[:,1], s=150)
for c in centroids:
	plt.scatter(centroids[c][0], centroids[c][1], color="k", marker="*", s=150)
plt.show()
